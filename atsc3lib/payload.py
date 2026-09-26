"""ATSC 3.0 payload chain: data symbols -> PLP FEC -> Baseband Packets.

This implements the receive side of the BICM chain for a data PLP (A/322
Sections 6, 7, 8):

    data OFDM symbols
      -> FFT, pilot channel estimate, equalise, data-cell extraction
      -> cell pool (Preamble spare cells + data symbols, in cell order)
      -> PLP cell slice -> QAM demap (Annex C) -> BICM de-interleave (6.2)
      -> LDPC (6.1.3) -> BCH (6.1.2.1) -> baseband packet descramble (5.2.3)
      -> Baseband Packets

The cell geometry follows A/322:
- scattered pilots: ``k mod (DX*DY) == DX*(l mod DY)`` (8.1.3.1)
- edge pilots at k = 0 and k = NoC-1 (8.1.5)
- common continual pilots CP8 (Table D.1.1/D.1.3) and the additional CPs
  (Table D.1.4)
- a subframe boundary symbol carries the same DX with DY = 1.

Reference: ATSC A/322:2024-04, Sections 6.2, 6.3, 7.2.6, 8.1.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from . import nuc
from . import spec
from . import l1_signaling
from .bch import BCHCode
from .frequency_interleaver import deinterleave as fi_deinterleave
from .group_interleaver import GroupInterleaver
from .ldpc_exact import (
    ATSC3LDPCExact, NINNER_SHORT, NINNER_NORMAL, RATE_MIN,
)
from .nuc import MODULATION_BITS, QPSK
from .pilot_reference import reference_sequence
from .signaling_fec import scramble_bits
from .cell_interleaver import CellInterleaver
from . import baseband
from . import alp
from . import ip as ip_layer
from . import twisted_block


# QPSK constellation (A/322 Table C.1.1): bit pair (y0,y1) -> (Re,Im).
QPSK_POINTS = np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]) / np.sqrt(2)


def _pilots(noc: int, dx: int, dy: int, l: int, sbs: bool,
            cp_rel: np.ndarray, add_cp=()) -> np.ndarray:
    """Relative carrier indices carrying a pilot in data symbol ``l``."""
    if sbs:
        sp = np.arange(0, noc, dx)
    else:
        sp = np.arange(dx * (l % dy), noc, dx * dy)
    out = np.unique(np.concatenate([
        sp, cp_rel, np.asarray(add_cp, dtype=int), [0, noc - 1]]))
    return out[(out >= 0) & (out < noc)]


def common_cp_relative(noc: int, fft_size: int) -> np.ndarray:
    """Common continual pilots (CP8/CP16/CP32) as relative carrier indices.

    A/322 8.1.4.1: the 8K/16K sets are derived from CP32; ``pilot_tables``
    holds the exact absolute set per FFT size.  ``noc`` selects which fall in
    the configured carrier range.
    """
    from . import pilot_tables
    origin = (spec.NOC_MAX[fft_size] - noc) // 2
    cps = pilot_tables.COMMON_CP[fft_size]
    return np.asarray([c - origin for c in cps if origin <= c < origin + noc],
                      dtype=int)


@dataclass
class DataSymbolCells:
    """Equalised data cells of one data OFDM symbol."""
    cells: np.ndarray
    pilot_coherence: float
    n_carriers: int


def data_symbol_cells(symbol: np.ndarray, fft_size: int, noc: int,
                      dx: int, dy: int, l: int, sbs: bool,
                      amplitude: float = 1.0,
                      add_cp=()) -> DataSymbolCells:
    """Equalised data cells of one data OFDM symbol, in relative-carrier order.

    Estimates the channel from the scattered + continual pilots and divides it
    out, then returns the non-pilot cells.  ``symbol`` is the FFT-length body
    of the symbol (guard interval removed).
    """
    x = np.asarray(symbol, dtype=np.complex128)
    spectrum = np.fft.fftshift(np.fft.fft(x))
    origin = (spec.NOC_MAX[fft_size] - noc) // 2
    spec_shift = fft_size // 2 + (origin - (spec.NOC_MAX[fft_size] - 1) // 2)
    carriers = spectrum[spec_shift:spec_shift + noc]

    pilots = _pilots(noc, dx, dy, l, sbs, common_cp_relative(noc, fft_size),
                     add_cp)
    ref = reference_sequence(int(pilots[-1]) + 1).astype(np.int64)
    known = (amplitude * (1.0 - 2.0 * ref[pilots])).astype(np.complex128)
    h_pilots = carriers[pilots] / known

    all_k = np.arange(noc)
    channel = (np.interp(all_k, pilots, h_pilots.real)
               + 1j * np.interp(all_k, pilots, h_pilots.imag))

    # Coherence of the pilot channel estimate (a decode-quality indicator).
    if len(h_pilots) > 1:
        coh = float(abs(np.sum(h_pilots[:-1] * np.conj(h_pilots[1:])))
                    / max(np.sum(np.abs(h_pilots) ** 2), 1e-30))
    else:
        coh = 0.0

    data_mask = np.ones(noc, dtype=bool)
    data_mask[pilots] = False
    return DataSymbolCells(cells=(carriers / channel)[data_mask],
                           pilot_coherence=coh, n_carriers=noc)


def qpsk_demap_llr(cells: np.ndarray) -> np.ndarray:
    """Max-log LLRs for QPSK, in q-stream order (q_{2*s+i}).

    Convention: LLR > 0 => bit 0 (the oracle's demap; the sign is inverted
    once when scattering into the LDPC decoder).
    """
    z = np.asarray(cells, dtype=np.complex128)
    d2 = np.abs(z[:, None] - QPSK_POINTS[None, :]) ** 2
    sigma2 = max(float(np.mean(d2.min(axis=1))), 1e-9)
    masks = np.array([[(k >> (1 - i)) & 1 for k in range(4)]
                      for i in range(2)], dtype=bool)
    out = np.empty((len(z), 2))
    for i in range(2):
        out[:, i] = (d2[:, masks[i]].min(1) - d2[:, ~masks[i]].min(1)) / sigma2
    return out.ravel()


@dataclass
class FecBlock:
    """Outcome of decoding one FEC block into a Baseband Packet."""
    bits: np.ndarray
    converged: bool
    bch_ok: bool
    bch_syndrome: int
    payload_bits: Optional[np.ndarray]
    packet: Optional[bytes]

    @property
    def ok(self) -> bool:
        return self.converged and self.bch_ok


#: L1D_plp_mod signalling value -> modulation name, restricted to the four
#: NUC modes the receive chain supports (A/322 Table 9.8).
MOD_NAME = {code: name for code, name in nuc.MOD_NAME.items()
            if name in MODULATION_BITS}

#: BCH correctable errors (A/322 6.1.2.1 / Table 6.3).
BCH_T = 12


class DataPlpChain:
    """Receive chain for one data PLP FEC block.

    Supports every QPSK/16QAM/64QAM/256QAM code rate tabulated in A/322
    Annexes A and B, at both frame lengths (Ninner=16200 short frames and
    Ninner=64800 normal frames).  The chain is::

        cells -> NUC/QAM demap (Annex C, 6.3.3)
              -> bit de-interleave (6.2, Annex B)
              -> LDPC (6.1.3) -> BCH (6.1.2.1) -> baseband descramble (5.2.3)
              -> Baseband Packet
    """

    #: The oracle's normalized-min-sum ladder.  alpha=0.75 suits the short
    #: codes' check degrees and is the fallback; 1.0/0.85 are tried first.
    ALPHA_LADDER = (1.0, 0.85, 0.75)

    def __init__(self, mod: str = QPSK, rate: int = RATE_MIN,
                 ninner: int = NINNER_SHORT, max_iterations: int = 100):
        if ninner not in (NINNER_SHORT, NINNER_NORMAL):
            raise NotImplementedError(
                f"only Ninner in ({NINNER_SHORT}, {NINNER_NORMAL}) is "
                f"implemented (got {ninner})")
        if mod not in MOD_NAME.values():
            raise NotImplementedError(f"modulation {mod!r} not supported")
        self.ninner = ninner
        self.rate = rate
        self.mod = mod
        self.mod_bits = MODULATION_BITS[mod]
        self.bch = BCHCode(ninner, BCH_T)
        self.ldpc = ATSC3LDPCExact(rate, n=ninner, max_iterations=max_iterations)
        self.Kldpc = self.ldpc.K
        self.Mouter = self.bch.mouter
        self.kpayload = self.Kldpc - self.Mouter
        self.gi = GroupInterleaver(rate, mod, n=ninner)
        self.order = np.asarray(self.gi.order)
        #: Data cells consumed by one FEC block at this modulation
        #: (A/322 Table 6.14).
        self.cells_per_fec = ninner // self.mod_bits

    def decode_cells(self, cells: np.ndarray, alpha: float = None) -> FecBlock:
        """Decode one FEC block (``cells`` of length ``cells_per_fec``)."""
        q = nuc.demap_llr(cells, self.mod_bits, self.rate)   # q > 0 => bit 0
        llr = np.empty(self.ninner)
        llr[self.order] = -q                                 # decoder: >0 => bit 1
        ladder = (alpha,) if alpha is not None else self.ALPHA_LADDER
        bits, converged = self.ldpc.decode(llr, alpha=ladder[0])
        for a in ladder[1:]:
            if converged:
                break
            bits, converged = self.ldpc.decode(llr, alpha=a)
        if not converged:
            return FecBlock(bits=bits, converged=False, bch_ok=False,
                            bch_syndrome=-1, payload_bits=None, packet=None)
        nouter = bits[:self.Kldpc]
        payload_bits, _nerr, bch_ok = self.bch.decode(nouter)
        packet = scramble_bits(
            np.asarray(payload_bits, dtype=np.uint8)[:self.kpayload])
        return FecBlock(bits=bits, converged=True, bch_ok=bool(bch_ok),
                        bch_syndrome=0 if bch_ok else -1,
                        payload_bits=packet,
                        packet=np.packbits(packet).tobytes())


class QPSKPlpChain(DataPlpChain):
    """Receive chain for a QPSK 2/15 PLP (the RF33 PLP-16 case)."""

    #: Fixed MODCOD: QPSK, rate 2/15 (A/322 Table 6.12).
    QPSK_RATE = 2

    def __init__(self, ninner: int = NINNER_SHORT, rate: int = QPSK_RATE,
                 max_iterations: int = 100):
        if rate != self.QPSK_RATE:
            raise NotImplementedError("only QPSK 2/15 is implemented so far")
        super().__init__(mod=QPSK, rate=rate, ninner=ninner,
                         max_iterations=max_iterations)


@dataclass
class CellPool:
    """The available data cells of one subframe, in cell order.

    ``cells`` is the concatenation of the Preamble's spare cells (subframe 0
    only, after L1-Basic/L1-Detail) and the data symbols' cells, each
    frequency-de-interleaved.  ``symbol_of`` maps each cell to its OFDM symbol
    index (0-based within the subframe; -1 for Preamble spare cells).
    """
    cells: np.ndarray
    symbol_of: np.ndarray
    n_preamble_spare: int
    n_null: int


def build_data_symbol_pool(y: np.ndarray, t0: int, fft_size: int,
                           guard_interval: int, noc: int, dx: int, dy: int,
                           n_data_symbols: int, sbs_symbols=(0,),
                           pattern: str = 'SP4_2', sbs_null: int = None,
                           cred_coeff: int = 0,
                           fi_offset: int = 0) -> CellPool:
    """Cells of a subframe's data symbols, frequency-de-interleaved in order.

    This is the generic (no-Preamble) half of :func:`build_cell_pool`; a
    subframe after the first inherits no Preamble cells, so its pool is exactly
    this.  ``fi_offset`` is the A/322 7.3 frequency-interleaver symbol counter
    origin for the first data symbol: 1 for subframe 0 (the Preamble is symbol
    0), and **0 for every subsequent subframe**, because the counter resets at
    each subframe boundary (A/322 7.3 rule 2; settled by experiment in the
    reference receiver).

    Args:
        y: Main-rate IQ starting at the bootstrap (sample 0).
        t0: Sample index of this subframe's first symbol (guard interval start).
        fft_size, guard_interval, noc: Subframe geometry.
        dx, dy: Scattered pilot pattern spacings.
        n_data_symbols: Symbols in the subframe (data + subframe boundary).
        sbs_symbols: Indices that are subframe boundary symbols.
        pattern: Scattered-pilot pattern (additional continual pilots).
        sbs_null: Null cells per SBS (signalled as L1D_sbs_null_cells); the
            8K cred-0 value is used when None (subframe-0 default).
        cred_coeff: Carrier-reduction coefficient (selects the CP/pilot sets).
        fi_offset: Frequency-interleaver symbol counter origin.

    Returns:
        CellPool for the subframe (``n_preamble_spare`` is 0).
    """
    n_null = spec.SBS_NULL_8K_CRED0 if sbs_null is None else int(sbs_null)
    lo_n = n_null // 2
    hi_n = n_null - lo_n
    add_cp = spec.additional_cp(pattern, fft_size, cred_coeff)

    parts, owner = [], []
    for l in range(n_data_symbols):
        sbs = l in tuple(sbs_symbols)
        start = t0 + (fft_size + guard_interval) * l + guard_interval
        result = data_symbol_cells(
            y[start:start + fft_size], fft_size, noc, dx, dy, l, sbs,
            add_cp=add_cp)
        x = fi_deinterleave(result.cells, fi_offset + l, fft_size)
        if sbs:
            x = x[lo_n:len(x) - hi_n]
        parts.append(x)
        owner.append(np.full(len(x), l, dtype=int))

    return CellPool(cells=np.concatenate(parts),
                    symbol_of=np.concatenate(owner),
                    n_preamble_spare=0, n_null=int(n_null))


def build_cell_pool(y: np.ndarray, t0: int, fft_size: int, guard_interval: int,
                    noc: int, dx: int, dy: int, n_data_symbols: int,
                    sbs_symbols=(0,), preamble_structure: int = 27,
                    l1_cells: int = 484 + 880,
                    pattern: str = None, sbs_null: int = None,
                    preamble_num_symbols: int = 1,
                    preamble_reduced_carriers: int = spec.PREAMBLE_FIRST_CRED,
                    l1b_cells: int = None) -> CellPool:
    """Build the subframe-0 cell pool: Preamble spare cells + data symbols.

    The first Preamble symbol carries L1-Basic at its start and L1-Detail after
    it; any later Preamble symbols carry the rest of L1-Detail (A/322 7.2.5.2).
    ``l1_cells`` is the number of cells consumed by signalling in the *first*
    Preamble symbol; the spare (PLP-available) cells are whatever is left in the
    last Preamble symbol.  Subframe-0 data symbols begin after all Preamble
    symbols, and the frequency-interleaver counter continues from the Preamble
    (the first Preamble symbol is frame symbol 0), so the first data symbol's
    counter origin is ``preamble_num_symbols``.

    Args:
        y: Main-rate IQ starting at the bootstrap (sample 0).
        t0: Sample index of the bootstrap (usually 0 for a frame-aligned ``y``).
        fft_size, guard_interval, noc: Subframe geometry.
        dx, dy: Scattered pilot pattern spacings.
        n_data_symbols: Number of symbols in the subframe (data + SBS).
        sbs_symbols: Indices that are subframe boundary symbols.
        preamble_structure: Bootstrap-decoded structure (for the preamble).
        l1_cells: Cells consumed by L1-Basic + L1-Detail in the first Preamble
            symbol (used when ``l1b_cells`` is None).
        pattern: Scattered-pilot pattern name; defaults to 8K SP4_2.
        sbs_null: Null cells per subframe boundary symbol; 8K cred-0 default.
        preamble_num_symbols: NP, the number of Preamble symbols.
        preamble_reduced_carriers: L1B_preamble_reduced_carriers, the
            cred_coeff used by the Preamble symbols after the first.
        l1b_cells: Number of L1-Basic cells at the start of the first Preamble
            symbol (used to locate L1-Detail); None falls back to ``l1_cells``.

    Returns:
        CellPool for the subframe.
    """
    from .preamble import (
        preamble_symbol_spectrum, estimate_preamble_channel,
        preamble_data_cells, preamble_noc)

    pre = spec.PREAMBLE_STRUCTURE[preamble_structure]

    # First Preamble symbol: cred-4 carriers, FI counter 0.
    sym_pre = y[t0 + pre.gi:t0 + pre.gi + pre.fft]
    carriers = preamble_symbol_spectrum(sym_pre, pre.fft,
                                        noc=preamble_noc(pre.fft))
    h_pre = estimate_preamble_channel(
        carriers, pre.dx, spec.PREAMBLE_PILOT_AMPLITUDE[(pre.fft, pre.gi)])
    x0 = fi_deinterleave(preamble_data_cells(carriers, h_pre, pre.dx), 0,
                         pre.fft)

    if preamble_num_symbols <= 1:
        spare = x0[l1_cells:]
        data_start = t0 + (pre.fft + pre.gi)
    else:
        # L1-Detail fills the first symbol after L1-Basic, then the later
        # Preamble symbols; the leftover of the LAST is spare for PLP data.
        n_l1b = l1_cells if l1b_cells is None else l1b_cells
        remaining = l1_cells - n_l1b
        sym0_detail = min(len(x0) - n_l1b, remaining)
        remaining -= sym0_detail
        spare = np.empty(0, dtype=x0.dtype)
        for k in range(1, preamble_num_symbols):
            start = t0 + (pre.fft + pre.gi) * k + pre.gi
            body = y[start:start + pre.fft].astype(np.complex128)
            n_sym = spec.noc(pre.fft, preamble_reduced_carriers)
            ck = preamble_symbol_spectrum(body, pre.fft, noc=n_sym)
            hk = estimate_preamble_channel(
                ck, pre.dx, spec.PREAMBLE_PILOT_AMPLITUDE[(pre.fft, pre.gi)])
            xk = fi_deinterleave(
                preamble_data_cells(ck, hk, pre.dx), k, pre.fft)
            if k == preamble_num_symbols - 1:
                used = min(len(xk), remaining)
                spare = xk[used:]
            else:
                remaining -= min(len(xk), remaining)
        data_start = t0 + (pre.fft + pre.gi) * preamble_num_symbols

    data = build_data_symbol_pool(
        y, data_start, fft_size, guard_interval, noc, dx, dy,
        n_data_symbols, sbs_symbols=sbs_symbols, pattern=pattern or 'SP4_2',
        sbs_null=sbs_null, fi_offset=preamble_num_symbols)

    return CellPool(cells=np.concatenate([spare, data.cells]),
                    symbol_of=np.concatenate([
                        np.full(len(spare), -1, dtype=int), data.symbol_of]),
                    n_preamble_spare=int(len(spare)), n_null=data.n_null)


@dataclass
class PlpPayload:
    """A decoded PLP payload (subframe 0)."""
    plp_id: int
    fec_blocks: list          # list[FecBlock], one per FEC block
    n_fec: int

    @property
    def n_converged(self) -> int:
        return sum(1 for b in self.fec_blocks if b.ok)

    @property
    def baseband_packets(self) -> list:
        """The decoded Baseband Packet bytes, converged blocks only."""
        return [b.packet for b in self.fec_blocks if b.ok and b.packet]


def decode_subframe0_qpsk_plp(y: np.ndarray, t0: int, fft_size: int,
                              guard_interval: int, noc: int, dx: int, dy: int,
                              n_data_symbols: int, sbs_symbols,
                              preamble_structure: int, l1_cells: int,
                              plp_id: int, plp_start: int, plp_size: int,
                              max_iterations: int = 100) -> PlpPayload:
    """Decode a QPSK 2/15 PLP from subframe 0 (the RF33 PLP-16 shape).

    The PLP's ``plp_size`` cells are taken from the subframe cell pool at
    ``plp_start``; each 16200-bit QPSK FEC block occupies 8100 cells.
    """
    pool = build_cell_pool(y, t0, fft_size, guard_interval, noc, dx, dy,
                           n_data_symbols, sbs_symbols=sbs_symbols,
                           preamble_structure=preamble_structure,
                           l1_cells=l1_cells)
    chain = QPSKPlpChain(max_iterations=max_iterations)
    #: QPSK: 2 bits per cell (A/322 Table 6.14).
    cells_per_block = chain.ninner // MODULATION_BITS[QPSK]
    n_fec = plp_size // cells_per_block
    blocks = []
    for j in range(n_fec):
        lo = plp_start + j * cells_per_block
        blocks.append(chain.decode_cells(pool.cells[lo:lo + cells_per_block]))
    return PlpPayload(plp_id=plp_id, fec_blocks=blocks, n_fec=n_fec)


#: L1D_plp_fec_type -> LDPC codeword length (A/322 Table 9.8).
PLP_NINNER = {
    l1_signaling.FEC_BCH_16K: NINNER_SHORT,
    l1_signaling.FEC_BCH_64K: NINNER_NORMAL,
    l1_signaling.FEC_CRC_16K: NINNER_SHORT,
    l1_signaling.FEC_CRC_64K: NINNER_NORMAL,
    l1_signaling.FEC_16K: NINNER_SHORT,
    l1_signaling.FEC_64K: NINNER_NORMAL,
}


def decode_plp_from_pool(pool: CellPool, plp, max_iterations: int = 100
                         ) -> PlpPayload:
    """Decode one PLP from an already-built subframe cell pool.

    Handles any tabulated QPSK/16QAM/64QAM/256QAM rate at either frame length
    (A/322 Table 9.8 ``fec_type``), the HTI twisted block interleaver (TI mode
    2) and the multi-FEC-block geometry.
    """
    cells = pool.cells[plp.start:plp.start + plp.size]
    mod = MOD_NAME.get(plp.modulation)
    if mod is None:
        raise NotImplementedError(
            f"modulation index {plp.modulation} not tabulated for data PLPs")
    ninner = PLP_NINNER.get(plp.fec_type)
    if ninner is None:
        raise NotImplementedError(
            f"fec_type {plp.fec_type} not tabulated for data PLPs")
    nti = (plp.hti_num_ti_blocks + 1) if plp.ti_mode == 2 else 1
    cell_inter = plp.hti_cell_interleaver or 0
    payload = decode_data_plp(cells, mod, RATE_MIN + plp.code_rate, ninner=ninner,
                              nti=nti, cell_interleaver=cell_inter,
                              max_iterations=max_iterations)
    payload.plp_id = plp.plp_id
    return payload


def decode_subframe0_plp(y: np.ndarray, t0: int, fft_size: int,
                         guard_interval: int, noc: int, dx: int, dy: int,
                         n_data_symbols: int, sbs_symbols,
                         preamble_structure: int, l1_cells: int,
                         plp, max_iterations: int = 100,
                         pattern: str = None, sbs_null: int = None,
                         preamble_num_symbols: int = 1,
                         preamble_reduced_carriers: int = spec.PREAMBLE_FIRST_CRED,
                         l1b_cells: int = None) -> PlpPayload:
    """Decode one PLP of subframe 0 from its L1-Detail ``plp`` config."""
    pool = build_cell_pool(y, t0, fft_size, guard_interval, noc, dx, dy,
                           n_data_symbols, sbs_symbols=sbs_symbols,
                           preamble_structure=preamble_structure,
                           l1_cells=l1_cells, pattern=pattern,
                           sbs_null=sbs_null,
                           preamble_num_symbols=preamble_num_symbols,
                           preamble_reduced_carriers=preamble_reduced_carriers,
                           l1b_cells=l1b_cells)
    return decode_plp_from_pool(pool, plp, max_iterations=max_iterations)


def decode_subframe_plp(y: np.ndarray, t0: int, fft_size: int,
                        guard_interval: int, noc: int, dx: int, dy: int,
                        n_data_symbols: int, sbs_symbols, plp,
                        max_iterations: int = 100, pattern: str = 'SP4_2',
                        sbs_null: int = None, cred_coeff: int = 0,
                        fi_offset: int = 0) -> PlpPayload:
    """Decode one PLP of a subframe after the first (no Preamble, FI reset).

    ``t0`` is this subframe's first symbol (guard-interval start); the frequency
    interleaver counter resets here, so ``fi_offset`` is 0 (A/322 7.3 rule 2).
    """
    pool = build_data_symbol_pool(
        y, t0, fft_size, guard_interval, noc, dx, dy, n_data_symbols,
        sbs_symbols=sbs_symbols, pattern=pattern, sbs_null=sbs_null,
        cred_coeff=cred_coeff, fi_offset=fi_offset)
    return decode_plp_from_pool(pool, plp, max_iterations=max_iterations)


def decode_data_plp(cells: np.ndarray, mod: str, rate: int, ninner: int = NINNER_SHORT,
                    nti: int = 1, n_fec: int = None, cell_interleaver: int = 0,
                    max_iterations: int = 100) -> PlpPayload:
    """Decode a data PLP from its cell slice, honouring the HTI interleaver.

    ``cells`` is the PLP's ``plp_size`` cells in cell order.  When TI mode is
    2 (HTI), the cells pass through the A/322 7.1.5.4 twisted block
    interleaver (``nrows = cells_per_fec``, ``ncols = n_fec``) and, if
    signalled, the A/322 7.1.5.2 cell interleaver before each FEC block is
    decoded.  ``NTI`` (``hti_num_ti_blocks + 1``) splits the slice into
    independent interleaving frames; the cell interleaver's permutation resets
    at each TI block.

    Args:
        cells: PLP cells in cell order.
        mod: QPSK, 16QAM, 64QAM or 256QAM (nuc.MOD_NAME).
        rate: LDPC code rate numerator over 15.
        ninner: LDPC codeword length (16200 or 64800).
        nti: Number of TI blocks (1 for no sub-frame division).
        n_fec: Total FEC blocks; derived from ``cells`` if omitted.
        cell_interleaver: A/322 7.1.5.2 HTI cell interleaver flag.
    """
    chain = DataPlpChain(mod=mod, rate=rate, ninner=ninner,
                         max_iterations=max_iterations)
    cpf = chain.cells_per_fec
    plp_id = -1
    if n_fec is None:
        n_fec = len(cells) // cpf
    per_ti = len(cells) // nti
    ncols = n_fec // nti

    blocks = []
    ci = CellInterleaver(cpf) if cell_interleaver else None
    for ti in range(nti):
        seg = cells[ti * per_ti:(ti + 1) * per_ti]
        mem = np.stack([twisted_block.fec_block(seg, j, cpf, ncols)
                        for j in range(ncols)])
        if ci is not None:
            mem = ci.deinterleave(mem)
        blocks.extend(chain.decode_cells(mem[j]) for j in range(ncols))
    return PlpPayload(plp_id=plp_id, fec_blocks=blocks, n_fec=n_fec)


@dataclass
class DecodedStreams:
    """The network-layer streams recovered from one PLP's Baseband Packets.

    Fields:
        packets: the de-encapsulated ALP packets (A/330 5.1).
        datagrams: UDP datagrams after IPv4 reassembly (RFC 791/768).
        lls: Low-Level Signaling tables (A/331 6.1), keyed by table_id.
        alp_stats: ALP walk bookkeeping (resyncs etc.), never hidden.
        ip_stats: IPv4 reassembly bookkeeping.
    """
    packets: list
    datagrams: list
    lls: list
    alp_stats: alp.AlpStats
    ip_stats: ip_layer.IpStats


def decode_streams(payload: PlpPayload) -> DecodedStreams:
    """Run Baseband Packets -> ALP -> IPv4/UDP -> LLS (A/322 5.2, A/330 5).

    Each converged FEC block yields one fixed-length Baseband Packet; their
    payloads are concatenated and the Baseband Packet pointers seed the ALP
    resynchronisation boundaries.
    """
    bpkts = [baseband.split_baseband_packet(p)
             for p in payload.baseband_packets]
    stream, boundaries = baseband.payload_stream(bpkts)
    alp_packets, alp_stats = alp.parse_alp(stream, boundaries)

    reasm = ip_layer.IpReassembler()
    datagrams = []
    for pkt in alp_packets:
        if pkt.packet_type == alp.PT_IPV4:
            datagrams.extend(reasm.feed(pkt.payload))
    lls = [ip_layer.parse_lls(d.payload) for d in datagrams
           if ip_layer.is_lls(d)]
    lls = [t for t in lls if t is not None]
    return DecodedStreams(packets=alp_packets, datagrams=datagrams, lls=lls,
                          alp_stats=alp_stats, ip_stats=reasm.stats)
