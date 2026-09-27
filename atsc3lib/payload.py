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
from .signaling_fec import scramble_bits, randomizer_bits
from .cell_interleaver import CellInterleaver
from . import baseband
from . import alp
from . import ip as ip_layer
from . import twisted_block


# QPSK constellation (A/322 Table C.1.1): bit pair (y0,y1) -> (Re,Im).
QPSK_POINTS = np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]) / np.sqrt(2)

#: L1D_plp_TI_mode values (A/322 Table 9.8 / l1_signaling.TI_MODE).
TI_NONE, TI_CTI, TI_HTI = 0, 1, 2


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


def _relative_carriers(symbol: np.ndarray, fft_size: int,
                       noc: int) -> np.ndarray:
    """FFT one symbol body and slice out its ``noc`` relative carriers.

    A/322 indexes carriers absolutely with the ``NoCmax`` set centred on DC, so
    relative carrier 0 sits at absolute ``(NoCmax - noc)//2``.
    """
    spectrum = np.fft.fftshift(np.fft.fft(np.asarray(symbol, dtype=np.complex128)))
    origin = (spec.NOC_MAX[fft_size] - noc) // 2
    shift = fft_size // 2 + (origin - (spec.NOC_MAX[fft_size] - 1) // 2)
    return spectrum[shift:shift + noc]


def scattered_pilots(noc: int, dx: int, dy: int, l: int,
                     sbs: bool) -> np.ndarray:
    """Relative indices of the scattered pilots of data symbol ``l``.

    A/322 8.1.3.1: ``k mod (DX*DY) == DX*(l mod DY)``, with an edge pilot at
    k = 0 and k = NoC-1 (8.1.5).  A subframe boundary symbol uses DY = 1
    (8.1.3.1).  These are the evenly spaced pilots whose phase drift under a
    timing error; the common continual pilots are deliberately excluded so the
    fine-timing statistic sees one uniform lattice.
    """
    sp = (np.arange(0, noc, dx) if sbs
          else np.arange(dx * (l % dy), noc, dx * dy))
    return np.unique(np.concatenate([sp, [0, noc - 1]]))


def scattered_pilot_coherence(carriers: np.ndarray, noc: int, dx: int, dy: int,
                              l: int, sbs: bool) -> float:
    """Coherence of the scattered-pilot channel estimate of one symbol.

    A/322 8.1.2: the scattered pilots carry the known reference values, so
    ``received / known`` is the channel response, smooth along ``k``.  Scoring

        C = |sum_j q_j conj(q_{j+1})| / sum_j |q_j|^2

    over consecutive scattered pilots is ~1 for the right FFT window and falls
    as the window slips, because a timing error ``tau`` rotates by
    ``2*pi*tau*(DX*DY)/N`` per pilot.  It fits no parameter, so the search that
    maximises it cannot inflate it.
    """
    pk = scattered_pilots(noc, dx, dy, l, sbs)
    ref = reference_sequence(int(pk[-1]) + 1).astype(np.int64)
    q = carriers[pk] / (1.0 - 2.0 * ref[pk])
    num = np.abs(np.sum(q[:-1] * np.conj(q[1:])))
    den = np.sum(np.abs(q) ** 2)
    return float(num / max(den, 1e-30))


def fine_timing(y: np.ndarray, t0: int, fft_size: int, gi: int, noc: int,
                dx: int, dy: int, symbol: int, sbs: bool,
                span: int = spec.FINE_TIMING_SPAN) -> "FineTiming":
    """Refine a subframe's FFT-window origin by scattered-pilot coherence.

    The bootstrap anchors the frame to within one guard interval; the exact
    sampling instant still has a residual of a few samples, and a residual of
    a few percent of the GI de-correlates the scattered pilots enough to
    randomise a QPSK constellation.  Scanning the pilot coherence of one
    non-SBS data symbol over ``+/-span`` samples pins the window to the sample
    that maximises it.  The correction is a property of the capture, so the
    caller applies the returned ``offset`` to every symbol of the subframe.

    Args:
        y: Main-rate IQ at or before ``t0``.
        t0: Nominal guard-interval start of the subframe's first symbol.
        fft_size, gi, noc, dx, dy: Subframe geometry.
        symbol: Data-symbol index whose scattered pilots are the ruler.
        sbs: Whether that symbol is a subframe boundary symbol.
        span: Half-width of the search, in samples.

    Returns:
        The refined :class:`FineTiming`; ``t0`` is the corrected subframe
        origin and ``offset`` the signed correction to the nominal one.
    """
    base = t0 + (fft_size + gi) * symbol + gi
    best = None
    for off in range(-span, span + 1):
        start = base + off
        if start < 0 or start + fft_size > len(y):
            continue
        carriers = _relative_carriers(y[start:start + fft_size], fft_size, noc)
        coh = scattered_pilot_coherence(carriers, noc, dx, dy, symbol, sbs)
        if best is None or coh > best[1]:
            best = (off, coh)
    return FineTiming(t0=t0 + best[0], coherence=best[1], symbol=symbol,
                      offset=best[0])


def subframe_fine_timing(y: np.ndarray, t0: int, fft_size: int, gi: int,
                         noc: int, dx: int, dy: int, n_data_symbols: int,
                         sbs_symbols=(0,),
                         span: int = spec.FINE_TIMING_SPAN) -> "FineTiming":
    """Fine-time a subframe off its first non-boundary data symbol.

    A subframe boundary symbol carries the denser DY = 1 pilot pattern and a
    different cell count (A/322 8.1.3.1), so a normal symbol is the cleaner
    ruler.  Every non-boundary symbol shares the same lattice, so one is
    enough.
    """
    sbs_set = tuple(sbs_symbols)
    symbol = next((l for l in range(n_data_symbols) if l not in sbs_set), 0)
    return fine_timing(y, t0, fft_size, gi, noc, dx, dy, symbol,
                       symbol in sbs_set, span=span)


def cpe_correct(cells: np.ndarray, symbol_of: np.ndarray,
                alphabet: np.ndarray,
                region: np.ndarray = None,
                dummy_values: np.ndarray = None,
                iterations: int = spec.CPE_ITERATIONS) -> np.ndarray:
    """Per-symbol residual complex-gain correction, decision-directed.

    After equalisation each OFDM symbol can still carry a residual common
    phase, because the channel estimate is built from that symbol's own pilots
    and the phase noise within the symbol is not.  For each symbol the two
    real parameters of a complex gain are fitted from thousands of cells
    against the alphabet each cell is already known to carry:

        c = <hard, z> / <hard, hard>,   z <- z / c

    repeated ``iterations`` times.  It is ordinary common-phase removal, not a
    search: it cannot rescue a wrong constellation, and LDPC remains the only
    oracle.

    Only cells whose true alphabet is known may feed the estimate: ``region``
    selects the cells carrying ``alphabet`` (one PLP's slice), and
    ``dummy_values`` marks the A/322 7.2.6.5 dummy tail, whose values are known
    exactly.  The correction itself — one complex gain per symbol — is applied
    to every cell, because the gain is common to the whole symbol.

    Args:
        cells: The subframe's equalised cells.
        symbol_of: OFDM symbol index per cell (-1 for Preamble spare cells).
        alphabet: Constellation the ``region`` cells carry.
        region: Boolean mask of cells known to carry ``alphabet``; None means
            every cell not marked by ``dummy_values``.
        dummy_values: Complex values of the known dummy tail, non-zero where
            known; zero elsewhere.
        iterations: Decision-directed iterations (``spec.CPE_ITERATIONS``).

    Returns:
        The corrected cells (unit-power normalised before and after).
    """
    cells = np.asarray(cells, dtype=np.complex128)
    out = cells / np.sqrt(np.mean(np.abs(cells) ** 2))
    pts = np.asarray(alphabet, dtype=np.complex128)
    n = len(out)
    if region is None:
        region = np.ones(n, dtype=bool)
    hard = np.zeros(n, dtype=np.complex128)
    known = np.asarray(region, dtype=bool).copy()
    if dummy_values is not None:
        dv = np.asarray(dummy_values, dtype=np.complex128)
        dsel = dv != 0
        hard[dsel] = dv[dsel]
        known |= dsel
    for _ in range(iterations):
        if region.any():
            z = out[region]
            hard[region] = pts[np.abs(z[:, None] - pts[None, :]).argmin(1)]
        for sym in np.unique(symbol_of):
            m = np.flatnonzero((symbol_of == sym) & known)
            if len(m) == 0:
                continue
            z = out[m]
            den = max(float(np.vdot(hard[m], hard[m]).real), 1e-12)
            c = np.vdot(hard[m], z) / den
            if abs(c) < 1e-12:
                continue
            out[symbol_of == sym] = out[symbol_of == sym] / c
    return out


@dataclass
class DataSymbolCells:
    """Equalised data cells of one data OFDM symbol."""
    cells: np.ndarray
    pilot_coherence: float
    n_carriers: int


@dataclass
class FineTiming:
    """Refined FFT-window origin of a subframe, from scattered-pilot coherence.

    ``t0`` is the refined guard-interval start of the first data symbol and
    ``offset`` the signed correction to the caller's nominal position.  The
    coherence is the value the search maximised, reported rather than assumed.
    """
    t0: int
    coherence: float
    symbol: int
    offset: int


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
    carriers = _relative_carriers(x, fft_size, noc)

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


def plp_alphabet(plp) -> np.ndarray:
    """Constellation the PLP's cells are known to carry (A/322 Annex C).

    The decision-directed CPE corrects each symbol against the alphabet the
    cells already carry; for a 64K normal frame the NUC position vectors
    depend on the code rate (Annex C), so both are taken from the L1 config.
    """
    mod = MOD_NAME.get(plp.modulation)
    if mod is None:
        raise NotImplementedError(
            f"modulation index {plp.modulation} not tabulated for data PLPs")
    return nuc.points(MODULATION_BITS[mod], RATE_MIN + plp.code_rate)


def cpe_spec(plp, dummy_start: int = None) -> "CpeSpec":
    """Build the CPE input for a PLP: its alphabet, slice and dummy tail.

    The decision-directed estimate may only use cells whose true alphabet is
    known: the PLP's own ``start..start+size`` slice, plus — when the caller
    supplies it — the A/322 7.2.6.5 dummy tail that follows *every* layer-0
    PLP of the subframe.  For subframe 0 the Preamble spare cells precede the
    slice and carry no known alphabet, so they are excluded.

    ``dummy_start`` is the index where the subframe's dummy tail begins, which
    is after the last layer-0 PLP, not after this one; None omits it.
    """
    return CpeSpec(alphabet=plp_alphabet(plp), region_start=plp.start,
                   region_stop=plp.start + plp.size, dummy_start=dummy_start)


def dummy_cell_values(pool_len: int, dummy_start: int) -> np.ndarray:
    """Known complex values of the A/322 7.2.6.5 dummy tail.

    The tail carries the baseband scrambling sequence mapped to ``+/-1``; its
    values are known exactly, so the CPE can use them without a decision.
    """
    vals = np.zeros(pool_len, dtype=np.complex128)
    if dummy_start < 0 or dummy_start >= pool_len:
        return vals
    bits = randomizer_bits(pool_len)[:pool_len].astype(np.int8)
    vals[dummy_start:] = 1.0 - 2.0 * bits[dummy_start:]
    return vals


@dataclass
class CpeSpec:
    """Input to the decision-directed common-phase correction.

    Attributes:
        alphabet: Constellation the region cells carry (A/322 Annex C).
        region_start: First pool index of the cells known to carry
            ``alphabet`` (one PLP's slice).
        region_stop: One past the last such cell.
        dummy_start: First pool index of the A/322 7.2.6.5 dummy tail whose
            values are known exactly; None when the subframe has no dummy tail.
    """
    alphabet: np.ndarray
    region_start: int
    region_stop: int
    dummy_start: Optional[int] = None

    def region(self, pool_len: int) -> np.ndarray:
        m = np.zeros(pool_len, dtype=bool)
        m[max(0, self.region_start):min(pool_len, self.region_stop)] = True
        return m

    def dummy_values(self, pool_len: int) -> Optional[np.ndarray]:
        if self.dummy_start is None:
            return None
        return dummy_cell_values(pool_len, self.dummy_start)

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
    index (0-based within the subframe; -1 for Preamble spare cells).  ``fine``
    records the scattered-pilot timing refinement applied to the data symbols,
    if any.
    """
    cells: np.ndarray
    symbol_of: np.ndarray
    n_preamble_spare: int
    n_null: int
    fine: Optional["FineTiming"] = None


def build_data_symbol_pool(y: np.ndarray, t0: int, fft_size: int,
                           guard_interval: int, noc: int, dx: int, dy: int,
                           n_data_symbols: int, sbs_symbols=(0,),
                           pattern: str = 'SP4_2', sbs_null: int = None,
                           cred_coeff: int = 0,
                           fi_offset: int = 0,
                           fi_enabled: bool = True,
                           fine_timing_enabled: bool = False,
                           fine: "FineTiming" = None,
                           cpe: "CpeSpec" = None) -> CellPool:
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
        fi_enabled: ``L1D_frequency_interleaver``; when 0 the subframe's cells
            are used in symbol order (A/322 Table 9.8, RF30 bypasses it).
        fine_timing_enabled: Run :func:`subframe_fine_timing` on the subframe
            and apply its correction to ``t0``.  The result is reported in the
            returned :class:`CellPool`.
        fine: A precomputed :class:`FineTiming`; overrides ``fine_timing``.
        cpe: A :class:`CpeSpec`; when given, the decision-directed per-symbol
            common-phase correction runs on the pool (:func:`cpe_correct`).

    Returns:
        CellPool for the subframe (``n_preamble_spare`` is 0).
    """
    n_null = spec.SBS_NULL_8K_CRED0 if sbs_null is None else int(sbs_null)
    lo_n = n_null // 2
    hi_n = n_null - lo_n
    add_cp = spec.additional_cp(pattern, fft_size, cred_coeff)
    if fine is None and fine_timing_enabled:
        fine = subframe_fine_timing(y, t0, fft_size, guard_interval, noc, dx, dy,
                                    n_data_symbols, sbs_symbols)
    if fine is not None:
        t0 = t0 + fine.offset

    parts, owner = [], []
    for l in range(n_data_symbols):
        sbs = l in tuple(sbs_symbols)
        start = t0 + (fft_size + guard_interval) * l + guard_interval
        result = data_symbol_cells(
            y[start:start + fft_size], fft_size, noc, dx, dy, l, sbs,
            add_cp=add_cp)
        x = (fi_deinterleave(result.cells, fi_offset + l, fft_size)
             if fi_enabled else result.cells)
        if sbs:
            x = x[lo_n:len(x) - hi_n]
        parts.append(x)
        owner.append(np.full(len(x), l, dtype=int))

    cells = np.concatenate(parts)
    symbol_of = np.concatenate(owner)
    if cpe is not None:
        cells = cpe_correct(cells, symbol_of, cpe.alphabet,
                            region=cpe.region(len(cells)),
                            dummy_values=cpe.dummy_values(len(cells)))
    return CellPool(cells=cells, symbol_of=symbol_of,
                    n_preamble_spare=0, n_null=int(n_null), fine=fine)


def build_cell_pool(y: np.ndarray, t0: int, fft_size: int, guard_interval: int,
                    noc: int, dx: int, dy: int, n_data_symbols: int,
                    sbs_symbols=(0,), preamble_structure: int = 27,
                    l1_cells: int = 484 + 880,
                    pattern: str = None, sbs_null: int = None,
                    preamble_num_symbols: int = 1,
                    preamble_reduced_carriers: int = spec.PREAMBLE_FIRST_CRED,
                    l1b_cells: int = None,
                    fi_enabled: bool = True,
                    fine_timing_enabled: bool = False,
                    cpe: "CpeSpec" = None) -> CellPool:
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
        fi_enabled: ``L1D_frequency_interleaver`` (A/322 Table 9.8).
        fine_timing_enabled: Run :func:`subframe_fine_timing` on the data
            symbols and apply its correction (A/322 8.1.3.1 scattered pilots).
        cpe: A :class:`CpeSpec`; when given, the decision-directed per-symbol
            common-phase correction runs on the data symbols (:func:`cpe_correct`).

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
        sbs_null=sbs_null, fi_offset=preamble_num_symbols,
        fi_enabled=fi_enabled, fine_timing_enabled=fine_timing_enabled)

    cells = np.concatenate([spare, data.cells])
    symbol_of = np.concatenate([
        np.full(len(spare), -1, dtype=int), data.symbol_of])
    if cpe is not None:
        cells = cpe_correct(cells, symbol_of, cpe.alphabet,
                            region=cpe.region(len(cells)),
                            dummy_values=cpe.dummy_values(len(cells)))
    return CellPool(cells=cells, symbol_of=symbol_of,
                    n_preamble_spare=int(len(spare)), n_null=data.n_null,
                    fine=data.fine)


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
PLP_NINNER = {    l1_signaling.FEC_BCH_16K: NINNER_SHORT,
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


@dataclass
class CtiDecode:
    """A CTI-mode PLP decoded across the frames supplied.

    Attributes:
        payload: the decoded FEC blocks.
        n_blocks: number of complete FEC blocks recovered.
        n_valid_cells: leading run of CTI-valid cells.
        c_offset: A/322 9.3.9.1 ``C``, the pre-CTI first-FEC-block offset.
        nrows: the CTI depth in delay lines (A/322 Table 9.24).
    """
    payload: PlpPayload
    n_blocks: int
    n_valid_cells: int
    c_offset: int
    nrows: int


def decode_cti_plp(stream_cells: np.ndarray, plp, max_iterations: int = 100,
                   max_blocks: int = None) -> CtiDecode:
    """Decode a CTI-mode PLP (A/322 7.1.4) from concatenated frame cells.

    The CTI never resets: ``stream_cells`` is the PLP's ``plp_size`` cells from
    each consecutive frame, in order, the first being the frame whose L1 carries
    ``plp.cti_start_row``.  The stream is CTI de-interleaved, then FEC blocks are
    taken from the offset ``C`` solved from ``L1D_plp_CTI_fec_block_start``
    (A/322 9.3.9.1); only cells fed by the initial delay-line state are dropped.
    """
    from . import cti

    mod = MOD_NAME.get(plp.modulation)
    if mod is None:
        raise NotImplementedError(
            f"modulation index {plp.modulation} not tabulated for data PLPs")
    ninner = PLP_NINNER.get(plp.fec_type)
    if ninner is None:
        raise NotImplementedError(
            f"fec_type {plp.fec_type} not tabulated for data PLPs")
    if plp.ti_mode != TI_CTI:
        raise NotImplementedError("decode_cti_plp requires TI mode 1 (CTI)")

    nrows = spec.cti_nrows(plp.cti_depth, bool(plp.ti_extended_interleaving))
    c_offset = spec.cti_start_c(plp.ti_fec_block_start, plp.cti_start_row,
                                nrows)
    deint = cti.deinterleave(stream_cells, nrows, plp.cti_start_row)
    n_valid = cti.first_invalid(deint.valid, len(deint.valid))
    cells_per_fec = ninner // MODULATION_BITS[mod]
    if c_offset < 0:
        n_blocks = 0
    else:
        n_blocks = max(0, (n_valid - c_offset) // cells_per_fec)
    if max_blocks is not None:
        n_blocks = min(n_blocks, max_blocks)

    chain = DataPlpChain(mod=mod, rate=RATE_MIN + plp.code_rate, ninner=ninner,
                         max_iterations=max_iterations)
    blocks = [chain.decode_cells(
        deint.cells[c_offset + m * cells_per_fec:
                    c_offset + (m + 1) * cells_per_fec])
        for m in range(n_blocks)]
    return CtiDecode(
        payload=PlpPayload(plp_id=plp.plp_id, fec_blocks=blocks,
                           n_fec=n_blocks),
        n_blocks=n_blocks, n_valid_cells=n_valid, c_offset=c_offset,
        nrows=nrows)


def decode_subframe0_plp(y: np.ndarray, t0: int, fft_size: int,
                         guard_interval: int, noc: int, dx: int, dy: int,
                         n_data_symbols: int, sbs_symbols,
                         preamble_structure: int, l1_cells: int,
                         plp, max_iterations: int = 100,
                         pattern: str = None, sbs_null: int = None,
                         preamble_num_symbols: int = 1,
                         preamble_reduced_carriers: int = spec.PREAMBLE_FIRST_CRED,
                         l1b_cells: int = None,
                         fi_enabled: bool = True,
                         fine_timing_enabled: bool = False,
                         cpe: bool = False,
                         dummy_start: int = None) -> PlpPayload:
    """Decode one PLP of subframe 0 from its L1-Detail ``plp`` config.

    With ``cpe`` the PLP's own alphabet (A/322 Annex C) drives the per-symbol
    common-phase correction; ``fine_timing_enabled`` refines the FFT window off
    the scattered pilots.  Both apply to the data symbols only; the Preamble
    spare cells, whose constellation is not known from ``plp``, are excluded
    from the CPE estimate.  ``dummy_start`` is the pool index where the A/322
    7.2.6.5 dummy tail begins (after the last layer-0 PLP).
    """
    pool = build_cell_pool(y, t0, fft_size, guard_interval, noc, dx, dy,
                           n_data_symbols, sbs_symbols=sbs_symbols,
                           preamble_structure=preamble_structure,
                           l1_cells=l1_cells, pattern=pattern,
                           sbs_null=sbs_null,
                           preamble_num_symbols=preamble_num_symbols,
                           preamble_reduced_carriers=preamble_reduced_carriers,
                           l1b_cells=l1b_cells, fi_enabled=fi_enabled,
                           fine_timing_enabled=fine_timing_enabled,
                           cpe=cpe_spec(plp, dummy_start) if cpe else None)
    return decode_plp_from_pool(pool, plp, max_iterations=max_iterations)


def decode_subframe_plp(y: np.ndarray, t0: int, fft_size: int,
                        guard_interval: int, noc: int, dx: int, dy: int,
                        n_data_symbols: int, sbs_symbols, plp,
                        max_iterations: int = 100, pattern: str = 'SP4_2',
                        sbs_null: int = None, cred_coeff: int = 0,
                        fi_offset: int = 0,
                        fi_enabled: bool = True,
                        fine_timing_enabled: bool = False,
                        cpe: bool = False,
                        dummy_start: int = None) -> PlpPayload:
    """Decode one PLP of a subframe after the first (no Preamble, FI reset).

    ``t0`` is this subframe's first symbol (guard-interval start); the frequency
    interleaver counter resets here, so ``fi_offset`` is 0 (A/322 7.3 rule 2).
    """
    pool = build_data_symbol_pool(
        y, t0, fft_size, guard_interval, noc, dx, dy, n_data_symbols,
        sbs_symbols=sbs_symbols, pattern=pattern, sbs_null=sbs_null,
        cred_coeff=cred_coeff, fi_offset=fi_offset, fi_enabled=fi_enabled,
        fine_timing_enabled=fine_timing_enabled,
        cpe=cpe_spec(plp, dummy_start) if cpe else None)
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
