"""ATSC 3.0 L1-Detail signalling protection (decode and encode).

L1-Detail is the variable-length signalling that follows L1-Basic in the
Preamble.  It uses the same protection blocks as L1-Basic (A/322 6.5.2) with
per-mode parameters, plus - for Modes 3-7 - a parity interleaver (6.5.2.6):

    scramble -> BCH(16200) encode -> zero-pad to Kldpc -> 16K LDPC
    -> parity interleave (Modes 3-7) -> group-wise parity permutation
    -> puncturing -> (repetition, Mode 1 only) -> bit demux / block interleave.

L1-Detail Mode 3 (rate 6/15 Type B) is what a typical multiplex uses.  The
information length Ksig = 8 * L1B_L1_Detail_size_bytes; if it exceeds Kseg the
signalling is split into several FEC frames (segmentation, 6.5.3.1) - not
implemented here.

The shared scrambling / padding / permutation primitives live in
:mod:`atsc3lib.signaling_fec`.

Reference: ATSC A/322:2024-04, Sections 6.5.2 and 6.5.3.
"""

from typing import Tuple

import numpy as np

from . import signaling_fec as sf
from . import spec
from .bch import BCHCode
from .crc import crc32_ok
from .ldpc_exact import ATSC3LDPCExact

NINNER = spec.L1B_NINNER
scramble_bits = sf.scramble_bits


def preamble_block_deinterleave(cells: np.ndarray, n_symbols: int) -> np.ndarray:
    """Invert the A/322 7.2.5.2 L1-Detail Preamble block interleaver.

    L1-Detail's ``L1B_L1_Detail_total_cells`` cells are spread across ``NP``
    Preamble symbols by a block interleaver with ``Lc = NP`` columns and
    ``Lr = floor(total/NP)`` rows: the first ``Lc*Lr`` cells are written
    row-wise and read out column-wise (``y(i*Lr+j) = M(j*Lc+i)``), and any
    remainder is appended unchanged.  This returns ``M`` (the pre-interleaver
    order), which is what the L1-Detail FEC chain expects.

    Args:
        cells: The received L1-Detail cells ``y`` in mapping order (first
            Preamble symbol's non-L1-Basic cells, then later symbols).
        n_symbols: NP, the number of Preamble symbols.

    Returns:
        The de-interleaved cells ``M``.
    """
    y = np.asarray(cells)
    total = len(y)
    lr = total // n_symbols
    full = n_symbols * lr
    out = np.empty(total, dtype=y.dtype)
    if full:
        m = np.arange(full)
        i = m % n_symbols
        j = m // n_symbols
        out[m] = y[i * lr + j]
    if total > full:
        out[full:] = y[full:]
    return out


class L1DetailCodec:
    """Encode/decode one L1-Detail FEC frame for a given mode.

    Args:
        mode: L1-Detail FEC mode (1-7, corresponding to
            ``L1B_L1_Detail_fec_type`` + 1).
        ksig: Information length in bits (8 * L1B_L1_Detail_size_bytes).
        max_iterations: LDPC decoder iteration cap.
    """

    def __init__(self, mode: int, ksig: int, max_iterations: int = 50):
        if mode not in spec.L1D_MODES:
            raise ValueError(f"L1-Detail mode {mode} not supported")
        self.mode = mode
        self.ksig = ksig
        m = spec.L1D_MODES[mode]
        self.mode_params = m
        g = spec.l1d_lengths(mode, ksig)
        self.geometry = g
        self.eta = g.eta
        self.Kldpc = g.kldpc
        self.Nouter = g.nouter
        self.rate = g.rate
        self.parity_interleaved = m.parity_interleaved

        self.pattern = list(m.shortening)
        self.groupwise = list(m.groupwise)
        self.first_group = m.groupwise_first

        self.bch = BCHCode(NINNER, 12)
        self.ldpc = ATSC3LDPCExact(self.rate, max_iterations=max_iterations)

        # Padding mask and the Nouter info positions within Kldpc.
        self._info_pos, self._padded = sf.info_positions(
            self.Kldpc, self.Nouter, self.pattern)

        # Transmitted-order clean-codeword parity positions.
        self._parity_pos = self._parity_positions()

        self.n_fec = g.n_fec
        self.n_parity_kept = g.n_parity_kept
        self.n_punc = g.n_punc
        self.n_repeat = g.n_repeat
        self.n_cells = g.n_cells
        self.n_tx = g.n_tx

    def _parity_positions(self) -> np.ndarray:
        """Clean-codeword parity positions in TRANSMISSION order.

        The transmitted parity is the group-wise interleaved ``Y`` (6.5.2.6),
        whose groups are the parity-interleaved ``u`` (Modes 3-7).  This maps
        each transmitted parity bit back to its index in the clean LDPC
        codeword ``c`` produced by our encoder (:mod:`ldpc_exact`, which is the
        6.1.3.2 form with no internal parity interleave).
        """
        u_positions = sf.groupwise_codeword_positions(
            self.Kldpc, NINNER, self.groupwise)
        if not self.parity_interleaved:
            return u_positions
        q = u_positions - self.Kldpc
        qldpc = spec.TYPE_B_QLDPC_16200[self.rate]
        return self.Kldpc + qldpc * (q % 360) + (q // 360)

    # --- encode ----------------------------------------------------------
    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """Encode Ksig L1-Detail bits into the transmitted bit sequence."""
        if len(info_bits) != self.ksig:
            raise ValueError(f"Expected {self.ksig} bits, got {len(info_bits)}")

        scrambled = sf.scramble_bits(np.asarray(info_bits))
        nouter = np.asarray(self.bch.encode(scrambled), dtype=np.uint8)
        assert len(nouter) == self.Nouter

        info = np.zeros(self.Kldpc, dtype=np.uint8)
        info[self._info_pos] = nouter
        codeword = self.ldpc.encode(info)

        permuted = codeword[self._parity_pos]
        parts = [nouter]
        if self.n_repeat:
            parts.append(permuted[:self.n_repeat])
        parts.append(permuted[:self.n_parity_kept])
        return np.concatenate(parts)

    # --- decode ----------------------------------------------------------
    def cells_to_llr(self, cells: np.ndarray) -> np.ndarray:
        """QPSK demap (Annex C.1.1) + 6.5.2.10 block de-interleave."""
        z = np.asarray(cells)
        ncells = self.n_tx // self.eta
        z = z[:ncells].astype(np.complex128)
        if not len(z):
            return np.zeros(self.n_tx, dtype=np.float64)
        z = z / np.sqrt(np.mean(np.abs(z) ** 2))
        hard = (np.sign(z.real) + 1j * np.sign(z.imag)) / np.sqrt(2)
        n0 = max(float(np.mean(np.abs(z - hard) ** 2)), 1e-6)
        k = 4.0 / n0
        llr_y0 = -k * z.imag
        llr_y1 = -k * z.real
        return np.concatenate([llr_y0, llr_y1])[:self.n_tx]

    def decode(self, llrs: np.ndarray, known_llr: float = 60.0
               ) -> Tuple[np.ndarray, bool, bool]:
        """Decode received LLRs into Ksig L1-Detail bits.

        Returns:
            (info_bits, bch_ok, crc_ok).  ``info_bits`` is the descrambled
            Ksig-bit payload (data followed by the 32-bit L1D_crc).
        """
        if len(llrs) != self.n_tx:
            raise ValueError(f"Expected {self.n_tx} LLRs, got {len(llrs)}")
        llrs = np.asarray(llrs, dtype=np.float64)

        codeword = np.zeros(NINNER, dtype=np.float64)
        codeword[:self.Kldpc][self._padded] = -known_llr
        codeword[self._info_pos] = llrs[:self.Nouter]

        # A/322 6.5.2.9: [info][repeated parity][punctured tail].  The repeat
        # block and the tail start at the same permuted-parity position, so
        # their LLRs add (6.5.2.7 Step 2).
        o = self.Nouter
        if self.n_repeat:
            n_rep = min(self.n_repeat, self._parity_pos.size)
            np.add.at(codeword, self._parity_pos[:n_rep],
                      llrs[o:o + n_rep])
            o += n_rep
        np.add.at(codeword, self._parity_pos[:self.n_parity_kept],
                  llrs[o:o + self.n_parity_kept])

        decoded, converged = self.ldpc.decode(codeword)
        nouter = decoded[self._info_pos]

        bch_bits, _nerr, bch_ok = self.bch.decode(nouter)
        info = sf.scramble_bits(np.asarray(bch_bits, dtype=np.uint8))[:self.ksig]
        self.last_ldpc_converged = bool(converged)
        return info, bool(bch_ok), crc32_ok(info)

    def decode_cells(self, cells: np.ndarray, known_llr: float = 60.0
                     ) -> Tuple[np.ndarray, bool, bool]:
        """Demap QPSK cells and decode."""
        return self.decode(self.cells_to_llr(cells), known_llr)
