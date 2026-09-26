"""ATSC 3.0 L1-Basic signalling protection (decode and encode).

Exact L1-Basic FEC chain from A/322 Section 6.5.2:

    scramble -> BCH(16200) encode (Nouter=368) -> zero-pad to Kldpc=3240
    -> 16K Type A LDPC (always rate 3/15 for L1-Basic) -> group-wise parity
    permutation (Table 6.21) -> puncturing (Table 6.24) -> repetition (Mode 1)
    -> bit demux / block interleave.

L1-Basic always carries Ksig = 200 information bits.  Modes 1-7 differ in
constellation and puncturing amount (and repetition for Mode 1); the LDPC code
is 3/15 for all modes because Kldpc = 3240 = 16200 * 3/15.

The shared scrambling / padding / permutation primitives live in
:mod:`atsc3lib.signaling_fec`.

Reference: ATSC A/322:2024-04 Physical Layer Protocol, Section 6.5.2
"""

from typing import Tuple

import numpy as np

from . import signaling_fec as sf
from . import spec
from .bch import BCHCode
from .ldpc_exact import ATSC3LDPCExact

KSIG_L1_BASIC = spec.L1B_KSIG
KLD_PC = spec.L1B_KLDPC
NOUTER = spec.L1B_NOUTER
NINNER = spec.L1B_NINNER
N_LDPC_PARITY = spec.L1B_NLDPC_PARITY
NFEC_BASE = NOUTER + N_LDPC_PARITY
L1_BASIC_LDPC_RATE = spec.L1B_LDPC_RATE

SHORTENING_PATTERN = spec.L1B_SHORTENING_PATTERN
GROUPWISE_PATTERN = spec.L1B_GROUPWISE_PATTERN
L1_BASIC_MODES = spec.L1B_MODES

scramble_bits = sf.scramble_bits
_randomizer_bits = sf.randomizer_bits

# Padded positions and the codeword positions carrying the Nouter bits.
_INFO_POS, _PADDED = sf.info_positions(KLD_PC, NOUTER, SHORTENING_PATTERN)
# Transmitted-order codeword positions for the group-wise permuted parity.
_PARITY_POS = sf.groupwise_codeword_positions(KLD_PC, NINNER, GROUPWISE_PATTERN)


class L1BasicCodec:
    """Encode/decode an L1-Basic block (200 bits) for a given mode."""

    def __init__(self, mode: int = 1, max_iterations: int = 50):
        if mode not in L1_BASIC_MODES:
            raise ValueError(f"L1-Basic mode {mode} not supported")
        self.mode = mode
        cfg = L1_BASIC_MODES[mode]
        self.eta = cfg.eta
        self.constellation = cfg.constellation
        self.bch = BCHCode(NINNER, 12)
        self.ldpc = ATSC3LDPCExact(L1_BASIC_LDPC_RATE,
                                   max_iterations=max_iterations)

        # Puncturing and repetition (A/322 6.5.2.7/6.5.2.8, Tables 6.23/6.24).
        lengths = spec.l1b_lengths(mode)
        self.n_fec = lengths.n_fec
        self.n_cells = lengths.n_cells
        self.n_punc = lengths.n_punc
        self.n_repeat = lengths.n_repeat
        self.n_tx = self.n_fec + self.n_repeat

    # --- encode ----------------------------------------------------------
    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """Encode 200 L1-Basic info bits into the transmitted bit sequence."""
        if len(info_bits) != KSIG_L1_BASIC:
            raise ValueError(f"Expected {KSIG_L1_BASIC} bits")

        scrambled = sf.scramble_bits(np.asarray(info_bits))
        nouter = np.asarray(self.bch.encode(scrambled), dtype=np.uint8)

        info = np.zeros(KLD_PC, dtype=np.uint8)
        info[_INFO_POS] = nouter
        codeword = self.ldpc.encode(info)              # 16200

        # Group-wise permuted parity, in transmission order.
        permuted = codeword[_PARITY_POS]
        n_base_parity = self.n_fec - NOUTER
        base = np.concatenate([nouter, permuted[:n_base_parity]])

        if self.n_repeat:
            base = np.concatenate([base, permuted[:self.n_repeat]])
        return base

    # --- decode ----------------------------------------------------------
    def cells_to_llr(self, cells: np.ndarray) -> np.ndarray:
        """QPSK demap (A/322 Annex C.1.1) + 6.5.2.10 block de-interleave.

        Annex C.1.1 maps bits (y0, y1) as 00->(1+j), 01->(-1+j),
        10->(+1-j), 11->(-1-j), so y1 sets the sign of I and y0 the sign of Q.
        6.5.2.10 writes the NFEC bits into a 2-column block interleaver
        column-wise and reads row-wise, so the symbol stream carries all y0
        bits first, then all y1 bits.

        Returns LLRs (library convention: LLR > 0 => bit 1) in transmitted
        order, ready for :meth:`decode`.
        """
        z = np.asarray(cells)
        ncells = self.n_fec // self.eta
        z = z[:ncells].astype(np.complex128)
        if not len(z):
            return np.zeros(self.n_tx, dtype=np.float64)
        z = z / np.sqrt(np.mean(np.abs(z) ** 2))
        hard = (np.sign(z.real) + 1j * np.sign(z.imag)) / np.sqrt(2)
        n0 = max(float(np.mean(np.abs(z - hard) ** 2)), 1e-6)
        k = 4.0 / n0
        # bit y0 = 0 when Im > 0, so (LLR>0 => bit1) gives llr_y0 = -k*Im.
        llr_y0 = -k * z.imag
        llr_y1 = -k * z.real
        # 6.5.2.10 block interleaver: column 0 = y0, column 1 = y1.
        return np.concatenate([llr_y0, llr_y1])[:self.n_fec]

    def decode_cells(self, cells: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Demap QPSK cells (with block de-interleave) and decode."""
        return self.decode(self.cells_to_llr(cells))

    def decode(self, llrs: np.ndarray,
               known_llr: float = 60.0) -> Tuple[np.ndarray, bool]:
        """Decode received LLRs into 200 info bits.

        Args:
            llrs: LLRs (length n_tx), library convention (LLR > 0 => bit 1),
                in transmitted order ``[nouter | permuted parity | repetition]``.
            known_llr: Magnitude used for known-zero padded info positions.

        Returns:
            (decoded_info_bits, converged) where ``converged`` reflects the BCH
            check, the definitive success criterion once punctured parity is
            unknown.
        """
        if len(llrs) != self.n_tx:
            raise ValueError(f"Expected {self.n_tx} LLRs, got {len(llrs)}")

        llrs = np.asarray(llrs, dtype=np.float64)

        codeword = np.zeros(NINNER, dtype=np.float64)
        # Known-zero padded info positions (LLR>0 => bit 1).
        codeword[:KLD_PC][_PADDED] = -known_llr
        codeword[_INFO_POS] = llrs[:NOUTER]

        # Scatter the received parity bits onto their codeword positions.
        n_base_parity = self.n_fec - NOUTER
        np.add.at(codeword, _PARITY_POS[:n_base_parity],
                  llrs[NOUTER:self.n_fec])

        # Repetition (Mode 1) adds onto the SAME parity positions.
        if self.n_repeat:
            n_rep = min(self.n_repeat, _PARITY_POS.size)
            np.add.at(codeword, _PARITY_POS[:n_rep],
                      llrs[self.n_fec:self.n_fec + n_rep])

        decoded, converged = self.ldpc.decode(codeword)
        nouter = decoded[_INFO_POS]

        bch_bits, _nerr, bch_ok = self.bch.decode(nouter)
        info = sf.scramble_bits(np.asarray(bch_bits, dtype=np.uint8))
        self.last_ldpc_converged = bool(converged)
        return info, bool(bch_ok)
