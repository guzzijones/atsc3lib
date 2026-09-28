"""Exact ATSC 3.0 bit interleaver (A/322 Section 6.2, short and normal frames).

The A/322 Section 6.2 bit interleaver consists of:

1. Parity interleaver (Type B LDPC rates only) - rearranges the LDPC parity
   section so the decoder sees it in natural order.
2. Group interleaver - permutes 360-bit groups using the A/322 Annex B tables
   (inlined in :mod:`atsc3lib.group_tables`: short frames use the Annex B.2
   numeric-modulation keys, normal frames the named-modulation keys).
3. Block interleaver - Type A (``Nr1``/``Nr2`` columns per Table 6.10) or
   Type B (``NQCB_IG``/``Npart1``/``Npart2`` per Table 6.11).

The table values match the A/322 Annex tables exactly.

Reference: ATSC A/322:2024-04 Physical Layer Protocol, Section 6.2
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .group_tables import GROUP_TABLES_16200, GROUP_TABLES_64800
from .ldpc_exact import (
    NINNER_SHORT, NINNER_NORMAL, RATE_DENOM, RATE_MIN, RATE_MAX, GROUP_SIZE,
    FEC_TYPE_A, FEC_TYPE_B, type_a_params, type_b_qldpc,
)
from .nuc import (
    MODULATION_BITS, QPSK, QAM16, QAM64, QAM256, QAM1024, QAM4096,
)

#: Block-interleaver type A/B (A/322 Table 6.8/6.9 code value).
BLOCK_TYPE_A, BLOCK_TYPE_B = 'A', 'B'


@dataclass(frozen=True)
class BlockParams:
    """Type A/B block-interleaver parameters (A/322 Tables 6.10/6.11).

    Type A uses ``(nr1, nr2, ncols = eta_MOD)`` and leaves the Type B
    ``npart1``/``npart2`` at 0; Type B uses ``(nqcb_ig = eta_MOD, npart1,
    npart2)`` and leaves ``nr1``/``nr2`` at 0.  Both rows and columns of the
    Type B part are derived from ``npart1`` and ``npart2``.
    """
    nr1: int
    nr2: int
    ncols: int
    npart1: int
    npart2: int


def _type_a(ninner: int, mod: str, nr1: int, nr2: int) -> BlockParams:
    return BlockParams(nr1, nr2, MODULATION_BITS[mod], 0, 0)


def _type_b(ninner: int, mod: str, npart1: int, npart2: int) -> BlockParams:
    return BlockParams(0, 0, MODULATION_BITS[mod], npart1, npart2)


#: Type A block interleaver (A/322 Table 6.10), keyed (Ninner, modulation).
TYPE_A_BLOCK: Dict[tuple, BlockParams] = {
    (NINNER_NORMAL, QPSK):   _type_a(NINNER_NORMAL, QPSK, 32400, 0),
    (NINNER_NORMAL, QAM16):  _type_a(NINNER_NORMAL, QAM16, 16200, 0),
    (NINNER_NORMAL, QAM64):  _type_a(NINNER_NORMAL, QAM64, 10800, 0),
    (NINNER_NORMAL, QAM256): _type_a(NINNER_NORMAL, QAM256, 7920, 180),
    (NINNER_NORMAL, QAM1024): _type_a(NINNER_NORMAL, QAM1024, 6480, 0),
    (NINNER_NORMAL, QAM4096): _type_a(NINNER_NORMAL, QAM4096, 5400, 0),
    (NINNER_SHORT, QPSK):    _type_a(NINNER_SHORT, QPSK, 7920, 180),
    (NINNER_SHORT, QAM16):   _type_a(NINNER_SHORT, QAM16, 3960, 90),
    (NINNER_SHORT, QAM64):   _type_a(NINNER_SHORT, QAM64, 2520, 180),
    (NINNER_SHORT, QAM256):  _type_a(NINNER_SHORT, QAM256, 1800, 225),
}

#: Type B block interleaver (A/322 Table 6.11), keyed (Ninner, modulation).
TYPE_B_BLOCK: Dict[tuple, BlockParams] = {
    (NINNER_NORMAL, QPSK):   _type_b(NINNER_NORMAL, QPSK, 64800, 0),
    (NINNER_NORMAL, QAM16):  _type_b(NINNER_NORMAL, QAM16, 64800, 0),
    (NINNER_NORMAL, QAM64):  _type_b(NINNER_NORMAL, QAM64, 64800, 0),
    (NINNER_NORMAL, QAM256): _type_b(NINNER_NORMAL, QAM256, 63360, 1440),
    (NINNER_NORMAL, QAM1024): _type_b(NINNER_NORMAL, QAM1024, 64800, 0),
    (NINNER_NORMAL, QAM4096): _type_b(NINNER_NORMAL, QAM4096, 64800, 0),
    (NINNER_SHORT, QPSK):    _type_b(NINNER_SHORT, QPSK, 15840, 360),
    (NINNER_SHORT, QAM16):   _type_b(NINNER_SHORT, QAM16, 15840, 360),
    (NINNER_SHORT, QAM64):   _type_b(NINNER_SHORT, QAM64, 15120, 1080),
    (NINNER_SHORT, QAM256):  _type_b(NINNER_SHORT, QAM256, 14400, 1800),
}

#: Block interleaver type per (modulation, rate) for normal frames; the rest
#: are Type A (A/322 Table 6.8).  Type A FEC does not imply Type A block.
_BI_TYPE_B_64800 = {
    (QAM16, 5), (QAM16, 8), (QAM16, 9),
    (QAM64, 7), (QAM64, 9), (QAM64, 10), (QAM64, 13),
    (QAM256, 5), (QAM256, 6), (QAM256, 7), (QAM256, 8),
    (QAM256, 10), (QAM256, 11), (QAM256, 13),
    (QAM1024, 5), (QAM1024, 7), (QAM1024, 9), (QAM1024, 10),
    (QAM1024, 11),
    (QAM4096, 7),
}

#: Block interleaver type per (modulation, rate) for short frames; the rest
#: are Type A (A/322 Table 6.9).
_BI_TYPE_B_16200 = {
    (QPSK, 6), (QPSK, 7), (QPSK, 9),
    (QAM16, 6), (QAM16, 7), (QAM16, 9), (QAM16, 11), (QAM16, 13),
    (QAM64, 6), (QAM64, 7), (QAM64, 9),
    (QAM256, 6), (QAM256, 11),
}

#: Modulation name -> short-frame group-table key, which numbers modulations
#: 1..6 (A/322 Annex B.2 order).
_MODULATION_TABLE_KEY = {
    QPSK: 1, QAM16: 2, QAM64: 3, QAM256: 4,
    QAM1024: 5, QAM4096: 6,
}

#: Data-PLP modulations the receive chain demaps (A/322 Annex C bank).
SUPPORTED_MODULATIONS = (QPSK, QAM16, QAM64, QAM256)


def _normalise_group_tables(raw: Dict, short: bool) -> Dict:
    """Normalise an inlined group table to {modulation name: {rate: tuple}}.

    The short-frame table keys modulations by number (A/322 Annex B.2 order,
    1..6); the normal-frame table keys them by name.  Rates are ints.
    """
    by_number = {v: k for k, v in _MODULATION_TABLE_KEY.items()}
    out = {}
    for mod, table in raw.items():
        name = by_number[int(mod)] if short else mod
        out[name] = {int(rate): list(perm) for rate, perm in table.items()}
    return out


_GROUP_TABLES = {
    NINNER_SHORT: _normalise_group_tables(GROUP_TABLES_16200, short=True),
    NINNER_NORMAL: _normalise_group_tables(GROUP_TABLES_64800, short=False),
}


def _bi_type(n: int) -> set:
    return _BI_TYPE_B_64800 if n == NINNER_NORMAL else _BI_TYPE_B_16200


def _bits_per_symbol(modulation: str) -> int:
    if modulation not in SUPPORTED_MODULATIONS:
        raise NotImplementedError(
            f"No A/322 bit interleaver for modulation {modulation!r}; "
            f"supported: {list(SUPPORTED_MODULATIONS)}"
        )
    return MODULATION_BITS[modulation]


class GroupInterleaver:
    """ATSC 3.0 bit interleaver for short and normal frames.

    The transmitter-side permutation is built once and cached as
    ``self.order``: the on-air bit at position ``i`` equals the LDPC
    codeword bit at index ``order[i]``. Interleaving applies ``order`` and
    deinterleaving inverts it.

    Args:
        rate: LDPC code rate numerator (2..13, denominator 15).
        modulation: One of QPSK, 16QAM, 64QAM, 256QAM.
        n: Codeword length (16200 short frames, 64800 normal frames).
    """

    def __init__(self, rate: int, modulation: str = QPSK,
                 n: int = NINNER_SHORT):
        if n not in _GROUP_TABLES:
            raise ValueError(f"Only N in {tuple(_GROUP_TABLES)} supported, "
                             f"got {n}")
        if not RATE_MIN <= rate <= RATE_MAX:
            raise ValueError(f"Rate {rate}/15 not supported "
                             f"(use {RATE_MIN}-{RATE_MAX})")

        self.rate = rate
        self.modulation = modulation
        self.n = n
        self.mod = _bits_per_symbol(modulation)
        self.K = n * rate // RATE_DENOM
        self.M = n - self.K
        self.ldpc_type = FEC_TYPE_A if rate in type_a_params(n) else FEC_TYPE_B
        self.q_val = type_b_qldpc(n).get(rate)
        self.n_group = n // GROUP_SIZE
        # Table 6.8/6.9: Type A FEC does not imply Type A block interleaving.
        self.block_type = (BLOCK_TYPE_B if (modulation, rate) in _bi_type(n)
                           else BLOCK_TYPE_A)
        self.block = (TYPE_A_BLOCK if self.block_type == BLOCK_TYPE_A
                      else TYPE_B_BLOCK)[(n, modulation)]

        try:
            self.group_table = _GROUP_TABLES[n][modulation][rate]
        except KeyError as exc:
            raise NotImplementedError(
                f"No A/322 group table for modulation {modulation} rate "
                f"{rate}/15 at Ninner={n}"
            ) from exc
        if len(self.group_table) != self.n_group:
            raise ValueError(
                f"Group table length {len(self.group_table)} != {self.n_group}"
            )

        self.order = self._build_order()
        # Permutation sanity check.
        if not np.array_equal(np.sort(self.order), np.arange(self.n)):
            raise ValueError("Internal error: group interleaver is not a permutation")

    def _build_order(self) -> np.ndarray:
        """Build the on-air -> codeword-index permutation."""
        src = np.arange(self.n, dtype=np.int64)

        # Step 1: parity interleaver (Type B only).
        if self.ldpc_type == FEC_TYPE_B:
            q = self.q_val
            tempu = np.empty(self.n, dtype=np.int64)
            tempu[:self.K] = src[:self.K]
            parity = src[self.K:]
            for t in range(q):
                for s in range(GROUP_SIZE):
                    tempu[self.K + GROUP_SIZE * t + s] = parity[q * s + t]
        else:
            tempu = src

        # Step 2: group interleaver.
        tempv = np.concatenate([
            tempu[g * GROUP_SIZE:(g + 1) * GROUP_SIZE]
            for g in self.group_table
        ])

        # Step 3: block interleaver.
        if self.block_type == BLOCK_TYPE_A:
            return self._block_interleave_type_a(tempv)
        return self._block_interleave_type_b(tempv)

    def _block_interleave_type_a(self, tempv: np.ndarray) -> np.ndarray:
        """A/322 6.2.3.1: write column-wise, read row-wise (Part 1 then 2)."""
        mod = self.mod
        nr1, nr2 = self.block.nr1, self.block.nr2

        out = np.empty(self.n, dtype=np.int64)
        cols = [tempv[i * nr1:(i + 1) * nr1] for i in range(mod)]
        idx = 0
        for j in range(nr1):
            for k in range(mod):
                out[idx] = cols[k][j]
                idx += 1
        if nr2:
            rows2 = nr1 * mod
            tail_cols = [
                tempv[rows2 + i * nr2:rows2 + (i + 1) * nr2] for i in range(mod)
            ]
            for j in range(nr2):
                for k in range(mod):
                    out[idx] = tail_cols[k][j]
                    idx += 1
        return out

    def _block_interleave_type_b(self, tempv: np.ndarray) -> np.ndarray:
        """A/322 6.2.3.2: 360-column row-write / column-read over Npart1."""
        mod = self.mod
        npart1, npart2 = self.block.npart1, self.block.npart2
        outer = npart1 // (GROUP_SIZE * mod)
        inner = GROUP_SIZE * mod

        out = np.empty(self.n, dtype=np.int64)
        idx = 0
        for nn in range(outer):
            indexn = nn * inner
            for _ in range(GROUP_SIZE):
                for k in range(mod):
                    out[idx] = tempv[indexn + GROUP_SIZE * k]
                    idx += 1
                indexn += 1
        if npart2:
            out[idx:idx + npart2] = tempv[self.n - npart2:]
            idx += npart2
        return out

    def interleave(self, codeword_bits: np.ndarray) -> np.ndarray:
        """Apply the transmitter bit interleaver (codeword -> on-air order)."""
        if len(codeword_bits) != self.n:
            raise ValueError(f"Expected {self.n} bits, got {len(codeword_bits)}")
        return np.asarray(codeword_bits)[self.order]

    def deinterleave(self, received_bits: np.ndarray) -> np.ndarray:
        """Invert the bit interleaver (on-air -> codeword order)."""
        if len(received_bits) != self.n:
            raise ValueError(f"Expected {self.n} bits, got {len(received_bits)}")
        out = np.empty(self.n, dtype=np.asarray(received_bits).dtype)
        out[self.order] = received_bits
        return out

    def deinterleave_llrs(self, llrs: np.ndarray) -> np.ndarray:
        """Deinterleave soft bits (LLRs) into LDPC codeword order."""
        return self.deinterleave(llrs)


def deinterleave_llrs(llrs: np.ndarray, rate: int,
                      modulation: str = QPSK,
                      n: int = NINNER_SHORT) -> np.ndarray:
    """Convenience function: deinterleave LLRs for a given MODCOD."""
    return GroupInterleaver(rate, modulation, n=n).deinterleave_llrs(llrs)
