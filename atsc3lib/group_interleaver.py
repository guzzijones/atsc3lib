"""Exact ATSC 3.0 bit interleaver for short FEC frames (N=16200).

Implements the A/322 Section 6.3 bit interleaver, which consists of:

1. Parity interleaver (Type B LDPC rates 6/15-13/15 only) - rearranges the
   LDPC parity section so the decoder sees it in natural order.
2. Group interleaver - permutes 360-bit groups using the official A/322
   tables (``data/group_interleaver_B2.json``).
3. Block interleaver - writes the bits into ``mod`` columns and reads them
   out row-wise, with a leftover tail (``nr2``/``npart2``) handled specially.

The tables in ``data/group_interleaver_B2.json`` are keyed by modulation
(1=QPSK, 2=16QAM, 3=64QAM, 4=256QAM) and code rate (2..13, denominator 15),
and match the A/322 Annex tables exactly.

Reference: ATSC A/322:2024-04 Physical Layer Protocol, Section 6.3
"""

import json
import os
from typing import Dict, List

import numpy as np

_N = 16200
_GROUP_SIZE = 360
_NUM_GROUPS = _N // _GROUP_SIZE  # 45

_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Modulation name -> JSON top-level key (only the four tabulated modes).
_MODULATION_KEYS = {
    'QPSK': 1,
    '16QAM': 2,
    '64QAM': 3,
    '256QAM': 4,
}

# Bits per symbol for the tabulated modes.
_MODULATION_ORDER = {
    'QPSK': 2,
    '16QAM': 4,
    '64QAM': 6,
    '256QAM': 8,
}

# (nr2, npart2) for short frames, per A/322 Section 6.3.
_TAIL_SHORT = {
    'QPSK': (180, 360),
    '16QAM': (90, 360),
    '64QAM': (180, 1080),
    '256QAM': (225, 1800),
}

# Block interleaver type for each (rate, modulation) at N=16200.
# Derived from A/322 Section 6.3 and cross-checked against the reference
# implementation (drmpeg/gr-atsc3, interleaver_bb_impl.cc).
_BLOCK_TYPE = {
    6:  {'QPSK': 'B', '16QAM': 'B', '64QAM': 'B', '256QAM': 'B'},
    7:  {'QPSK': 'B', '16QAM': 'B', '64QAM': 'B', '256QAM': 'A'},
    8:  {'QPSK': 'A', '16QAM': 'A', '64QAM': 'A', '256QAM': 'A'},
    9:  {'QPSK': 'B', '16QAM': 'B', '64QAM': 'B', '256QAM': 'A'},
    10: {'QPSK': 'A', '16QAM': 'A', '64QAM': 'A', '256QAM': 'A'},
    11: {'QPSK': 'A', '16QAM': 'B', '64QAM': 'A', '256QAM': 'B'},
    12: {'QPSK': 'A', '16QAM': 'A', '64QAM': 'A', '256QAM': 'A'},
    13: {'QPSK': 'A', '16QAM': 'B', '64QAM': 'A', '256QAM': 'A'},
}

# Type B parity-interleaver lifting factor Qldpc for short frames.
_TYPE_B_QLDPC = {6: 27, 7: 24, 8: 21, 9: 18, 10: 15, 11: 12, 12: 9, 13: 6}


def _load_group_tables() -> Dict[int, Dict[int, List[int]]]:
    with open(os.path.join(_DATA_DIR, 'group_interleaver_B2.json')) as f:
        raw = json.load(f)
    return {
        int(mod): {int(rate): perm for rate, perm in table.items()}
        for mod, table in raw.items()
    }


_GROUP_TABLES = _load_group_tables()


def _modulation_key(modulation: str) -> int:
    if modulation not in _MODULATION_KEYS:
        raise NotImplementedError(
            f"No A/322 group interleaver table for modulation {modulation!r}; "
            f"supported: {sorted(_MODULATION_KEYS)}"
        )
    return _MODULATION_KEYS[modulation]


def _bits_per_symbol(modulation: str) -> int:
    if modulation not in _MODULATION_ORDER:
        raise NotImplementedError(
            f"No A/322 bit interleaver for modulation {modulation!r}; "
            f"supported: {sorted(_MODULATION_ORDER)}"
        )
    return _MODULATION_ORDER[modulation]


class GroupInterleaver:
    """ATSC 3.0 bit interleaver for short frames (N=16200).

    The transmitter-side permutation is built once and cached as
    ``self.order``: the on-air bit at position ``i`` equals the LDPC
    codeword bit at index ``order[i]``. Interleaving applies ``order`` and
    deinterleaving inverts it.

    Args:
        rate: LDPC code rate numerator (2..13, denominator 15).
        modulation: One of QPSK, 16QAM, 64QAM, 256QAM.
        n: Codeword length (only 16200 supported).
    """

    def __init__(self, rate: int, modulation: str = 'QPSK', n: int = 16200):
        if n != _N:
            raise ValueError(f"Only N=16200 (short frames) supported, got {n}")
        if rate not in range(2, 14):
            raise ValueError(f"Rate {rate}/15 not supported (use 2-13)")

        self.rate = rate
        self.modulation = modulation
        self.n = n
        self.mod = _bits_per_symbol(modulation)
        self.K = n * rate // 15
        self.M = n - self.K
        self.ldpc_type = 'A' if rate in (2, 3, 4, 5) else 'B'
        self.q_val = _TYPE_B_QLDPC.get(rate)
        # Type A LDPC rates always use block type A; Type B rates vary.
        self.block_type = 'A' if self.ldpc_type == 'A' else _BLOCK_TYPE[rate][modulation]

        mod_key = _modulation_key(modulation)
        try:
            self.group_table = _GROUP_TABLES[mod_key][rate]
        except KeyError as exc:
            raise NotImplementedError(
                f"No group table for {modulation} rate {rate}/15"
            ) from exc
        if len(self.group_table) != _NUM_GROUPS:
            raise ValueError(
                f"Group table length {len(self.group_table)} != {_NUM_GROUPS}"
            )

        self.order = self._build_order()
        # Permutation sanity check.
        if not np.array_equal(np.sort(self.order), np.arange(self.n)):
            raise ValueError("Internal error: group interleaver is not a permutation")

    def _build_order(self) -> np.ndarray:
        """Build the on-air -> codeword-index permutation."""
        src = np.arange(self.n, dtype=np.int64)

        # Step 1: parity interleaver (Type B only).
        if self.ldpc_type == 'B':
            q = self.q_val
            tempu = np.empty(self.n, dtype=np.int64)
            tempu[:self.K] = src[:self.K]
            parity = src[self.K:]
            for t in range(q):
                for s in range(360):
                    tempu[self.K + 360 * t + s] = parity[q * s + t]
        else:
            tempu = src

        # Step 2: group interleaver.
        tempv = np.concatenate([
            tempu[g * _GROUP_SIZE:(g + 1) * _GROUP_SIZE]
            for g in self.group_table
        ])

        # Step 3: block interleaver.
        if self.block_type == 'A':
            return self._block_interleave_type_a(tempv)
        return self._block_interleave_type_b(tempv)

    def _block_interleave_type_a(self, tempv: np.ndarray) -> np.ndarray:
        mod = self.mod
        nr2, _ = _TAIL_SHORT[self.modulation]
        packed = self.n // mod
        rows = packed - nr2

        out = np.empty(self.n, dtype=np.int64)
        cols = [tempv[i * rows:(i + 1) * rows] for i in range(mod)]
        idx = 0
        for j in range(rows):
            for k in range(mod):
                out[idx] = cols[k][j]
                idx += 1
        if nr2:
            rows2 = rows * mod
            tail_cols = [
                tempv[rows2 + i * nr2:rows2 + (i + 1) * nr2] for i in range(mod)
            ]
            for j in range(nr2):
                for k in range(mod):
                    out[idx] = tail_cols[k][j]
                    idx += 1
        return out

    def _block_interleave_type_b(self, tempv: np.ndarray) -> np.ndarray:
        mod = self.mod
        _, npart2 = _TAIL_SHORT[self.modulation]
        outer = self.n // (360 * mod)
        inner = 360 * mod

        out = np.empty(self.n, dtype=np.int64)
        idx = 0
        for nn in range(outer):
            indexn = nn * inner
            for _ in range(360):
                for k in range(mod):
                    out[idx] = tempv[indexn + 360 * k]
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
                      modulation: str = 'QPSK') -> np.ndarray:
    """Convenience function: deinterleave LLRs for a given MODCOD."""
    return GroupInterleaver(rate, modulation).deinterleave_llrs(llrs)
