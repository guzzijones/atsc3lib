"""ATSC 3.0 frequency interleaver (A/322 Section 7.3).

The frequency interleaver reorders the ``Ndata`` data cells of an OFDM symbol
using a per-symbol permutation ``H_l(p)`` generated from two LFSRs (an
interleaving sequence generator with a wire permutation, and a symbol offset
generator). It is always applied to Preamble symbols.

This module implements the exact address generator and the inverse
(deinterleaver) used by the receiver.

Wire permutations and LFSR taps are taken from A/322 Tables 7.12-7.14 and
cross-checked against the reference implementation ``drmpeg/gr-atsc3``
(``freqinterleaver_cc_impl.cc``).
"""

from typing import Dict, List

import numpy as np


class _FIParams:
    """Per-FFT-size constants for the frequency interleaver."""

    def __init__(self, pn_degree, pn_mask, max_states, logic, logic2,
                 bitperm, bitperm_odd):
        self.pn_degree = pn_degree          # R' is (pn_degree) bits
        self.pn_mask = pn_mask
        self.max_states = max_states
        self.logic = logic
        self.logic2 = logic2
        self.bitperm = bitperm
        self.bitperm_odd = bitperm_odd


_PARAMS: Dict[int, _FIParams] = {
    8192: _FIParams(
        pn_degree=12, pn_mask=0xFFF, max_states=8192,
        logic=[0, 1, 4, 6],
        logic2=[0, 1, 4, 5, 9, 11],
        bitperm=[7, 1, 4, 2, 9, 6, 8, 10, 0, 3, 11, 5],
        bitperm_odd=[11, 4, 9, 3, 1, 2, 5, 0, 6, 7, 10, 8],
    ),
    16384: _FIParams(
        pn_degree=13, pn_mask=0x1FFF, max_states=16384,
        logic=[0, 1, 4, 5, 9, 11],
        logic2=[0, 1, 2, 12],
        bitperm=[9, 7, 6, 10, 12, 5, 1, 11, 0, 2, 3, 4, 8],
        bitperm_odd=[6, 8, 10, 12, 2, 0, 4, 1, 11, 3, 5, 9, 7],
    ),
    32768: _FIParams(
        pn_degree=14, pn_mask=0x3FFF, max_states=32768,
        logic=[0, 1, 2, 12],
        logic2=[0, 1],
        bitperm=[7, 13, 3, 4, 9, 2, 12, 11, 1, 8, 10, 0, 5, 6],
        bitperm_odd=[7, 13, 3, 4, 9, 2, 12, 11, 1, 8, 10, 0, 5, 6],
    ),
}


def _permute_bits(word: int, table: List[int]) -> int:
    out = 0
    for n, dst in enumerate(table):
        out |= ((word >> n) & 1) << dst
    return out


def generate_addresses(symbol_index: int, fft_size: int,
                       n_data: int) -> np.ndarray:
    """Generate the frequency interleaver address sequence H_l(p).

    Args:
        symbol_index: OFDM symbol index within the frame/subframe (0-based).
        fft_size: 8192, 16384 or 32768.
        n_data: Number of data cells Ndata in this symbol.

    Returns:
        int array of length ``n_data`` mapping output cell index -> input index.
    """
    if fft_size not in _PARAMS:
        raise ValueError(f"Unsupported FFT size {fft_size}")
    p = _PARAMS[fft_size]
    if n_data > p.max_states:
        raise ValueError(f"Ndata {n_data} exceeds {p.max_states}")

    # Symbol offset generator G: constant for two consecutive symbols.  G is
    # Nr = log2(Nmax) = pn_degree + 1 bits wide, initialised to all ones.
    g_word = 0
    for k in range(0, (symbol_index // 2) + 1):
        if k == 0:
            g_word = (1 << (p.pn_degree + 1)) - 1
        else:
            result = 0
            for bit in p.logic2:
                result ^= (g_word >> bit) & 1
            g_word &= (1 << (p.pn_degree + 1)) - 1
            g_word >>= 1
            g_word |= result << p.pn_degree

    use_odd = (symbol_index % 2) == 1
    table = p.bitperm_odd if use_odd else p.bitperm

    addresses = np.empty(n_data, dtype=np.int64)
    count = 0
    lfsr = 0
    for j in range(p.max_states):
        if j == 0 or j == 1:
            lfsr = 0
        elif j == 2:
            lfsr = 1
        else:
            result = 0
            for bit in p.logic:
                result ^= (lfsr >> bit) & 1
            lfsr &= p.pn_mask
            lfsr >>= 1
            lfsr |= result << (p.pn_degree - 1)

        value = _permute_bits(lfsr, table)
        value += (j % 2) * (p.max_states // 2)
        value ^= g_word
        if value < n_data:
            addresses[count] = value
            count += 1
            if count == n_data:
                break
    if count != n_data:
        raise RuntimeError(
            f"FI generated {count} addresses, expected {n_data}")
    return addresses


def interleave(cells: np.ndarray, symbol_index: int, fft_size: int) -> np.ndarray:
    """Apply the frequency interleaver: A[n] = X[H[n]]."""
    n_data = len(cells)
    h = generate_addresses(symbol_index, fft_size, n_data)
    return np.asarray(cells)[h]


def deinterleave(cells: np.ndarray, symbol_index: int,
                 fft_size: int) -> np.ndarray:
    """Invert the frequency interleaver: X[H[n]] = A[n]."""
    n_data = len(cells)
    h = generate_addresses(symbol_index, fft_size, n_data)
    out = np.empty(n_data, dtype=np.asarray(cells).dtype)
    out[h] = cells
    return out
