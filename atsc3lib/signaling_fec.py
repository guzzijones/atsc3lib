"""Shared FEC helpers for ATSC 3.0 L1-Basic and L1-Detail signalling.

L1-Basic (A/322 6.5.2) and L1-Detail (6.5.2/6.5.3) share most of their
protection chain: baseband scrambling, BCH outer coding, zero-padding
(shortening), LDPC inner coding, group-wise parity permutation and puncturing.
They differ only in their parameters (Ksig, Kldpc, code rate, patterns).

This module holds the parameterised building blocks; :mod:`atsc3lib.l1_basic`
and :mod:`atsc3lib.l1_detail` feed them the per-case tables from
:mod:`atsc3lib.spec`.

Bit groups are 360 bits throughout (A/322 6.5.2.4).
"""

from typing import Tuple

import numpy as np

from . import spec

GROUP = spec.BCH_GROUP_SIZE  # 360


# ---------------------------------------------------------------------------
# Baseband scrambler (A/322 5.2.3, applied to L1 by 6.5.2.2)
# ---------------------------------------------------------------------------

def randomizer_bits(length: int) -> np.ndarray:
    """The baseband scrambling sequence (A/322 5.2.3)."""
    sr = spec.SCRAMBLER_SEED
    bits = np.zeros(length, dtype=np.uint8)
    idx = 0
    while idx < length:
        packed = (((sr & 0x004) << 5) | ((sr & 0x008) << 3)
                  | ((sr & 0x010) << 1) | ((sr & 0x020) >> 1)
                  | ((sr & 0x200) >> 6) | ((sr & 0x1000) >> 10)
                  | ((sr & 0x2000) >> 12) | ((sr & 0x8000) >> 15))
        for n in range(7, -1, -1):
            if idx < length:
                bits[idx] = (packed >> n) & 1
                idx += 1
        feedback = sr & 1
        sr >>= 1
        if feedback:
            sr ^= spec.SCRAMBLER_POLY
    return bits


def scramble_bits(bits: np.ndarray) -> np.ndarray:
    """Self-inverse baseband scrambler/descrambler for L1 signalling."""
    return np.bitwise_xor(np.asarray(bits, dtype=np.uint8),
                          randomizer_bits(len(bits)))


# ---------------------------------------------------------------------------
# Shortening / zero padding (A/322 6.5.2.4)
# ---------------------------------------------------------------------------

def zero_pad_mask(kldpc: int, nouter: int, pattern) -> np.ndarray:
    """Boolean mask of the zero-padded (shortened) positions in Kldpc bits.

    ``Npad`` whole groups are padded in the order given by ``pattern``, then
    the first part of the next group in that order is padded.
    """
    padded = np.zeros(kldpc, dtype=bool)
    npad_bits = kldpc - nouter
    npad = npad_bits // GROUP
    for j in range(npad):
        g = pattern[j]
        padded[GROUP * g:GROUP * (g + 1)] = True
    partial = npad_bits - GROUP * npad
    if partial > 0:
        g = pattern[npad]
        padded[GROUP * g:GROUP * g + partial] = True
    return padded


def info_positions(kldpc: int, nouter: int, pattern) -> Tuple[np.ndarray, np.ndarray]:
    """Codeword positions of the Nouter info+BCH bits, ascending.

    Returns ``(positions, padded_mask)``.
    """
    padded = zero_pad_mask(kldpc, nouter, pattern)
    return np.flatnonzero(~padded), padded


# ---------------------------------------------------------------------------
# Group-wise parity permutation (A/322 6.5.2.6, Tables 6.21/6.22)
# ---------------------------------------------------------------------------

def groupwise_codeword_positions(kldpc: int, ninner: int, pattern) -> np.ndarray:
    """Codeword positions of the transmitted parity bits, in transmission order.

    The group-wise interleaver sets ``Y_j = X_{pi_p(j)}`` for output parity
    groups ``j = Kldpc/360 .. Ninner/360 - 1``; ``pattern`` holds the absolute
    source group indices ``pi_p(j)``.  The transmitted parity is the first
    ``n_keep`` bits of ``Y`` (puncturing removes the tail), so this returns the
    ``X`` (parity-interleaved codeword) positions in that order.

    This is the mapping *before* any parity interleaver (6.5.2.6); callers for
    L1-Detail Modes 3-7 must additionally apply :func:`u_to_c_parity`.
    """
    ninfo_g = kldpc // GROUP
    ngroup = ninner // GROUP
    positions = []
    for j in range(ninfo_g, ngroup):
        src = pattern[j - ninfo_g]
        positions.extend(range(GROUP * src, GROUP * (src + 1)))
    return np.asarray(positions, dtype=int)


def parity_interleave_positions(kldpc: int, ninner: int, qldpc: int) -> np.ndarray:
    """Map parity-interleaved codeword index to clean-codeword parity index.

    A/322 6.5.2.6: ``u_{K+360t+s} = c_{K+Qldpc*s+t}`` for ``0 <= s < 360``,
    ``0 <= t < Qldpc``.  Returns an array ``m`` of length ``M = Ninner-Kldpc``
    such that ``u_parity[q] = c_parity[m[q]]``, i.e. the clean-codeword parity
    index for each interleaved-parity index.  Applies to L1-Detail Modes 3-7
    only; for other modes it is the identity.
    """
    m = ninner - kldpc
    q = np.arange(m)
    t, s = np.divmod(q, 360)
    return qldpc * s + t
