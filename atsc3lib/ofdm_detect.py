"""OFDM parameter detection for ATSC 3.0 (FFT size, guard interval, timing).

The legacy ``ofdm.py`` implementation picks symbol boundaries with an O(n*cp)
loop and assumes a fixed (FFT, GI). This module provides a correct, efficient
detector:

- ``detect_guard_interval``: for each candidate (FFT, GI) pair, computes the
  normalized cyclic-prefix correlation *at the correct lag* and compares.
- ``find_symbol_start``: locates the first OFDM symbol after the bootstrap.

The cyclic-prefix metric for a candidate symbol length L = FFT + GI and lag
FFT is:  rho = |<x[n] x*[n+FFT]>| / sqrt(<|x|^2><|x[n+FFT]|^2>) averaged over
sample positions. A correct candidate gives a sharp peak; an incorrect one
does not.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

# ATSC 3.0 guard interval lengths (samples) per FFT size (A/322 Table H.1.1).
GUARD_INTERVALS = {
    8192: [192, 384, 512, 768, 1024, 1536, 2048],
    16384: [192, 384, 512, 768, 1024, 1536, 2048, 2432, 3072, 3648, 4096],
    32768: [192, 384, 512, 768, 1024, 1536, 2048, 2432, 3072, 3648, 4096, 4864],
}

FFT_SIZES = [8192, 16384, 32768]


def cp_correlation(x: np.ndarray, fft_size: int, gi: int,
                   max_samples: int = 400_000) -> float:
    """Normalized cyclic-prefix correlation at lag ``fft_size``.

    Averages |x[n] conj(x[n+FFT])| over n in the CP region of each symbol.
    Returns a value in [0, 1]; correct (FFT, GI) yields a clear peak.
    """
    if max_samples and len(x) > max_samples:
        x = x[:max_samples]
    symbol = fft_size + gi
    n_symbols = (len(x) - symbol) // symbol
    if n_symbols < 4:
        return 0.0

    num = 0.0
    den_a = 0.0
    den_b = 0.0
    for s in range(n_symbols):
        start = s * symbol
        a = x[start:start + gi]
        b = x[start + fft_size:start + fft_size + gi]
        num += np.abs(np.vdot(a, b))
        den_a += np.vdot(a, a).real
        den_b += np.vdot(b, b).real
    if den_a == 0 or den_b == 0:
        return 0.0
    return float(num / np.sqrt(den_a * den_b))


def detect_guard_interval(x: np.ndarray, fft_size: int,
                          max_samples: int = 400_000) -> Tuple[int, float]:
    """Return the best (guard_interval, correlation) for a given FFT size."""
    best_gi = GUARD_INTERVALS[fft_size][-1]
    best_val = -1.0
    for gi in GUARD_INTERVALS[fft_size]:
        val = cp_correlation(x, fft_size, gi, max_samples)
        if val > best_val:
            best_val = val
            best_gi = gi
    return best_gi, best_val


def detect_ofdm_params(x: np.ndarray, fft_candidates: Optional[List[int]] = None,
                       max_samples: int = 400_000
                       ) -> Tuple[int, int, float, Dict[int, float]]:
    """Detect (FFT size, guard interval) among ATSC 3.0 candidates.

    Returns (fft_size, guard_interval, best_corr, scores_by_fft).
    """
    if fft_candidates is None:
        fft_candidates = FFT_SIZES
    scores: Dict[int, float] = {}
    best = (-1, -1, -1.0)
    for fft in fft_candidates:
        gi, val = detect_guard_interval(x, fft, max_samples)
        scores[fft] = val
        if val > best[2]:
            best = (fft, gi, val)
    return best[0], best[1], best[2], scores


def find_symbol_start(x: np.ndarray, fft_size: int, gi: int,
                      search: int = 100_000) -> int:
    """Find the sample offset of the first OFDM symbol (max CP correlation).

    Scans candidate offsets in [0, search) and returns the offset maximizing
    the local CP correlation over a few symbols.
    """
    symbol = fft_size + gi
    limit = min(search, len(x) - 3 * symbol)
    if limit <= 0:
        return 0
    step = max(1, gi // 4)
    best_offset = 0
    best_val = -1.0
    for off in range(0, limit, step):
        num = 0.0
        den_a = 0.0
        den_b = 0.0
        for s in range(3):
            start = off + s * symbol
            a = x[start:start + gi]
            b = x[start + fft_size:start + fft_size + gi]
            if len(a) < gi or len(b) < gi:
                break
            num += np.abs(np.vdot(a, b))
            den_a += np.vdot(a, a).real
            den_b += np.vdot(b, b).real
        if den_a and den_b:
            val = num / np.sqrt(den_a * den_b)
            if val > best_val:
                best_val = val
                best_offset = off
    return best_offset
