"""Tests for OFDM parameter detection (FFT size / guard interval)."""

import numpy as np
import pytest

from atsc3lib.ofdm_detect import (
    cp_correlation, detect_guard_interval, detect_ofdm_params,
    find_symbol_start, GUARD_INTERVALS, FFT_SIZES,
)


def _make_ofdm(fft, gi, n_symbols=15, seed=0):
    rng = np.random.default_rng(seed)
    syms = []
    for _ in range(n_symbols):
        d = (rng.normal(size=fft) + 1j * rng.normal(size=fft)) / np.sqrt(2)
        td = np.fft.ifft(d)
        syms.append(np.concatenate([td[-gi:], td]))
    return np.concatenate(syms)


class TestCPCorrelation:
    @pytest.mark.parametrize('fft,gi', [(8192, 2048), (8192, 1536),
                                        (8192, 512), (16384, 1024)])
    def test_correct_params_high_correlation(self, fft, gi):
        x = _make_ofdm(fft, gi)
        assert cp_correlation(x, fft, gi) > 0.95

    def test_wrong_gi_lower(self):
        x = _make_ofdm(8192, 1536)
        correct = cp_correlation(x, 8192, 1536)
        wrong = cp_correlation(x, 8192, 256)
        assert correct > wrong

    def test_noise_low(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=400_000) + 1j * rng.normal(size=400_000)
        assert cp_correlation(x, 8192, 1536) < 0.05


class TestDetect:
    @pytest.mark.parametrize('fft,gi', [(8192, 2048), (8192, 1536),
                                        (8192, 512), (16384, 384)])
    def test_detect_exact(self, fft, gi):
        x = _make_ofdm(fft, gi)
        dfft, dgi, val, _ = detect_ofdm_params(x, fft_candidates=[fft])
        assert dfft == fft
        assert dgi == gi
        assert val > 0.9

    def test_detect_fft_and_gi(self):
        x = _make_ofdm(8192, 1536)
        fft, gi, val, scores = detect_ofdm_params(x)
        assert fft == 8192
        assert gi == 1536
        # Other FFT sizes should score much lower.
        assert scores[8192] > scores.get(16384, 0)

    def test_guard_interval_options_defined(self):
        for fft in FFT_SIZES:
            assert 192 in GUARD_INTERVALS[fft]


class TestFindSymbolStart:
    def test_finds_aligned_start(self):
        x = _make_ofdm(8192, 512, n_symbols=20)
        offset = find_symbol_start(x, 8192, 512, search=256)
        # Should be at (or very near) a symbol boundary.
        assert offset % (8192 + 512) < 32
