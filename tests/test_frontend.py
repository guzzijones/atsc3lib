"""Unit tests for the ATSC 3.0 front-end (resampling + acquisition)."""

import numpy as np
import pytest

from atsc3lib.bootstrap import generate_bootstrap
from atsc3lib.frontend import (
    resample_iq, acquire_frame, bootstrap_span_main_rate,
    BOOTSTRAP_RATE_HZ, MAIN_RATE_HZ,
)


class TestResample:
    def test_identity(self):
        iq = (np.random.default_rng(0).normal(size=1000) +
              1j * np.random.default_rng(1).normal(size=1000)).astype(np.complex64)
        out = resample_iq(iq, BOOTSTRAP_RATE_HZ, BOOTSTRAP_RATE_HZ)
        assert np.array_equal(out, iq)

    def test_upsample_length(self):
        iq = np.ones(1000, dtype=np.complex64)
        out = resample_iq(iq, 1_000_000, 2_000_000)
        assert abs(len(out) - 2000) < 5

    def test_downsample_length(self):
        iq = np.ones(2000, dtype=np.complex64)
        out = resample_iq(iq, 2_000_000, 1_000_000)
        assert abs(len(out) - 1000) < 5


class TestBootstrapSpan:
    def test_span_ratio(self):
        # Native bootstrap is 12288 samples at 6.144 MHz -> 9/8 at 6.912 MHz.
        assert bootstrap_span_main_rate() == 13824


class TestAcquireFrame:
    @pytest.mark.parametrize('structure', [6, 10, 30])
    def test_acquire_from_10mhz(self, structure):
        wf = generate_bootstrap(structure, major=0, minor=0)
        iq_fs = resample_iq(wf, BOOTSTRAP_RATE_HZ, 10e6).astype(np.complex64)
        rng = np.random.default_rng(structure)
        noise = (rng.normal(0, 0.05, len(iq_fs)) +
                 1j * rng.normal(0, 0.05, len(iq_fs))).astype(np.complex64)
        pre = (rng.normal(0, 0.05, 30000) +
               1j * rng.normal(0, 0.05, 30000)).astype(np.complex64)
        rx = np.concatenate([pre, (iq_fs + noise).astype(np.complex64)])

        info = acquire_frame(rx, 10e6)
        assert info.bootstrap.structure == structure
        assert info.fft_size in (8192, 16384, 32768)
        assert info.symbol_size == info.fft_size + info.guard_interval
        assert info.bootstrap_main_start >= 0
