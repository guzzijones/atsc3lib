"""Unit tests for ATSC 3.0 bootstrap generation and detection."""

import numpy as np
import pytest

from atsc3lib import spec
from atsc3lib.bootstrap import (
    generate_bootstrap, detect_bootstrap, decode_preamble_structure,
    encode_bootstrap_signaling, SHIFT_FOR_BYTE, BYTE_FOR_SHIFT,
    MINOR_VERSION_SEEDS, BOOTSTRAP_FFT_SIZE, fine_cfo,
    NUM_BOOTSTRAP_SYMBOLS, B_SIZE, C_SIZE,
)

SYMBOL_LEN = C_SIZE + BOOTSTRAP_FFT_SIZE + B_SIZE
BOOTSTRAP_LEN = SYMBOL_LEN * NUM_BOOTSTRAP_SYMBOLS


class TestShiftMapping:
    def test_injective(self):
        assert len(SHIFT_FOR_BYTE) == 256
        assert len(BYTE_FOR_SHIFT) == 256

    def test_roundtrip(self):
        for byte in range(256):
            assert BYTE_FOR_SHIFT[SHIFT_FOR_BYTE[byte]] == byte


class TestPreambleStructures:
    def test_first_structures(self):
        p0 = decode_preamble_structure(0)
        assert p0.fft_size == 8192 and p0.guard_interval == 192
        assert p0.l1_fec_mode == 1

        p4 = decode_preamble_structure(4)
        assert p4.fft_size == 8192 and p4.l1_fec_mode == 5

    def test_16k_and_32k(self):
        assert decode_preamble_structure(35).fft_size == 16384
        assert decode_preamble_structure(90).fft_size == 32768
        assert decode_preamble_structure(159).fft_size == 32768

    def test_unknown_structure(self):
        with pytest.raises(ValueError, match="preamble_structure"):
            decode_preamble_structure(200)

    def test_all_structures_defined(self):
        for s in range(160):
            assert s in spec.PREAMBLE_STRUCTURE


class TestSignalingBytes:
    def test_structure_in_byte2(self):
        sig = encode_bootstrap_signaling(42)
        assert sig[2] == 42
        assert sig[1] == 2  # BSR coefficient

    def test_out_of_range(self):
        with pytest.raises(ValueError, match="preamble_structure"):
            encode_bootstrap_signaling(160)


class TestGeneration:
    def test_length(self):
        wf = generate_bootstrap(0)
        assert len(wf) == BOOTSTRAP_LEN

    def test_all_versions_generate(self):
        for major, minor in [(0, 0), (0, 3), (1, 0), (1, 7)]:
            wf = generate_bootstrap(10, major=major, minor=minor)
            assert len(wf) == BOOTSTRAP_LEN
            assert np.all(np.isfinite(wf))


class TestDetection:
    @pytest.mark.parametrize('structure', [0, 6, 10, 30, 90, 120, 159])
    def test_detect_clean(self, structure):
        wf = generate_bootstrap(structure, major=0, minor=0)
        rng = np.random.default_rng(structure)
        rx = np.concatenate([
            rng.normal(0, 0.1, 700) + 1j * rng.normal(0, 0.1, 700),
            wf,
            rng.normal(0, 0.1, 700) + 1j * rng.normal(0, 0.1, 700),
        ])
        det = detect_bootstrap(rx)
        assert det.start == 700
        assert det.structure == structure
        assert det.major == 0 and det.minor == 0

    def test_detect_recovers_fft_gi(self):
        wf = generate_bootstrap(35, major=0, minor=0)
        rx = np.concatenate([np.zeros(300, dtype=np.complex128), wf])
        det = detect_bootstrap(rx)
        params = decode_preamble_structure(det.structure)
        assert params.fft_size == 16384
        assert params.guard_interval == 192

    def test_detect_requires_length(self):
        with pytest.raises(ValueError, match="samples"):
            detect_bootstrap(np.zeros(100, dtype=np.complex128))

    def test_version_restriction(self):
        # Restricting the hypotheses still finds a matching bootstrap...
        wf = generate_bootstrap(27, major=0, minor=0)
        rx = np.concatenate([np.zeros(300, dtype=np.complex128), wf])
        det = detect_bootstrap(rx, versions=[(0, 0)])
        assert det.structure == 27
        # ...and an empty hypothesis set fails cleanly.
        with pytest.raises(ValueError, match="No valid bootstrap"):
            detect_bootstrap(rx, versions=[])


class TestFineCfo:
    @pytest.mark.parametrize('cfo_hz', [0.0, 250.0, -700.0])
    def test_recovers_injected_cfo(self, cfo_hz):
        wf = generate_bootstrap(27, major=0, minor=0)
        n = np.arange(len(wf))
        rx = wf * np.exp(1j * 2 * np.pi * cfo_hz * n / spec.BOOTSTRAP_RATE_HZ)
        rx = np.concatenate([np.zeros(500, dtype=np.complex128), rx])
        est = fine_cfo(rx, 500)
        assert abs(est - cfo_hz) < 5.0

    def test_zero_for_empty(self):
        assert fine_cfo(np.zeros(10, dtype=np.complex128), 0) == 0.0
