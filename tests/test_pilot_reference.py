"""Tests for the pilot reference sequence and preamble pilot helpers."""

import numpy as np
import pytest

from atsc3lib.pilot_reference import (
    reference_sequence, pilot_values, preamble_amplitude,
    preamble_dx, preamble_pilot_indices,
)


class TestReferenceSequence:
    def test_first_24_values(self):
        # A/322 8.1.2: 1101 1000 0000 0001 0100 0000
        expected = [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
                    0, 0, 0, 0, 0, 0]
        assert list(reference_sequence(24)) == expected

    def test_binary(self):
        r = reference_sequence(1000)
        assert set(np.unique(r)).issubset({0, 1})

    def test_pilot_values_bpsk(self):
        vals = pilot_values(50, amplitude=1.23)
        assert np.allclose(vals.imag, 0.0)
        assert set(np.round(np.abs(vals.real), 6)).issubset({1.23})


class TestPreamble:
    def test_amplitude_structure_27(self):
        # Structure 27 = 8K, GI 1536 -> A=1.230
        assert preamble_amplitude(8192, 1536) == 1.230

    def test_amplitude_8k_192(self):
        assert preamble_amplitude(8192, 192) == 1.841

    def test_dx_structure_27(self):
        # Table H.1.1: structure 27 uses preamble pilot DX=4
        assert preamble_dx(27) == 4

    def test_dx_structure_0(self):
        assert preamble_dx(0) == 16

    def test_pilot_indices(self):
        idx = preamble_pilot_indices(100, 4)
        assert list(idx[:5]) == [0, 4, 8, 12, 16]
        assert np.all(idx % 4 == 0)

    def test_missing_amplitude_raises(self):
        with pytest.raises(KeyError):
            preamble_amplitude(4096, 100)
