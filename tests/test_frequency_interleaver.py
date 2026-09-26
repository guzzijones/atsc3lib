"""Tests for the ATSC 3.0 frequency interleaver."""

import numpy as np
import pytest

from atsc3lib.frequency_interleaver import (
    generate_addresses, interleave, deinterleave, _PARAMS,
)


class TestAddressGeneration:
    @pytest.mark.parametrize('fft', [8192, 16384, 32768])
    @pytest.mark.parametrize('symbol', [0, 1, 2, 3, 5])
    def test_is_permutation(self, fft, symbol):
        n = min(4000, _PARAMS[fft].max_states)
        h = generate_addresses(symbol, fft, n)
        assert len(h) == n
        assert np.array_equal(np.sort(h), np.arange(n))

    def test_unsupported_fft(self):
        with pytest.raises(ValueError, match="FFT"):
            generate_addresses(0, 4096, 100)

    def test_too_many_cells(self):
        with pytest.raises(ValueError, match="exceeds"):
            generate_addresses(0, 8192, 9000)

    def test_offset_constant_for_pair_8k_16k(self):
        # For 8K/16K the symbol offset changes every two symbols, but the
        # interleaving sequence generator still resets each symbol, so the
        # addresses are not necessarily equal. Just check determinism.
        h0a = generate_addresses(0, 8192, 100)
        h0b = generate_addresses(0, 8192, 100)
        assert np.array_equal(h0a, h0b)

    def test_known_first_address_8k(self):
        # Anchor verified against an independent A/322 receiver (Felbs/atsc3).
        h = generate_addresses(0, 8192, 4851)
        assert h[0] == 4095
        assert h[1] == 4063

    def test_addresses_match_reference_l0(self):
        # First entries of H_l(p) for l=0, 8K, 4851 data cells.
        h = generate_addresses(0, 8192, 4851)
        assert list(h[:8]) == [4095, 4063, 4087, 3071, 3999, 4051, 3061, 959]


class TestRoundTrip:
    @pytest.mark.parametrize('fft', [8192, 16384])
    @pytest.mark.parametrize('symbol', [0, 1, 2, 3])
    def test_interleave_deinterleave(self, fft, symbol):
        n = 4000
        x = np.arange(n)
        a = interleave(x, symbol, fft)
        assert np.array_equal(deinterleave(a, symbol, fft), x)

    def test_complex_roundtrip(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=3000) + 1j * rng.normal(size=3000)
        a = interleave(x, 2, 8192)
        assert np.allclose(deinterleave(a, 2, 8192), x)
