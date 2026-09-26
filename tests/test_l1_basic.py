"""Unit tests for the L1-Basic signaling FEC chain."""

import numpy as np
import pytest

from atsc3lib.l1_basic import (
    L1BasicCodec, scramble_bits, KSIG_L1_BASIC, NOUTER, KLD_PC,
    L1_BASIC_MODES, _randomizer_bits,
)


class TestRandomizer:
    def test_first_values(self):
        # A/322 5.2.3: first scrambling bits are 1100 0000 0110 1101 ...
        bits = _randomizer_bits(16)
        assert list(bits[:8]) == [1, 1, 0, 0, 0, 0, 0, 0]
        assert list(bits[8:16]) == [0, 1, 1, 0, 1, 1, 0, 1]

    def test_scramble_self_inverse(self):
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, 500, dtype=np.uint8)
        assert np.array_equal(scramble_bits(scramble_bits(bits)), bits)


class TestCodecConstruction:
    def test_all_modes(self):
        for mode in range(1, 8):
            codec = L1BasicCodec(mode, max_iterations=15)
            assert codec.eta == L1_BASIC_MODES[mode].eta

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            L1BasicCodec(8)

    def test_cell_counts_match_table_6_17(self):
        expected_cells = {1: 3820, 2: 934, 3: 484, 4: 259,
                          5: 163, 6: 112, 7: 69}
        for mode, cells in expected_cells.items():
            codec = L1BasicCodec(mode, max_iterations=15)
            assert codec.n_tx // codec.eta == cells

    def test_mode1_repeats(self):
        assert L1BasicCodec(1).n_repeat == 3672
        assert L1BasicCodec(2).n_repeat == 0


class TestRoundTrip:
    @pytest.mark.parametrize('mode', [1, 2, 3, 4, 5, 6, 7])
    def test_clean_roundtrip(self, mode):
        codec = L1BasicCodec(mode, max_iterations=15)
        rng = np.random.default_rng(mode)
        info = rng.integers(0, 2, KSIG_L1_BASIC, dtype=np.uint8)
        tx = codec.encode(info)
        assert len(tx) == codec.n_tx
        llrs = np.where(tx == 1, 8.0, -8.0)
        out, ok = codec.decode(llrs)
        assert ok
        assert np.array_equal(out, info)

    def test_known_pattern(self):
        codec = L1BasicCodec(3, max_iterations=15)
        info = np.zeros(KSIG_L1_BASIC, dtype=np.uint8)
        tx = codec.encode(info)
        out, ok = codec.decode(np.where(tx == 1, 6.0, -6.0))
        assert ok and np.array_equal(out, info)

    def test_wrong_llr_length(self):
        codec = L1BasicCodec(2, max_iterations=15)
        with pytest.raises(ValueError, match="LLRs"):
            codec.decode(np.zeros(10))

    def test_wrong_info_length(self):
        codec = L1BasicCodec(2, max_iterations=15)
        with pytest.raises(ValueError, match="200"):
            codec.encode(np.zeros(10, dtype=np.uint8))

    def test_noise_tolerance(self):
        codec = L1BasicCodec(1, max_iterations=15)
        rng = np.random.default_rng(42)
        info = rng.integers(0, 2, KSIG_L1_BASIC, dtype=np.uint8)
        tx = codec.encode(info)
        llrs = np.where(tx == 1, 6.0, -6.0)
        llrs += rng.normal(0, 1.0, len(llrs))
        out, ok = codec.decode(llrs)
        assert ok
        assert np.array_equal(out, info)


class TestShortening:
    def test_padded_positions_known_zero(self):
        from atsc3lib.signaling_fec import info_positions
        from atsc3lib import spec
        positions, padded = info_positions(
            KLD_PC, NOUTER, spec.L1B_SHORTENING_PATTERN)
        assert len(padded) == KLD_PC
        assert int(padded.sum()) == KLD_PC - NOUTER
        assert len(positions) == NOUTER
        assert not np.any(padded[positions])
