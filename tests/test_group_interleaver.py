"""Unit tests for the exact A/322 group/bit interleaver (N=16200)."""

import numpy as np
import pytest

from atsc3lib.group_interleaver import (
    GroupInterleaver, deinterleave_llrs, _GROUP_TABLES
)


MODULATIONS = ['QPSK', '16QAM', '64QAM', '256QAM']


class TestTables:
    def test_tables_cover_modulations_and_rates(self):
        for mod_key in (1, 2, 3, 4):
            assert mod_key in _GROUP_TABLES
            for rate in range(2, 14):
                perm = _GROUP_TABLES[mod_key][rate]
                assert len(perm) == 45
                assert sorted(perm) == list(range(45))

    def test_official_short_frame_entries(self):
        # A/322 Annex table values (verified against drmpeg/gr-atsc3).
        assert _GROUP_TABLES[1][2][:6] == [0, 2, 4, 6, 8, 10]
        assert _GROUP_TABLES[1][3][:6] == [15, 22, 34, 19, 7, 17]
        assert _GROUP_TABLES[2][2][:6] == [5, 33, 18, 8, 29, 10]
        assert _GROUP_TABLES[1][6][:6] == [7, 4, 0, 5, 27, 30]


class TestInterleaverConstruction:
    @pytest.mark.parametrize('modulation', MODULATIONS)
    @pytest.mark.parametrize('rate', [4, 6, 8, 10, 13])
    def test_order_is_permutation(self, rate, modulation):
        gi = GroupInterleaver(rate, modulation)
        assert len(gi.order) == 16200
        assert np.array_equal(np.sort(gi.order), np.arange(16200))

    def test_unsupported_modulation(self):
        with pytest.raises(NotImplementedError, match="modulation"):
            GroupInterleaver(6, '1024QAM')

    def test_unsupported_length(self):
        with pytest.raises(ValueError, match="16200"):
            GroupInterleaver(6, 'QPSK', n=64800)

    def test_type_a_rates_use_block_type_a(self):
        assert GroupInterleaver(4, 'QPSK').block_type == 'A'

    def test_type_b_parity_interleaver_enabled(self):
        gi = GroupInterleaver(6, 'QPSK')
        assert gi.ldpc_type == 'B'
        assert gi.q_val == 27


class TestRoundTrip:
    @pytest.mark.parametrize('modulation', MODULATIONS)
    @pytest.mark.parametrize('rate', [4, 6, 7, 10, 13])
    def test_interleave_deinterleave_bits(self, rate, modulation):
        gi = GroupInterleaver(rate, modulation)
        rng = np.random.default_rng(rate)
        bits = rng.integers(0, 2, gi.n, dtype=np.uint8)
        assert np.array_equal(gi.deinterleave(gi.interleave(bits)), bits)

    def test_deinterleave_llrs_matches_bits(self):
        gi = GroupInterleaver(10, '64QAM')
        rng = np.random.default_rng(0)
        llrs = rng.normal(size=gi.n)
        out = gi.deinterleave_llrs(llrs)
        assert out.dtype == llrs.dtype
        assert np.array_equal(out, gi.deinterleave(llrs))


class TestConvenienceFunctions:
    def test_deinterleave_llrs_function(self):
        llrs = np.random.default_rng(1).normal(size=16200)
        out = deinterleave_llrs(llrs, 6, 'QPSK')
        assert len(out) == 16200


class TestLDPCIntegration:
    @pytest.mark.parametrize('modulation,rate,error_rate',
                             [('QPSK', 4, 0.03), ('16QAM', 7, 0.02),
                              ('64QAM', 10, 0.01)])
    def test_end_to_end_interleaved_decode(self, modulation, rate, error_rate):
        from atsc3lib.ldpc_exact import ATSC3LDPCExact

        codec = ATSC3LDPCExact(rate)
        gi = GroupInterleaver(rate, modulation)
        rng = np.random.default_rng(rate)

        info = rng.integers(0, 2, codec.K, dtype=np.uint8)
        codeword = codec.encode(info)
        on_air = gi.interleave(codeword)

        received = on_air.copy()
        received[rng.random(codec.n) < error_rate] ^= 1
        llrs = np.where(received == 1, 4.0, -4.0)

        decoded, converged = codec.decode(gi.deinterleave_llrs(llrs))
        assert converged
        assert np.array_equal(decoded, info)
