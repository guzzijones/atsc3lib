"""Unit tests for the exact A/322 group/bit interleaver (short and normal)."""

import numpy as np
import pytest

from atsc3lib.group_interleaver import (
    GroupInterleaver, deinterleave_llrs, _GROUP_TABLES
)
from atsc3lib.ldpc_exact import NINNER_SHORT, NINNER_NORMAL


MODULATIONS = ['QPSK', '16QAM', '64QAM', '256QAM']


class TestTables:
    def test_tables_cover_both_frame_lengths(self):
        for n, ngroup in ((NINNER_SHORT, 45), (NINNER_NORMAL, 180)):
            assert n in _GROUP_TABLES
            for mod in MODULATIONS:
                assert mod in _GROUP_TABLES[n]
                for rate in range(2, 14):
                    perm = _GROUP_TABLES[n][mod][rate]
                    assert len(perm) == ngroup
                    assert sorted(perm) == list(range(ngroup))

    def test_official_short_frame_entries(self):
        # A/322 Annex B.2 values (verified against drmpeg/gr-atsc3).
        assert _GROUP_TABLES[NINNER_SHORT]['QPSK'][2][:6] == [0, 2, 4, 6, 8, 10]
        assert _GROUP_TABLES[NINNER_SHORT]['QPSK'][3][:6] == [15, 22, 34, 19, 7, 17]
        assert _GROUP_TABLES[NINNER_SHORT]['16QAM'][2][:6] == [5, 33, 18, 8, 29, 10]
        assert _GROUP_TABLES[NINNER_SHORT]['QPSK'][6][:6] == [7, 4, 0, 5, 27, 30]

    def test_official_normal_frame_entries(self):
        # A/322 Annex B.1 Table B.1.1 (QPSK, 2/15) printed vector.
        assert _GROUP_TABLES[NINNER_NORMAL]['QPSK'][2][:8] == [
            70, 149, 136, 153, 104, 110, 134, 61]


class TestInterleaverConstruction:
    @pytest.mark.parametrize('modulation', MODULATIONS)
    @pytest.mark.parametrize('rate', [4, 6, 8, 10, 13])
    @pytest.mark.parametrize('n', [NINNER_SHORT, NINNER_NORMAL])
    def test_order_is_permutation(self, rate, modulation, n):
        gi = GroupInterleaver(rate, modulation, n=n)
        assert len(gi.order) == n
        assert np.array_equal(np.sort(gi.order), np.arange(n))

    def test_unsupported_modulation(self):
        with pytest.raises(NotImplementedError, match="modulation"):
            GroupInterleaver(6, '1024QAM')

    def test_unsupported_length(self):
        with pytest.raises(ValueError, match="Only N"):
            GroupInterleaver(6, 'QPSK', n=99)

    def test_type_a_rates_use_block_type_a(self):
        assert GroupInterleaver(4, 'QPSK').block_type == 'A'

    def test_normal_frame_type_b_overrides_fec_type(self):
        # A/322 Table 6.8: 16QAM 5/15 is Type A FEC but Type B block.
        gi = GroupInterleaver(5, '16QAM', n=NINNER_NORMAL)
        assert gi.ldpc_type == 'A'
        assert gi.block_type == 'B'

    def test_type_b_parity_interleaver_enabled(self):
        gi = GroupInterleaver(6, 'QPSK')
        assert gi.ldpc_type == 'B'
        assert gi.q_val == 27


class TestRoundTrip:
    @pytest.mark.parametrize('modulation', MODULATIONS)
    @pytest.mark.parametrize('rate', [4, 6, 7, 10, 13])
    @pytest.mark.parametrize('n', [NINNER_SHORT, NINNER_NORMAL])
    def test_interleave_deinterleave_bits(self, rate, modulation, n):
        gi = GroupInterleaver(rate, modulation, n=n)
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


class TestPrintedExamples:
    def test_type_a_block_256qam_normal(self):
        # A/322 6.2.3.1 worked example: q0.. = v0,v7920,... and the Part 2 tail
        # begins at q63360 = v63360,v63540,...
        from atsc3lib.group_interleaver import TYPE_A_BLOCK
        gi = GroupInterleaver(2, '256QAM', n=NINNER_NORMAL)
        gi.block_type, gi.block = 'A', TYPE_A_BLOCK[(NINNER_NORMAL, '256QAM')]
        out = gi._block_interleave_type_a(np.arange(NINNER_NORMAL))
        assert out[:8].tolist() == [0, 7920, 15840, 23760, 31680, 39600,
                                    47520, 55440]
        assert out[63360:63368].tolist() == [63360, 63540, 63720, 63900,
                                             64080, 64260, 64440, 64620]


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

    @pytest.mark.parametrize('modulation,rate,error_rate',
                             [('QPSK', 4, 0.03), ('64QAM', 11, 0.01),
                              ('256QAM', 11, 0.005)])
    def test_end_to_end_normal_frame(self, modulation, rate, error_rate):
        from atsc3lib.ldpc_exact import ATSC3LDPCExact

        codec = ATSC3LDPCExact(rate, n=NINNER_NORMAL)
        gi = GroupInterleaver(rate, modulation, n=NINNER_NORMAL)
        rng = np.random.default_rng(rate)

        info = rng.integers(0, 2, codec.K, dtype=np.uint8)
        on_air = gi.interleave(codec.encode(info))
        received = on_air.copy()
        received[rng.random(codec.n) < error_rate] ^= 1
        llrs = np.where(received == 1, 4.0, -4.0)

        decoded, converged = codec.decode(gi.deinterleave_llrs(llrs))
        assert converged
        assert np.array_equal(decoded, info)
