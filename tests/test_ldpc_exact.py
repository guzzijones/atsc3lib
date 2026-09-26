"""Unit tests for the exact A/322 LDPC codec (short and normal frames)."""

import numpy as np
import pytest

from atsc3lib.ldpc_exact import (
    ATSC3LDPCExact, NINNER_SHORT, NINNER_NORMAL,
    TYPE_A_PARAMS_16200, TYPE_B_QLDPC_16200, get_code_params, load_tables,
)


class TestCodeParams:
    def test_type_a_rates(self):
        for rate in TYPE_A_PARAMS_16200:
            k, m, ctype = get_code_params(rate)
            assert ctype == 'A'
            assert k == NINNER_SHORT * rate // 15
            assert m == NINNER_SHORT - k

    def test_type_b_rates(self):
        for rate in TYPE_B_QLDPC_16200:
            k, m, ctype = get_code_params(rate)
            assert ctype == 'B'
            assert k == NINNER_SHORT * rate // 15
            assert m == NINNER_SHORT - k

    def test_normal_frame_rate_7_is_type_a(self):
        # A/322 Table 6.5: 7/15 is Type A at Ninner=64800.
        _, _, ctype = get_code_params(7, n=NINNER_NORMAL)
        assert ctype == 'A'

    def test_tables_present_for_all_rates(self):
        for n in (NINNER_SHORT, NINNER_NORMAL):
            tables = load_tables(n)
            for rate in range(2, 14):
                assert tables[rate]['rate'] == rate


class TestEncoder:
    @pytest.mark.parametrize('rate', [2, 4, 6, 7, 10, 13])
    def test_encode_produces_valid_codeword(self, rate):
        codec = ATSC3LDPCExact(rate)
        rng = np.random.default_rng(rate)
        info = rng.integers(0, 2, codec.K, dtype=np.uint8)
        cw = codec.encode(info)
        assert len(cw) == codec.n
        assert codec.check_syndrome(cw) is True

    @pytest.mark.parametrize('rate', [2, 5, 6, 7, 11, 13])
    def test_encode_produces_valid_normal_codeword(self, rate):
        codec = ATSC3LDPCExact(rate, n=NINNER_NORMAL)
        rng = np.random.default_rng(100 + rate)
        info = rng.integers(0, 2, codec.K, dtype=np.uint8)
        cw = codec.encode(info)
        assert len(cw) == codec.n
        assert codec.check_syndrome(cw) is True

    def test_encode_wrong_length(self):
        codec = ATSC3LDPCExact(6)
        with pytest.raises(ValueError, match="Expected"):
            codec.encode(np.zeros(10, dtype=np.uint8))

    def test_unsupported_length(self):
        with pytest.raises(ValueError, match="Ninner"):
            ATSC3LDPCExact(6, n=99)

    def test_unsupported_rate(self):
        with pytest.raises(ValueError, match="Rate"):
            ATSC3LDPCExact(14)


class TestDecoder:
    @pytest.mark.parametrize('rate', [2, 4, 6, 10, 13])
    def test_decode_clean_codeword(self, rate):
        codec = ATSC3LDPCExact(rate)
        rng = np.random.default_rng(1000 + rate)
        info = rng.integers(0, 2, codec.K, dtype=np.uint8)
        cw = codec.encode(info)
        llrs = np.where(cw == 1, 5.0, -5.0)
        out, converged = codec.decode(llrs)
        assert converged
        assert np.array_equal(out, info)

    @pytest.mark.parametrize('rate,error_rate', [(4, 0.03), (6, 0.03), (10, 0.01)])
    def test_decode_corrects_errors(self, rate, error_rate):
        codec = ATSC3LDPCExact(rate, max_iterations=50)
        rng = np.random.default_rng(2000 + rate)
        info = rng.integers(0, 2, codec.K, dtype=np.uint8)
        cw = codec.encode(info)
        noisy = cw.copy()
        flip = rng.random(codec.n) < error_rate
        noisy[flip] ^= 1
        llrs = np.where(noisy == 1, 4.0, -4.0)
        out, converged = codec.decode(llrs)
        assert converged
        assert np.array_equal(out, info)

    def test_decode_wrong_length(self):
        codec = ATSC3LDPCExact(6)
        with pytest.raises(ValueError, match="LLRs"):
            codec.decode(np.zeros(100))

    def test_llr_convention_positive_is_one(self):
        codec = ATSC3LDPCExact(10)
        rng = np.random.default_rng(7)
        info = rng.integers(0, 2, codec.K, dtype=np.uint8)
        cw = codec.encode(info)
        llrs = np.where(cw == 1, 5.0, -5.0)
        out, converged = codec.decode(llrs)
        assert converged and np.array_equal(out, info)
