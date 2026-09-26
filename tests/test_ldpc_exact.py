"""Unit tests for the exact A/322 LDPC codec (N=16200 short frames)."""

import numpy as np
import pytest

from atsc3lib.ldpc_exact import (
    ATSC3LDPCExact, TYPE_A_PARAMS, TYPE_B_QLDPC, get_code_params, load_tables
)


class TestCodeParams:
    def test_type_a_rates(self):
        for rate in TYPE_A_PARAMS:
            k, m, ctype = get_code_params(rate)
            assert ctype == 'A'
            assert k == 16200 * rate // 15
            assert m == 16200 - k

    def test_type_b_rates(self):
        for rate in TYPE_B_QLDPC:
            k, m, ctype = get_code_params(rate)
            assert ctype == 'B'
            assert k == 16200 * rate // 15
            assert m == 16200 - k

    def test_tables_present_for_all_rates(self):
        tables = load_tables()
        for rate in range(2, 14):
            assert rate - 1 in tables
            assert tables[rate - 1]['rate'] == rate


class TestEncoder:
    @pytest.mark.parametrize('rate', [2, 4, 6, 7, 10, 13])
    def test_encode_produces_valid_codeword(self, rate):
        codec = ATSC3LDPCExact(rate)
        rng = np.random.default_rng(rate)
        info = rng.integers(0, 2, codec.K, dtype=np.uint8)
        cw = codec.encode(info)
        assert len(cw) == codec.n
        assert codec.check_syndrome(cw) is True

    def test_encode_wrong_length(self):
        codec = ATSC3LDPCExact(6)
        with pytest.raises(ValueError, match="Expected"):
            codec.encode(np.zeros(10, dtype=np.uint8))

    def test_unsupported_length(self):
        with pytest.raises(ValueError, match="16200"):
            ATSC3LDPCExact(6, n=64800)

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
