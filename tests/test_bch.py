"""Unit tests for the ATSC 3.0 BCH outer code."""

import numpy as np
import pytest

from atsc3lib.bch import BCHCode


class TestConstruction:
    def test_16200_parameters(self):
        bch = BCHCode(16200, 12)
        assert bch.mouter == 168
        assert bch.bits == 14
        assert bch.n_full == 16383
        assert bch.k_full == 16215

    def test_64800_parameters(self):
        bch = BCHCode(64800, 12)
        assert bch.mouter == 192
        assert bch.bits == 16

    def test_unsupported_length(self):
        with pytest.raises(ValueError, match="Unsupported"):
            BCHCode(1000)

    def test_primitive_polynomial(self):
        # g1 must be primitive for GF(2^14): alpha has order 2^14 - 1.
        bch = BCHCode(16200)
        assert bch.gf.exp[bch.gf.order] == 1


class TestEncode:
    def test_parity_length(self):
        bch = BCHCode(16200)
        msg = np.random.default_rng(0).integers(0, 2, 500, dtype=np.uint8)
        cw = bch.encode(msg)
        assert len(cw) == 500 + 168

    def test_message_too_long(self):
        bch = BCHCode(16200)
        with pytest.raises(ValueError, match="too long"):
            bch.encode(np.zeros(bch.k_full + 1, dtype=np.uint8))

    def test_clean_codeword_has_zero_syndrome(self):
        bch = BCHCode(16200)
        msg = np.random.default_rng(3).integers(0, 2, 200, dtype=np.uint8)
        cw = bch.encode(msg)
        synd = bch._syndromes(list(cw))
        assert all(s == 0 for s in synd[1:])


class TestDecode:
    @pytest.mark.parametrize('kpayload', [200, 1000, 5000, 10000])
    def test_clean(self, kpayload):
        bch = BCHCode(16200)
        msg = np.random.default_rng(kpayload).integers(0, 2, kpayload, dtype=np.uint8)
        cw = bch.encode(msg)
        out, nerr, ok = bch.decode(cw)
        assert ok and nerr == 0
        assert np.array_equal(out, msg)

    @pytest.mark.parametrize('num_errors', [1, 6, 12])
    def test_corrects_errors(self, num_errors):
        bch = BCHCode(16200)
        rng = np.random.default_rng(num_errors)
        msg = rng.integers(0, 2, 2000, dtype=np.uint8)
        cw = np.array(bch.encode(msg))
        pos = rng.choice(len(cw), num_errors, replace=False)
        cw[pos] ^= 1
        out, nerr, ok = bch.decode(cw)
        assert ok
        assert nerr == num_errors
        assert np.array_equal(out, msg)

    def test_l1_basic_block_size(self):
        bch = BCHCode(16200)
        rng = np.random.default_rng(7)
        # L1-Basic: Kpayload=200, Nouter=368
        msg = rng.integers(0, 2, 200, dtype=np.uint8)
        cw = np.array(bch.encode(msg))
        cw[rng.choice(len(cw), 12, replace=False)] ^= 1
        out, nerr, ok = bch.decode(cw)
        assert ok and np.array_equal(out, msg)

    def test_uncorrectable_detected(self):
        bch = BCHCode(16200)
        rng = np.random.default_rng(11)
        msg = rng.integers(0, 2, 2000, dtype=np.uint8)
        cw = np.array(bch.encode(msg))
        rejected = 0
        for _ in range(20):
            rx = cw.copy()
            rx[rng.choice(len(rx), 16, replace=False)] ^= 1
            out, nerr, ok = bch.decode(rx)
            if not ok:
                rejected += 1
            else:
                assert np.array_equal(out, msg)
        assert rejected > 0

    def test_too_short(self):
        bch = BCHCode(16200)
        with pytest.raises(ValueError, match="shorter"):
            bch.decode(np.zeros(10, dtype=np.uint8))

    def test_64800_corrects(self):
        bch = BCHCode(64800)
        rng = np.random.default_rng(2)
        msg = rng.integers(0, 2, 3000, dtype=np.uint8)
        cw = np.array(bch.encode(msg))
        cw[rng.choice(len(cw), 12, replace=False)] ^= 1
        out, nerr, ok = bch.decode(cw)
        assert ok and np.array_equal(out, msg)
