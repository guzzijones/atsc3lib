"""Tests for the L1-Detail FEC codec (A/322 6.5.2, Mode 3)."""

import numpy as np
import pytest

from atsc3lib import spec
from atsc3lib.crc import crc32
from atsc3lib.l1_detail import L1DetailCodec


def _with_crc(payload: np.ndarray) -> np.ndarray:
    c = crc32(payload)
    crc = np.array([(c >> (31 - i)) & 1 for i in range(32)], dtype=np.uint8)
    return np.concatenate([payload, crc])


class TestGeometry:
    def test_mode3_rf33(self):
        g = spec.l1d_lengths(3, 512)
        assert g.nouter == 680
        assert g.kldpc == 6480
        assert g.n_fec == 1760
        assert g.n_cells == 880
        assert g.n_punc == 8640
        assert g.rate == 6

    def test_segmentation_rejected(self):
        with pytest.raises(ValueError, match="segmentation"):
            spec.l1d_lengths(3, spec.L1D_MODES[3].kseg + 1)


class TestRoundTrip:
    @pytest.mark.parametrize('mode,ksig', [(3, 512), (4, 512), (5, 512),
                                           (6, 512), (7, 512)])
    def test_synthetic_roundtrip(self, mode, ksig):
        rng = np.random.default_rng(mode)
        payload = rng.integers(0, 2, ksig - 32, dtype=np.uint8)
        info = _with_crc(payload)
        codec = L1DetailCodec(mode, ksig, max_iterations=50)
        tx = codec.encode(info)
        assert len(tx) == codec.n_tx

        llrs = np.where(tx == 1, 8.0, -8.0)
        out, bch_ok, crc_ok = codec.decode(llrs)
        assert bch_ok and crc_ok
        assert np.array_equal(out, info)

    def test_cells_roundtrip(self):
        # Full cells path: block-interleave (6.5.2.10) + QPSK map, then the
        # receiver's demap + de-interleave.
        mode, ksig = 3, 512
        codec = L1DetailCodec(mode, ksig, max_iterations=50)
        rng = np.random.default_rng(0)
        info = _with_crc(rng.integers(0, 2, ksig - 32, dtype=np.uint8))
        tx = codec.encode(info)
        n = len(tx)
        y0 = tx[:n // 2].astype(int)   # column 0 (sets Q sign)
        y1 = tx[n // 2:].astype(int)   # column 1 (sets I sign)
        cells = ((1 - 2 * y1) + 1j * (1 - 2 * y0)) / np.sqrt(2)
        out, bch_ok, crc_ok = codec.decode_cells(cells)
        assert bch_ok and crc_ok
        assert np.array_equal(out, info)


@pytest.mark.skipif(not __import__('os').path.exists(
    __import__('os').path.join(__import__('os').path.dirname(__file__),
                              'data', 'rf33_l1detail_info.npy')),
    reason="real-air fixture not present")
class TestAir:
    def test_mode3_from_real_broadcast(self):
        import os
        base = os.path.join(os.path.dirname(__file__), 'data')
        payload = np.load(os.path.join(base, 'rf33_l1detail_info.npy'))
        codec = L1DetailCodec(3, 512, max_iterations=100)
        # The fixture is the info bits (payload+CRC); encoding must be a
        # fixed point of decode at high SNR.
        tx = codec.encode(payload)
        out, bch_ok, crc_ok = codec.decode(np.where(tx == 1, 8.0, -8.0))
        assert bch_ok and crc_ok
        assert np.array_equal(out, payload)
