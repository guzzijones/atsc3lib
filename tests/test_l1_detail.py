"""Tests for the L1-Detail FEC codec (A/322 6.5.2, Mode 3)."""

import numpy as np
import pytest

from atsc3lib import spec
from atsc3lib.crc import crc32
from atsc3lib.l1_detail import L1DetailCodec, preamble_block_deinterleave


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

    def test_mode1_repetition_geometry(self):
        # A/322 Table 6.23: Nrepeat = 2*floor(61/16 * Nouter) - 508 (Mode 1).
        # Ksig 336 bits (42 bytes) gives exactly the 3611 cells an independent
        # receiver reads as L1B_L1_Detail_total_cells on RF30 (WIAV 569 MHz),
        # which is the free signalled/derived gate M8 used.
        g = spec.l1d_lengths(1, 336)
        assert g.nouter == 504
        assert g.n_repeat == 2 * (61 * 504 // 16) - 508 == 3334
        assert g.n_tx == g.n_fec + g.n_repeat == 7222
        assert g.n_cells == 3611
        # Modes 2-7 have no repetition (Table 6.23).
        for mode in range(2, 8):
            assert spec.l1d_lengths(mode, 512).n_repeat == 0

    def test_mode1_repetition_roundtrip(self):
        ksig = 336
        rng = np.random.default_rng(1)
        payload = rng.integers(0, 2, ksig - 32, dtype=np.uint8)
        info = _with_crc(payload)
        codec = L1DetailCodec(1, ksig, max_iterations=60)
        tx = codec.encode(info)
        assert len(tx) == codec.n_tx == codec.n_fec + codec.n_repeat
        llrs = np.where(tx == 1, 8.0, -8.0)
        out, bch_ok, crc_ok = codec.decode(llrs)
        assert bch_ok and crc_ok and np.array_equal(out[:ksig - 32], payload)


class TestPreambleBlockInterleaver:
    """A/322 7.2.5.2: L1-Detail spread over NP Preamble symbols."""

    @staticmethod
    def _interleave(m, np_sym):
        total = len(m)
        lr = total // np_sym
        full = np_sym * lr
        y = np.empty(total, dtype=m.dtype)
        if full:
            n = np.arange(full)
            i = n // lr
            j = n % lr
            y[n] = m[j * np_sym + i]
        if total > full:
            y[full:] = m[full:]
        return y

    @pytest.mark.parametrize('total,np_sym', [
        (3708, 2), (3611, 2), (7222, 2), (100, 1), (101, 2), (5, 2)])
    def test_inverse(self, total, np_sym):
        m = np.arange(total)
        y = self._interleave(m, np_sym)
        assert np.array_equal(preamble_block_deinterleave(y, np_sym), m)

    def test_rf30_geometry(self):
        # RF30: total_cells 3708 over NP = 2 -> Lr = 1854, full coverage.
        total, np_sym = 3708, 2
        assert total % np_sym == 0
        assert total // np_sym == 1854


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
