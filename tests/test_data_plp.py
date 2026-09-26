"""Tests for the data-PLP BICM chain (NUC, HTI, LDPC/BCH/descramble)."""

import json
import os

import numpy as np
import pytest

from atsc3lib import nuc, twisted_block, cell_interleaver
from atsc3lib.payload import DataPlpChain

_DATA = os.path.join(os.path.dirname(__file__), 'data')
_HTI_CELLS = os.path.join(_DATA, 'plp0_hti_cells.npy')
_HTI_CI_CELLS = os.path.join(_DATA, 'plp0_hti_ci_cells.npy')
_HTI_PAYLOAD = os.path.join(_DATA, 'plp0_hti_payload.npy')
_HTI_META = os.path.join(_DATA, 'plp0_hti_meta.json')
_HTI_CI_META = os.path.join(_DATA, 'plp0_hti_ci_meta.json')


class TestNuc:
    def test_qpsk_is_uniform(self):
        p = nuc.points(2, 2)
        assert len(p) == 4
        assert np.allclose(np.abs(p.real), 1 / np.sqrt(2))
        assert np.allclose(np.abs(p.imag), 1 / np.sqrt(2))

    @pytest.mark.parametrize("mo,rate", [(4, 11), (6, 11), (8, 11),
                                         (6, 2), (8, 9)])
    def test_nuc_mean_power_is_one(self, mo, rate):
        p = nuc.points(mo, rate)
        assert len(p) == (1 << mo)
        assert abs(float(np.mean(np.abs(p) ** 2)) - 1.0) < 1e-3

    def test_demap_ideal_point_is_confident(self):
        p = nuc.points(6, 11)
        llr = nuc.demap_llr(p[:1], 6, 11)
        assert np.all(llr > 0)          # label 0 => all bits 0


class TestTwistedBlock:
    def test_a327_gold_vector(self):
        got, expect = twisted_block.gold_vector()
        assert got == expect

    @pytest.mark.parametrize("nr,nc,nv", [(2700, 37, 0), (8100, 1, 0),
                                          (2025, 16, 7), (2700, 37, 5)])
    def test_read_order_is_bijection(self, nr, nc, nv):
        o = twisted_block.read_order(nr, nc, nv)
        assert np.array_equal(np.sort(o), np.arange(nr * (nc - nv)))

    def test_roundtrip_and_fec_block(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(2700 * 37)
        mem = twisted_block.deinterleave(x, 2700, 37)
        assert np.array_equal(twisted_block.interleave(mem, 2700, 37), x)
        for j in range(37):
            assert np.array_equal(
                twisted_block.fec_block(x, j, 2700, 37), mem[j * 2700:(j + 1) * 2700])


class TestCellInterleaver:
    def test_a322_gold_shift_vector(self):
        got, expect = cell_interleaver.gold_vector()
        assert got == expect

    def test_nd_rejects_untapped_widths(self):
        with pytest.raises(ValueError):
            cell_interleaver.CellInterleaver(100)      # N_d = 7

    @pytest.mark.parametrize("ncells", [8100, 10800, 5400, 16200])
    def test_basic_permutation_is_bijection(self, ncells):
        c0 = cell_interleaver.CellInterleaver(ncells).l0
        assert len(c0) == ncells
        assert np.array_equal(np.sort(c0), np.arange(ncells))

    @pytest.mark.parametrize("ncells,nblocks", [(8100, 5), (10800, 3),
                                                (16200, 2)])
    def test_roundtrip(self, ncells, nblocks):
        rng = np.random.default_rng(0)
        x = (rng.standard_normal((nblocks, ncells))
             + 1j * rng.standard_normal((nblocks, ncells)))
        ci = cell_interleaver.CellInterleaver(ncells)
        assert np.allclose(ci.deinterleave(ci.interleave(x)), x)


class TestDataPlpChain:
    @pytest.mark.skipif(not os.path.exists(_HTI_CELLS),
                        reason="oracle fixture not present")
    def test_plp0_hti_roundtrip(self):
        cells = np.load(_HTI_CELLS)
        meta = json.load(open(_HTI_META))
        from atsc3lib.payload import decode_data_plp
        r = decode_data_plp(cells, meta['mod'], meta['rate'],
                            nti=meta['nti'], n_fec=meta['n_fec'])
        assert r.n_fec == meta['n_fec']
        assert r.n_converged == meta['n_fec']
        expect = np.load(_HTI_PAYLOAD)
        for block, ref in zip(r.fec_blocks, expect):
            assert np.array_equal(np.asarray(block.payload_bits), ref)

    @pytest.mark.skipif(not os.path.exists(_HTI_CI_CELLS),
                        reason="oracle fixture not present")
    def test_plp0_hti_cell_interleaver_roundtrip(self):
        cells = np.load(_HTI_CI_CELLS)
        meta = json.load(open(_HTI_CI_META))
        from atsc3lib.payload import decode_data_plp
        r = decode_data_plp(cells, meta['mod'], meta['rate'],
                            nti=meta['nti'], n_fec=meta['n_fec'],
                            cell_interleaver=1)
        assert r.n_converged == meta['n_fec']
        expect = np.load(_HTI_PAYLOAD)
        for block, ref in zip(r.fec_blocks, expect):
            assert np.array_equal(np.asarray(block.payload_bits), ref)

    @pytest.mark.skipif(not os.path.exists(_HTI_CI_CELLS),
                        reason="oracle fixture not present")
    def test_plp0_hti_cell_interleaver_is_required(self):
        # Decoding the cell-interleaved stream with the flag off must fail:
        # otherwise the fixture proves nothing about the stage.
        cells = np.load(_HTI_CI_CELLS)
        meta = json.load(open(_HTI_CI_META))
        from atsc3lib.payload import decode_data_plp
        r = decode_data_plp(cells, meta['mod'], meta['rate'],
                            nti=meta['nti'], n_fec=meta['n_fec'],
                            cell_interleaver=0)
        assert r.n_converged < meta['n_fec']
