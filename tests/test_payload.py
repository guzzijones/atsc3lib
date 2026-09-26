"""Tests for the ATSC 3.0 payload chain (cell pool -> PLP FEC -> Baseband Packet)."""

import json
import os

import numpy as np
import pytest

from atsc3lib import spec
from atsc3lib.payload import (
    QPSKPlpChain, build_cell_pool, data_symbol_cells, qpsk_demap_llr,
    QPSK_POINTS,
)

_DATA = os.path.join(os.path.dirname(__file__), 'data')
_FRAME = os.path.join(_DATA, 'rf33_frame_y.npy')
_FRAME_META = os.path.join(_DATA, 'rf33_frame_meta.json')
_PLP16_CELLS = os.path.join(_DATA, 'rf33_plp16_cells.npy')
_PLP16_BB = os.path.join(_DATA, 'rf33_plp16_bb.bin')


class TestGeometry:
    def test_data_symbol_cell_counts(self):
        # 8K, cred 0, SP4_2, normal data symbol and SBS.
        add = spec.additional_cp('SP4_2')
        assert spec.AVAIL_DATA_8K['SP4_2'] == 5999
        assert spec.SBS_ACTIVE_8K_CRED0 == 5009
        assert spec.SBS_NULL_8K_CRED0 == 127

    def test_additional_cp_known(self):
        assert spec.additional_cp('SP4_2') == (1732,)


class TestQpskDemap:
    def test_ideal_points(self):
        # LLR > 0 => bit 0: point labelled 00 sits at (+1+j)/sqrt2.
        llr = qpsk_demap_llr(QPSK_POINTS[:1])
        assert llr[0] > 0 and llr[1] > 0     # both bits 0
        llr = qpsk_demap_llr(QPSK_POINTS[3:4])
        assert llr[0] < 0 and llr[1] < 0     # both bits 1


@pytest.mark.skipif(not os.path.exists(_PLP16_CELLS),
                    reason="real-air fixture not present")
class TestPlp16Chain:
    def test_decodes_oracle_cells(self):
        cells = np.load(_PLP16_CELLS)
        r = QPSKPlpChain().decode_cells(cells)
        assert r.ok
        assert r.packet == open(_PLP16_BB, 'rb').read()


@pytest.mark.skipif(not os.path.exists(_FRAME),
                    reason="real-air frame fixture not present")
class TestAirFrame:
    def test_full_payload_chain(self):
        y = np.load(_FRAME)
        meta = json.load(open(_FRAME_META))
        pool = build_cell_pool(
            y, meta['t0'], meta['fft'], meta['gi'], meta['noc'],
            meta['dx'], meta['dy'], meta['n_data_symbols'],
            sbs_symbols=tuple(meta['sbs_symbols']),
            preamble_structure=meta['preamble_structure'])

        # The subframe-0 cell budget is a closed identity (preamble spare +
        # 34 data symbols + 2 SBS symbols).
        assert len(pool.cells) == 211472
        assert pool.n_preamble_spare == 3487

        p16 = meta['plp16']
        r = QPSKPlpChain().decode_cells(
            pool.cells[p16['start']:p16['start'] + p16['size']])
        assert r.ok
        assert len(r.packet) == meta['plp16_bb_bytes']

    def test_cell_counts_match_tables(self):
        y = np.load(_FRAME)
        meta = json.load(open(_FRAME_META))
        add = spec.additional_cp('SP4_2')
        # normal symbol
        l = 1
        start = meta['t0'] + (meta['fft'] + meta['gi']) * (1 + l) + meta['gi']
        result = data_symbol_cells(
            y[start:start + meta['fft']], meta['fft'], meta['noc'],
            meta['dx'], meta['dy'], l, False, add_cp=add)
        assert len(result.cells) == spec.AVAIL_DATA_8K['SP4_2']
        assert result.pilot_coherence > 0.5
