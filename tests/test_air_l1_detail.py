"""End-to-end L1-Basic -> L1-Detail decode from a real ATSC 3.0 broadcast.

The fixture is one 8K Preamble symbol captured off air on RF channel 33
(587 MHz NextGen TV multiplex) with a HackRF at 10 MS/s.  This test runs the
complete preamble-to-signalling chain and checks the decoded per-PLP
configuration against an independent receiver's parse.

Chain exercised:

    real Preamble symbol
      -> FFT + pilot channel estimate + equalize + frequency de-interleave
      -> L1-Basic cells (0..483): LDPC 3/15 + BCH + CRC
      -> L1-Detail cells (484..1363): LDPC 6/15 + BCH + CRC
      -> A/322 Table 9.2 / 9.8 parse -> PLP configuration
"""

import json
import os

import numpy as np
import pytest

from atsc3lib import spec
from atsc3lib.crc import crc32_ok
from atsc3lib.l1_basic import L1BasicCodec
from atsc3lib.l1_detail import L1DetailCodec
from atsc3lib.l1_signaling import parse_l1_basic, parse_l1_detail
from atsc3lib.preamble import (
    estimate_preamble_channel, preamble_data_cells, preamble_l1_cells,
    preamble_noc, preamble_symbol_spectrum,
)

_DATA = os.path.join(os.path.dirname(__file__), 'data')
_SYMBOL = os.path.join(_DATA, 'rf33_preamble_symbol.npy')
_L1D = os.path.join(_DATA, 'rf33_l1detail_info.npy')
_META = os.path.join(_DATA, 'rf33_preamble_meta.json')

pytestmark = pytest.mark.skipif(
    not (os.path.exists(_SYMBOL) and os.path.exists(_META)),
    reason="real-air fixture not present")


def _cells():
    symbol = np.load(_SYMBOL).astype(np.complex128)
    structure = json.load(open(_META))['preamble_structure']
    params = spec.PREAMBLE_STRUCTURE[structure]
    fft, gi, dx = params.fft, params.gi, params.dx
    carriers = preamble_symbol_spectrum(symbol, fft, noc=preamble_noc(fft))
    channel = estimate_preamble_channel(
        carriers, dx, spec.PREAMBLE_PILOT_AMPLITUDE[(fft, gi)])
    data = preamble_data_cells(carriers, channel, dx)
    # frequency de-interleave (frequency_interleaver module)
    from atsc3lib.frequency_interleaver import deinterleave as fi_deinterleave
    return fi_deinterleave(data, 0, fft), params


def test_l1_basic_then_detail_plp_config():
    cells, params = _cells()
    assert params.l1b_mode == 3

    # --- L1-Basic: cells 0..483 ---
    lb_codec = L1BasicCodec(3, max_iterations=100)
    lb_bits, lb_ok = lb_codec.decode_cells(cells[:484])
    assert lb_ok, "L1-Basic did not verify on the real-air fixture"
    assert crc32_ok(lb_bits)

    lb = parse_l1_basic(lb_bits)
    assert lb.crc_ok
    assert lb.l1_detail_size_bytes == 64
    assert lb.l1_detail_fec_type == 2          # Mode 3
    assert lb.l1_detail_total_cells == 880
    assert lb.preamble_num_symbols == 0        # NP = 1 -> identity interleaver
    assert lb.time_info_flag == 3

    # --- L1-Detail: cells 484..1363 ---
    ksig = lb.l1_detail_size_bytes * 8
    ld_codec = L1DetailCodec(lb.l1_detail_fec_type + 1, ksig,
                             max_iterations=100)
    assert ld_codec.n_cells == lb.l1_detail_total_cells

    ld_bits, bch_ok, crc_ok = ld_codec.decode_cells(
        cells[484:484 + ld_codec.n_cells])
    assert bch_ok and crc_ok, "L1-Detail did not verify"

    # must match the independent receiver's decoded bits
    if os.path.exists(_L1D):
        assert np.array_equal(ld_bits, np.load(_L1D))

    # --- A/322 Table 9.8 parse ---
    ld = parse_l1_detail(ld_bits, lb)
    assert ld.crc_ok
    assert ld.version == 1
    assert ld.bsid == 540
    assert ld.reserved_all_ones and ld.reserved_len == 7

    assert len(ld.subframes) == 2
    sf0, sf1 = ld.subframes
    assert [p.plp_id for p in sf0['plps']] == [0, 16]
    assert sf1['fft_size'] == 1                 # 16K
    assert sf1['num_ofdm_symbols'] == 74
    assert [p.plp_id for p in sf1['plps']] == [1]

    # Per-PLP MODCODs (the deliverable)
    p0, p16 = sf0['plps']
    assert (p0.fec_type, p0.modulation, p0.code_rate) == (0, 2, 9)
    assert (p16.fec_type, p16.modulation, p16.code_rate) == (0, 0, 0)
    p1 = sf1['plps'][0]
    assert (p1.fec_type, p1.modulation, p1.code_rate) == (1, 3, 9)
