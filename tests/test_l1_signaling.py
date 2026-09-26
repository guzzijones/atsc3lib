"""Tests for the L1-Basic and L1-Detail signalling parsers.

The real-air fixtures are 200-bit L1-Basic and 512-bit L1-Detail blocks decoded
from an RF33 (587 MHz) broadcast and independently confirmed by a second
receiver implementation.  Field values and the CRC are ground truth.
"""

import json
import os

import numpy as np
import pytest

from atsc3lib import spec
from atsc3lib.l1_signaling import parse_l1_basic, parse_l1_detail

_DATA = os.path.join(os.path.dirname(__file__), 'data')
_L1B = os.path.join(_DATA, 'rf33_l1basic_info.npy')
_L1D = os.path.join(_DATA, 'rf33_l1detail_info.npy')
_FIELDS = os.path.join(_DATA, 'rf33_l1detail_fields.json')


class TestParseL1BasicAir:
    def _lb(self):
        if not os.path.exists(_L1B):
            pytest.skip("real-air fixture not present")
        return parse_l1_basic(np.load(_L1B))

    def test_header_fields(self):
        lb = self._lb()
        assert lb.version == 0
        assert lb.crc_ok
        assert lb.frame_length_mode == 1

    def test_detail_pointer_fields(self):
        lb = self._lb()
        assert lb.l1_detail_size_bytes == 64
        assert lb.l1_detail_fec_type == 2      # Mode 3
        assert lb.l1_detail_total_cells == 880
        assert lb.preamble_num_symbols == 0     # NP = 1
        assert lb.l1_detail_additional_parity_mode == 0
        assert lb.time_info_flag == 3

    def test_subframe_fields(self):
        lb = self._lb()
        assert lb.num_subframes == 1
        assert lb.first_sub_fft_size == 0       # 8K
        assert lb.first_sub_guard_interval == 6
        assert lb.first_sub_num_ofdm_symbols == 34

    def test_reserved_default_ones(self):
        lb = self._lb()
        assert lb.raw['L1B_reserved'] == (1 << 48) - 1


class TestParseL1DetailAir:
    def _parse(self):
        if not (os.path.exists(_L1B) and os.path.exists(_L1D)):
            pytest.skip("real-air fixtures not present")
        lb = parse_l1_basic(np.load(_L1B))
        return parse_l1_detail(np.load(_L1D), lb)

    def test_header_and_crc(self):
        ld = self._parse()
        assert ld.version == 1
        assert ld.num_rf == 0
        assert ld.bsid == 540
        assert ld.crc_ok
        assert ld.reserved_len == 7
        assert ld.reserved_all_ones

    def test_subframe_and_plp_structure(self):
        ld = self._parse()
        assert len(ld.subframes) == 2

        sf0 = ld.subframes[0]
        assert sf0['num_plp'] == 1          # 2 PLPs
        assert len(sf0['plps']) == 2
        assert [p.plp_id for p in sf0['plps']] == [0, 16]

        sf1 = ld.subframes[1]
        assert sf1['fft_size'] == 1         # 16K
        assert sf1['guard_interval'] == 6
        assert sf1['num_ofdm_symbols'] == 74
        assert len(sf1['plps']) == 1
        assert sf1['plps'][0].plp_id == 1

    def test_plp_modcods(self):
        ld = self._parse()
        p0, p16 = ld.subframes[0]['plps']
        assert p0.fec_type == 0 and p0.modulation == 2 and p0.code_rate == 9
        assert p16.fec_type == 0 and p16.modulation == 0 and p16.code_rate == 0
        p1 = ld.subframes[1]['plps'][0]
        assert p1.fec_type == 1 and p1.modulation == 3 and p1.code_rate == 9

    def test_matches_independent_parse(self):
        if not os.path.exists(_FIELDS):
            pytest.skip("oracle field list not present")
        expected = {(r['path'], r['name']): r['value']
                    for r in json.load(open(_FIELDS))}
        ld = self._parse()
        # Spot-check a representative spread of the independent parse.
        assert ld.bsid == expected[('', 'L1D_bsid')]
        assert ld.subframes[0]['plps'][1].size == expected[
            ('i=0/j=1/', 'L1D_plp_size')]
        assert ld.subframes[1]['plps'][0].size == expected[
            ('i=1/j=0/', 'L1D_plp_size')]
        assert ld.subframes[1]['plps'][0].modulation == expected[
            ('i=1/j=0/', 'L1D_plp_mod')]


class TestSyntheticRoundTrip:
    def test_l1_detail_field_parse(self):
        # Build a minimal L1-Detail bit string for a single trivial subframe
        # with no PLPs where possible, and confirm the parser reads fields in
        # order rather than by fixed offset.
        bits = []
        def put(value, width):
            bits.extend((value >> (width - 1 - i)) & 1 for i in range(width))

        put(1, 4)          # L1D_version
        put(0, 3)          # L1D_num_rf
        # L1B_time_info_flag = 0 -> no time block
        # subframe 0
        put(0, 1)          # L1D_frequency_interleaver
        # no SBS (first_sub_sbs_first/last = 0)
        put(0, 6)          # L1D_num_plp  (=> 1 PLP)
        # PLP
        put(5, 6)          # id
        put(0, 1)          # lls
        put(0, 2)          # layer = 0
        put(100, 24)       # start
        put(200, 24)       # size
        put(0, 2)          # scrambler
        put(0, 4)          # fec_type = 0 (16K LDPC) -> mod/cod present
        put(0, 4)          # mod QPSK
        put(0, 4)          # cod 2/15
        put(0, 2)          # TI_mode 0 -> 15-bit start
        put(7, 15)         # fec_block_start
        put(0, 1)          # plp_type
        put(0x1234, 16)    # bsid
        # reserved to fill out to a 32-bit CRC boundary
        while (len(bits) + 32) % 8 != 0:
            bits.append(1)
        reserved = 8
        bits.extend([1] * reserved)
        # CRC (computed with the all-ones init over data+CRC -> residue 0)
        from atsc3lib.crc import crc32
        crc = crc32(bits)
        put(crc, 32)

        from atsc3lib.l1_signaling import L1Basic
        lb = L1Basic(
            version=0, mimo_scattered_pilot_encoding=0, lls_flag=0,
            time_info_flag=0, return_channel_flag=0, papr_reduction=0,
            frame_length_mode=1, time_offset=0, num_subframes=0,
            preamble_num_symbols=0, preamble_reduced_carriers=0,
            l1_detail_content_tag=0, l1_detail_size_bytes=0,
            l1_detail_fec_type=2, l1_detail_additional_parity_mode=0,
            l1_detail_total_cells=0, first_sub_mimo=0, first_sub_miso=0,
            first_sub_fft_size=0, first_sub_reduced_carriers=0,
            first_sub_guard_interval=0, first_sub_num_ofdm_symbols=0,
            first_sub_scattered_pilot_pattern=0,
            first_sub_scattered_pilot_boost=0, first_sub_sbs_first=0,
            first_sub_sbs_last=0, crc_ok=True, raw={})

        ld = parse_l1_detail(np.array(bits, dtype=np.uint8), lb)
        assert ld.version == 1
        assert ld.bsid == 0x1234
        assert ld.crc_ok
        assert len(ld.subframes) == 1
        assert ld.subframes[0]['plps'][0].plp_id == 5
        assert ld.subframes[0]['plps'][0].size == 200
