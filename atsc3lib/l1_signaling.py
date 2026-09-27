"""ATSC 3.0 L1-Basic and L1-Detail signalling parsers.

L1-Basic is the fixed 200-bit block at the start of the Preamble (A/322
Table 9.2).  L1-Detail is the variable-length block that carries the per-PLP
configuration (A/322 Table 9.8).  Both are plain MSB-first bit fields; this
module turns decoded bits into named values.

The syntax follows A/322:2024-04 Tables 9.2 and 9.8 directly.  Note that
several fields are "value minus one" (`L1B_num_subframes`, `L1D_num_plp`), and
that L1-Detail's structure depends on field values read earlier in the same
block (for example `L1B_num_subframes`, `L1B_first_sub_sbs_last`), so parsing
is sequential rather than a fixed offset table.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .crc import crc32_ok


# ---------------------------------------------------------------------------
# Small label maps (A/322 Tables 9.x), for readable output only.
# ---------------------------------------------------------------------------
FFT_SIZE = {0: 8192, 1: 16384, 2: 32768, 3: 4096, 4: 2048, 5: 1024}
FFT_SIZE_2BIT = {0: 8192, 1: 16384, 2: 32768, 3: 8192}
MODULATION = {0: 'QPSK', 1: '16QAM', 2: '64QAM', 3: '256QAM',
              4: '1024QAM', 5: '4096QAM'}
L1D_MODULATION = {0: 'QPSK', 1: '16QAM-NUC', 2: '64QAM-NUC', 3: '256QAM-NUC',
                  4: '1024QAM-NUC', 5: '4096QAM-NUC'}
CODE_RATE = {i: f"{i + 2}/15" for i in range(12)}          # 2/15 .. 13/15
#: L1D_plp_fec_type code values (A/322 Table 9.8).
FEC_BCH_16K, FEC_BCH_64K = 0, 1
FEC_CRC_16K, FEC_CRC_64K = 2, 3
FEC_16K, FEC_64K = 4, 5
FEC_NONE = 6
L1D_FEC_TYPE = {FEC_BCH_16K: 'BCH + 16K LDPC', FEC_BCH_64K: 'BCH + 64K LDPC',
                FEC_CRC_16K: 'CRC + 16K LDPC', FEC_CRC_64K: 'CRC + 64K LDPC',
                FEC_16K: '16K LDPC', FEC_64K: '64K LDPC', FEC_NONE: 'No FEC'}
TI_MODE = {0: 'No time interleaving', 1: 'Convolutional time interleaving',
           2: 'Hybrid time interleaving', 3: 'Reserved'}
PLP_LAYER = {0: 'Core layer', 1: 'Enhanced layer', 2: 'Reserved',
             3: 'Reserved'}


class _BitReader:
    """Sequential MSB-first bit reader with bounds checking."""

    def __init__(self, bits):
        self.bits = np.asarray(bits)
        self.pos = 0

    def read(self, name: str, width: int) -> int:
        if self.pos + width > len(self.bits):
            raise ValueError(
                f"{name}: need {width} bits at {self.pos}, "
                f"only {len(self.bits) - self.pos} remain")
        value = 0
        for i in range(width):
            value = (value << 1) | int(self.bits[self.pos + i])
        self.pos += width
        return value

    def remaining(self) -> int:
        return len(self.bits) - self.pos


# ---------------------------------------------------------------------------
# L1-Basic (A/322 Table 9.2)
# ---------------------------------------------------------------------------

@dataclass
class L1Basic:
    """Decoded L1-Basic fields (A/322 Table 9.2)."""
    version: int
    mimo_scattered_pilot_encoding: int
    lls_flag: int
    time_info_flag: int
    return_channel_flag: int
    papr_reduction: int
    frame_length_mode: int
    time_offset: int
    num_subframes: int
    preamble_num_symbols: int
    preamble_reduced_carriers: int
    l1_detail_content_tag: int
    l1_detail_size_bytes: int
    l1_detail_fec_type: int
    l1_detail_additional_parity_mode: int
    l1_detail_total_cells: int
    first_sub_mimo: int
    first_sub_miso: int
    first_sub_fft_size: int
    first_sub_reduced_carriers: int
    first_sub_guard_interval: int
    first_sub_num_ofdm_symbols: int
    first_sub_scattered_pilot_pattern: int
    first_sub_scattered_pilot_boost: int
    first_sub_sbs_first: int
    first_sub_sbs_last: int
    crc_ok: bool
    raw: Dict[str, int] = field(default_factory=dict)


def parse_l1_basic(bits) -> L1Basic:
    """Parse a 200-bit L1-Basic block (A/322 Table 9.2)."""
    r = _BitReader(bits)
    f: Dict[str, int] = {}
    f['L1B_version'] = r.read('L1B_version', 3)
    f['L1B_mimo_scattered_pilot_encoding'] = r.read('L1B_mimo_scattered_pilot_encoding', 1)
    f['L1B_lls_flag'] = r.read('L1B_lls_flag', 1)
    f['L1B_time_info_flag'] = r.read('L1B_time_info_flag', 2)
    f['L1B_return_channel_flag'] = r.read('L1B_return_channel_flag', 1)
    f['L1B_papr_reduction'] = r.read('L1B_papr_reduction', 2)
    f['L1B_frame_length_mode'] = r.read('L1B_frame_length_mode', 1)
    if f['L1B_frame_length_mode'] == 0:
        f['L1B_frame_length'] = r.read('L1B_frame_length', 10)
        f['L1B_excess_samples_per_symbol'] = r.read('L1B_excess_samples_per_symbol', 13)
    else:
        f['L1B_time_offset'] = r.read('L1B_time_offset', 16)
        f['L1B_additional_samples'] = r.read('L1B_additional_samples', 7)
    f['L1B_num_subframes'] = r.read('L1B_num_subframes', 8)
    f['L1B_preamble_num_symbols'] = r.read('L1B_preamble_num_symbols', 3)
    f['L1B_preamble_reduced_carriers'] = r.read('L1B_preamble_reduced_carriers', 3)
    f['L1B_L1_Detail_content_tag'] = r.read('L1B_L1_Detail_content_tag', 2)
    f['L1B_L1_Detail_size_bytes'] = r.read('L1B_L1_Detail_size_bytes', 13)
    f['L1B_L1_Detail_fec_type'] = r.read('L1B_L1_Detail_fec_type', 3)
    f['L1B_L1_Detail_additional_parity_mode'] = r.read('L1B_L1_Detail_additional_parity_mode', 2)
    f['L1B_L1_Detail_total_cells'] = r.read('L1B_L1_Detail_total_cells', 19)
    f['L1B_first_sub_mimo'] = r.read('L1B_first_sub_mimo', 1)
    f['L1B_first_sub_miso'] = r.read('L1B_first_sub_miso', 2)
    f['L1B_first_sub_fft_size'] = r.read('L1B_first_sub_fft_size', 2)
    f['L1B_first_sub_reduced_carriers'] = r.read('L1B_first_sub_reduced_carriers', 3)
    f['L1B_first_sub_guard_interval'] = r.read('L1B_first_sub_guard_interval', 4)
    f['L1B_first_sub_num_ofdm_symbols'] = r.read('L1B_first_sub_num_ofdm_symbols', 11)
    f['L1B_first_sub_scattered_pilot_pattern'] = r.read('L1B_first_sub_scattered_pilot_pattern', 5)
    f['L1B_first_sub_scattered_pilot_boost'] = r.read('L1B_first_sub_scattered_pilot_boost', 3)
    f['L1B_first_sub_sbs_first'] = r.read('L1B_first_sub_sbs_first', 1)
    f['L1B_first_sub_sbs_last'] = r.read('L1B_first_sub_sbs_last', 1)
    f['L1B_reserved'] = r.read('L1B_reserved', 48)
    f['L1B_crc'] = r.read('L1B_crc', 32)

    return L1Basic(
        version=f['L1B_version'],
        mimo_scattered_pilot_encoding=f['L1B_mimo_scattered_pilot_encoding'],
        lls_flag=f['L1B_lls_flag'],
        time_info_flag=f['L1B_time_info_flag'],
        return_channel_flag=f['L1B_return_channel_flag'],
        papr_reduction=f['L1B_papr_reduction'],
        frame_length_mode=f['L1B_frame_length_mode'],
        time_offset=f.get('L1B_time_offset', 0),
        num_subframes=f['L1B_num_subframes'],
        preamble_num_symbols=f['L1B_preamble_num_symbols'],
        preamble_reduced_carriers=f['L1B_preamble_reduced_carriers'],
        l1_detail_content_tag=f['L1B_L1_Detail_content_tag'],
        l1_detail_size_bytes=f['L1B_L1_Detail_size_bytes'],
        l1_detail_fec_type=f['L1B_L1_Detail_fec_type'],
        l1_detail_additional_parity_mode=f['L1B_L1_Detail_additional_parity_mode'],
        l1_detail_total_cells=f['L1B_L1_Detail_total_cells'],
        first_sub_mimo=f['L1B_first_sub_mimo'],
        first_sub_miso=f['L1B_first_sub_miso'],
        first_sub_fft_size=f['L1B_first_sub_fft_size'],
        first_sub_reduced_carriers=f['L1B_first_sub_reduced_carriers'],
        first_sub_guard_interval=f['L1B_first_sub_guard_interval'],
        first_sub_num_ofdm_symbols=f['L1B_first_sub_num_ofdm_symbols'],
        first_sub_scattered_pilot_pattern=f['L1B_first_sub_scattered_pilot_pattern'],
        first_sub_scattered_pilot_boost=f['L1B_first_sub_scattered_pilot_boost'],
        first_sub_sbs_first=f['L1B_first_sub_sbs_first'],
        first_sub_sbs_last=f['L1B_first_sub_sbs_last'],
        crc_ok=crc32_ok(bits),
        raw=f,
    )


# ---------------------------------------------------------------------------
# L1-Detail (A/322 Table 9.8)
# ---------------------------------------------------------------------------

@dataclass
class PLPConfig:
    """Per-PLP configuration from L1-Detail (A/322 Table 9.8)."""
    plp_id: int
    lls_flag: int
    layer: int
    start: int
    size: int
    scrambler_type: int
    fec_type: int
    modulation: Optional[int]
    code_rate: Optional[int]
    ti_mode: int
    ti_fec_block_start: Optional[int]
    hti_inter_subframe: Optional[int]
    hti_num_ti_blocks: Optional[int]
    hti_num_fec_blocks: Optional[int]
    hti_cell_interleaver: Optional[int]
    cti_depth: Optional[int]
    cti_start_row: Optional[int]
    ti_extended_interleaving: Optional[int]
    ldm_injection_level: Optional[int]
    raw: Dict[str, int] = field(default_factory=dict)


@dataclass
class L1Detail:
    """Decoded L1-Detail (A/322 Table 9.8)."""
    version: int
    num_rf: int
    time_sec: Optional[int]
    bsid: int
    subframes: List[dict]
    reserved_len: int
    reserved_all_ones: bool
    crc_ok: bool
    raw: Dict[str, int] = field(default_factory=dict)


def _parse_plp(r: _BitReader, n_rf: int, first_sub_mimo: int,
               l1d_mimo: int, subframe_index: int) -> PLPConfig:
    """Parse one L1D_plp entry (A/322 Table 9.8)."""
    plp_id = r.read('L1D_plp_id', 6)
    lls = r.read('L1D_plp_lls_flag', 1)
    layer = r.read('L1D_plp_layer', 2)
    start = r.read('L1D_plp_start', 24)
    size = r.read('L1D_plp_size', 24)
    scrambler = r.read('L1D_plp_scrambler_type', 2)
    fec_type = r.read('L1D_plp_fec_type', 4)
    mod = cod = None
    if fec_type in (0, 1, 2, 3, 4, 5):
        mod = r.read('L1D_plp_mod', 4)
        cod = r.read('L1D_plp_cod', 4)
    ti_mode = r.read('L1D_plp_TI_mode', 2)
    ti_start = None
    if ti_mode == 0:
        ti_start = r.read('L1D_plp_fec_block_start', 15)
    elif ti_mode == 1:
        ti_start = r.read('L1D_plp_CTI_fec_block_start', 22)
    if n_rf > 0:
        n_bonded = r.read('L1D_plp_num_channel_bonded', 3)
        if n_bonded > 0:
            r.read('L1D_plp_channel_bonding_format', 2)
            for _ in range(n_bonded):
                r.read('L1D_plp_bonded_rf_id', 3)
    if (subframe_index == 0 and first_sub_mimo == 1) or \
            (subframe_index > 0 and l1d_mimo == 1):
        r.read('L1D_plp_mimo_stream_combining', 1)
        r.read('L1D_plp_mimo_IQ_interleaving', 1)
        r.read('L1D_plp_mimo_PH', 1)
    hti_inter = hti_ti = hti_fec = hti_cell = None
    cti_depth = cti_start_row = ti_extended = None
    ldm_level = None
    if layer == 0:
        plp_type = r.read('L1D_plp_type', 1)
        if plp_type == 1:
            r.read('L1D_plp_num_subslices', 14)
            r.read('L1D_plp_subslice_interval', 24)
        if ti_mode in (1, 2) and mod == 0:
            ti_extended = r.read('L1D_plp_TI_extended_interleaving', 1)
        if ti_mode == 1:
            cti_depth = r.read('L1D_plp_CTI_depth', 3)
            cti_start_row = r.read('L1D_plp_CTI_start_row', 11)
        elif ti_mode == 2:
            hti_inter = r.read('L1D_plp_HTI_inter_subframe', 1)
            hti_ti = r.read('L1D_plp_HTI_num_ti_blocks', 4)
            r.read('L1D_plp_HTI_num_fec_blocks_max', 12)
            if hti_inter == 0:
                hti_fec = r.read('L1D_plp_HTI_num_fec_blocks', 12)
            else:
                for _ in range(hti_ti):
                    r.read('L1D_plp_HTI_num_fec_blocks', 12)
            hti_cell = r.read('L1D_plp_HTI_cell_interleaver', 1)
    else:
        ldm_level = r.read('L1D_plp_ldm_injection_level', 5)

    return PLPConfig(
        plp_id=plp_id, lls_flag=lls, layer=layer, start=start, size=size,
        scrambler_type=scrambler, fec_type=fec_type, modulation=mod,
        code_rate=cod, ti_mode=ti_mode, ti_fec_block_start=ti_start,
        hti_inter_subframe=hti_inter, hti_num_ti_blocks=hti_ti,
        hti_num_fec_blocks=hti_fec, hti_cell_interleaver=hti_cell,
        cti_depth=cti_depth, cti_start_row=cti_start_row,
        ti_extended_interleaving=ti_extended,
        ldm_injection_level=ldm_level, raw={})


def parse_l1_detail(bits, l1b: L1Basic) -> L1Detail:
    """Parse an L1-Detail block given its decoded L1-Basic (A/322 Table 9.8)."""
    r = _BitReader(bits)
    raw: Dict[str, int] = {}

    version = r.read('L1D_version', 4)
    num_rf = r.read('L1D_num_rf', 3)
    raw['L1D_version'] = version
    raw['L1D_num_rf'] = num_rf
    for _ in range(1, num_rf + 1):
        r.read('L1D_bonded_bsid', 16)
        r.read('reserved', 3)

    time_sec: Optional[int] = None
    if l1b.time_info_flag != 0:
        time_sec = r.read('L1D_time_sec', 32)
        r.read('L1D_time_msec', 10)
        if l1b.time_info_flag != 1:
            r.read('L1D_time_usec', 10)
            if l1b.time_info_flag != 2:
                r.read('L1D_time_nsec', 10)

    subframes = []
    # L1B_num_subframes is the count minus one.
    for i in range(l1b.num_subframes + 1):
        sf: Dict[str, int] = {'index': i}
        if i > 0:
            sf['mimo'] = r.read('L1D_mimo', 1)
            sf['miso'] = r.read('L1D_miso', 2)
            sf['fft_size'] = r.read('L1D_fft_size', 2)
            sf['reduced_carriers'] = r.read('L1D_reduced_carriers', 3)
            sf['guard_interval'] = r.read('L1D_guard_interval', 4)
            sf['num_ofdm_symbols'] = r.read('L1D_num_ofdm_symbols', 11)
            sf['scattered_pilot_pattern'] = r.read('L1D_scattered_pilot_pattern', 5)
            sf['scattered_pilot_boost'] = r.read('L1D_scattered_pilot_boost', 3)
            sf['sbs_first'] = r.read('L1D_sbs_first', 1)
            sf['sbs_last'] = r.read('L1D_sbs_last', 1)
        if l1b.num_subframes > 0:
            sf['subframe_multiplex'] = r.read('L1D_subframe_multiplex', 1)
        sf['frequency_interleaver'] = r.read('L1D_frequency_interleaver', 1)
        has_sbs = (
            (i == 0 and (l1b.first_sub_sbs_first or l1b.first_sub_sbs_last)) or
            (i > 0 and (sf.get('sbs_first') or sf.get('sbs_last')))
        )
        if has_sbs:
            sf['sbs_null_cells'] = r.read('L1D_sbs_null_cells', 13)
        num_plp = r.read('L1D_num_plp', 6)
        sf['num_plp'] = num_plp
        plps = []
        for _ in range(num_plp + 1):
            first_mimo = l1b.first_sub_mimo
            l1d_mimo = sf.get('mimo', 0)
            plps.append(_parse_plp(r, num_rf, first_mimo, l1d_mimo, i))
        sf['plps'] = plps
        subframes.append(sf)

    bsid = r.read('L1D_bsid', 16)

    # L1D_reserved is "as needed": everything up to the trailing 32-bit CRC.
    reserved_len = max(0, r.remaining() - 32)
    reserved = bits[r.pos:r.pos + reserved_len] if reserved_len else np.array([], dtype=np.uint8)
    reserved_all_ones = bool(len(reserved)) and bool(np.all(reserved == 1))
    r.read('L1D_crc', 32)

    return L1Detail(
        version=version, num_rf=num_rf, time_sec=time_sec, bsid=bsid,
        subframes=subframes, reserved_len=int(reserved_len),
        reserved_all_ones=reserved_all_ones, crc_ok=crc32_ok(bits), raw=raw,
    )


# ---------------------------------------------------------------------------
# Backwards-compatible facade
# ---------------------------------------------------------------------------

class L1SignalingParser:
    """Parse L1-Basic and L1-Detail signalling (A/322 Tables 9.2 and 9.8)."""

    def parse_l1_basic(self, bits) -> L1Basic:
        return parse_l1_basic(bits)

    def parse_l1_detail(self, bits, l1b: L1Basic) -> L1Detail:
        return parse_l1_detail(bits, l1b)
