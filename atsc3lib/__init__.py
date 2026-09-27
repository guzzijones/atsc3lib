"""ATSC 3.0 physical-layer receiver library.

Hardware-agnostic: works with any SDR (HackRF, RTL-SDR, Airspy, SDRplay, USRP).

The canonical entry point is :func:`decode_capture`, which runs the full
validated signalling chain (bootstrap -> Preamble -> L1-Basic -> L1-Detail ->
per-PLP configuration) on a saved IQ capture.
"""

__version__ = "0.1.0"
__author__ = "OpenATSC3"

from . import spec
from . import crc

# Capture and front-end
from .capture import capture
from .frontend import (
    resample_iq, acquire_frame, read_hackrf_iq, FrameInfo,
    BOOTSTRAP_RATE_HZ, MAIN_RATE_HZ,
)
from .ofdm_detect import (
    detect_ofdm_params, detect_guard_interval, find_symbol_start,
    cp_correlation, GUARD_INTERVALS, FFT_SIZES,
)

# Bootstrap and Preamble
from .bootstrap import (
    generate_bootstrap, detect_bootstrap, decode_preamble_structure,
    BootstrapParams, BootstrapDetection,
)
from .pilot_reference import (
    reference_sequence, pilot_values, preamble_amplitude,
    preamble_dx, preamble_pilot_indices,
)
from .preamble import (
    carrier_shift, preamble_noc, common_continual_pilots, preamble_data_mask,
    preamble_pilot_values, estimate_preamble_channel, preamble_data_cells,
    preamble_symbol_spectrum, preamble_l1_cells, pilot_coherence,
)
from .frequency_interleaver import (
    generate_addresses, interleave as freq_interleave,
    deinterleave as freq_deinterleave,
)

# FEC
from .ldpc_exact import ATSC3LDPCExact
from .bch import BCHCode
from .group_interleaver import GroupInterleaver
from .prbs_descrambler import PRBSDescrambler, BitScrambler
from . import nuc
from . import twisted_block
from . import cell_interleaver
from . import baseband
from . import alp
from . import ip
from . import signaling_fec

# Signalling
from .l1_signaling import (
    L1SignalingParser, parse_l1_basic, parse_l1_detail,
    L1Basic, L1Detail, PLPConfig,
)
from .l1_basic import L1BasicCodec, scramble_bits
from .l1_detail import L1DetailCodec

# Payload
from .payload import (
    QPSKPlpChain, DataPlpChain, build_cell_pool, build_data_symbol_pool,
    data_symbol_cells, qpsk_demap_llr, CellPool, QPSK_POINTS, PlpPayload,
    decode_subframe0_qpsk_plp, decode_subframe0_plp, decode_subframe_plp,
    decode_plp_from_pool, DecodedStreams, decode_streams,
    FineTiming, CpeSpec, fine_timing, subframe_fine_timing, cpe_correct,
    scattered_pilot_coherence, cpe_spec, plp_alphabet, dummy_cell_values,
)

# Pilot / data-cell tables (A/322 Annex D/F), fetched from the pinned source
from . import pilot_tables

# Receiver (full chain)
from .receiver import (
    decode_signaling, decode_capture, ReceiverResult,
    decode_first_plp_payload, decode_plp_payload, decode_plp_streams,
)

__all__ = [
    # capture / front-end
    "capture",
    "resample_iq",
    "acquire_frame",
    "read_hackrf_iq",
    "FrameInfo",
    "BOOTSTRAP_RATE_HZ",
    "MAIN_RATE_HZ",
    "detect_ofdm_params",
    "detect_guard_interval",
    "find_symbol_start",
    "cp_correlation",
    "GUARD_INTERVALS",
    "FFT_SIZES",
    # bootstrap / preamble
    "generate_bootstrap",
    "detect_bootstrap",
    "decode_preamble_structure",
    "BootstrapParams",
    "BootstrapDetection",
    "reference_sequence",
    "pilot_values",
    "preamble_amplitude",
    "preamble_dx",
    "preamble_pilot_indices",
    "carrier_shift",
    "preamble_noc",
    "common_continual_pilots",
    "preamble_data_mask",
    "preamble_pilot_values",
    "estimate_preamble_channel",
    "preamble_data_cells",
    "preamble_symbol_spectrum",
    "preamble_l1_cells",
    "pilot_coherence",
    "generate_addresses",
    "freq_interleave",
    "freq_deinterleave",
    # FEC
    "ATSC3LDPCExact",
    "BCHCode",
    "GroupInterleaver",
    "PRBSDescrambler",
    "BitScrambler",
    "signaling_fec",
    "nuc",
    "twisted_block",
    "cell_interleaver",
    "baseband",
    "alp",
    "ip",
    # signalling
    "L1SignalingParser",
    "parse_l1_basic",
    "parse_l1_detail",
    "L1Basic",
    "L1Detail",
    "PLPConfig",
    "L1BasicCodec",
    "L1DetailCodec",
    "scramble_bits",
    # payload
    "QPSKPlpChain",
    "DataPlpChain",
    "build_cell_pool",
    "build_data_symbol_pool",
    "data_symbol_cells",
    "qpsk_demap_llr",
    "CellPool",
    "QPSK_POINTS",
    "PlpPayload",
    "decode_subframe0_qpsk_plp",
    "decode_subframe0_plp",
    "decode_subframe_plp",
    "decode_plp_from_pool",
    "DecodedStreams",
    "decode_streams",
    "FineTiming",
    "CpeSpec",
    "fine_timing",
    "subframe_fine_timing",
    "cpe_correct",
    "scattered_pilot_coherence",
    "cpe_spec",
    "plp_alphabet",
    "dummy_cell_values",
    # receiver
    "decode_signaling",
    "decode_capture",
    "ReceiverResult",
    "decode_first_plp_payload",
    "decode_plp_payload",
    "decode_plp_streams",
    # modules
    "spec",
    "crc",
]
