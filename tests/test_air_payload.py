"""End-to-end payload decode from a real ATSC 3.0 broadcast.

A 1.3 Msample 10 MS/s slice of the RF33 (587 MHz) capture, saved around the
bootstrap.  This exercises the full receive path including the bootstrap
fractional-CFO correction and the A/322 guarded-interval mapping that must be
right for OFDM symbols to line up:

    raw IQ -> resample 6.144M -> bootstrap -> fine CFO -> resample 6.912M
           -> Preamble -> L1-Basic/L1-Detail -> subframe cell pool
           -> QPSK 2/15 PLP-16 -> LDPC/BCH/descramble -> Baseband Packet

The decoded PLP-16 Baseband Packet must equal the independent receiver's.
"""

import os

import numpy as np
import pytest

from atsc3lib.receiver import decode_signaling, decode_plp_payload
from atsc3lib.payload import decode_streams

_DATA = os.path.join(os.path.dirname(__file__), 'data')
_SLICE = os.path.join(_DATA, 'rf33_acquire_slice.npy')
_BB = os.path.join(_DATA, 'rf33_plp16_bb.bin')

pytestmark = pytest.mark.skipif(
    not os.path.exists(_SLICE), reason="real-air capture slice not present")


def test_live_plp16_payload_matches_oracle():
    iq = np.load(_SLICE)
    result = decode_signaling(iq, 10e6)
    assert result.l1_basic_ok and result.l1_detail_ok
    result, payload = decode_plp_payload(iq, 10e6, result=result)
    assert payload is not None
    assert payload.plp_id == 16
    assert payload.n_converged == 1
    packets = payload.baseband_packets
    assert len(packets) == 1
    if os.path.exists(_BB):
        assert packets[0] == open(_BB, 'rb').read()


def test_live_plp16_bbp_header_and_streams():
    # PLP-16's one Baseband Packet is padding-only (no ALP starts): this gates
    # the A/322 5.2.2 header parse and the A/330 ALP walk on real air.
    iq = np.load(_SLICE)
    result = decode_signaling(iq, 10e6)
    result, payload = decode_plp_payload(iq, 10e6, result=result)
    assert payload is not None
    streams = decode_streams(payload)
    assert streams.packets == []
    assert streams.datagrams == []
    assert streams.alp_stats.resync == 0
    assert streams.alp_stats.single == 0
