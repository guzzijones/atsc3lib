"""Tests for the link layer: A/322 5.2.2 Baseband Packets, A/330 ALP, IP/UDP.

Header constructors here follow the standards' own syntax tables, so the
round trip is gated on the spec's field layout rather than on our reader.
"""

import os

import numpy as np
import pytest

from atsc3lib import alp, baseband
from atsc3lib import ip as ip_layer

_DATA = os.path.join(os.path.dirname(__file__), 'data')
_PLP16_BB = os.path.join(_DATA, 'rf33_plp16_bb.bin')


class TestBasebandHeader:
    @pytest.mark.skipif(not os.path.exists(_PLP16_BB),
                        reason="real-air fixture not present")
    def test_plp16_all_padding(self):
        # RF33 PLP-16: MODE=1, pointer=all-ones, OFI=10, EXT_TYPE=111 (all
        # padding), EXT_LEN=245 -> 249-byte Baseband Packet.
        b = open(_PLP16_BB, 'rb').read()
        p = baseband.split_baseband_packet(b)
        assert p.mode == 1
        assert p.pointer == baseband.POINTER_NONE
        assert not p.starts_alp
        assert p.ofi == baseband.OFI_LONG
        assert p.ext_type == baseband.EXT_TYPE_ALL_PADDING
        # Base+Optional = 4 bytes, then a 245-byte all-padding Extension.
        assert p.header_len == 4 + 245
        assert p.header_len == len(b)
        assert len(p.payload) == 0

    def test_short_mode_pointer(self):
        # A/322 5.2.2.1 example: pointer 130 -> MODE=1, bytes 82 04.
        data = bytes([0b10000010, 0b00000100]) + bytes(200)
        p = baseband.split_baseband_packet(data)
        assert p.mode == 1
        assert p.pointer == 130
        assert p.ofi == baseband.OFI_NONE
        assert p.header_len == 2
        assert p.starts_alp

    def test_short_extension(self):
        # OFI=01: EXT_TYPE(3) + EXT_LEN(5).
        ext_type, ext_len = 0b101, 5
        data = bytes([0x80, 0b01 << 0 | baseband.OFI_SHORT]) + \
            bytes([(ext_type << 5) | ext_len]) + bytes(ext_len) + b'XY'
        p = baseband.split_baseband_packet(data)
        assert p.ofi == baseband.OFI_SHORT
        assert p.ext_type == ext_type
        assert p.ext_len == ext_len
        assert p.header_len == 3 + ext_len
        assert p.payload == b'XY'

    def test_payload_stream_boundaries(self):
        a = baseband.BasebandPacket(0, 0, 0, None, 0, None, 1, b'abc')
        b = baseband.BasebandPacket(1, 2, 0, None, 0, None, 2, b'de')
        stream, bounds = baseband.payload_stream([a, b])
        assert stream == b'abcde'
        assert bounds == [0, 3 + 2]
        stream, bounds = baseband.payload_stream(
            [baseband.BasebandPacket(1, baseband.POINTER_NONE, 0, None, 0,
                                     None, 2, b'de')])
        assert bounds == []


def _base(ptype, pc):
    """16-bit A/330 Base Header prefix: packet_type(3) at bits 15..13, PC bit 12."""
    return (ptype << PACKET_TYPE_FIELD_BITS) | (pc << PAYLOAD_CONFIG_FIELD_BITS)


#: A/330 Table 5.1 field positions within the 16-bit Base Header.
PACKET_TYPE_FIELD_BITS = 13
PAYLOAD_CONFIG_FIELD_BITS = 12
SEGMENTATION_CONCAT_BIT = 11


def _alp_single(ptype, payload, sid=None, hef=None):
    """A/330 Table 5.1/5.4 single-packet writer."""
    n = len(payload)
    hm = 0 if (n <= alp.SINGLE_PAYLOAD_MAX and sid is None and not hef) else 1
    head = _base(ptype, 0) | (hm << (alp.LENGTH_BITS)) | (n & 0x7FF)
    out = head.to_bytes(2, 'big')
    if hm:
        sif = 1 if sid is not None else 0
        out += bytes([((n >> alp.LENGTH_BITS) << 3) | (sif << 1) | (hef or 0)])
        if sif:
            out += bytes([sid])
        if hef:
            out += bytes([hef]) + bytes([0])       # extension_type, len-1=0
    return out + payload


def _alp_segment(ptype, ssn, last, payload, sid=None):
    """A/330 Table 5.5 segmentation writer (PC=1, S/C=0)."""
    n = len(payload)
    head = _base(ptype, 1) | (n & 0x7FF)
    ctl = (ssn << 3) | (last << 2) | ((1 if sid is not None else 0) << 1)
    out = head.to_bytes(2, 'big') + bytes([ctl])
    if sid is not None:
        out += bytes([sid])
    return out + payload


def _alp_concat(ptype, packets, sid=None):
    """A/330 Table 5.6 concatenation writer (PC=1, S/C=1), (count+1) lengths."""
    total = sum(len(p) for p in packets)
    count = len(packets) - 2
    head = _base(ptype, 1) | (1 << SEGMENTATION_CONCAT_BIT) | (total & 0x7FF)
    ctl = ((total >> alp.LENGTH_BITS) << 4) | (count << 1) | \
        (1 if sid is not None else 0)
    lens = [len(p) for p in packets[:-1]]
    bits = 0
    for L in lens:
        bits = (bits << 12) | L
    nbytes = (len(lens) * 12 + 7) // 8
    bits <<= nbytes * 8 - len(lens) * 12
    out = head.to_bytes(2, 'big') + bytes([ctl]) + bits.to_bytes(nbytes, 'big')
    if sid is not None:
        out += bytes([sid])
    return out + b''.join(packets)


def _alp_signalling(payload):
    """A/330 Table 5.11 signalling writer with the LMT values (Table 7.1)."""
    n = len(payload)
    head = _base(alp.PT_SIGNALLING, 0) | (n & 0x7FF)
    hdr = bytes([0x01]) + b'\xff\xff' + bytes([1, 0x0f])
    return head.to_bytes(2, 'big') + hdr + payload


def _wrap(alppkt):
    """A minimal MODE=0 Baseband Packet carrying one ALP packet."""
    return baseband.split_baseband_packet(bytes([0x00]) + alppkt)


def _walk(alppkt):
    pkt = _wrap(alppkt)
    stream, bounds = baseband.payload_stream([pkt])
    return alp.parse_alp(stream, bounds)


class TestAlpWalk:
    def test_single_short(self):
        pay = b'\x45' + bytes(19)
        pkts, st = _walk(_alp_single(alp.PT_IPV4, pay))
        assert len(pkts) == 1 and pkts[0].payload == pay
        assert pkts[0].packet_type == alp.PT_IPV4
        assert st.single == 1 and st.resync == 0

    def test_single_long_with_sid(self):
        pay = bytes(range(256)) * 10
        pkts, st = _walk(_alp_single(alp.PT_IPV4, pay, sid=7))
        assert pkts[0].payload == pay and pkts[0].sid == 7

    def test_segmentation_roundtrip(self):
        data = bytes(range(256)) * 3
        parts = [data[:300], data[300:600], data[600:]]
        stream = b''.join(_alp_segment(alp.PT_IPV4, i, i == 2, p)
                          for i, p in enumerate(parts))
        pkts, st = _walk(stream)
        assert len(pkts) == 1 and pkts[0].payload == data
        assert st.segmented == 3 and st.reassembled == 1

    def test_segmentation_incomplete(self):
        a, b = _alp_segment(alp.PT_IPV4, 0, 0, b'x' * 10), \
            _alp_segment(alp.PT_IPV4, 2, 1, b'y' * 10)
        pkts, st = _walk(a + b)
        assert pkts == [] and st.incomplete == 1

    def test_concatenation_roundtrip(self):
        ips = [b'\x45' + bytes(19), b'\x45' + bytes(39), b'\x45' + bytes(9)]
        pkts, st = _walk(_alp_concat(alp.PT_IPV4, ips))
        assert [p.payload for p in pkts] == ips
        assert st.concatenated == 1 and st.components == len(ips)

    def test_signalling(self):
        pay = b'\x01\x00\x00\x00' + b'LMT!'
        pkts, st = _walk(_alp_signalling(pay))
        assert pkts[0].packet_type == alp.PT_SIGNALLING
        assert pkts[0].payload == pay
        assert st.signalling == 1

    def test_empty_stream(self):
        pkts, st = alp.parse_alp(b'', [])
        assert pkts == [] and st.resync == 0


class TestIpUdp:
    @staticmethod
    def _ipv4_udp(src, dst, sport, dport, payload, ident=0, mf=0, foff=0):
        udp = (sport.to_bytes(2, 'big') + dport.to_bytes(2, 'big')
               + (8 + len(payload)).to_bytes(2, 'big') + b'\x00\x00' + payload)
        total = 20 + len(udp)
        flags_frag = (mf << 13) | foff
        return (bytes([0x45, 0, total >> 8, total & 0xFF,
                       ident >> 8, ident & 0xFF,
                       flags_frag >> 8, flags_frag & 0xFF, 64, 17, 0, 0])
                + src + dst + udp)

    def test_udp(self):
        r = ip_layer.IpReassembler()
        dgs = r.feed(self._ipv4_udp(b'\x0a\x00\x00\x01',
                                    b'\xe0\x00\x17\x3c', 1, 4937, b'hi'))
        assert len(dgs) == 1 and dgs[0].payload == b'hi'
        assert ip_layer.is_lls(dgs[0])

    def test_fragment_reassembly(self):
        body = bytes(range(256)) * 4
        src, dst = b'\x0a\x00\x00\x01', b'\xe0\x00\x00\x01'
        whole = self._ipv4_udp(src, dst, 1, 2, body)
        # Split the UDP datagram at an 8-byte boundary (RFC 791 requirement).
        split = 8 + 200
        f1, f2 = whole[20:20 + split], whole[20 + split:]
        p1 = (bytes([0x45, 0]) + (20 + len(f1)).to_bytes(2, 'big')
              + bytes([0, 9]) + (((1 << 13) | 0) >> 8).to_bytes(1, 'big')
              + bytes([0, 64, 17, 0, 0]) + src + dst + f1)
        offset = (split // 8) & 0x1FFF
        p2 = (bytes([0x45, 0]) + (20 + len(f2)).to_bytes(2, 'big')
              + bytes([0, 9, offset >> 8, offset & 0xFF, 64, 17, 0, 0])
              + src + dst + f2)
        r = ip_layer.IpReassembler()
        assert r.feed(p1) == []
        dgs = r.feed(p2)
        assert len(dgs) == 1 and dgs[0].payload == body
        assert r.stats.reassembled == 1

    def test_non_ipv4_ignored(self):
        r = ip_layer.IpReassembler()
        assert r.feed(b'\x60' + bytes(19)) == []
        assert r.stats.not_ipv4 == 1

    def test_lls_table(self):
        payload = b'\x01\x00\x00\x00SLTbody'
        t = ip_layer.parse_lls(payload)
        assert t.table_id == 0x01 and t.name == 'SLT' and t.data == b'SLTbody'


class TestDecodeStreams:
    def test_bbp_to_lls(self):
        from atsc3lib.payload import PlpPayload, decode_streams

        def ipv4_udp(payload):
            udp = (1234).to_bytes(2, 'big') + (4937).to_bytes(2, 'big') \
                + (8 + len(payload)).to_bytes(2, 'big') + b'\x00\x00' + payload
            total = 20 + len(udp)
            return (bytes([0x45, 0, total >> 8, total & 0xFF, 0, 0, 0x40, 0,
                           64, 17, 0, 0, 0x0A, 0, 0, 1])
                    + bytearray(b'\xe0\x00\x17\x3c') + udp)

        lls_payload = b'\x01\x00\x00\x00' + b'SLTbody'
        alppkt = _alp_single(alp.PT_IPV4, ipv4_udp(lls_payload), sid=3)
        bbp = bytes([0x00]) + alppkt

        class _Block:
            ok = True

            def __init__(self, packet):
                self.packet = packet

        payload = PlpPayload(plp_id=16, fec_blocks=[_Block(bbp)], n_fec=1)
        streams = decode_streams(payload)
        assert len(streams.packets) == 1
        assert len(streams.datagrams) == 1
        assert len(streams.lls) == 1
        assert streams.lls[0].name == 'SLT'
        assert streams.lls[0].data == b'SLTbody'
        # 'SLTbody' is not valid SLT XML, so no parsed SLT is attached.
        assert streams.slt is None

    def test_bbp_to_parsed_slt(self):
        import gzip
        from pathlib import Path

        from atsc3lib.payload import PlpPayload, decode_streams

        def ipv4_udp(payload):
            udp = (1234).to_bytes(2, 'big') + (4937).to_bytes(2, 'big') \
                + (8 + len(payload)).to_bytes(2, 'big') + b'\x00\x00' + payload
            total = 20 + len(udp)
            return (bytes([0x45, 0, total >> 8, total & 0xFF, 0, 0, 0x40, 0,
                           64, 17, 0, 0, 0x0A, 0, 0, 1])
                    + bytearray(b'\xe0\x00\x17\x3c') + udp)

        slt_body = gzip.compress(b'<SLT bsid="540"><Service serviceId="1"/></SLT>')
        lls_payload = b'\x01\x00\x00\x01' + slt_body
        alppkt = _alp_single(alp.PT_IPV4, ipv4_udp(lls_payload), sid=3)
        bbp = bytes([0x00]) + alppkt

        class _Block:
            ok = True

            def __init__(self, packet):
                self.packet = packet

        payload = PlpPayload(plp_id=16, fec_blocks=[_Block(bbp)], n_fec=1)
        streams = decode_streams(payload)
        assert streams.slt is not None
        assert streams.slt.bsid == (540,)
        assert streams.slt.services[0].service_id == 1
