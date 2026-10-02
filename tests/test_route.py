"""Tests for A/331 ROUTE/LCT parsing, gated on RF33 off-air fixtures."""

import gzip
from pathlib import Path

import pytest

from atsc3lib.route import (
    CP_NRT_FILE,
    CP_NRT_UNSIGNED_PACKAGE,
    Efdt,
    RouteAssembler,
    RouteError,
    RouteObject,
    SLS_TSI,
    parse_efdt,
    parse_lct_header,
    parse_route_packet,
)

DATA = Path(__file__).parent / "data"
EFDT = DATA / "route_efdt_gmloop.bin"
SG_OBJECTS = [DATA / ("route_object_sgfe00_%d.bin" % i) for i in range(3)]
CP3 = DATA / "route_sls_sgfe00_cp3.bin"


class TestLctHeader:
    def test_efdt_header_fields(self):
        h = parse_route_packet(EFDT.read_bytes()).header
        assert h.version == 1
        assert h.congestion_control == 0
        assert h.psi == 0b10
        assert h.tsi_present == 1
        assert h.toi_present == 1
        assert h.half_word == 0
        assert h.codepoint == CP_NRT_FILE
        assert h.header_bytes == 24
        assert h.is_route_source

    def test_truncated_header(self):
        with pytest.raises(RouteError):
            parse_lct_header(b"\x12")


class TestRoutePacket:
    def test_efdt_packet(self):
        pkt = parse_route_packet(EFDT.read_bytes())
        assert pkt.header.is_route_source
        assert pkt.tsi == SLS_TSI
        assert pkt.toi == 0
        assert pkt.start_offset == 0
        assert pkt.is_sls
        assert pkt.header.close_object == 1

    def test_efdt_is_fdt_instance_xml(self):
        payload = parse_route_packet(EFDT.read_bytes()).payload
        assert payload.startswith(b'<?xml version="1.0"')
        text = payload.decode("utf-8")
        assert "<FDT-Instance" in text
        assert 'Content-Location="sls"' in text
        assert 'TOI="4390914"' in text
        assert 'Content-Length="1853"' in text
        assert 'Content-Type="multipart/related"' in text

    def test_sgfe00_object_packets(self):
        for path in SG_OBJECTS:
            pkt = parse_route_packet(path.read_bytes())
            assert pkt.header.is_route_source
            assert pkt.tsi == 2
            assert pkt.header.codepoint == CP_NRT_FILE
            assert pkt.start_offset == 0
            assert pkt.payload[:2] == b"\x1f\x8b"

    def test_cp3_is_sls_unsigned_package(self):
        pkt = parse_route_packet(CP3.read_bytes())
        assert pkt.header.is_route_source
        assert pkt.tsi == SLS_TSI
        assert pkt.header.codepoint == CP_NRT_UNSIGNED_PACKAGE
        assert pkt.header.close_object == 1
        assert pkt.start_offset == 4344
        assert pkt.payload.lstrip()[:9] == b"<fdt:File"

    def test_too_short_for_fec_payload_id(self):
        header_only = bytes.fromhex("12a005010000000000000000")
        with pytest.raises(RouteError):
            parse_route_packet(header_only)


class TestRouteAssembler:
    def test_feed_returns_object(self):
        asm = RouteAssembler()
        obj = asm.feed(SG_OBJECTS[0].read_bytes())
        assert isinstance(obj, RouteObject)
        assert obj.tsi == 2
        assert asm.packets == 1

    def test_sls_objects_filters_tsi_zero(self):
        asm = RouteAssembler()
        asm.feed(EFDT.read_bytes())
        for path in SG_OBJECTS:
            asm.feed(path.read_bytes())
        assert [o.tsi for o in asm.sls_objects()] == [SLS_TSI]
        assert asm.sls_objects()[0].toi == 0

    def test_reassemble_from_chunks(self):
        obj = RouteObject(tsi=0, toi=1)
        obj.add(0, b"hello ")
        obj.add(6, b"world")
        assert obj.reassemble() == b"hello world"

    def test_reassemble_gzip_object(self):
        obj = RouteObject(tsi=0, toi=2)
        blob = gzip.compress(b"<LS>signaling</LS>")
        obj.add(0, blob)
        assert obj.is_gzip
        assert obj.inflate() == b"<LS>signaling</LS>"

    def test_inflate_passthrough_when_not_gzip(self):
        obj = RouteObject(tsi=0, toi=3)
        obj.add(0, b"plain")
        assert obj.inflate() == b"plain"


class TestEfdt:
    """The Extended FDT (SLS bootstrap object) off RF33 GMLOOP TSI=0."""

    def test_efdt_from_sls_object(self):
        pkt = parse_route_packet(EFDT.read_bytes())
        efdt = parse_efdt(pkt.payload)
        assert isinstance(efdt, Efdt)
        assert efdt.efdt_version == 3
        assert efdt.expires == 4294967295
        assert len(efdt.files) == 1
        f = efdt.files[0]
        assert f.content_location == "sls"
        assert f.toi == 4390914
        assert f.content_length == 1853
        assert f.content_type == "multipart/related"

    def test_rejects_non_fdt_root(self):
        with pytest.raises(RouteError):
            parse_efdt(b"<NotFDT/>")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
