"""Tests for A/331 SLT parsing, gated on the RF33 off-air fixture."""

import gzip
from pathlib import Path

import pytest

from atsc3lib import ip as ip_layer
from atsc3lib.slt import (
    BroadcastSvcSignaling,
    SLS_PROTOCOL_MMTP,
    SLS_PROTOCOL_ROUTE,
    Slt,
    SltService,
    parse_slt_xml,
    slt_from_lls,
)

FIXTURE = Path(__file__).parent / "data" / "rf33_slt_lls.bin"


@pytest.fixture(scope="module")
def slt() -> Slt:
    return slt_from_lls(FIXTURE.read_bytes())


class TestRf33Slt:
    """The RF33 (BSID 540) lighthouse SLT as received off air."""

    def test_bsid(self, slt):
        assert slt.bsid == (540,)

    def test_service_count(self, slt):
        assert len(slt.services) == 10

    def test_service_ids(self, slt):
        assert [s.service_id for s in slt.services] == [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 65024]

    def test_whut_route_signaling(self, slt):
        whut = next(s for s in slt.services if s.service_id == 1)
        assert whut.short_service_name == "WHUT"
        assert whut.major_channel_no == 32
        assert whut.minor_channel_no == 1
        assert whut.category_name == "Linear A/V Service"
        sig = whut.signaling
        assert sig is not None
        assert sig.sls_protocol == SLS_PROTOCOL_ROUTE
        assert sig.protocol_name == "ROUTE"
        assert sig.destination_ip == "239.255.32.1"
        assert sig.destination_port == 8321
        assert sig.source_ip == "172.18.129.20"

    def test_wjla_is_mmtp(self, slt):
        wjla = next(s for s in slt.services if s.service_id == 2)
        assert wjla.short_service_name == "WJLA"
        assert wjla.category_name == "Linear A/V Service"
        sig = wjla.signaling
        assert sig.sls_protocol == SLS_PROTOCOL_MMTP
        assert sig.protocol_name == "MMTP"
        assert sig.destination_ip == "239.255.7.1"
        assert sig.destination_port == 8071

    def test_protected_and_drm(self, slt):
        wttg = next(s for s in slt.services if s.service_id == 3)
        assert wttg.protected
        assert wttg.drm_system_id == "urn:uuid:edef8ba9-79d6-4ace-a3c8-27dcd51d21ed"
        whut = next(s for s in slt.services if s.service_id == 1)
        assert not whut.protected
        assert whut.drm_system_id is None

    def test_broadband_service(self, slt):
        t2 = next(s for s in slt.services if s.service_id == 6)
        assert t2.broadband_access_required
        assert t2.configuration == "Broadband"

    def test_data_service_category(self, slt):
        sg = next(s for s in slt.services if s.service_id == 65024)
        assert sg.category_name == "Data Service"
        assert sg.signaling.destination_ip == "239.255.0.255"
        assert sg.signaling.destination_port == 8000


class TestSltParsing:
    def test_gzip_and_plain_agree(self, slt):
        plain = gzip.decompress(FIXTURE.read_bytes())
        assert parse_slt_xml(plain) == slt

    def test_parses_lls_table_header(self, slt):
        body = FIXTURE.read_bytes()
        table = ip_layer.parse_lls(
            bytes([0x01, 0x00, 0x00, 0x0D]) + body)
        assert table.table_id == 0x01
        assert table.name == "SLT"
        assert table.table_version == 0x0D
        assert slt_from_lls(table.data) == slt

    def test_rejects_non_slt_root(self):
        with pytest.raises(ValueError):
            parse_slt_xml(b'<NotSLT bsid="1"/>')

    def test_all_services_have_signaling(self, slt):
        for svc in slt.services:
            assert svc.signaling is not None
            assert svc.signaling.sls_protocol in (SLS_PROTOCOL_ROUTE,
                                                  SLS_PROTOCOL_MMTP)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
