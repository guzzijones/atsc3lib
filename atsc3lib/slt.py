"""A/331 Service List Table (SLT) parsing.

The SLT is the entry point of ATSC 3.0 service discovery: a gzip-compressed
XML document delivered as an LLS table (``table_id`` 0x01) that lists every
Service in the Broadcast Stream together with the bootstrap information a
receiver needs to acquire that Service's Service Layer Signaling (SLS).

This module inflates and parses the XML into :class:`Slt` / :class:`SltService`
records.  It does *not* acquire the SLS itself — the advertised
``BroadcastSvcSignaling`` address/port is the input to the ROUTE (A/331 7.1) or
MMTP (A/331 7.2) SLS path.

Gated on the RF33 (587 MHz, BSID 540) capture's SLT in
``tests/data/rf33_slt_lls.bin``.

Reference: ATSC A/331:2026-04, Section 6.3 and Tables 6.2/6.4/6.5/6.6.
"""

import gzip
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

#: RFC 1952 gzip magic (A/331 6.3: the SLT XML is gzip-compressed).
GZIP_MAGIC = b"\x1f\x8b"

#: SLT.Service.BroadcastSvcSignaling@slsProtocol (A/331 Table 6.6).
SLS_PROTOCOL_ROUTE = 1
SLS_PROTOCOL_MMTP = 2
SLS_PROTOCOL_NAME = {SLS_PROTOCOL_ROUTE: "ROUTE", SLS_PROTOCOL_MMTP: "MMTP"}

#: SLT.Service@serviceCategory (A/331 Table 6.4).
SERVICE_CATEGORY = {
    0: "ATSC Reserved",
    1: "Linear A/V Service",
    2: "Linear Audio-only Service",
    3: "App-based Service",
    4: "Data Service",
    5: "ATSC Reserved",
    6: "DRM Data Service",
}

#: Default SLT XML namespace (A/331 6.3).
SLT_NAMESPACE = "tag:atsc.org,2016:XMLSchemas/ATSC3/Delivery/SLT/1.0/"


def _local(tag: str) -> str:
    """Element local name with any ``{namespace}`` prefix removed."""
    return tag.rsplit("}", 1)[-1]


def _find(elem: ET.Element, name: str) -> Optional[ET.Element]:
    """First direct child of ``elem`` whose local name is ``name``."""
    for child in elem:
        if _local(child.tag) == name:
            return child
    return None


def _findall(elem: ET.Element, name: str) -> List[ET.Element]:
    """All direct children of ``elem`` whose local name is ``name``."""
    return [child for child in elem if _local(child.tag) == name]


def _int(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    return int(text.strip())


@dataclass(frozen=True)
class BroadcastSvcSignaling:
    """SLT.Service.BroadcastSvcSignaling (A/331 Table 6.2).

    The attributes give the receiver the IP/UDP coordinates of the LCT channel
    (ROUTE, ``sls_protocol`` 1) or MMTP session (``sls_protocol`` 2) that
    carries this Service's SLS (A/331 6.3).
    """
    sls_protocol: int
    destination_ip: Optional[str]
    destination_port: Optional[int]
    source_ip: Optional[str]

    @property
    def protocol_name(self) -> str:
        return SLS_PROTOCOL_NAME.get(self.sls_protocol,
                                     f"reserved({self.sls_protocol})")


@dataclass(frozen=True)
class SltService:
    """SLT.Service (A/331 Table 6.2)."""
    service_id: int
    slt_svc_seq_num: Optional[int]
    service_category: Optional[int]
    major_channel_no: Optional[int]
    minor_channel_no: Optional[int]
    short_service_name: Optional[str]
    global_service_id: Optional[str]
    configuration: Optional[str]
    protected: bool
    broadband_access_required: bool
    drm_system_id: Optional[str]
    signaling: Optional[BroadcastSvcSignaling]

    @property
    def category_name(self) -> str:
        return SERVICE_CATEGORY.get(self.service_category,
                                    f"reserved({self.service_category})")


@dataclass(frozen=True)
class Slt:
    """SLT root element (A/331 6.3)."""
    bsid: Tuple[int, ...]
    services: Tuple[SltService, ...] = field(default_factory=tuple)


def slt_from_lls(data: bytes) -> Slt:
    """Inflate (if needed) and parse an SLT from LLS body bytes."""
    raw = gzip.decompress(data) if data[:2] == GZIP_MAGIC else data
    return parse_slt_xml(raw)


def parse_slt_xml(xml_bytes: bytes) -> Slt:
    """Parse uncompressed SLT XML bytes (A/331 6.3)."""
    root = ET.fromstring(xml_bytes)
    if _local(root.tag) != "SLT":
        raise ValueError(f"not an SLT root element: {_local(root.tag)}")
    bsid = tuple(int(v) for v in (root.get("bsid") or "").split())
    services = tuple(_parse_service(s) for s in _findall(root, "Service"))
    return Slt(bsid=bsid, services=services)


def _parse_service(elem: ET.Element) -> SltService:
    sig_elem = _find(elem, "BroadcastSvcSignaling")
    signaling = None
    if sig_elem is not None:
        signaling = BroadcastSvcSignaling(
            sls_protocol=int(sig_elem.get("slsProtocol")),
            destination_ip=sig_elem.get("slsDestinationIpAddress"),
            destination_port=_int(sig_elem.get("slsDestinationUdpPort")),
            source_ip=sig_elem.get("slsSourceIpAddress"),
        )
    return SltService(
        service_id=int(elem.get("serviceId")),
        slt_svc_seq_num=_int(elem.get("sltSvcSeqNum")),
        service_category=_int(elem.get("serviceCategory")),
        major_channel_no=_int(elem.get("majorChannelNo")),
        minor_channel_no=_int(elem.get("minorChannelNo")),
        short_service_name=elem.get("shortServiceName"),
        global_service_id=elem.get("globalServiceID"),
        configuration=elem.get("configuration"),
        protected=elem.get("protected") == "true",
        broadband_access_required=elem.get("broadbandAccessRequired") == "true",
        drm_system_id=elem.get("drmSystemID"),
        signaling=signaling,
    )
