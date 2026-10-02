"""A/331 ROUTE delivery: LCT/ALC packet parsing and object reassembly.

ROUTE (Real-time Object delivery over Unidirectional Transport) carries Service
Layer Signaling and media as objects over ALC/LCT sessions (A/331 Annex A.3).
The packet is an IP/UDP datagram whose payload is::

    LCT Header | FEC Payload ID (start_offset) | Payload Data

The LCT header is RFC 5651 (A/331 A.3.6); for ROUTE the Congestion Control
flag C is 0, the Protocol-Specific Indication PSI is 0b10 (source packet), the
Transport Session Identifier flag S is 1 (32-bit TSI) and the Transport Object
Identifier flag O is 1 (32-bit TOI).  The FEC Payload ID for source flows is a
32-bit ``start_offset`` giving the byte position of this packet's payload
within the delivery object (A/331 A.3.5.1), so an object is reassembled by
placing each payload at its ``start_offset``.

The Codepoint (CP) gives the delivery-object type (A/331 Table A.3.6); the SLS
itself is delivered on the TSI=0 LCT channel (A/331 7.1), and its fragments are
gzip-compressed (A/331 A.3.3.4).

Gated on the RF33 (587 MHz) capture: ``tests/data/route_efdt_gmloop.bin``
(EFDT bootstrap, TSI=0/TOI=0) and the SG-FE00 Data Service objects in
``tests/data/route_object_sgfe00_*.bin`` / ``route_sls_sgfe00_cp3.bin``.

Reference: ATSC A/331:2026-04 Annex A.3.5, A.3.6, Tables A.3.6/A.3.7;
RFC 5651 (LCT), RFC 5775 (ALC).
"""

import gzip
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

#: LCT header fixed part before the optional CCI/TSI/TOI (RFC 5651 Figure 1).
LCT_BASE_BYTES = 4
#: Congestion Control Information is 32*(C+1) bits (RFC 5651 5.1).
CCI_WORD_BYTES = 4
#: S/TOI half-word unit (RFC 5651 5.1).
HALF_WORD_BYTES = 2
#: FEC Payload ID for source flows is a 32-bit start_offset (A/331 A.3.5.1).
START_OFFSET_BYTES = 4

#: ROUTE constrains the LCT flags (A/331 A.3.6).
ROUTE_VERSION = 1
ROUTE_PSI_SOURCE = 0b10
ROUTE_C = 0
ROUTE_S = 1
ROUTE_O = 1

#: LCT Header Extension Type values (A/331 A.3.7.1, RFC 5651 5.2).
HET_EXT_TIME = 1
HET_EXT_ROUTE_PRESENTATION_TIME = 66
HET_EXT_TOL = 67
HET_EXT_FTI = 194

#: Payload Codepoint (A/331 Table A.3.6).
CP_NRT_FILE = 1
CP_NRT_ENTITY = 2
CP_NRT_UNSIGNED_PACKAGE = 3
CP_NRT_SIGNED_PACKAGE = 4
CP_NEW_IS_TIMELINE_CHANGED = 5
CP_NEW_IS_TIMELINE_CONTINUED = 6
CP_REDUNDANT_IS = 7
CP_MEDIA_SEGMENT_FILE = 8
CP_MEDIA_SEGMENT_ENTITY = 9
CP_NAME = {
    CP_NRT_FILE: "NRT File Mode",
    CP_NRT_ENTITY: "NRT Entity Mode",
    CP_NRT_UNSIGNED_PACKAGE: "NRT Unsigned Package Mode",
    CP_NRT_SIGNED_PACKAGE: "NRT Signed Package Mode",
    CP_NEW_IS_TIMELINE_CHANGED: "New IS, timeline changed",
    CP_NEW_IS_TIMELINE_CONTINUED: "New IS, timeline continued",
    CP_REDUNDANT_IS: "Redundant IS",
    CP_MEDIA_SEGMENT_FILE: "Media Segment, File Mode",
    CP_MEDIA_SEGMENT_ENTITY: "Media Segment, Entity Mode",
}

#: The LCT channel that carries ROUTE SLS (A/331 7.1: dedicated, TSI=0).
SLS_TSI = 0

#: gzip magic (RFC 1952) used to mark gzip-compressed SLS fragments.
GZIP_MAGIC = b"\x1f\x8b"


class RouteError(ValueError):
    """Malformed ROUTE packet."""


@dataclass(frozen=True)
class LctHeader:
    """LCT header fields (RFC 5651 5.1, A/331 A.3.6)."""
    version: int
    congestion_control: int
    psi: int
    tsi_present: int
    toi_present: int
    half_word: int
    close_session: int
    close_object: int
    header_len: int
    codepoint: int
    cci: bytes
    tsi: Optional[int]
    toi: Optional[int]
    extensions: bytes

    @property
    def is_route_source(self) -> bool:
        """True when the flags match a ROUTE source packet (A/331 A.3.6)."""
        return (self.version == ROUTE_VERSION
                and self.congestion_control == ROUTE_C
                and self.psi == ROUTE_PSI_SOURCE
                and self.tsi_present == ROUTE_S
                and self.toi_present == ROUTE_O
                and self.half_word == 0)

    @property
    def codepoint_name(self) -> str:
        return CP_NAME.get(self.codepoint, f"reserved({self.codepoint})")

    @property
    def header_bytes(self) -> int:
        return self.header_len * 4


@dataclass(frozen=True)
class RoutePacket:
    """One ROUTE/ALC source packet (A/331 A.3.5)."""
    header: LctHeader
    start_offset: int
    payload: bytes

    @property
    def tsi(self) -> int:
        return self.header.tsi if self.header.tsi is not None else 0

    @property
    def toi(self) -> int:
        return self.header.toi if self.header.toi is not None else 0

    @property
    def is_sls(self) -> bool:
        return self.tsi == SLS_TSI


def parse_lct_header(data: bytes) -> LctHeader:
    """Parse an LCT header from the first bytes of a ROUTE payload."""
    if len(data) < LCT_BASE_BYTES:
        raise RouteError("truncated LCT header")
    b0, b1, header_len, codepoint = data[0], data[1], data[2], data[3]
    version = b0 >> 4
    congestion_control = (b0 >> 2) & 0b11
    psi = b0 & 0b11
    # RFC 5651 5.1 Figure 1: S(1) O(2) H(1) Res(2) A(1) B(1).
    tsi_present = (b1 >> 7) & 1
    toi_present = (b1 >> 5) & 0b11
    half_word = (b1 >> 4) & 1
    close_session = (b1 >> 1) & 1
    close_object = b1 & 1

    offset = LCT_BASE_BYTES + CCI_WORD_BYTES * (congestion_control + 1)
    if header_len * 4 < offset or offset > len(data):
        raise RouteError("LCT header length inconsistent with CCI")

    cci = data[LCT_BASE_BYTES:offset]
    tsi_bytes = 4 * tsi_present + HALF_WORD_BYTES * half_word
    tsi = (int.from_bytes(data[offset:offset + tsi_bytes], "big")
           if tsi_bytes else None)
    offset += tsi_bytes
    toi_bytes = 4 * toi_present + HALF_WORD_BYTES * half_word
    toi = (int.from_bytes(data[offset:offset + toi_bytes], "big")
           if toi_bytes else None)
    offset += toi_bytes

    extensions = data[offset:header_len * 4]
    return LctHeader(version=version, congestion_control=congestion_control,
                     psi=psi, tsi_present=tsi_present, toi_present=toi_present,
                     half_word=half_word, close_session=close_session,
                     close_object=close_object, header_len=header_len,
                     codepoint=codepoint, cci=cci, tsi=tsi, toi=toi,
                     extensions=extensions)


def parse_route_packet(data: bytes) -> RoutePacket:
    """Parse a ROUTE/ALC source packet (A/331 A.3.5)."""
    header = parse_lct_header(data)
    after = header.header_bytes
    if len(data) < after + START_OFFSET_BYTES:
        raise RouteError("ROUTE packet too short for FEC Payload ID")
    start_offset = int.from_bytes(
        data[after:after + START_OFFSET_BYTES], "big")
    return RoutePacket(header=header, start_offset=start_offset,
                       payload=data[after + START_OFFSET_BYTES:])


@dataclass
class RouteObject:
    """A delivery object reassembled across ROUTE packets (A/331 A.3.10.2)."""
    tsi: int
    toi: int
    data: bytes = b""
    _chunks: Dict[int, bytes] = field(default_factory=dict, repr=False)

    def add(self, start_offset: int, payload: bytes) -> None:
        self._chunks[start_offset] = bytes(payload)

    def reassemble(self) -> bytes:
        if not self._chunks:
            return b""
        end = max(off + len(chunk) for off, chunk in self._chunks.items())
        buf = bytearray(end)
        for off, chunk in self._chunks.items():
            buf[off:off + len(chunk)] = chunk
        self.data = bytes(buf)
        return self.data

    def _ensure(self) -> bytes:
        if not self.data and self._chunks:
            self.reassemble()
        return self.data

    @property
    def is_gzip(self) -> bool:
        return self._ensure()[:2] == GZIP_MAGIC

    def inflate(self) -> bytes:
        """gzip-decompress the object, or return it unchanged if not gzip."""
        data = self._ensure()
        if data[:2] == GZIP_MAGIC:
            return gzip.decompress(data)
        return data


@dataclass
class RouteAssembler:
    """Reassemble ROUTE delivery objects keyed by (TSI, TOI)."""
    objects: Dict[Tuple[int, int], RouteObject] = field(default_factory=dict)
    packets: int = 0

    def feed(self, data: bytes) -> RouteObject:
        pkt = parse_route_packet(data)
        self.packets += 1
        key = (pkt.tsi, pkt.toi)
        obj = self.objects.get(key)
        if obj is None:
            obj = RouteObject(tsi=pkt.tsi, toi=pkt.toi)
            self.objects[key] = obj
        obj.add(pkt.start_offset, pkt.payload)
        return obj

    def sls_objects(self) -> List[RouteObject]:
        """Objects delivered on the SLS LCT channel (TSI=0, A/331 7.1)."""
        return [obj for (tsi, _), obj in self.objects.items()
                if tsi == SLS_TSI]


#: ATSC-FDT namespace for the Extended FDT (A/331 A.3.3.2).
AFDT_NAMESPACE = "tag:atsc.org,2016:XMLSchemas/ATSC3/Delivery/ATSC-FDT/1.0/"


@dataclass(frozen=True)
class EfdtFile:
    """One ``File`` entry of an Extended FDT Instance (A/331 Table A.3.1)."""
    content_location: str
    toi: int
    content_length: Optional[int]
    content_type: Optional[str]


@dataclass(frozen=True)
class Efdt:
    """Extended File Delivery Table instance (A/331 A.3.3.2).

    Announces the files (SLS fragments, segments) a source flow delivers; a
    file's ``toi`` links it to the ROUTE delivery object of the same TOI.
    """
    efdt_version: Optional[int]
    expires: Optional[int]
    files: Tuple[EfdtFile, ...] = field(default_factory=tuple)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_efdt(xml_bytes: bytes) -> Efdt:
    """Parse an Extended FDT Instance (A/331 A.3.3.2, FDT per RFC 6726)."""
    root = ET.fromstring(xml_bytes)
    if _local(root.tag) != "FDT-Instance":
        raise RouteError(f"not an FDT-Instance: {_local(root.tag)}")
    files = []
    for elem in root:
        if _local(elem.tag) != "File":
            continue
        length = elem.get("Content-Length")
        files.append(EfdtFile(
            content_location=elem.get("Content-Location", ""),
            toi=int(elem.get("TOI", "0")),
            content_length=int(length) if length is not None else None,
            content_type=elem.get("Content-Type"),
        ))
    return Efdt(
        efdt_version=_int_attr(root, "efdtVersion", AFDT_NAMESPACE),
        expires=_int_attr(root, "Expires", None),
        files=tuple(files),
    )


def _int_attr(elem: ET.Element, name: str, namespace: Optional[str]):
    key = f"{{{namespace}}}{name}" if namespace else name
    value = elem.get(key)
    if value is None:
        value = elem.get(name)
    return int(value) if value is not None else None
