"""IPv4 / UDP parsing and fragment reassembly.

The ALP layer (A/330) hands up network-layer packets: IPv4 datagrams,
optionally fragmented, carrying UDP.  This module turns them into complete
``(src, dst, sport, dport, payload)`` UDP datagrams, reassembling IPv4
fragments (flags/fragment-offset, RFC 791).

This is not itself an ATSC specification — it is the IP/UDP layer above
A/330 — so the fields follow RFC 791 and RFC 768 directly.

Reference: RFC 791 (IPv4), RFC 768 (UDP).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

#: IPv4 fixed header / UDP header lengths (RFC 791, RFC 768).
IPV4_MIN_HEADER = 20
UDP_HEADER_BYTES = 8
IPV4_VERSION = 4
PROTO_UDP = 17
FRAG_OFFSET_UNIT = 8
MORE_FRAGMENTS = 0x1


@dataclass(frozen=True)
class UdpDatagram:
    """A reassembled UDP datagram (RFC 768)."""
    src_ip: bytes
    dst_ip: bytes
    src_port: int
    dst_port: int
    payload: bytes


@dataclass
class IpStats:
    """Reassembly bookkeeping."""
    datagrams: int = 0
    fragments: int = 0
    reassembled: int = 0
    not_ipv4: int = 0
    not_udp: int = 0


@dataclass
class IpReassembler:
    """Reassemble IPv4 fragments into UDP datagrams (RFC 791)."""
    stats: IpStats = field(default_factory=IpStats)
    _frags: Dict[Tuple, Dict] = field(default_factory=dict)

    def feed(self, packet: bytes) -> List[UdpDatagram]:
        """Feed one IPv4 packet; return any complete UDP datagrams."""
        if len(packet) < IPV4_MIN_HEADER or (packet[0] >> 4) != IPV4_VERSION:
            self.stats.not_ipv4 += 1
            return []
        ihl = (packet[0] & 0x0F) * 4
        total_length = (packet[2] << 8) | packet[3]
        ident = (packet[4] << 8) | packet[5]
        flags = packet[6] >> 5
        frag_offset = (((packet[6] & 0x1F) << 8) | packet[7]) * FRAG_OFFSET_UNIT
        proto = packet[9]
        src, dst = bytes(packet[12:16]), bytes(packet[16:20])
        if proto != PROTO_UDP:
            self.stats.not_udp += 1
            return []
        body = packet[ihl:total_length] if 0 < total_length <= len(packet) \
            else packet[ihl:]
        more = flags & MORE_FRAGMENTS
        if not more and frag_offset == 0:
            self.stats.datagrams += 1
            return self._udp(src, dst, body)
        self.stats.fragments += 1
        key = (src, dst, ident, proto)
        entry = self._frags.setdefault(key, {})
        entry[frag_offset] = bytes(body)
        if not more:
            entry["_end"] = frag_offset + len(body)
        end = entry.get("_end")
        if end is None:
            return []
        chunks, off = [], 0
        while off < end:
            if off not in entry:
                return []
            chunks.append(entry[off])
            off += len(entry[off])
        del self._frags[key]
        self.stats.reassembled += 1
        self.stats.datagrams += 1
        return self._udp(src, dst, b"".join(chunks))

    def _udp(self, src: bytes, dst: bytes, udp: bytes) -> List[UdpDatagram]:
        if len(udp) < UDP_HEADER_BYTES:
            return []
        ulen = (udp[4] << 8) | udp[5]
        payload = udp[UDP_HEADER_BYTES:ulen] \
            if UDP_HEADER_BYTES <= ulen <= len(udp) else udp[UDP_HEADER_BYTES:]
        return [UdpDatagram(src_ip=src, dst_ip=dst,
                            src_port=(udp[0] << 8) | udp[1],
                            dst_port=(udp[2] << 8) | udp[3],
                            payload=payload)]


#: LLS multicast / port (A/331).
LLS_IP = b"\xe0\x00\x17\x3c"        # 224.0.23.60
LLS_PORT = 4937

#: LLS table_id -> name (A/331 Table 6.1).
LLS_TABLE_NAME = {
    0x01: "SLT", 0x02: "RRT", 0x03: "SystemTime", 0x04: "AEAT",
    0x05: "OnscreenMessageNotification", 0x06: "CertificationData",
    0x07: "SignedMultiTable",
}


def is_lls(datagram: UdpDatagram) -> bool:
    """True for a Low-Level Signaling datagram (A/331 6.1)."""
    return datagram.dst_ip == LLS_IP and datagram.dst_port == LLS_PORT


@dataclass(frozen=True)
class LlsTable:
    """A Low-Level Signaling table (A/331 6.1)."""
    table_id: int
    name: str
    data: bytes


def parse_lls(payload: bytes) -> Optional[LlsTable]:
    """Parse an LLS table's 4-byte header (A/331 6.1)."""
    if len(payload) < 4:
        return None
    table_id = payload[0]
    return LlsTable(table_id=table_id,
                    name=LLS_TABLE_NAME.get(table_id, f"0x{table_id:02x}"),
                    data=payload[4:])
