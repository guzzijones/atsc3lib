"""A/330 ALP packet de-encapsulation.

ALP (ATSC Link-layer Protocol) wraps network-layer packets — IPv4, compressed
IP, MPEG-2 TS and link-layer signalling — into a Base Header plus an optional
Additional Header and Extension Header (A/330 5.1).

Header order (A/330 5.1.2, 5.2, 5.3)::

    Base Header
    Additional Header        single / segmentation / concatenation
    type-specific header     signalling_information_hdr() or type_extension_hdr()
    Extension Header         sub_stream_identification() then header_extension()
    Payload

``packet_type`` (A/330 Table 5.2): 000 IPv4, 010 compressed IP, 100 link-layer
signalling, 110 type extension, 111 MPEG-2 TS.  ``payload_configuration``
(A/330 5.1.1): 0 single whole packet, 1 segmentation or concatenation.

``length`` is always the 11 LSBs of the payload length; the Additional Header
carries the remaining MSBs.  The transmitter's own Baseband Packet pointers
(A/322 5.2.2) resynchronise the walk after a malformed header, and resyncs are
counted rather than silently swallowed.

Reference: ATSC A/330:2026-04, Section 5 and Annex B.
"""

import bisect
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

#: Base Header (A/330 Tables 5.1/5.2, 5.1.1).
PACKET_TYPE_BITS = 3
PAYLOAD_CONFIG_BITS = 1
LENGTH_BITS = 11
BASE_HEADER_BYTES = 2
PC_SINGLE = 0

#: packet_type code values (A/330 Table 5.2).
PT_IPV4 = 0b000
PT_COMPRESSED_IP = 0b010
PT_SIGNALLING = 0b100
PT_TYPE_EXTENSION = 0b110
PT_MPEG2_TS = 0b111

#: Additional Header for single packets (A/330 Table 5.4):
#: length_MSB(5) reserved(1) SIF(1) HEF(1).
SINGLE_PAYLOAD_MAX = 2047
SINGLE_MSB_SHIFT = 3
SINGLE_SIF_SHIFT = 1

#: Additional Header for segmentation (A/330 Table 5.5), control-byte bit
#: positions: segment_sequence_number(5) last_segment_indicator(1) SIF(1) HEF(1).
SEGMENT_SSN_SHIFT = 3
SEGMENT_LAST_SHIFT = 2
SEGMENT_SIF_SHIFT = 1

#: Additional Header for concatenation (A/330 Table 5.6), control-byte bit
#: positions: length_MSB(4) count(3) SIF(1), then (count+1) component_length.
CONCAT_MSB_SHIFT = 4
CONCAT_COUNT_BITS = 3
CONCAT_COUNT_SHIFT = 1
CONCAT_SIF_SHIFT = 0
CONCAT_COMPONENT_BITS = 12
CONCAT_MIN_PACKETS = 2

#: type-specific Additional Headers (A/330 Tables 5.11/5.15).
SIGNALLING_HDR_BYTES = (8 + 16 + 8 + 2 + 2 + 4) // 8
TYPE_EXTENSION_HDR_BYTES = 16 // 8

#: Extension Header (A/330 5.1.3.1/5.1.3.2).
SID_BYTES = 1
HEADER_EXT_LEN_BYTES = 2

#: A/330 5.1.1 / 5.1.2.3 payload bounds.
MAX_ALP_PAYLOAD = 65535
MAX_CONCAT_PAYLOAD = 32767


@dataclass(frozen=True)
class AlpPacket:
    """One de-encapsulated ALP packet (A/330 5.1)."""
    packet_type: int
    payload: bytes
    sid: Optional[int] = None


@dataclass
class AlpStats:
    """Walk bookkeeping, reported rather than hidden."""
    single: int = 0
    segmented: int = 0
    concatenated: int = 0
    components: int = 0
    reassembled: int = 0
    incomplete: int = 0
    signalling: int = 0
    resync: int = 0

    def merge(self, other: 'AlpStats') -> 'AlpStats':
        for key in self.__dataclass_fields__:
            setattr(self, key, getattr(self, key) + getattr(other, key))
        return self


def _bits(data: bytes, bitpos: int, width: int) -> int:
    value = 0
    for i in range(width):
        value = (value << 1) | ((data[(bitpos + i) >> 3]
                                 >> (7 - ((bitpos + i) & 7))) & 1)
    return value


def _header_extension_bytes(data: bytes, pos: int) -> int:
    """Length of ``header_extension()`` at ``pos`` (A/330 Table 5.9)."""
    if pos + HEADER_EXT_LEN_BYTES > len(data):
        return 0
    return HEADER_EXT_LEN_BYTES + data[pos + 1]


@dataclass
class AlpWalker:
    """Stateful ALP walk over a concatenated Baseband Packet payload stream."""

    stream: bytes
    boundaries: List[int] = field(default_factory=list)
    stats: AlpStats = field(default_factory=AlpStats)
    _segments: Dict[int, Dict[int, bytes]] = field(default_factory=dict)

    def __post_init__(self):
        self.boundaries = sorted(set(self.boundaries))

    def _anchor_clean(self, start: int, end: int) -> bool:
        """No Baseband Packet pointer may fall strictly inside [start, end)."""
        i = bisect.bisect_right(self.boundaries, start)
        return not (i < len(self.boundaries) and self.boundaries[i] < end)

    def _payload_len(self, pc: int, sc: int, b: bytes, pos: int):
        """Return (payload_length, additional_header_bytes)."""
        h = (b[pos] << 8) | b[pos + 1]
        length = h & ((1 << LENGTH_BITS) - 1)
        if pc == PC_SINGLE:
            hm = (h >> LENGTH_BITS) & 1
            if not hm:
                return length, BASE_HEADER_BYTES
            if pos + BASE_HEADER_BYTES + 1 > len(b):
                return None
            ctl = b[pos + BASE_HEADER_BYTES]
            length |= (ctl >> 3) << LENGTH_BITS
            return length, BASE_HEADER_BYTES + 1
        if sc == 1:
            return None                       # handled by _concat
        if pos + BASE_HEADER_BYTES + 1 > len(b):
            return None
        return length, BASE_HEADER_BYTES + 1

    def _tail_headers(self, b: bytes, pos: int, hdr: int, ptype: int,
                      sif: int, hef: int):
        """Consume type-specific then Extension Headers.

        Returns ``(sid, header_bytes)`` or None on truncation.
        """
        sid = None
        if ptype == PT_SIGNALLING:
            hdr += SIGNALLING_HDR_BYTES
        elif ptype == PT_TYPE_EXTENSION:
            hdr += TYPE_EXTENSION_HDR_BYTES
        if sif:
            if pos + hdr + SID_BYTES > len(b):
                return None
            sid = b[pos + hdr]
            hdr += SID_BYTES
        if hef:
            if pos + hdr + HEADER_EXT_LEN_BYTES > len(b):
                return None
            hdr += _header_extension_bytes(b, pos + hdr)
        return sid, hdr

    def _single(self, pos: int) -> Optional[Tuple[List[AlpPacket], int]]:
        """payload_configuration == 0 (A/330 5.1.2.1)."""
        b = self.stream
        h = (b[pos] << 8) | b[pos + 1]
        ptype = h >> (BASE_HEADER_BYTES * 8 - PACKET_TYPE_BITS)
        hm = (h >> LENGTH_BITS) & 1
        length = h & ((1 << LENGTH_BITS) - 1)
        hdr = BASE_HEADER_BYTES
        sif = hef = 0
        if hm:
            if pos + hdr + 1 > len(b):
                return None
            ctl = b[pos + hdr]
            length |= (ctl >> SINGLE_MSB_SHIFT) << LENGTH_BITS
            sif, hef = (ctl >> SINGLE_SIF_SHIFT) & 1, ctl & 1
            hdr += 1
        tail = self._tail_headers(b, pos, hdr, ptype, sif, hef)
        if tail is None:
            return None
        sid, hdr = tail
        if length == 0 or length > MAX_ALP_PAYLOAD or pos + hdr + length > len(b):
            return None
        if not self._anchor_clean(pos, pos + hdr + length):
            return None
        self.stats.single += 1
        if ptype == PT_SIGNALLING:
            self.stats.signalling += 1
        return ([AlpPacket(ptype, bytes(b[pos + hdr:pos + hdr + length]), sid)],
                hdr + length)

    def _segment(self, pos: int) -> Optional[Tuple[List[AlpPacket], int]]:
        """payload_configuration == 1, segmentation_concatenation == 0."""
        b = self.stream
        if pos + BASE_HEADER_BYTES + 1 > len(b):
            return None
        h = (b[pos] << 8) | b[pos + 1]
        ptype = h >> (BASE_HEADER_BYTES * 8 - PACKET_TYPE_BITS)
        length = h & ((1 << LENGTH_BITS) - 1)
        ctl = b[pos + BASE_HEADER_BYTES]
        ssn = ctl >> SEGMENT_SSN_SHIFT
        last = (ctl >> SEGMENT_LAST_SHIFT) & 1
        sif = (ctl >> SEGMENT_SIF_SHIFT) & 1
        hef = ctl & 1
        hdr = BASE_HEADER_BYTES + 1
        tail = self._tail_headers(b, pos, hdr, ptype, sif, hef)
        if tail is None:
            return None
        sid, hdr = tail
        if length == 0 or length > MAX_ALP_PAYLOAD or pos + hdr + length > len(b):
            return None
        if ssn == 0 and last:
            return None
        if not self._anchor_clean(pos, pos + hdr + length):
            return None
        self.stats.segmented += 1
        if ptype == PT_SIGNALLING:
            self.stats.signalling += 1
        key = sid if sid is not None else 0
        buf = self._segments.setdefault(key, {})
        buf[ssn] = bytes(b[pos + hdr:pos + hdr + length])
        out: List[AlpPacket] = []
        if last:
            keys = sorted(buf)
            if keys == list(range(len(keys))):
                out = [AlpPacket(ptype, b"".join(buf[k] for k in keys), sid)]
                self.stats.reassembled += 1
            else:
                self.stats.incomplete += 1
            self._segments[key] = {}
        return out, hdr + length

    def _concat(self, pos: int) -> Optional[Tuple[List[AlpPacket], int]]:
        """payload_configuration == 1, segmentation_concatenation == 1."""
        b = self.stream
        if pos + BASE_HEADER_BYTES + 1 > len(b):
            return None
        h = (b[pos] << 8) | b[pos + 1]
        ptype = h >> (BASE_HEADER_BYTES * 8 - PACKET_TYPE_BITS)
        length = h & ((1 << LENGTH_BITS) - 1)
        ctl = b[pos + BASE_HEADER_BYTES]
        length_msb = ctl >> CONCAT_MSB_SHIFT
        count = (ctl >> CONCAT_COUNT_SHIFT) & ((1 << CONCAT_COUNT_BITS) - 1)
        sif = (ctl >> CONCAT_SIF_SHIFT) & 1
        packets = count + CONCAT_MIN_PACKETS
        total = (length_msb << LENGTH_BITS) | length
        nfields = packets - 1
        nbytes = (nfields * CONCAT_COMPONENT_BITS + 7) // 8
        hdr = BASE_HEADER_BYTES + 1 + nbytes
        if total == 0 or total > MAX_CONCAT_PAYLOAD or pos + hdr + total > len(b):
            return None
        tail = self._tail_headers(b, pos, hdr, ptype, sif, 0)
        if tail is None:
            return None
        sid, hdr = tail
        if not self._anchor_clean(pos, pos + hdr + total):
            return None
        start = (BASE_HEADER_BYTES + 1) * 8
        lens = [_bits(b, start + i * CONCAT_COMPONENT_BITS,
                      CONCAT_COMPONENT_BITS) for i in range(nfields)]
        last_len = total - sum(lens)
        if last_len <= 0 or any(L <= 0 for L in lens):
            return None
        lens.append(last_len)
        self.stats.concatenated += 1
        self.stats.components += packets
        if ptype == PT_SIGNALLING:
            self.stats.signalling += 1
        out, off = [], pos + hdr
        for L in lens:
            out.append(AlpPacket(ptype, bytes(b[off:off + L]), sid))
            off += L
        return out, hdr + total

    def run(self) -> Tuple[List[AlpPacket], AlpStats]:
        """Walk the stream from the first signalled boundary."""
        b, out = self.stream, []
        pos = self.boundaries[0] if self.boundaries else 0
        while pos + BASE_HEADER_BYTES <= len(b):
            head = (b[pos] << 8) | b[pos + 1]
            pc = (head >> (LENGTH_BITS + PAYLOAD_CONFIG_BITS)) & 1
            if pc == PC_SINGLE:
                r = self._single(pos)
            else:
                sc = (head >> LENGTH_BITS) & 1
                r = self._concat(pos) if sc else self._segment(pos)
            if r is None:
                nxt = [x for x in self.boundaries if x > pos]
                if not nxt:
                    break
                pos = nxt[0]
                self.stats.resync += 1
                continue
            pkts, consumed = r
            out += pkts
            pos += consumed
        return out, self.stats


def parse_alp(stream: bytes,
              boundaries: Iterable[int] = ()) -> Tuple[List[AlpPacket], AlpStats]:
    """De-encapsulate a Baseband Packet payload stream (A/330 5.1)."""
    return AlpWalker(stream=stream, boundaries=list(boundaries)).run()
