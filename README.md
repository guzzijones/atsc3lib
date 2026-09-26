# atsc3lib

**ATSC 3.0 physical-layer receiver library.**

Hardware-agnostic: works with raw IQ from any SDR (HackRF, RTL-SDR, Airspy,
SDRplay, USRP).

Implements, from the A/322 specification:

- Bootstrap detection and signalling decode
- Preamble OFDM parameter detection, channel estimation and equalization
- Frequency de-interleaving
- L1-Basic and L1-Detail FEC (BCH + LDPC) and field parsing
- **Per-PLP configuration** decoded from L1-Detail
- **Data-PLP payload chain**: NUC/QAM demap, bit de-interleave, HTI twisted
  block de-interleave and cell interleaver, LDPC/BCH, descramble to Baseband
  Packets
- **Link/network layer**: A/322 5.2.2 Baseband Packet headers, A/330 ALP
  de-encapsulation (single / segmentation / concatenation / signalling),
  IPv4 fragment reassembly and UDP, A/331 Low-Level Signaling

Validated against real off-air captures and an independent receiver.

## Installation

```bash
cd atsc3lib
pip install -e .
```

## Capture IQ Samples

```bash
atsc3-capture -f 569 -o out/capture.iq          # auto-detect SDR
atsc3-capture -f 569 -t hackrf -g 28            # force HackRF
```

## Decode Signalling and PLP Configuration

```bash
atsc3-decode out/capture.iq --rate 10e6 --fmt cs8
```

Example output:

```
Capture: out/capture.iq @ 10.000 MHz
  L1-Basic: version 0, CRC OK
    subframes           : 2
    L1-Detail fec type  : 2 (mode 3)
    L1-Detail total cells: 880
  L1-Detail: version 1, BSID 540, CRC OK
  Per-PLP configuration (3 PLP(s)):
    subframe 0  PLP 0   layer=0 start=0 size=199800 fec=0 mod=2 cod=9 TI=2
    subframe 0  PLP 16  layer=0 start=199800 size=8100 fec=0 mod=0 cod=0 TI=2
    subframe 1  PLP 1   layer=0 start=0 size=947700 fec=1 mod=3 cod=9 TI=2
```

## Python API

```python
from atsc3lib import decode_capture, decode_plp_streams

result = decode_capture('out/capture.iq', fs_main=10e6, fmt='cs8')
if result.l1_detail_ok:
    for subframe, plp in result.plps:
        print(subframe, plp.plp_id, plp.modulation, plp.code_rate)

# Decode a subframe-0 PLP through to ALP/UDP/LLS
iq = ...  # the same capture IQ
result, streams = decode_plp_streams(iq, fs_main=10e6, plp_id=16, result=result)
for table in streams.lls:
    print(table.name, len(table.data))
for datagram in streams.datagrams:
    ...
```

Raw Baseband Packet bytes remain available from `decode_plp_payload` ->
`PlpPayload.baseband_packets`; `payload.decode_streams` runs the link layer on
them alone.

## Supported Hardware

- **HackRF** (via `hackrf_transfer`)
- **RTL-SDR** (via `rtl_sdr`)
- **Airspy** (via `airspy_rx`)
- **Any SoapySDR device**

The library is hardware-agnostic: it consumes raw IQ samples from any source.

## Link-layer limitations

The A/330 layer implements Base Headers, single/segmentation/concatenation
payloads, the type-specific and Extension Headers, IPv4 fragment reassembly
and UDP, and LLS table extraction.  ROHC header decompression (A/330 §6,
compressed-IP `packet_type = 010`) is not implemented, so compressed streams
are surfaced as raw ALP packets rather than decompressed datagrams.

## Payload limitations

The payload chain supports Ninner = 16200 (short frames) only, and the
tabulated QPSK/16QAM/64QAM/256QAM MODCODs.  TI mode 2 supports the A/322
7.1.5.4 twisted block interleaver and the optional A/322 7.1.5.2 **cell**
interleaver (`L1D_plp_HTI_cell_interleaver`); TI modes 0/1 are supported.

PLP 0 (64QAM-NUC 11/15) of the RF33 multiplex sits at the ~18.8 dB AWGN
threshold and the available captures measure ~15.9 dB MER, so it does not
converge in either this receiver or the independent reference (0/74 FEC
blocks).  This is a link-margin limit, not a chain error; the chain itself is
verified bit-exact against reference-encoded 64QAM 11/15 cells.
