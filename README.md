# atsc3lib

**ATSC 3.0 physical-layer receiver library.**

Hardware-agnostic: works with raw IQ from any SDR wide enough for the 6 MHz
channel (HackRF, Airspy, SDRplay, USRP).

Implements, from the A/322 specification:

- Bootstrap detection and signalling decode
- Preamble OFDM parameter detection, channel estimation and equalization
- Frequency de-interleaving
- L1-Basic and L1-Detail FEC (BCH + LDPC) and field parsing
- **Per-PLP configuration** decoded from L1-Detail
- **Data-PLP payload chain**: NUC/QAM demap, bit de-interleave, HTI twisted
  block de-interleave and cell interleaver, LDPC/BCH, descramble to Baseband
  Packets — for both short-frame (Ninner=16200) and normal-frame
  (Ninner=64800) codes, all 12 rates and all four NUC modes
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
atsc3-capture -f 599 -o out/capture.iq          # auto-detect SDR
atsc3-capture -f 599 -t hackrf -g 28            # force HackRF
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
  L1-Detail: version 1, CRC OK
  Per-PLP configuration:
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
- **Airspy** (via `airspy_rx`)
- **Any SoapySDR device**

The library is hardware-agnostic: it consumes raw IQ samples from any source
wide enough for the 6 MHz ATSC 3.0 channel.

## Link-layer limitations

The A/330 layer implements Base Headers, single/segmentation/concatenation
payloads, the type-specific and Extension Headers, IPv4 fragment reassembly
and UDP, and LLS table extraction.  ROHC header decompression (A/330 §6,
compressed-IP `packet_type = 010`) is not implemented, so compressed streams
are surfaced as raw ALP packets rather than decompressed datagrams.

## Payload limitations

**Ground rule: every rung must be validated on air.**  A feature is only
implemented when a receivable stream carries it; a structural or synthetic
round-trip gate does not qualify a rung.  If the link cannot deliver the
feature, record the blocker and move on rather than building an unprovable
stage.

The payload chain supports both Ninner = 16200 (short frames) and
Ninner = 64800 (normal frames), and the tabulated QPSK/16QAM/64QAM/256QAM
MODCODs.  TI mode 2 supports the A/322 7.1.5.4 twisted block interleaver and
the optional A/322 7.1.5.2 **cell** interleaver
(`L1D_plp_HTI_cell_interleaver`); TI modes 0/1 are supported.

**A lighthouse multiplex decodes off air.**  PLP-0 (64QAM-NUC 11/15) converges 53-60 of 74 FEC blocks from a clean capture and yields real LLS: the
A/331 **SLT** and SystemTime.  The SLT lists the major services carried by the
multiplex.  This is the first off-air service list, and it replaces the earlier
"padding-only, link-limited" conclusion.

The scale bug that hid this: the A/322 Annex C NUC alphabets have unit average
power, but an equalised cell block does not, and the max-log metric is not
invariant to that scale.  Every data FEC block is therefore normalised to unit
mean power before demapping (`DataPlpChain.decode_cells`); without it PLP-0
decodes 0/74, with it 53-60/74.
The decision-directed CPE happened to normalise internally, which masked the
defect whenever CPE was on.  The property is gated by
`tests/test_data_plp.py::test_decode_cells_scale_invariant`.

Payload demodulation works on any subframe.  Subframe 0 carries the Preamble
spare cells; later subframes are demodulated at their own FFT/GI/pilot geometry
with the A/322 7.3 frequency-interleaver counter reset at the subframe boundary.
Pass `subframe=` to `decode_plp_payload` (and `--plp` plus `--subframe` to the
CLI).  The per-FFT pilot and data-cell tables (A/322 Tables 7.3-7.6, Annex F,
D.1.4/D.1.5) are inlined in `atsc3lib/pilot_data.py`, generated by
`tools/fetch_pilot_tables.py` from the pinned independent transcription and
gated by the constant-data-carrier identity.

The Preamble may span **multiple OFDM symbols** (A/322 7.2.5).  L1-Basic sits
at the start of the first symbol; L1-Detail fills the rest of it and the later
symbols, block-de-interleaved with Lc = NP columns (7.2.5.2).  All Preamble
symbols share the bootstrap geometry, later ones use
`L1B_preamble_reduced_carriers`, and the frequency-interleaver counter keeps
counting frame symbols (so subframe 0's first data symbol has origin NP).  A
two-symbol Preamble (NP = 2, L1-Basic Mode 1 + L1-Detail) is verified off air.

LDM and CTI multiplexes are supported: `cti.py` implements the A/322 7.1.4
convolutional time de-interleaver and its 9.3.9.1 signalled identity, `spec.py`
carries the Table 9.24 `Nrows` menu and the Table 9.22/6.15/6.16 LDM power
ratios, and the demodulator includes scattered-pilot **fine timing**
(8.1.3.1, `subframe_fine_timing`) and a decision-directed per-symbol **CPE**
(7.2.6.5 dummy tail, `cpe_correct`).  Both are on by default for the CTI path
(`decode_cti_plp_streams`; `--no-fine-timing`, `--no-cpe` to disable).
