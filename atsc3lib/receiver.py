"""ATSC 3.0 receiver: the validated physical-layer chain in one place.

This is the canonical receive path, exercised end to end on real air.  It runs:

    raw IQ capture
      -> resample to 6.144 MHz, detect the bootstrap        (frontend, bootstrap)
      -> locate the first Preamble symbol, FFT, equalize,
         frequency-deinterleave                              (preamble)
      -> L1-Basic FEC decode + parse                       (l1_basic, l1_signaling)
      -> L1-Detail FEC decode + parse -> per-PLP config     (l1_detail, l1_signaling)
      -> PLP FEC -> Baseband Packets                        (payload)
      -> ALP -> IPv4/UDP -> Low-Level Signaling             (baseband, alp, ip)

:func:`decode_capture` stops at the per-PLP configuration; :func:`decode_plp_payload`
and :func:`decode_plp_streams` extend the chain through the payload and the
link/network layers.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import spec
from .bootstrap import detect_bootstrap, fine_cfo
from .crc import crc32_ok
from .frontend import read_hackrf_iq, resample_iq
from .l1_basic import L1BasicCodec
from .l1_detail import L1DetailCodec
from .l1_signaling import L1Basic, L1Detail, parse_l1_basic, parse_l1_detail
from .preamble import preamble_l1_cells


@dataclass
class ReceiverResult:
    """Outcome of decoding the signalling of one frame."""
    preamble_structure: int
    l1_basic: Optional[L1Basic]
    l1_basic_ok: bool
    l1_detail: Optional[L1Detail]
    l1_detail_ok: bool
    bootstrap_start: int
    frame_start: int
    error: Optional[str] = None

    @property
    def plps(self):
        """Flat list of (subframe_index, PLPConfig) for the frame."""
        if self.l1_detail is None:
            return []
        out = []
        for sf in self.l1_detail.subframes:
            for plp in sf['plps']:
                out.append((sf['index'], plp))
        return out


def _bootstrap_to_preamble(iq_main: np.ndarray, fs_main: float):
    """Detect the bootstrap and return (preamble body, main frame, structure, start).

    ``main`` is the main-rate IQ beginning at the bootstrap, de-rotated by the
    fractional carrier-frequency offset estimated from the bootstrap; the first
    Preamble OFDM symbol body follows after the bootstrap span and the guard
    interval.
    """
    boot = resample_iq(iq_main, fs_main, spec.BOOTSTRAP_RATE_HZ)
    det = detect_bootstrap(boot)
    cfo = fine_cfo(boot, det.start)

    main = resample_iq(iq_main, fs_main, spec.MAIN_RATE_HZ)
    start = int(round(det.start * spec.MAIN_RATE_HZ / spec.BOOTSTRAP_RATE_HZ))
    main = main[start:]
    if cfo:
        n = np.arange(len(main))
        main = (main * np.exp(-1j * 2 * np.pi * cfo * n
                              / spec.MAIN_RATE_HZ)).astype(np.complex64)
    boot_span = int(round(spec.BOOTSTRAP_TOTAL_SAMPLES
                          * spec.MAIN_RATE_HZ / spec.BOOTSTRAP_RATE_HZ))
    params = spec.PREAMBLE_STRUCTURE[det.structure]
    symbol = main[boot_span + params.gi:boot_span + params.gi
                  + params.fft].astype(np.complex128)
    return symbol, main, det.structure, start


def decode_signaling(iq_main: np.ndarray, fs_main: float,
                     max_iterations: int = 100) -> ReceiverResult:
    """Decode L1-Basic and L1-Detail from a raw IQ capture.

    Args:
        iq_main: Complex IQ samples at the capture sample rate.
        fs_main: Capture sample rate in Hz (e.g. 10e6 for the HackRF).
        max_iterations: LDPC iteration cap for both signalling blocks.

    Returns:
        ReceiverResult with the parsed L1-Basic, L1-Detail (and therefore the
        per-PLP configuration), plus the bootstrap/frame sample offsets.
    """
    try:
        symbol, main, structure, start = _bootstrap_to_preamble(iq_main, fs_main)
    except Exception as exc:  # noqa: BLE001 - report, do not crash the caller
        return ReceiverResult(preamble_structure=-1, l1_basic=None,
                              l1_basic_ok=False, l1_detail=None,
                              l1_detail_ok=False, bootstrap_start=-1,
                              frame_start=-1,
                              error=f"bootstrap/preamble: {exc}")

    params = spec.PREAMBLE_STRUCTURE[structure]
    cells, _ = preamble_l1_cells(symbol, structure)

    # --- L1-Basic: first 484 cells (Mode 3), then parse ---
    lb_codec = L1BasicCodec(params.l1b_mode, max_iterations=max_iterations)
    lb_bits, lb_ok = lb_codec.decode_cells(cells[:lb_codec.n_cells])
    if not (lb_ok and crc32_ok(lb_bits)):
        return ReceiverResult(preamble_structure=structure, l1_basic=None,
                              l1_basic_ok=False, l1_detail=None,
                              l1_detail_ok=False, bootstrap_start=start,
                              frame_start=start,
                              error="L1-Basic did not verify")
    l1b = parse_l1_basic(lb_bits)

    # --- L1-Detail: next L1B_L1_Detail_total_cells cells ---
    ksig = l1b.l1_detail_size_bytes * 8
    ld_codec = L1DetailCodec(l1b.l1_detail_fec_type + 1, ksig,
                             max_iterations=max_iterations)
    lo = lb_codec.n_cells
    ld_bits, bch_ok, ld_crc_ok = ld_codec.decode_cells(
        cells[lo:lo + ld_codec.n_cells])
    if not (bch_ok and ld_crc_ok):
        return ReceiverResult(preamble_structure=structure, l1_basic=l1b,
                              l1_basic_ok=True, l1_detail=None,
                              l1_detail_ok=False, bootstrap_start=start,
                              frame_start=start,
                              error="L1-Detail did not verify")
    l1d = parse_l1_detail(ld_bits, l1b)

    return ReceiverResult(preamble_structure=structure, l1_basic=l1b, l1_basic_ok=True,
                          l1_detail=l1d, l1_detail_ok=True,
                          bootstrap_start=start, frame_start=start)


def decode_capture(path: str, fs_main: float, fmt: str = 'auto',
                   max_iterations: int = 100) -> ReceiverResult:
    """Decode the signalling from a saved IQ capture file.

    ``fmt`` selects the sample format: 'cs8' (HackRF int8, the default for
    unknown files), 'cs16' (int16) or 'cf32' (float32).  'auto' guesses by
    file extension and falls back to int8.
    """
    if fmt == 'auto':
        fmt = 'cs8'
    if fmt in ('cs8', 'int8', 'hackrf'):
        iq = read_hackrf_iq(path)
    elif fmt in ('cf32', 'float32'):
        raw = np.fromfile(path, dtype=np.float32)
        iq = (raw[::2] + 1j * raw[1::2]).astype(np.complex64)
    elif fmt in ('cs16', 'int16'):
        raw = np.fromfile(path, dtype=np.int16)
        iq = ((raw[::2] + 1j * raw[1::2]) / 32768.0).astype(np.complex64)
    else:
        raise ValueError(f"Unknown capture format {fmt!r}")
    return decode_signaling(iq, fs_main, max_iterations=max_iterations)


def _subframe0_geometry(result):
    """Resolve subframe-0 demod geometry from a decoded ReceiverResult."""
    lb = result.l1_basic
    sf0 = result.l1_detail.subframes[0]
    fft = {0: 8192, 1: 16384, 2: 32768}.get(lb.first_sub_fft_size, 8192)
    gi = spec.guard_interval(fft, lb.first_sub_guard_interval)
    noc = spec.noc(fft, lb.first_sub_reduced_carriers)
    pattern = spec.SP_PATTERN_SIGNALING.get(
        lb.first_sub_scattered_pilot_pattern, 'SP4_2')
    dx, dy = spec.SP_DXDY.get(pattern, (4, 2))
    n_sym = sf0.get('num_ofdm_symbols')
    if n_sym is None:
        n_sym = lb.first_sub_num_ofdm_symbols
    n_data_symbols = n_sym + 1
    sbs = []
    if sf0.get('sbs_first') or lb.first_sub_sbs_first:
        sbs.append(0)
    if sf0.get('sbs_last') or lb.first_sub_sbs_last:
        sbs.append(n_data_symbols - 1)
    sbs_null = sf0.get('sbs_null_cells')
    return dict(fft=fft, gi=gi, noc=noc, dx=dx, dy=dy,
                n_data_symbols=n_data_symbols, sbs=tuple(sbs),
                pattern=pattern, sbs_null=sbs_null)


def decode_plp_payload(iq_main: np.ndarray, fs_main: float, plp_id: int = None,
                       max_iterations: int = 100, result=None):
    """Decode one subframe-0 data PLP's payload.

    ``plp_id`` selects the PLP; when None the smallest subframe-0 PLP is
    chosen (the RF33 PLP-16 signalling shape).  ``result`` may supply an
    already-decoded :class:`ReceiverResult` to avoid re-running the signalling
    chain.  Returns ``(result, payload)`` with ``payload`` a
    :class:`~atsc3lib.payload.PlpPayload` or None.
    """
    from .payload import decode_subframe0_plp

    if result is None:
        result = decode_signaling(iq_main, fs_main,
                                  max_iterations=max_iterations)
    if not result.l1_detail_ok:
        return result, None

    sf0 = result.l1_detail.subframes[0]
    candidates = [p for p in sf0['plps'] if p.layer == 0]
    if not candidates:
        return result, None
    if plp_id is None:
        target = min(candidates, key=lambda p: p.size)
    else:
        target = next((p for p in candidates if p.plp_id == plp_id), None)
    if target is None:
        return result, None

    g = _subframe0_geometry(result)
    _, main, structure, _ = _bootstrap_to_preamble(iq_main, fs_main)
    l1_cells = 484 + result.l1_basic.l1_detail_total_cells
    boot_span = int(round(spec.BOOTSTRAP_TOTAL_SAMPLES
                          * spec.MAIN_RATE_HZ / spec.BOOTSTRAP_RATE_HZ))
    payload = decode_subframe0_plp(
        main, boot_span, g['fft'], g['gi'], g['noc'], g['dx'], g['dy'],
        g['n_data_symbols'], g['sbs'], preamble_structure=structure,
        l1_cells=l1_cells, plp=target, max_iterations=max_iterations,
        pattern=g['pattern'], sbs_null=g['sbs_null'])
    return result, payload


def decode_first_plp_payload(iq_main: np.ndarray, fs_main: float,
                             max_iterations: int = 100, result=None):
    """Decode signalling and the smallest subframe-0 PLP payload.

    This is the RF33-class case (a small QPSK PLP alongside a large payload
    PLP).  Returns ``(result, payload)`` where ``payload`` is a
    :class:`~atsc3lib.payload.PlpPayload` or None.
    """
    return decode_plp_payload(iq_main, fs_main, plp_id=None,
                              max_iterations=max_iterations, result=result)


def decode_plp_streams(iq_main: np.ndarray, fs_main: float, plp_id: int = None,
                       max_iterations: int = 100, result=None):
    """Decode one PLP all the way to network-layer streams.

    Runs :func:`decode_plp_payload` and then Baseband Packets -> ALP -> IPv4/6
    -> UDP -> Low-Level Signaling (A/322 5.2, A/330 5, A/331 6.1).  Returns
    ``(result, streams)`` with ``streams`` a
    :class:`~atsc3lib.payload.DecodedStreams` or None.
    """
    from .payload import decode_streams

    result, payload = decode_plp_payload(
        iq_main, fs_main, plp_id=plp_id, max_iterations=max_iterations,
        result=result)
    if payload is None:
        return result, None
    return result, decode_streams(payload)
