"""ATSC 3.0 receiver: the validated physical-layer chain in one place.

This is the canonical receive path, exercised end to end on real air.  It runs:

    raw IQ capture
      -> resample to 6.144 MHz, detect the bootstrap        (frontend, bootstrap)
      -> locate the first Preamble symbol, FFT, equalize,
         frequency-deinterleave                              (preamble)
      -> L1-Basic FEC decode + parse                       (l1_basic, l1_signaling)
      -> L1-Detail FEC decode + parse -> per-PLP config     (l1_detail, l1_signaling)

The payload chain (PLP -> ALP -> IP) is not implemented yet; this stops at the
per-PLP configuration decoded from L1-Detail.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import spec
from .bootstrap import detect_bootstrap
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
    """Detect the bootstrap and return (preamble time samples, structure, start).

    The bootstrap occupies ``BOOTSTRAP_TOTAL_SAMPLES`` at 6.144 MHz; the first
    Preamble OFDM symbol begins immediately after it.  We return the Preamble
    symbol's FFT-length body (guard interval removed) at the main sample rate.
    """
    boot = resample_iq(iq_main, fs_main, spec.BOOTSTRAP_RATE_HZ)
    det = detect_bootstrap(boot)

    main = resample_iq(iq_main, fs_main, spec.MAIN_RATE_HZ)
    start = int(round(det.start * spec.MAIN_RATE_HZ / spec.BOOTSTRAP_RATE_HZ))
    boot_span = int(round(spec.BOOTSTRAP_TOTAL_SAMPLES
                          * spec.MAIN_RATE_HZ / spec.BOOTSTRAP_RATE_HZ))
    params = spec.PREAMBLE_STRUCTURE[det.structure]
    sym_start = start + boot_span + params.gi
    symbol = main[sym_start:sym_start + params.fft].astype(np.complex128)
    return symbol, det.structure, start


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
        symbol, structure, start = _bootstrap_to_preamble(iq_main, fs_main)
    except Exception as exc:  # noqa: BLE001 - report, do not crash the caller
        return ReceiverResult(preamble_structure=-1, l1_basic=None, l1_basic_ok=False,
                              l1_detail=None, l1_detail_ok=False,
                              bootstrap_start=-1, frame_start=-1,
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
