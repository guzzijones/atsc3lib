"""ATSC 3.0 front-end: resampling and bootstrap-based frame acquisition.

Bridges raw SDR captures to the bootstrap/L1 decoders.

Sample rates (A/322 Annex N.2.2):
- Bootstrap signaling is defined at 6.144 MHz.
- The main OFDM signal uses 0.384 * (bsr_coefficient + 16) MHz; for the
  default bsr_coefficient = 2 this is 6.912 MHz.

A capture taken at an arbitrary SDR rate (e.g. 10 MHz for the SDRplay) must be
resampled to 6.144 MHz before bootstrap detection, then resampled again /
decimated to 6.912 MHz for OFDM demodulation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import resample_poly

from . import spec
from .bootstrap import (
    detect_bootstrap, decode_preamble_structure, BootstrapDetection,
    BootstrapParams, C_SIZE, BOOTSTRAP_FFT_SIZE, B_SIZE, NUM_BOOTSTRAP_SYMBOLS,
)

BOOTSTRAP_RATE_HZ = spec.BOOTSTRAP_RATE_HZ
MAIN_RATE_HZ = spec.MAIN_RATE_HZ
BOOTSTRAP_SAMPLES_NATIVE = spec.BOOTSTRAP_TOTAL_SAMPLES


def resample_iq(iq: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Rational polyphase resample complex IQ from fs_in to fs_out."""
    if abs(fs_in - fs_out) < 1e-6:
        return iq
    from fractions import Fraction
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(iq, ratio.numerator, ratio.denominator).astype(np.complex64)


def read_hackrf_iq(path: str, limit: Optional[int] = None) -> np.ndarray:
    """Read an int8 interleaved IQ file (SDRplay/HackRF CS8) into complex64."""
    data = np.fromfile(path, dtype=np.int8)
    if limit is not None:
        data = data[:limit * 2]
    return (data[::2].astype(np.float32) +
            1j * data[1::2].astype(np.float32)) / 128.0


def read_cs8_iq(path: str, limit: Optional[int] = None) -> np.ndarray:
    """Read an interleaved int8 (CS8) IQ capture into complex64."""
    return read_hackrf_iq(path, limit)


@dataclass
class FrameInfo:
    """Acquired frame parameters."""
    bootstrap: BootstrapDetection
    params: BootstrapParams
    bootstrap_main_start: int
    symbol_size: int  # FFT + guard, at the main sample rate

    @property
    def fft_size(self) -> int:
        return self.params.fft_size

    @property
    def guard_interval(self) -> int:
        return self.params.guard_interval


def acquire_frame(iq_main: np.ndarray, fs_main: float,
                  search_limit: int = 4_000_000) -> FrameInfo:
    """Resample, detect the bootstrap, and compute frame timing.

    Args:
        iq_main: Complex IQ at the capture sample rate.
        fs_main: Capture sample rate in Hz.
        search_limit: Max main-rate samples to resample for the search.

    Returns:
        FrameInfo with bootstrap detection and OFDM symbol geometry.
    """
    iq_a = iq_main if search_limit is None else iq_main[:search_limit]

    iq_boot = resample_iq(iq_a, fs_main, BOOTSTRAP_RATE_HZ)
    det = detect_bootstrap(iq_boot)
    params = decode_preamble_structure(det.structure)

    # Map the bootstrap start into the main-rate sample index.
    bootstrap_main_start = int(round(det.start * MAIN_RATE_HZ / BOOTSTRAP_RATE_HZ))
    symbol_size = params.fft_size + params.guard_interval

    return FrameInfo(
        bootstrap=det, params=params,
        bootstrap_main_start=bootstrap_main_start,
        symbol_size=symbol_size,
    )


def bootstrap_span_main_rate() -> int:
    """Length of the bootstrap in main-rate samples (9/8 of native)."""
    return int(round(BOOTSTRAP_SAMPLES_NATIVE * MAIN_RATE_HZ / BOOTSTRAP_RATE_HZ))
