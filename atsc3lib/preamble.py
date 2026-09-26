"""ATSC 3.0 Preamble demodulation.

Extracts the data cells of a Preamble OFDM symbol, equalizes them using the
Preamble pilots (dense, ``k mod DX == 0``, DY = 1) and common continual pilots,
applies the frequency deinterleaver, and demaps QPSK for the L1 signalling.

All constants come from :mod:`atsc3lib.spec`, which cites the A/322 clause for
each one.

Reference: ATSC A/322:2024-04, Sections 7.2.5, 7.3, 8.1.2, 8.1.6.
"""

from typing import Tuple

import numpy as np

from . import spec
from .frequency_interleaver import deinterleave as fi_deinterleave
from .pilot_reference import reference_sequence


def carrier_shift(fft_size: int, noc: int) -> int:
    """Absolute FFT-bin index of relative carrier 0.

    A/322 indexes carriers absolutely, with the full ``NoCmax`` carrier set
    centred on DC: absolute carrier ``a`` sits at FFT bin
    ``fft_size//2 + (a - (NoCmax - 1)//2)``.  A cred_coeff-reduced signal uses
    the middle ``noc`` carriers, so relative carrier 0 is absolute carrier
    ``(NoCmax - noc)//2``.
    """
    centre = (spec.NOC_MAX[fft_size] - 1) // 2
    origin = (spec.NOC_MAX[fft_size] - noc) // 2
    return fft_size // 2 + (origin - centre)


def preamble_noc(fft_size: int) -> int:
    """Number of carriers in the first Preamble symbol.

    The first Preamble symbol always uses the ``cred_coeff = 4`` carrier count
    regardless of the payload's cred_coeff (A/322 7.2.5).
    """
    return spec.noc(fft_size, spec.PREAMBLE_FIRST_CRED)


def decode_preamble_structure(preamble_structure: int) -> spec.PreambleStructure:
    """Look up a ``preamble_structure`` value (A/322 Table H.1.1)."""
    if preamble_structure not in spec.PREAMBLE_STRUCTURE:
        raise ValueError(f"Unknown preamble_structure {preamble_structure}")
    return spec.PREAMBLE_STRUCTURE[preamble_structure]


def preamble_symbol_spectrum(symbol_time: np.ndarray, fft_size: int,
                             cfo: float = 0.0,
                             noc: int = None) -> np.ndarray:
    """FFT one Preamble symbol (GI already removed) to relative carriers.

    Args:
        symbol_time: The FFT-length time-domain samples (guard interval
            removed by the caller).
        fft_size: 8192, 16384 or 32768.
        cfo: Residual carrier frequency offset in Hz to de-rotate.
        noc: Number of carriers; defaults to the first-Preamble count.

    Returns:
        ``noc`` complex carriers in relative-carrier order (index 0 is the
        lowest).
    """
    x = np.asarray(symbol_time, dtype=np.complex128)
    if cfo:
        x = x * np.exp(-1j * 2 * np.pi * cfo * np.arange(len(x)))
    spectrum = np.fft.fftshift(np.fft.fft(x))
    if noc is None:
        noc = preamble_noc(fft_size)
    shift = carrier_shift(fft_size, noc)
    return spectrum[shift:shift + noc]


def preamble_pilot_indices(noc: int, dx: int) -> np.ndarray:
    """Relative carrier indices of the Preamble pilots: ``k mod DX == 0``.

    Preamble pilots use DY = 1 (A/322 8.1.6.1), so they recur in every
    Preamble symbol.
    """
    return np.arange(0, noc, dx)


def preamble_pilot_values(noc: int, dx: int, amplitude: float) -> np.ndarray:
    """Known complex values of the Preamble pilots (A/322 8.1.6.3).

    The pilot phase comes from the reference sequence ``r_k``; the value is
    ``A * (1 - 2 r_k)`` (real BPSK, +/-A).
    """
    k = preamble_pilot_indices(noc, dx)
    r = reference_sequence(int(k[-1]) + 1).astype(np.int64)
    return (amplitude * (1.0 - 2.0 * r[k])).astype(np.complex128)


def preamble_data_mask(noc: int, dx: int) -> np.ndarray:
    """Boolean mask of data cells in the first Preamble symbol.

    Data cells are the carriers that are neither Preamble pilots
    (``k mod DX == 0``) nor common continual pilots (A/322 Table D.1.1).
    For 8K / GI 1536 this yields exactly 4851 cells, matching Table 7.2.
    """
    mask = (np.arange(noc) % dx) != 0
    mask[common_continual_pilots(noc)] = False
    return mask


def common_continual_pilots(noc: int) -> np.ndarray:
    """Relative indices of the common continual pilots (A/322 Tables D.1.1/D.1.3).

    The common set is published for 32K as ``CP32`` (Table D.1.1).  For 8K it
    is derived as ``CP8 = ceil(CP32[4k] / 4)`` (Table D.1.3), giving absolute
    carrier indices; those are then shifted into the relative grid by the
    carrier origin ``(NoCmax - NoC) // 2``.
    """
    cp8 = np.ceil(np.asarray(spec.CP32[::4], dtype=float) / 4.0).astype(int)
    origin = (spec.NOC_MAX[8192] - noc) // 2
    cp8 = cp8[(cp8 >= origin) & (cp8 < origin + noc)]
    return cp8 - origin


def estimate_preamble_channel(carriers: np.ndarray, dx: int,
                              amplitude: float) -> np.ndarray:
    """Estimate the channel frequency response across all carriers.

    Least-squares at the known Preamble pilots, then linear interpolation in
    frequency (real and imaginary parts separately) over every carrier.
    """
    noc = len(carriers)
    pilots = preamble_pilot_indices(noc, dx)
    known = preamble_pilot_values(noc, dx, amplitude)
    h_pilots = carriers[pilots] / known

    all_k = np.arange(noc)
    return (np.interp(all_k, pilots, h_pilots.real)
            + 1j * np.interp(all_k, pilots, h_pilots.imag))


def preamble_data_cells(carriers: np.ndarray, channel: np.ndarray,
                        dx: int) -> np.ndarray:
    """Equalized data-cell values of a Preamble symbol, in carrier order."""
    equalized = carriers / channel
    return equalized[preamble_data_mask(len(carriers), dx)]


def preamble_l1_cells(symbol_time: np.ndarray, preamble_structure: int) -> Tuple[np.ndarray, dict]:
    """Full front-end for the first Preamble symbol: FFT -> equalize -> deinterleave.

    Args:
        symbol_time: FFT-length time samples of the first Preamble symbol.
        preamble_structure: The value decoded from the bootstrap.

    Returns:
        (cells, params) where ``cells`` are the frequency-deinterleaved data
        cells ready for QPSK demapping, and ``params`` is the Table H.1.1 row.
    """
    params = decode_preamble_structure(preamble_structure)
    fft_size, gi, dx = params.fft, params.gi, params.dx
    noc = preamble_noc(fft_size)
    carriers = preamble_symbol_spectrum(symbol_time, fft_size, noc=noc)
    amplitude = spec.PREAMBLE_PILOT_AMPLITUDE[(fft_size, gi)]
    channel = estimate_preamble_channel(carriers, dx, amplitude)
    data = preamble_data_cells(carriers, channel, dx)
    return fi_deinterleave(data, 0, fft_size), params
