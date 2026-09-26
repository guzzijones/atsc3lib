"""ATSC 3.0 pilot reference sequence.

Every pilot cell (scattered, continual, edge, Preamble, subframe-boundary) is
modulated from a single real reference sequence ``r_k`` (A/322 Section 8.1.2):

- LFSR generator G(x) = 1 + x^9 + x^10 + x^12 + x^13.
- Seed and printed first values are in :mod:`atsc3lib.spec`.
- Pilot value: ``Re{c} = A * (1 - 2 r_k)``, ``Im{c} = 0`` (real BPSK, +/-A).

The amplitude and spacing tables now live in :mod:`atsc3lib.spec`; this module
only computes the sequence itself.
"""

import numpy as np

from . import spec


def reference_sequence(length: int) -> np.ndarray:
    """The pilot reference sequence ``r_k`` (0/1) of the given length.

    A/322 Section 8.1.2.  The first 24 values are 1101 1000 0000 0001 0100
    0000.
    """
    sr = spec.REFERENCE_SEED
    bits = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        feedback = (sr ^ (sr >> 1) ^ (sr >> 3) ^ (sr >> 4)) & 1
        bits[i] = sr & 1
        sr >>= 1
        if feedback:
            sr |= 0x1000
    return bits


def pilot_values(length: int, amplitude: float) -> np.ndarray:
    """Complex BPSK pilot values of the given length: ``+/- amplitude``."""
    r = reference_sequence(length)
    return (amplitude * (1.0 - 2.0 * r)).astype(np.complex128)


def preamble_amplitude(fft_size: int, guard_interval: int) -> float:
    """Preamble pilot amplitude A_Preamble (A/322 Table 8.6)."""
    return spec.PREAMBLE_PILOT_AMPLITUDE[(fft_size, guard_interval)]


def preamble_dx(preamble_structure: int) -> int:
    """Preamble pilot spacing DX for a ``preamble_structure`` (Table H.1.1)."""
    return spec.PREAMBLE_STRUCTURE[preamble_structure].dx


def preamble_pilot_indices(noc: int, dx: int) -> np.ndarray:
    """Relative carrier indices k in [0, NoC) with ``k mod DX == 0``."""
    return np.arange(0, noc, dx)
