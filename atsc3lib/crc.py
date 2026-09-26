"""ATSC 3.0 CRC-32 (A/322 Section 6.1.2.2).

The L1-Basic and L1-Detail signalling blocks each end with a 32-bit CRC.  The
generator polynomial is:

    G(x) = x^32 + x^21 + x^16 + x^11 + 1

with all register stages initialised to one, data applied MSB first, and the
32 register contents appended most-significant bit first.  Because the
initialisation is a fixed all-ones value, the CRC is affine: the effect of any
constant XOR mask (for example an unknown descrambler) cancels in a difference.
"""

import numpy as np

from . import spec

CRC_POLY = spec.CRC32_POLY
CRC_INIT = spec.CRC32_INIT
CRC_BITS = 32


def crc32(bits, init: int = CRC_INIT) -> int:
    """Return the CRC-32 of a bit sequence (MSB first).

    Args:
        bits: Iterable of 0/1 values.
        init: Initial register value (default all ones).

    Returns:
        The 32-bit CRC value as an integer.
    """
    reg = init & 0xFFFFFFFF
    for b in bits:
        feedback = ((reg >> 31) & 1) ^ (int(b) & 1)
        reg = (reg << 1) & 0xFFFFFFFF
        if feedback:
            reg ^= CRC_POLY
    return reg


def crc32_ok(bits) -> bool:
    """True if the trailing 32 bits are the CRC of the preceding bits.

    That is, the CRC register over the entire block (data followed by its CRC,
    with all-ones init) reads zero.
    """
    return crc32(bits) == 0
