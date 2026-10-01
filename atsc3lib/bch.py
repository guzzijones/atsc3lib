"""ATSC 3.0 BCH outer code for FEC frames.

Implements the 12-bit correctable systematic binary BCH code defined in
ATSC A/322 Section 6.1.2.1, Table 6.3.

- Ninner = 16200 (short frames): GF(2^14), generator degree Mouter = 168.
- Ninner = 64800 (normal frames): GF(2^16), generator degree Mouter = 192.

The code is a shortened primitive BCH code: information of Kpayload bits is
encoded into a codeword of Nouter = Kpayload + Mouter bits, appended as the
BCH parity. Decoding corrects up to t = 12 errors (Berlekamp-Massey +
Chien search + Forney for binary, i.e. root flipping).

Reference: ATSC A/322:2024-04 Physical Layer Protocol, Section 6.1.2.1
"""

from typing import List, Optional, Tuple

import numpy as np

# ATSC Ninner=16200 component polynomials (degree 14, roots alpha^1..alpha^24),
# given as exponent lists from Table 6.3.
_BCH_EXPS_16200 = [
    [14, 5, 3, 1, 0],
    [14, 11, 8, 6, 0],
    [14, 10, 9, 6, 2, 1, 0],
    [14, 12, 10, 8, 7, 4, 0],
    [14, 13, 11, 9, 8, 6, 4, 2, 0],
    [14, 13, 9, 8, 7, 3, 0],
    [14, 13, 11, 10, 7, 6, 5, 2, 0],
    [14, 11, 10, 9, 8, 5, 0],
    [14, 10, 9, 3, 2, 1, 0],
    [14, 12, 11, 9, 6, 3, 0],
    [14, 12, 11, 4, 0],
    [14, 13, 10, 8, 7, 6, 5, 3, 2, 1, 0],
]

# ATSC Ninner=64800 component polynomials (degree 16, roots alpha^1..alpha^24).
_BCH_EXPS_64800 = [
    [16, 5, 3, 2, 0],
    [16, 8, 6, 5, 4, 1, 0],
    [16, 11, 10, 9, 8, 7, 5, 4, 3, 2, 0],
    [16, 14, 12, 11, 9, 6, 4, 2, 0],
    [16, 12, 11, 10, 9, 8, 5, 3, 2, 1, 0],
    [16, 15, 14, 13, 12, 10, 9, 8, 7, 5, 4, 2, 0],
    [16, 15, 13, 11, 10, 9, 8, 6, 5, 2, 0],
    [16, 14, 13, 12, 9, 8, 6, 5, 2, 1, 0],
    [16, 11, 10, 9, 7, 5, 0],
    [16, 14, 13, 12, 10, 8, 7, 5, 2, 1, 0],
    [16, 13, 12, 11, 9, 5, 3, 2, 0],
    [16, 12, 11, 9, 7, 6, 5, 1, 0],
]


def _exps_to_poly(exps: List[int]) -> int:
    value = 0
    for e in exps:
        value |= 1 << e
    return value


_BCH_POLYS_16200 = [_exps_to_poly(e) for e in _BCH_EXPS_16200]
_BCH_POLYS_64800 = [_exps_to_poly(e) for e in _BCH_EXPS_64800]


class _GF:
    """Finite field GF(2^m) with exp/log tables for a given primitive poly."""

    def __init__(self, bits: int, prim_poly: int):
        self.bits = bits
        self.order = (1 << bits) - 1
        self.prim_poly = prim_poly
        self.exp = [0] * (self.order + 1)
        self.log = [0] * (1 << bits)
        x = 1
        for i in range(self.order):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & (1 << bits):
                x ^= prim_poly
        if x != 1:
            raise ValueError("Provided polynomial is not primitive")
        self.exp[self.order] = self.exp[0]
        self.exp_np = np.array(self.exp, dtype=np.int64)
        self.log_np = np.array(self.log, dtype=np.int64)

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.exp[(self.log[a] + self.log[b]) % self.order]

    def inv(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("inverse of zero")
        return self.exp[(self.order - self.log[a]) % self.order]


def _poly_mul(a: int, b: int) -> int:
    result = 0
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
    return result


def _poly_mod(dividend: int, divisor: int) -> int:
    while dividend.bit_length() >= divisor.bit_length() and dividend:
        dividend ^= divisor << (dividend.bit_length() - divisor.bit_length())
    return dividend


def _ints_to_bits(value: int, length: int) -> List[int]:
    return [(value >> (length - 1 - i)) & 1 for i in range(length)]


def _bits_to_int(bits: List[int]) -> int:
    value = 0
    for b in bits:
        value = (value << 1) | (b & 1)
    return value


# Cache fields per bit size.
_FIELDS = {}


def _get_field(bits: int, prim_poly: int) -> _GF:
    if bits not in _FIELDS:
        _FIELDS[bits] = _GF(bits, prim_poly)
    return _FIELDS[bits]


class BCHCode:
    """Systematic shortened BCH codec for ATSC 3.0.

    Args:
        n: Inner code length (16200 or 64800).
        t: Correctable errors (12 for ATSC 3.0).
    """

    #: Cached syndrome matrices, keyed by ``(bits, codeword length, t)``.
    #: ``E[j, i] = alpha^(j * (n-1-i))`` for syndrome index ``j`` (1..2t) and
    #: MSB-first bit position ``i``, so a syndrome is the GF(2^m) XOR of the
    #: matrix columns selected by the set bits of the received word.
    _SYND_CACHE = {}

    def __init__(self, n: int = 16200, t: int = 12):
        if n == 16200:
            polys = _BCH_POLYS_16200
            bits = 14
        elif n == 64800:
            polys = _BCH_POLYS_64800
            bits = 16
        else:
            raise ValueError(f"Unsupported Ninner={n} (use 16200 or 64800)")

        self.n = n
        self.t = t
        self.bits = bits
        self.g = 1
        for p in polys[:t]:
            self.g = _poly_mul(self.g, p)
        self.mouter = self.g.bit_length() - 1
        self.n_full = (1 << bits) - 1
        self.k_full = self.n_full - self.mouter
        self.gf = _get_field(bits, polys[0])

    def encode(self, message_bits) -> List[int]:
        """Encode a message of any length <= k_full into a shortened codeword."""
        msg = [int(b) & 1 for b in message_bits]
        kpayload = len(msg)
        if kpayload > self.k_full:
            raise ValueError(f"Message too long: {kpayload} > {self.k_full}")

        msg_int = _bits_to_int(msg)
        # A/322 6.1.2.1: s(x) = m(x) x^Mouter - p(x), so the codeword is
        # directly the shortened length Nouter = Kpayload + Mouter.
        rem = _poly_mod(msg_int << self.mouter, self.g)
        return msg + _ints_to_bits(rem, self.mouter)

    def _syndromes(self, rx) -> List[int]:
        """Syndromes S[1..2t] = r(alpha^j) for the shortened codeword r(x).

        Field addition is integer XOR (GF(2^m) as a vector space over GF(2)),
        so each syndrome is the XOR of the precomputed ``alpha^(j*(n-1-i))``
        columns selected by the set bits of ``rx``.
        """
        gf = self.gf
        n = len(rx)
        key = (self.bits, n, self.t)
        e = self._SYND_CACHE.get(key)
        if e is None:
            j = np.arange(1, 2 * self.t + 1, dtype=np.int64)
            pos = np.arange(n - 1, -1, -1, dtype=np.int64)
            exps = (j[:, None] * pos[None, :]) % gf.order
            e = gf.exp_np[exps]
            self._SYND_CACHE[key] = e
        rxb = np.asarray(rx, dtype=np.uint8) & 1
        ones = np.flatnonzero(rxb)
        synd = [0] * (2 * self.t + 1)
        if ones.size == 0:
            return synd
        synd[1:] = np.bitwise_xor.reduce(e[:, ones], axis=1).tolist()
        return synd

    def _berlekamp_massey(self, synd: List[int]) -> Optional[List[int]]:
        """Error locator polynomial sigma(x), constant term 1."""
        gf = self.gf
        c = [1]
        b_poly = [1]
        L = 0
        m = 1
        b = 1

        def add_scaled_shift(target, src, coef, shift):
            out = target[:]
            for i, v in enumerate(src):
                idx = i + shift
                while len(out) <= idx:
                    out.append(0)
                out[idx] ^= gf.mul(coef, v)
            return out

        for nn in range(0, 2 * self.t):
            d = synd[nn + 1]
            for i in range(1, L + 1):
                if i < len(c):
                    d ^= gf.mul(c[i], synd[nn + 1 - i])

            if d == 0:
                m += 1
            elif 2 * L <= nn:
                t_poly = c[:]
                coef = gf.mul(d, gf.inv(b))
                c = add_scaled_shift(c, b_poly, coef, m)
                L = nn + 1 - L
                b_poly = t_poly
                b = d
                m = 1
            else:
                coef = gf.mul(d, gf.inv(b))
                c = add_scaled_shift(c, b_poly, coef, m)
                m += 1

        if len(c) - 1 != L:
            return None
        return c

    def _chien_search(self, sigma: List[int], n: int) -> Optional[List[int]]:
        """Find error positions p in [0, n) of an n-bit codeword (MSB first).

        Vectorized Horner evaluation of the reversed locator polynomial over
        the candidate field elements ``alpha^(-location)``, ``location =
        n-1-p``.
        """
        gf = self.gf
        sigma_rev = list(reversed(sigma))
        location = np.arange(n - 1, -1, -1, dtype=np.int64)
        x_inv = gf.exp_np[(-location) % gf.order]
        value = np.zeros(n, dtype=np.int64)
        for coeff in sigma_rev:
            c = int(coeff)
            nz = value != 0
            prod = np.zeros(n, dtype=np.int64)
            prod[nz] = gf.exp_np[(gf.log_np[value[nz]] + gf.log_np[x_inv[nz]])
                                 % gf.order]
            value = prod ^ c
        positions = np.flatnonzero(value == 0).tolist()
        if len(positions) > self.t:
            return None
        return positions

    def decode(self, received_bits) -> Tuple[List[int], int, bool]:
        """Decode a shortened codeword.

        Args:
            received_bits: Nouter = Kpayload + Mouter bits, MSB first.

        Returns:
            (corrected_message_bits, num_errors_corrected, success)
        """
        rx = [int(b) & 1 for b in received_bits]
        n = len(rx)
        kpayload = n - self.mouter
        if kpayload <= 0:
            raise ValueError("Received vector shorter than BCH parity")

        synd = self._syndromes(rx)
        if all(s == 0 for s in synd[1:]):
            return rx[:kpayload], 0, True

        sigma = self._berlekamp_massey(synd)
        if sigma is None:
            return rx[:kpayload], 0, False

        positions = self._chien_search(sigma, n)
        if positions is None or len(positions) == 0 or len(positions) > self.t:
            return rx[:kpayload], 0, False

        corrected = rx[:]
        for p in positions:
            corrected[p] ^= 1

        check = self._syndromes(corrected)
        if any(s != 0 for s in check[1:]):
            return rx[:kpayload], 0, False

        return corrected[:kpayload], len(positions), True
