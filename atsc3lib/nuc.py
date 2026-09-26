"""ATSC 3.0 non-uniform constellation (NUC) alphabets and demapper.

Implements A/322 Annex C constellation position vectors and the Section 6.3
max-log demapper used by data PLPs.

The numeric tables are shipped as a banked artifact
(``data/nuc_a322.npz``) extracted from the A/322 specification; the position
vectors are facts and the quadrant rule below reconstructs the full alphabet
from the ``w`` vectors the tables print.

A/322 6.3.4.2 quadrant rule, with ``b = M/4``::

    x[0:b]   =  w
    x[b:2b]  = -conj(w)
    x[2b:3b] =  conj(w)
    x[3b:4b] = -w

and index k is the decimal value of (y0 .. y_{MOD-1}) with y0 the MSB.

QPSK (A/322 Table C.1.1) is the uniform four-point constellation and is not
tabulated in Annex C.

Reference: ATSC A/322:2024-04, Section 6.3, Annex C.
"""

import os
from typing import Dict, Tuple

import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
_BANK = os.path.join(_DATA_DIR, 'nuc_a322.npz')

# Modulation order (bits per symbol) by L1D_plp_mod signalling value.
# A/322 Table 9.8: 0 QPSK, 1 16QAM-NUC, 2 64QAM-NUC, 3 256QAM-NUC,
# 4 1024QAM-NUC, 5 4096QAM-NUC.
MOD_ORDER: Dict[int, int] = {0: 2, 1: 4, 2: 6, 3: 8, 4: 10, 5: 12}
MOD_NAME: Dict[int, str] = {
    0: 'QPSK', 1: '16QAM', 2: '64QAM', 3: '256QAM',
    4: '1024QAM', 5: '4096QAM',
}

# table tag per NUC order (Annex C): 16 -> C.1.2/C.1.3, 64 -> C.1.4/C.1.5,
# 256 -> C.1.6/C.1.7.  Rates 2..7 and 8..13 use different tables.
_TABLE = {16: ('C.1.2', 'C.1.3'), 64: ('C.1.4', 'C.1.5'),
          256: ('C.1.6', 'C.1.7')}

# A/322 Table C.1.1 QPSK: bit pair (y0,y1) -> (Re,Im), y0 = MSB.
QPSK_POINTS = np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]) / np.sqrt(2)


def _bank_key(table: str, rate: int) -> str:
    return f"{table.replace('.', '_')}__{rate}_15"


def _quadrant(w: np.ndarray) -> np.ndarray:
    """A/322 6.3.4.2: reconstruct the M-point alphabet from its w vector."""
    w = np.asarray(w)
    return np.concatenate([w, -np.conj(w), np.conj(w), -w])


_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def points(mod_order: int, rate: int) -> np.ndarray:
    """Constellation positions for a (modulation order, code rate).

    Args:
        mod_order: bits per symbol (2, 4, 6, 8, 10 or 12).
        rate: LDPC code rate numerator over 15 (2..13).

    Returns:
        Complex array of ``2**mod_order`` points, index = label with y0 MSB.
    """
    if mod_order == 2:
        return QPSK_POINTS
    m = 1 << mod_order
    if m not in _TABLE:
        raise NotImplementedError(
            f"No NUC table for modulation order {mod_order}")
    if not 2 <= rate <= 13:
        raise ValueError(f"Rate {rate}/15 out of range 2..13")
    key = (mod_order, rate)
    if key not in _CACHE:
        table = _TABLE[m][0 if rate <= 7 else 1]
        bank = np.load(_BANK)
        w = bank[_bank_key(table, rate)]
        _CACHE[key] = _quadrant(w)
    return _CACHE[key]


def bit_masks(nbits: int) -> np.ndarray:
    """masks[i] -> boolean over labels, True where y_i = 1 (y0 the MSB)."""
    k = np.arange(1 << nbits)
    return np.array([((k >> (nbits - 1 - i)) & 1).astype(bool)
                     for i in range(nbits)])


def demap_llr(cells: np.ndarray, mod_order: int, rate: int,
              sigma2: float = None) -> np.ndarray:
    """Max-log LLRs in q-stream order (A/322 6.3.3, y_i = q_{MOD*s+i}).

    Convention: ``llr > 0`` means bit 0 at the corresponding bit level.
    """
    pts = points(mod_order, rate)
    nb = mod_order
    z = np.asarray(cells, dtype=np.complex128)
    d2 = np.abs(z[:, None] - pts[None, :]) ** 2
    if sigma2 is None:
        sigma2 = max(float(np.mean(d2.min(axis=1))), 1e-9)
    masks = bit_masks(nb)
    out = np.empty((len(z), nb))
    for i in range(nb):
        out[:, i] = (d2[:, masks[i]].min(1) - d2[:, ~masks[i]].min(1)) / sigma2
    return out.ravel()
