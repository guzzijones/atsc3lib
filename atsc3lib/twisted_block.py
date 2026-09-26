"""A/322 7.1.5.4 Twisted Block Interleaver (HTI, TI mode 2).

The time interleaver of a data PLP with ``L1D_plp_TI_mode == 2`` is a
"twisted" block interleaver over the PLP's FEC blocks: the cells of
``N_FEC_TI`` FEC blocks are written into an ``N_r x N_c`` memory and read out
along a twisted diagonal.

A/322 7.1.5.4 defines the read order for cell ``i`` (0-based within the
interleaver memory)::

    R_i     = i mod N_r
    T_i     = R_i mod N_c
    C_i     = (T_i + floor(i / N_r)) mod N_c
    theta_i = N_r * C_i + R_i

with the cell skipped when ``theta_i < N_FEC_TI_DUMMY * N_r`` (virtual FEC
blocks sit first in the memory).  The memory is written column-wise, so the
linear index ``theta = N_r * C + R`` holds cell ``R`` of FEC block ``C``.

This formulation is gated on the worked example A/327 Figure 6.5 prints
(N_r=4, N_c=3, one virtual block -> ``b g a f d e c h``), which exercises the
virtual-cell skip rule the on-air RF33 case (no virtual blocks) does not.

Reference: ATSC A/322:2024-04, Section 7.1.5.4; ATSC A/327 Figure 6.5.
"""

from typing import Tuple

import numpy as np


def read_order(nrows: int, ncols: int, n_virtual: int = 0) -> np.ndarray:
    """Memory indices ``theta_i`` in TBI output order, with virtuals removed.

    Returns an int array of length ``nrows * (ncols - n_virtual)`` whose
    entries are indices into the compacted (virtual-free) memory, so that
    ``out[order] = transmitted`` de-interleaves a received cell vector.
    """
    i = np.arange(nrows * ncols)
    r = i % nrows
    t = r % ncols
    c = (t + i // nrows) % ncols
    theta = nrows * c + r
    keep = theta >= n_virtual * nrows
    return theta[keep] - n_virtual * nrows


def deinterleave(cells: np.ndarray, nrows: int, ncols: int,
                 n_virtual: int = 0) -> np.ndarray:
    """Cells in transmitted order -> memory order (FEC block major)."""
    order = read_order(nrows, ncols, n_virtual)
    if len(order) != len(cells):
        raise ValueError(f"expected {len(order)} cells, got {len(cells)}")
    out = np.empty_like(cells)
    out[order] = cells
    return out


def interleave(mem: np.ndarray, nrows: int, ncols: int,
               n_virtual: int = 0) -> np.ndarray:
    """Memory order -> transmitted order (transmitter side)."""
    return np.asarray(mem)[read_order(nrows, ncols, n_virtual)]


def fec_block(cells: np.ndarray, j: int, nrows: int, ncols: int,
              n_virtual: int = 0) -> np.ndarray:
    """The ``j``-th data FEC block's cells, without materialising the memory.

    ``j`` is 0-based over the DATA blocks; block ``j`` occupies memory columns
    ``j + n_virtual``.
    """
    r = np.arange(nrows)
    c = j + n_virtual
    i = nrows * ((c - (r % ncols)) % ncols) + r
    if n_virtual:
        i = i - np.searchsorted(
            np.sort(_virtual_positions(nrows, ncols, n_virtual)), i)
    return np.asarray(cells)[i]


def _virtual_positions(nrows: int, ncols: int, n_virtual: int) -> np.ndarray:
    i = np.arange(nrows * ncols)
    r = i % nrows
    c = ((r % ncols) + i // nrows) % ncols
    return i[(nrows * c + r) < n_virtual * nrows]


# A/327 Figure 6.5 worked example: N_r=4, N_c=3, one virtual FEC block.
GOLD = dict(nrows=4, ncols=3, n_virtual=1,
            expect="b g a f d e c h".split())


def gold_vector() -> Tuple[list, list]:
    """Reproduce A/327 Figure 6.5's printed output."""
    nr, nc, nv = GOLD["nrows"], GOLD["ncols"], GOLD["n_virtual"]
    mem = [""] * (nr * nv) + list("abcd") + list("efgh")
    got = [mem[nv * nr + t] for t in read_order(nr, nc, nv)]
    return got, list(GOLD["expect"])
