"""ATSC 3.0 Convolutional Time Interleaver (CTI), receive side.

A/322 7.1.4: the CTI is a convolutional (Ramsey/Forney) interleaver with
``Nrows`` delay lines, line ``k`` holding ``k`` delay elements.  Two
commutators step one row per cell.  Tracing one cell shows it is written into
row ``k`` at step ``q``, advances one element per visit to row ``k`` (every
``Nrows`` steps), and leaves on the ``k``-th visit::

    out[q] = in[q - Nrows * k_q]        k_q = (start_row + q) mod Nrows

Inverting, the pre-CTI cell with index ``i`` leaves the interleaver at::

    q(i) = i + Nrows * ((start_row + i) mod Nrows)

A/322 9.3.9.1 prints the transmitter's own form of that map for the signalled
field ``L1D_plp_CTI_fec_block_start`` (evaluated at ``i = C``)::

    L1D_plp_CTI_fec_block_start = C + Nrows * ((start_row + C) mod Nrows)

so the reading above is gated on the air by solving for ``C`` and requiring
``0 <= C < cells_per_FEC_Block`` (:func:`spec.cti_start_c`).  The CTI never
resets: consecutive subframes concatenate with ``start_row`` advancing by the
PLP size, which is signalled per frame.

Reference: ATSC A/322:2024-04, Sections 7.1.3/7.1.4/9.3.9.1, Table 9.24.
"""

from dataclasses import dataclass

import numpy as np


def out_index(i, nrows: int, start_row: int) -> np.ndarray:
    """Output position of each pre-CTI cell index ``i`` (A/322 7.1.4.1)."""
    i = np.asarray(i, np.int64)
    return i + nrows * ((start_row + i) % nrows)


def in_index(q, nrows: int, start_row: int) -> np.ndarray:
    """Pre-CTI cell index leaving at output position ``q`` (inverse map)."""
    q = np.asarray(q, np.int64)
    return q - nrows * ((start_row + q) % nrows)


@dataclass
class Deinterleaved:
    """Result of a CTI de-interleave.

    Attributes:
        cells: Reconstructed pre-CTI cells; entries with ``valid`` False are
            fed by the delay-line initial state (A/322 7.1.4.2) and are zero.
        valid: Boolean mask; False where the output position lay beyond the
            received stream.
    """
    cells: np.ndarray
    valid: np.ndarray


def interleave(cells, nrows: int, start_row: int, fill=None,
               n_out: int = None) -> np.ndarray:
    """Transmit-side CTI, for round-trip tests.

    ``cells[i]`` is placed at output position ``out_index(i)``, so a payload of
    length ``n`` needs an output at least ``n + nrows**2`` long.  Output
    positions whose pre-CTI index is negative carry the delay-line initial state
    (A/322 7.1.4.2); ``fill`` may supply those, else zero.
    """
    cells = np.asarray(cells)
    n = len(cells)
    L = (n + nrows * nrows) if n_out is None else n_out
    out = np.zeros(L, cells.dtype) if fill is None else np.asarray(fill).copy()
    q = np.arange(L, dtype=np.int64)
    i = in_index(q, nrows, start_row)
    ok = (i >= 0) & (i < n)
    out[q[ok]] = cells[i[ok]]
    return out


def deinterleave(recv, nrows: int, start_row: int,
                 n_out: int = None) -> Deinterleaved:
    """Receive-side CTI de-interleave (A/322 7.1.4.1).

    Args:
        recv: Cells as they leave the CTI, the first being the one whose
            commutator position is ``start_row``.
        nrows: Number of delay lines (A/322 Table 9.24).
        start_row: ``L1D_plp_CTI_start_row`` for this subframe.
        n_out: Output length; defaults to ``len(recv)``.

    Returns:
        :class:`Deinterleaved`; ``valid`` marks the reconstructed cells.
    """
    recv = np.asarray(recv)
    length = len(recv)
    n = length if n_out is None else n_out
    i = np.arange(n, dtype=np.int64)
    q = out_index(i, nrows, start_row)
    valid = q < length
    out = np.zeros(n, recv.dtype)
    out[valid] = recv[q[valid]]
    return Deinterleaved(cells=out, valid=valid)


def first_invalid(valid: np.ndarray, length: int) -> int:
    """Length of the leading run of valid cells (FEC blocks must fit inside)."""
    bad = np.flatnonzero(~np.asarray(valid))
    return int(bad[0]) if len(bad) else length
