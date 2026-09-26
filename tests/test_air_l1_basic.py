"""End-to-end L1-Basic decode from a real ATSC 3.0 broadcast.

This is the integration gate for the whole physical-layer front-end.  The
fixture is a single 8K Preamble symbol captured off air on RF channel 33
(587 MHz, WRC/WHUT NextGen TV multiplex, Baltimore/Washington) with a HackRF at
10 MS/s.  ``rf33_l1basic_info.npy`` holds the 200 L1-Basic information bits
independently confirmed by a second, unrelated receiver implementation, so a
regression here is a real spec-conformance failure, not a self-consistency
check.

Chain exercised in one shot:

    real Preamble symbol
      -> FFT + preamble pilot channel estimate + equalize (preamble.py)
      -> frequency de-interleave (frequency_interleaver.py, A/322 7.3)
      -> QPSK demap + 6.5.2.10 block de-interleave (l1_basic.py)
      -> LDPC 3/15 + BCH (ldpc_exact.py, bch.py)
      -> 200 L1-Basic bits

If this test fails, do not "fix" it by changing the expected bits: they are
ground truth from the air.
"""

import json
import os

import numpy as np
import pytest

from atsc3lib import spec
from atsc3lib.preamble import preamble_l1_cells
from atsc3lib.l1_basic import L1BasicCodec

_DATA = os.path.join(os.path.dirname(__file__), 'data')
_SYMBOL = os.path.join(_DATA, 'rf33_preamble_symbol.npy')
_INFO = os.path.join(_DATA, 'rf33_l1basic_info.npy')
_META = os.path.join(_DATA, 'rf33_preamble_meta.json')


@pytest.mark.skipif(not os.path.exists(_SYMBOL),
                    reason="real-air fixture not present")
def test_l1_basic_from_real_broadcast():
    symbol = np.load(_SYMBOL).astype(np.complex128)
    expected = np.load(_INFO)
    meta = json.load(open(_META))
    structure = meta['preamble_structure']

    params = spec.PREAMBLE_STRUCTURE[structure]
    assert params.fft == 8192
    assert structure == 27  # 8K, GI 1536, DX 4, L1-Basic Mode 3

    cells, out_params = preamble_l1_cells(symbol, structure)
    assert out_params.l1b_mode == 3

    info, ok = L1BasicCodec(3, max_iterations=100).decode_cells(cells)
    assert ok, "LDPC/BCH did not converge on the real-air fixture"
    assert np.array_equal(info, expected), "decoded bits differ from ground truth"


@pytest.mark.skipif(not os.path.exists(_SYMBOL),
                    reason="real-air fixture not present")
def test_fixture_is_self_consistent():
    # A quick sanity check that the saved symbol really is a Preamble OFDM
    # symbol at the expected carrier geometry.
    symbol = np.load(_SYMBOL)
    assert symbol.dtype == np.complex64
    assert len(symbol) == 8192
    assert np.isfinite(symbol).all()
