"""L1-Detail decode from a real two-symbol Preamble (A/322 7.2.5.2).

RF30 (WIAV-CD, 569 MHz) transmits a normal 8K Preamble with
``L1B_preamble_num_symbols = 1`` (NP = 2), L1-Basic Mode 1 and
``L1B_L1_Detail_total_cells = 3708``.  L1-Basic fills the first 3820 cells of
the first Preamble symbol; L1-Detail continues in the rest of that symbol and
finishes in the second, then is block-de-interleaved with Lc = NP = 2 columns
(7.2.5.2) before the FEC chain.

Fixture ``rf30_preamble_pair.npy`` holds the two Preamble symbol spans
(each ``GI + FFT``) at main rate, starting at the bootstrap; the first span
begins with the bootstrap-relative offset.  This is the wall that blocked every
multi-symbol-Preamble station before; the test asserts BCH + CRC verify.
"""

import json
import os

import numpy as np
import pytest

from atsc3lib import spec
from atsc3lib.l1_basic import L1BasicCodec
from atsc3lib.l1_detail import L1DetailCodec, preamble_block_deinterleave
from atsc3lib.l1_signaling import parse_l1_basic
from atsc3lib.preamble import preamble_l1_cells, preamble_symbol_cells

_DATA = os.path.join(os.path.dirname(__file__), 'data')
_PAIR = os.path.join(_DATA, 'rf30_preamble_pair.npy')
_META = os.path.join(_DATA, 'rf30_preamble_pair_meta.json')

pytestmark = pytest.mark.skipif(
    not (os.path.exists(_PAIR) and os.path.exists(_META)),
    reason="real-air two-symbol-Preamble fixture not present")


def _meta():
    return json.load(open(_META))


def test_l1_basic_mode1_first_symbol():
    m = _meta()
    pair = np.load(_PAIR)
    fft, gi = m['fft'], m['gi']
    sym0 = pair[gi:gi + fft].astype(np.complex128)
    cells, params = preamble_l1_cells(sym0, m['preamble_structure'])
    assert params.l1b_mode == 1
    assert len(cells) == spec.PREAMBLE_DATA_CELLS_CRED4[(fft, gi)]
    lb = L1BasicCodec(1, max_iterations=80)
    assert lb.n_cells == m['l1b_cells']
    bits, ok = lb.decode_cells(cells[:lb.n_cells])
    assert ok
    parsed = parse_l1_basic(bits)
    assert parsed.preamble_num_symbols + 1 == m['preamble_num_symbols'] == 2
    assert parsed.preamble_reduced_carriers == m['preamble_reduced_carriers']
    assert parsed.l1_detail_total_cells == m['l1_detail_total_cells']


def test_l1_detail_across_two_preamble_symbols():
    m = _meta()
    pair = np.load(_PAIR)
    fft, gi = m['fft'], m['gi']
    sym0 = pair[gi:gi + fft].astype(np.complex128)
    sym1 = pair[(fft + gi) + gi:(fft + gi) + gi + fft].astype(np.complex128)

    cells0, _ = preamble_l1_cells(sym0, m['preamble_structure'])
    lb = L1BasicCodec(1, max_iterations=80)
    bits, ok = lb.decode_cells(cells0[:lb.n_cells])
    assert ok
    parsed = parse_l1_basic(bits)

    cells1 = preamble_symbol_cells(
        sym1, m['preamble_structure'], fi_index=1,
        cred_coeff=m['preamble_reduced_carriers'])
    y = np.concatenate([cells0[lb.n_cells:], cells1])[:m['l1_detail_total_cells']]
    assert len(y) == parsed.l1_detail_total_cells

    ld = L1DetailCodec(parsed.l1_detail_fec_type + 1,
                       parsed.l1_detail_size_bytes * 8, max_iterations=80)
    m_deint = preamble_block_deinterleave(y, parsed.preamble_num_symbols + 1)
    out, bch_ok, crc_ok = ld.decode_cells(m_deint)
    assert bch_ok and crc_ok
    assert ld.last_ldpc_converged
    assert out.shape[0] == parsed.l1_detail_size_bytes * 8
