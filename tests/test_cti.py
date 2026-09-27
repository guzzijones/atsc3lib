"""Convolutional Time Interleaver (A/322 7.1.4) gates.

The CTI's de-interleaver index map is not an assumption: A/322 9.3.9.1 prints
the transmitter's own formula for the signalled ``L1D_plp_CTI_fec_block_start``,
which is the map evaluated at ``i = C``.  Solving that for ``C`` and requiring
``0 <= C < cells_per_FEC_Block`` therefore gates the reading on real air, with
the whole Table 9.24 ``Nrows`` menu run as controls that must fail.

RF30 (WIAV-CD, 569 MHz) signals PLP 0 Core as QPSK 6/15, Ninner 64800, CTI
depth 3 -> Nrows 1024, start_row 146, fec_block_start 863918.  Only Nrows 1024
solves to a ``C`` inside its 32400-cell FEC block, and that ``C`` is 11950.

Reference: ATSC A/322:2024-04, Sections 7.1.4/9.3.9.1, Tables 9.24/9.26.
"""

import os

import numpy as np
import pytest

from atsc3lib import cti, spec
from atsc3lib.l1_signaling import FEC_BCH_16K, PLPConfig
from atsc3lib.payload import decode_cti_plp

_DATA = os.path.join(os.path.dirname(__file__), 'data')
_PLP16_CELLS = os.path.join(_DATA, 'rf33_plp16_cells.npy')
_PLP16_BB = os.path.join(_DATA, 'rf33_plp16_bb.bin')

#: RF30 PLP 0 Core layer, read off an LDPC-decoded L1-Detail.
RF30_PLP0 = dict(ninner=64800, mod_bits=2, depth=3,
                 extended=False, start_row=146, fec_block_start=863918)


class TestIndexMap:
    def test_out_index_matches_spec_9_3_9_1(self):
        """The printed identity is exactly ``out_index`` at i = C."""
        nrows, start_row, C = 1024, 146, 11950
        assert cti.out_index(C, nrows, start_row) == \
            C + nrows * ((start_row + C) % nrows)

    @pytest.mark.parametrize('nrows,start', [(8, 0), (8, 5), (512, 191),
                                             (1024, 344), (1024, 1002)])
    def test_round_trip_identity(self, nrows, start):
        rng = np.random.default_rng(nrows * 131 + start)
        n = max(4 * nrows * nrows, 4096)
        src = rng.integers(0, 1 << 30, n).astype(np.int64)
        tx = cti.interleave(src, nrows, start)
        rx = cti.deinterleave(tx, nrows, start, n_out=n)
        assert rx.valid.mean() > 0.5
        assert np.array_equal(rx.cells[rx.valid], src[rx.valid])

    @pytest.mark.parametrize('nrows,start', [(512, 191), (1024, 344)])
    def test_structure_injective_delay(self, nrows, start):
        n = 3 * nrows * nrows
        i = np.arange(n, dtype=np.int64)
        q = cti.out_index(i, nrows, start)
        assert len(np.unique(q)) == n
        d = q - i
        assert np.all(d % nrows == 0)
        assert d.min() >= 0 and d.max() < nrows * nrows
        assert np.all(((start + q) % nrows) == d // nrows)

    @pytest.mark.parametrize('nrows,start', [(8, 3), (512, 191)])
    def test_in_index_inverts_out_index(self, nrows, start):
        """The receive map and the transmit map are the same equation."""
        q = np.arange(2 * nrows * nrows, dtype=np.int64)
        assert np.all(cti.out_index(cti.in_index(q, nrows, start),
                                    nrows, start) == q)


class TestTable924:
    @pytest.mark.parametrize('depth,rows', sorted(spec.CTI_NROWS.items()))
    def test_non_extended(self, depth, rows):
        assert spec.cti_nrows(depth, False) == rows

    @pytest.mark.parametrize('depth,rows',
                             sorted(spec.CTI_NROWS_EXTENDED.items()))
    def test_extended(self, depth, rows):
        assert spec.cti_nrows(depth, True) == rows

    def test_reserved_depth_rejected(self):
        with pytest.raises(ValueError):
            spec.cti_nrows(4)


class TestAirGateRf30:
    """A/322 9.3.9.1 solved for C: only the signalled Nrows lands in range."""

    #: Table 9.24 menu, non-extended and extended, as controls.
    MENU = sorted(set(list(spec.CTI_NROWS.values())
                      + list(spec.CTI_NROWS_EXTENDED.values())))

    def test_only_signalled_nrows_solves_inside_a_fec_block(self):
        p = RF30_PLP0
        cpf = p['ninner'] // p['mod_bits']
        signalled = spec.cti_nrows(p['depth'], p['extended'])
        hits = []
        for nrows in self.MENU:
            if p['start_row'] >= nrows:
                continue
            C = spec.cti_start_c(p['fec_block_start'], p['start_row'], nrows)
            if 0 <= C < cpf:
                hits.append(nrows)
        assert hits == [signalled] == [1024]

    def test_solved_c_is_the_known_value(self):
        p = RF30_PLP0
        nrows = spec.cti_nrows(p['depth'], p['extended'])
        assert spec.cti_start_c(p['fec_block_start'], p['start_row'],
                                nrows) == 11950


class TestLdmTables:
    @pytest.mark.parametrize('code,db', [(0, 0.0), (8, 4.0), (10, 5.0),
                                         (30, 25.0)])
    def test_injection_signalling_table_9_22(self, code, db):
        assert spec.LDM_INJECTION_DB[code] == db

    def test_power_ratios_sum_to_one(self):
        for db in (0.0, 4.0, 5.0, 25.0):
            p = spec.ldm_power(db)
            assert p.core_ratio + p.enhanced_ratio == pytest.approx(1.0)

    def test_rf30_injection_4db(self):
        """RF30 PLP 1 signals level 8 -> 4.0 dB (A/322 Table 9.22)."""
        p = spec.ldm_power(spec.LDM_INJECTION_DB[8])
        assert p.injection_db == 4.0
        assert p.alpha == pytest.approx(0.6309573)
        assert p.beta == pytest.approx(0.8457262)
        assert p.core_ratio == pytest.approx(0.7154, abs=1e-3)

    def test_table_6_16_alpha_beta_pinned(self):
        assert spec.ldm_power(5.0).alpha == pytest.approx(0.5623413)
        assert spec.ldm_power(5.0).beta == pytest.approx(0.8716346)


class TestParserCtiFields:
    """L1-Detail parse of the CTI fields (A/322 Table 9.8, TI mode 1)."""

    def _plp_bits(self, depth, start_row, fec_block_start):
        bits = []

        def put(value, width):
            for i in range(width - 1, -1, -1):
                bits.append((value >> i) & 1)

        put(3, 6)                    # L1D_plp_id
        put(0, 1)                    # lls
        put(0, 2)                    # layer 0 (Core)
        put(0, 24)                   # start
        put(1044305, 24)             # size
        put(0, 2)                    # scrambler
        put(1, 4)                    # fec_type 1 (64K LDPC) -> mod/cod present
        put(0, 4)                    # mod QPSK (0) -> extended bit present
        put(4, 4)                    # cod 6/15
        put(1, 2)                    # TI_mode 1 (CTI)
        put(fec_block_start, 22)     # L1D_plp_CTI_fec_block_start
        put(0, 1)                    # L1D_plp_type 0
        put(0, 1)                    # TI_extended_interleaving 0
        put(depth, 3)                # L1D_plp_CTI_depth
        put(start_row, 11)           # L1D_plp_CTI_start_row
        return np.array(bits, dtype=np.uint8)

    def _parse(self, bits):
        from atsc3lib.l1_signaling import _BitReader, _parse_plp
        return _parse_plp(_BitReader(bits), 0, 0, 0, 0)

    def test_cti_fields_round_trip(self):
        plp = self._parse(self._plp_bits(3, 146, 863918))
        assert plp.ti_mode == 1
        assert plp.cti_depth == 3
        assert plp.cti_start_row == 146
        assert plp.ti_fec_block_start == 863918
        assert plp.ti_extended_interleaving == 0
        assert spec.cti_nrows(plp.cti_depth, False) == 1024


@pytest.mark.skipif(not os.path.exists(_PLP16_CELLS),
                    reason="real-air QPSK FEC-block fixture not present")
class TestCtiEndToEnd:
    """Interleave a real decodable FEC block, then recover it through CTI.

    Reuses the RF33 PLP-16 QPSK 2/15 block (8100 cells, Ninner 16200): the TX
    side is a synthetic CTI whose ``C`` solves to 0, the RX side is the shipped
    :func:`decode_cti_plp`.  Controls run with the CTI bypassed and with a wrong
    ``start_row``; both must fail to converge.
    """

    NROWS = spec.CTI_NROWS[0]          # depth 0 -> 512
    START_ROW = 191

    def _plp(self, fec_block_start):
        return PLPConfig(
            plp_id=0, lls_flag=0, layer=0, start=0, size=8100,
            scrambler_type=0, fec_type=FEC_BCH_16K, modulation=0, code_rate=0,
            ti_mode=1, ti_fec_block_start=fec_block_start,
            hti_inter_subframe=None, hti_num_ti_blocks=None,
            hti_num_fec_blocks=None, hti_cell_interleaver=None,
            cti_depth=0, cti_start_row=self.START_ROW,
            ti_extended_interleaving=0, ldm_injection_level=None)

    def test_round_trip_recovers_the_fec_block(self):
        src = np.load(_PLP16_CELLS)
        tx = cti.interleave(src, self.NROWS, self.START_ROW)
        # extend so every pre-CTI cell of the first block is contiguous-valid
        stream = np.concatenate([tx, np.zeros(self.NROWS * self.NROWS,
                                              dtype=tx.dtype)])
        c = spec.cti_start_c(self.NROWS * self.START_ROW, self.START_ROW,
                             self.NROWS)
        assert c == 0
        r = decode_cti_plp(stream, self._plp(self.NROWS * self.START_ROW),
                           max_iterations=80, max_blocks=1)
        assert r.n_blocks == 1
        assert r.payload.n_converged == 1
        assert r.payload.fec_blocks[0].packet == open(_PLP16_BB, 'rb').read()

    def test_bypassing_the_cti_fails(self):
        src = np.load(_PLP16_CELLS)
        stream = np.concatenate([src, np.zeros(self.NROWS * self.NROWS,
                                               dtype=src.dtype)])
        r = decode_cti_plp(stream, self._plp(self.NROWS * self.START_ROW),
                           max_iterations=80, max_blocks=1)
        assert r.payload.n_converged == 0

    def test_wrong_start_row_fails(self):
        src = np.load(_PLP16_CELLS)
        tx = cti.interleave(src, self.NROWS, self.START_ROW)
        stream = np.concatenate([tx, np.zeros(self.NROWS * self.NROWS,
                                              dtype=tx.dtype)])
        plp = self._plp(self.NROWS * self.START_ROW)
        plp.cti_start_row = (self.START_ROW + 1) % self.NROWS
        r = decode_cti_plp(stream, plp, max_iterations=80, max_blocks=1)
        assert r.payload.n_converged == 0
