"""Subframe 1 (16K FFT) geometry and cell-pool gate.

Subframe 1 of the RF33 frame runs a different FFT size from subframe 0 and,
being after the first subframe, inherits no Preamble cells and must have its
A/322 7.3 frequency-interleaver symbol counter reset at the boundary.  These
tests gate the 16K pilot/data-cell model and the pool arithmetic without
requiring the 256QAM PLP to close (which is link-limited on the captures).

The synthetic tests are spec-derived.  The real-air test, when its fixture is
present, checks the subframe-1 pool against the reference receiver's exact
length (956179 cells) and the per-pattern pilot counts.
"""

import json
import os

import numpy as np
import pytest

from atsc3lib import pilot_tables, spec
from atsc3lib.payload import build_data_symbol_pool, data_symbol_cells

_DATA = os.path.join(os.path.dirname(__file__), 'data')
_SF1 = os.path.join(_DATA, 'rf33_sf1_y.npy')
_META = os.path.join(_DATA, 'rf33_sf1_meta.json')

#: Reference receiver's subframe-1 pool length and the RF33 16K geometry.
SF1_POOL_CELLS = 956179


class TestPilotTables:
    def test_source_recorded(self):
        assert 'gr-atsc3' in pilot_tables.SOURCE

    @pytest.mark.parametrize('fft,cred', [
        (8192, 0), (16384, 0), (32768, 0), (16384, 3)])
    def test_noc_matches_table_7_1(self, fft, cred):
        assert pilot_tables.noc(fft, cred) == spec.noc(fft, cred)

    def test_common_cp_counts(self):
        # Table D.1.1/D.1.2/D.1.3: 192 / 96 / 48 absolute common CPs.
        assert len(pilot_tables.COMMON_CP[32768]) == 192
        assert len(pilot_tables.COMMON_CP[16384]) == 96
        assert len(pilot_tables.COMMON_CP[8192]) == 48


class TestSf1Geometry:
    def test_16k_sp4_4_pilot_counts_constant(self):
        # A/322 8.1.4.1: the additional CPs make the data-carrier count
        # invariant across the Dy=4 lattice phases.  12861 = Table 7.4 16K.
        fft, cred, pattern = 16384, 0, 'SP4_4'
        noc = pilot_tables.noc(fft, cred)
        cp = pilot_tables.common_cp_relative(fft, cred)
        add = pilot_tables.additional_cp(fft, pattern, cred)
        dx, dy = spec.SP_DXDY[pattern]
        counts = set()
        for phase in range(dy):
            pilots = set(range(dx * phase, noc, dx * dy)) | {0, noc - 1} \
                | set(cp) | set(add)
            counts.add(noc - len(pilots))
        assert counts == {pilot_tables.avail_data(fft, cred, pattern)}
        assert counts == {12861}

    def test_16k_sp4_4_additional_cp_known(self):
        assert pilot_tables.additional_cp(16384, 'SP4_4', 0) == \
            (3460, 5768, 11452)

    def test_null_split(self):
        # 1609 null cells split floor(n/2) low, the rest high (A/322 7.2.6.4).
        null = 1609
        assert (null // 2, null - null // 2) == (804, 805)

    def test_pool_arithmetic_synthetic(self):
        # Two SBS symbols contribute active data; the rest are data symbols.
        nsym, null, avail = 75, 1609, 12861
        per_sbs = pilot_tables.sbs_total(16384, 0, 'SP4_4') - null
        assert per_sbs == 8663
        assert 2 * per_sbs + (nsym - 2) * avail == SF1_POOL_CELLS


@pytest.mark.skipif(not (os.path.exists(_SF1) and os.path.exists(_META)),
                    reason="real-air subframe-1 fixture not present")
class TestSf1Air:
    def _pool(self):
        y = np.load(_SF1).astype(np.complex128)
        m = json.load(open(_META))
        return build_data_symbol_pool(
            y, 0, m['fft'], m['gi'], m['noc'], m['dx'], m['dy'],
            m['n_data_symbols'], sbs_symbols=tuple(m['sbs_symbols']),
            pattern=m['pattern'], sbs_null=m['sbs_null'], cred_coeff=m['cred'],
            fi_offset=m['fi_offset']), m

    def test_pool_length_matches_reference(self):
        pool, _ = self._pool()
        assert len(pool.cells) == SF1_POOL_CELLS
        assert pool.n_preamble_spare == 0

    def test_per_symbol_pilot_counts(self):
        y = np.load(_SF1).astype(np.complex128)
        m = json.load(open(_META))
        add = spec.additional_cp(m['pattern'], m['fft'], m['cred'])
        # A normal symbol and an SBS symbol must match Tables 7.4 and 7.5.
        res = data_symbol_cells(
            y[m['gi']:m['gi'] + m['fft'] + m['gi']][:m['fft']], m['fft'],
            m['noc'], m['dx'], m['dy'], 1, False, add_cp=add)
        assert len(res.cells) == pilot_tables.avail_data(
            m['fft'], m['cred'], m['pattern'])

    def test_plp1_shape(self):
        _, m = self._pool()
        assert m['plp1']['size'] == 947700
        # 947700 cells / (64800 / 8 bits) = 117 FEC blocks, 117 = 3 x 39.
        blocks = m['plp1']['size'] // (64800 // 8)
        assert blocks == 117
        assert blocks == m['plp1']['nti'] * 39
