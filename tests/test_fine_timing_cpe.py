"""Front-end stages the RF30-class path needs: fine timing and CPE.

Two stages the oracle's core path applies and the RF33-class path never needed:

- **Fine timing** (A/322 7.2.5.1 / 8.1.3.1): the bootstrap anchors the frame to
  within one guard interval; the residual sampling instant is pinned by
  maximising the scattered-pilot coherence of one data symbol.  Gated
  synthetically (a known shift must be recovered exactly) and on real air
  (the RF33 frame fixture's pilot coherence must be high).
- **CPE** (decision-directed per-symbol common gain): each OFDM symbol's
  residual complex gain is fitted from the cells whose alphabet is known.
  Gated synthetically (a per-symbol phase must be removed to recover the
  FEC block) and with a control (a wrong/absent region must not improve it).

Reference: ATSC A/322:2024-04, Sections 7.2.5, 8.1.2/8.1.3.1.
"""

import json
import os

import numpy as np
import pytest

from atsc3lib import spec
from atsc3lib import nuc
from atsc3lib.payload import (
    build_cell_pool, cpe_correct, CpeSpec, scattered_pilot_coherence,
    scattered_pilots, subframe_fine_timing, _relative_carriers,
)
from atsc3lib.preamble import (
    preamble_symbol_spectrum, preamble_noc, pilot_coherence,
)
from atsc3lib.pilot_reference import reference_sequence

_DATA = os.path.join(os.path.dirname(__file__), 'data')
_FRAME = os.path.join(_DATA, 'rf33_frame_y.npy')
_FRAME_META = os.path.join(_DATA, 'rf33_frame_meta.json')


class TestScatteredPilotGeometry:
    def test_normal_symbol_lattice(self):
        # A/322 8.1.3.1: k mod (DX*DY) == DX*(l mod DY), plus edge pilots.
        p = scattered_pilots(6913, 4, 2, 1, False)
        assert p[0] == 0 and p[-1] == 6912
        core = p[(p != 0) & (p != 6912)]
        assert np.all(core % (4 * 2) == 4 * (1 % 2))

    def test_sbs_symbol_lattice(self):
        # A subframe boundary symbol uses DY = 1.
        p = scattered_pilots(6913, 4, 2, 0, True)
        core = p[(p != 0) & (p != 6912)]
        assert np.all(core % 4 == 0)


class TestFineTimingSynthetic:
    """A known whole-sample shift must be recovered exactly."""

    def _coherence(self, y, start, noc, dx, dy, l=1, sbs=False):
        car = _relative_carriers(y[start:start + 8192], 8192, noc)
        return scattered_pilot_coherence(car, noc, dx, dy, l, sbs)

    @pytest.mark.skipif(not os.path.exists(_FRAME),
                        reason="real-air frame fixture not present")
    def test_recovers_a_known_shift(self):
        y = np.load(_FRAME)
        meta = json.load(open(_FRAME_META))
        t0, fft, gi = meta['t0'], meta['fft'], meta['gi']
        noc, dx, dy = meta['noc'], meta['dx'], meta['dy']
        # Shift the whole stream by a known amount; the nominal t0 is now wrong.
        shift = 11
        ys = np.roll(y, shift)
        ft = subframe_fine_timing(ys, t0, fft, gi, noc, dx, dy,
                                  meta['n_data_symbols'],
                                  tuple(meta['sbs_symbols']), span=40)
        assert ft.offset == shift

    @pytest.mark.skipif(not os.path.exists(_FRAME),
                        reason="real-air frame fixture not present")
    def test_nominal_is_already_optimal(self):
        y = np.load(_FRAME)
        meta = json.load(open(_FRAME_META))
        ft = subframe_fine_timing(
            y, meta['t0'], meta['fft'], meta['gi'], meta['noc'],
            meta['dx'], meta['dy'], meta['n_data_symbols'],
            tuple(meta['sbs_symbols']), span=40)
        assert ft.offset == 0
        assert ft.coherence > 0.9


@pytest.mark.skipif(not os.path.exists(_FRAME),
                    reason="real-air frame fixture not present")
class TestPreambleCoherenceRealAir:
    def test_fixture_preamble_coherence_is_high(self):
        y = np.load(_FRAME)
        meta = json.load(open(_FRAME_META))
        pre = spec.PREAMBLE_STRUCTURE[meta['preamble_structure']]
        t = meta['t0'] + pre.gi
        car = preamble_symbol_spectrum(y[t:t + pre.fft].astype(np.complex128),
                                       pre.fft, noc=preamble_noc(pre.fft))
        amp = spec.PREAMBLE_PILOT_AMPLITUDE[(pre.fft, pre.gi)]
        assert pilot_coherence(car, pre.dx, amp) > 0.9


class TestCpeSynthetic:
    """A per-symbol complex gain on ideal cells must be removed."""

    @pytest.mark.parametrize('phase', [0.0, 0.35, -0.6])
    def test_removes_a_per_symbol_phase(self, phase):
        pts = nuc.points(2, 6)          # QPSK
        rng = np.random.default_rng(7)
        n = 5999
        idx = rng.integers(0, len(pts), n)
        cells = pts[idx].astype(np.complex128)
        symbol_of = np.zeros(n, dtype=int)
        symbol_of[n // 2:] = 1
        cells[symbol_of == 1] *= np.exp(1j * phase)
        out = cpe_correct(cells, symbol_of, pts,
                          region=np.ones(n, dtype=bool))
        # every symbol must land on the constellation again
        for s in (0, 1):
            z = out[symbol_of == s]
            err = np.abs(z[:, None] - pts[None, :]).min(1)
            assert np.mean(err) < 1e-6

    def test_wrong_alphabet_does_not_improve(self):
        """CPE cannot rescue a wrong constellation hypothesis."""
        pts = nuc.points(2, 6)
        rng = np.random.default_rng(3)
        n = 5999
        cells = pts[rng.integers(0, len(pts), n)].astype(np.complex128)
        symbol_of = np.zeros(n, dtype=int)
        cells *= np.exp(1j * 0.4)
        wrong = np.array([1 + 0j, -1 + 0j, 0 + 1j, 0 - 1j])
        before = np.mean(np.abs(cells[:, None] - pts[None, :]).min(1))
        out = cpe_correct(cells, symbol_of, wrong,
                          region=np.ones(n, dtype=bool))
        after = np.mean(np.abs(out[:, None] - pts[None, :]).min(1))
        assert after >= before

    def test_region_restricts_the_estimate(self):
        """Cells outside the region must not steer the fitted gain."""
        pts = nuc.points(2, 6)
        rng = np.random.default_rng(11)
        n = 400
        cells = pts[rng.integers(0, len(pts), n)].astype(np.complex128)
        symbol_of = np.zeros(n, dtype=int)
        cells *= np.exp(1j * 0.5)
        # region covers only the first half; the second half is corrupt.
        cells[200:] += 0.9 * np.exp(1j * 1.1)
        region = np.zeros(n, dtype=bool)
        region[:200] = True
        out = cpe_correct(cells, symbol_of, pts, region=region,
                          iterations=5)
        err = np.abs(out[:200, None] - pts[None, :]).min(1)
        assert np.mean(err) < 1e-6


class TestDummyValues:
    def test_matches_the_baseband_scrambler(self):
        from atsc3lib.signaling_fec import randomizer_bits
        from atsc3lib.payload import dummy_cell_values
        n, start = 1000, 600
        dv = dummy_cell_values(n, start)
        assert np.all(dv[:start] == 0)
        bits = randomizer_bits(n)[start:].astype(np.int8)
        assert np.array_equal(dv[start:], 1.0 - 2.0 * bits)


@pytest.mark.skipif(not os.path.exists(_FRAME),
                    reason="real-air frame fixture not present")
class TestRealAirIntegration:
    """The two stages must not regress the decodable RF33 PLP-16 block."""

    def _pool(self, fine, cpe):
        y = np.load(_FRAME)
        m = json.load(open(_FRAME_META))
        p16 = m['plp16']
        spec_obj = None
        if cpe:
            spec_obj = CpeSpec(alphabet=nuc.points(2, 2),
                               region_start=p16['start'],
                               region_stop=p16['start'] + p16['size'],
                               dummy_start=p16['start'] + p16['size'])
        return build_cell_pool(
            y, m['t0'], m['fft'], m['gi'], m['noc'], m['dx'], m['dy'],
            m['n_data_symbols'], sbs_symbols=tuple(m['sbs_symbols']),
            preamble_structure=m['preamble_structure'],
            fine_timing_enabled=fine, cpe=spec_obj), m

    def test_nominal_and_staged_decode_identically(self):
        from atsc3lib.payload import QPSKPlpChain
        nominal, m = self._pool(False, False)
        staged, _ = self._pool(True, True)
        p16 = m['plp16']
        sl = slice(p16['start'], p16['start'] + p16['size'])
        r0 = QPSKPlpChain().decode_cells(nominal.cells[sl])
        r1 = QPSKPlpChain().decode_cells(staged.cells[sl])
        assert r0.ok and r1.ok
        assert r0.packet == r1.packet

