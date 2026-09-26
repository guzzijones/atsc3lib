"""Tests for the Preamble front-end (channel estimate, cell mask, decode)."""

import numpy as np
import pytest

from atsc3lib import spec
from atsc3lib.preamble import (
    carrier_shift, preamble_noc, common_continual_pilots, preamble_data_mask,
    preamble_pilot_indices, preamble_pilot_values, estimate_preamble_channel,
    preamble_data_cells, preamble_l1_cells,
)
from atsc3lib.pilot_reference import reference_sequence


# A/322 Table 7.2, first Preamble symbol, cred_coeff = 4.
TABLE_7_2_CRED4 = {
    (8192, 192): 6075, (8192, 384): 5667, (8192, 512): 5395,
    (8192, 768): 4851, (8192, 1024): 4307, (8192, 1536): 4851,
    (8192, 2048): 4307,
}


class TestGeometry:
    def test_preamble_noc_is_cred4(self):
        assert preamble_noc(8192) == 6529

    def test_carrier_shift_centres_carriers(self):
        # 6529 carriers centred in an 8192-point FFT start at bin (8192-6529)//2.
        assert carrier_shift(8192, 6529) == 832

    def test_common_continual_pilots_are_relative(self):
        cps = common_continual_pilots(6529)
        assert len(cps) == 45
        assert cps.min() >= 0 and cps.max() < 6529


class TestDataMask:
    @pytest.mark.parametrize('gi,expected', [
        (192, 6075), (384, 5667), (512, 5395), (768, 4851),
        (1024, 4307), (1536, 4851), (2048, 4307),
    ])
    def test_cell_count_matches_table_7_2(self, gi, expected):
        struct = {192: 2, 384: 7, 512: 12, 768: 17,
                  1024: 22, 1536: 27, 2048: 32}[gi]
        dx = spec.PREAMBLE_STRUCTURE[struct].dx
        mask = preamble_data_mask(preamble_noc(8192), dx)
        assert int(mask.sum()) == expected == TABLE_7_2_CRED4[(8192, gi)]

    def test_pilots_excluded(self):
        mask = preamble_data_mask(6529, 4)
        assert not np.any(mask[preamble_pilot_indices(6529, 4)])
        assert not np.any(mask[common_continual_pilots(6529)])

    @pytest.mark.parametrize('gi,row', [
        (192, [6432, 6342, 6253, 6164, 6075]),
        (384, [6000, 5916, 5833, 5750, 5667]),
        (512, [5712, 5632, 5553, 5474, 5395]),
        (768, [5136, 5064, 4993, 4922, 4851]),
        (1024, [4560, 4496, 4433, 4370, 4307]),
        (1536, [5136, 5064, 4993, 4922, 4851]),
        (2048, [4560, 4496, 4433, 4370, 4307]),
    ])
    def test_all_creds_match_table_7_2(self, gi, row):
        """Later Preamble symbols use L1B_preamble_reduced_carriers (0..4)."""
        struct = {192: 2, 384: 7, 512: 12, 768: 17,
                  1024: 22, 1536: 27, 2048: 32}[gi]
        dx = spec.PREAMBLE_STRUCTURE[struct].dx
        got = [int(preamble_data_mask(spec.noc(8192, c), dx).sum())
               for c in range(5)]
        assert got == row


class TestPilotValues:
    def test_pilot_values_are_bpsk(self):
        v = preamble_pilot_values(6529, 4, amplitude=1.23)
        assert set(np.round(np.abs(v), 6)).issubset({1.23})

    def test_pilot_value_is_amplitude_times_sign(self):
        v = preamble_pilot_values(100, 4, amplitude=1.0)
        r = reference_sequence(100)
        for n, k in enumerate(preamble_pilot_indices(100, 4)):
            assert v[n] == (1 - 2 * int(r[k]))


class TestChannelEstimate:
    def _synthetic_preamble(self, dx=4, amplitude=1.23, seed=0, channel=None):
        noc = 6529
        rng = np.random.default_rng(seed)
        k = np.arange(noc)
        if channel is None:
            channel = 1.0 + 0.3 * np.exp(2j * np.pi * k / 300)
        data = ((rng.integers(0, 2, noc) * 2 - 1)
                + 1j * (rng.integers(0, 2, noc) * 2 - 1)) / np.sqrt(2)
        x = data.copy()
        x[preamble_pilot_indices(noc, dx)] = preamble_pilot_values(
            noc, dx, amplitude)
        return x * channel, data, channel

    def test_recovers_channel(self):
        carriers, _, channel = self._synthetic_preamble()
        h = estimate_preamble_channel(carriers, dx=4, amplitude=1.23)
        corr = np.abs(np.vdot(h, channel)) / np.sqrt(
            np.vdot(h, h).real * np.vdot(channel, channel).real)
        assert corr > 0.99

    def test_data_cells_recover_qpsk(self):
        carriers, data, _ = self._synthetic_preamble()
        h = estimate_preamble_channel(carriers, dx=4, amplitude=1.23)
        cells = preamble_data_cells(carriers, h, dx=4)
        mask = preamble_data_mask(len(carriers), 4)
        truth = data[mask]
        corr = np.abs(np.vdot(cells, truth)) / np.sqrt(
            np.vdot(cells, cells).real * np.vdot(truth, truth).real)
        assert corr > 0.99
