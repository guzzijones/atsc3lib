"""Normal-frame FEC (Ninner = 64800) gate: A/322 Annex A.1/B.1.

The tables are machine-extracted from the A/322 PDF by
``tools/extract_ldpc64k.py`` and ``tools/extract_bicm.py``.  These tests gate
the banked tables structurally (the extraction's own arithmetic checks) and
gate the decoder end to end at every rate.
"""

import json
import os

import numpy as np
import pytest

from atsc3lib.ldpc_exact import (
    ATSC3LDPCExact, NINNER_NORMAL, RATE_DENOM, TYPE_A_PARAMS_64800,
    TYPE_B_QLDPC_64800, get_code_params,
)
from atsc3lib.group_interleaver import (
    GroupInterleaver, SUPPORTED_MODULATIONS,
)

_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                     'atsc3lib', 'data')
_LDPC = os.path.join(_DATA, 'ldpc_tables_N64800.json')
_BICM = os.path.join(_DATA, 'group_interleaver_64800.json')

RATES = list(range(2, 14))
NGROUP = NINNER_NORMAL // 360


class TestAnnexA1Tables:
    @pytest.fixture(scope='class')
    @classmethod
    def tables(cls):
        with open(_LDPC) as f:
            return {int(k): v for k, v in json.load(f).items()}

    def test_all_rates_present(self, tables):
        assert sorted(tables) == RATES

    @pytest.mark.parametrize('rate', RATES)
    def test_row_count_from_section_6(self, tables, rate):
        # G1: Type B rows == K/360; Type A rows == K/360 + Q1 (6.1.3.1).
        table = tables[rate]
        assert table['Kldpc'] == NINNER_NORMAL * rate // RATE_DENOM
        if rate in TYPE_A_PARAMS_64800:
            expect = table['Kldpc'] // 360 + TYPE_A_PARAMS_64800[rate].q1
        else:
            expect = table['Kldpc'] // 360
        assert len(table['rows']) == expect

    @pytest.mark.parametrize('rate', RATES)
    def test_addresses_in_parity_range(self, tables, rate):
        m = NINNER_NORMAL - tables[rate]['Kldpc']
        for row in tables[rate]['rows']:
            assert 0 <= min(row) and max(row) < m
            assert row == sorted(row) and len(set(row)) == len(row)


class TestAnnexB1Tables:
    @pytest.fixture(scope='class')
    @classmethod
    def tables(cls):
        with open(_BICM) as f:
            return json.load(f)

    def test_header(self, tables):
        assert tables['Ninner'] == NINNER_NORMAL
        assert tables['Ngroup'] == NGROUP

    @pytest.mark.parametrize('mod', SUPPORTED_MODULATIONS)
    def test_all_rates_are_permutations(self, tables, mod):
        for rate in RATES:
            perm = tables['tables'][mod][str(rate)]
            assert sorted(perm) == list(range(NGROUP))


class TestCodeParamsNormal:
    def test_type_a_normal_only_rates(self):
        assert 7 in TYPE_A_PARAMS_64800
        assert 7 not in TYPE_B_QLDPC_64800

    @pytest.mark.parametrize('rate', RATES)
    def test_geometry(self, rate):
        k, m, ctype = get_code_params(rate, n=NINNER_NORMAL)
        assert k == NINNER_NORMAL * rate // RATE_DENOM
        assert m == NINNER_NORMAL - k
        assert ctype == ('A' if rate in TYPE_A_PARAMS_64800 else 'B')


class TestEncoderAllRates:
    @pytest.mark.parametrize('rate', RATES)
    def test_valid_codeword(self, rate):
        codec = ATSC3LDPCExact(rate, n=NINNER_NORMAL)
        rng = np.random.default_rng(rate)
        info = rng.integers(0, 2, codec.K, dtype=np.uint8)
        assert codec.check_syndrome(codec.encode(info)) is True


class TestEndToEndAllModcods:
    @pytest.mark.parametrize('mod,rate,error_rate', [
        ('QPSK', 2, 0.02), ('QPSK', 11, 0.02),
        ('16QAM', 5, 0.02), ('16QAM', 11, 0.01),
        ('64QAM', 7, 0.01), ('64QAM', 11, 0.01),
        ('256QAM', 8, 0.01), ('256QAM', 11, 0.005),
    ])
    def test_interleaved_error_correction(self, mod, rate, error_rate):
        codec = ATSC3LDPCExact(rate, n=NINNER_NORMAL, max_iterations=50)
        gi = GroupInterleaver(rate, mod, n=NINNER_NORMAL)
        rng = np.random.default_rng(rate * 100 + codec.K)
        info = rng.integers(0, 2, codec.K, dtype=np.uint8)
        on_air = gi.interleave(codec.encode(info))
        received = on_air.copy()
        received[rng.random(codec.n) < error_rate] ^= 1
        llrs = np.where(received == 1, 4.0, -4.0)
        decoded, converged = codec.decode(gi.deinterleave_llrs(llrs))
        assert converged
        assert np.array_equal(decoded, info)
