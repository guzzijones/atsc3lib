"""Network-gated check that the fetch tool reproduces the banked pilot tables.

Fetches the pinned gr-atsc3 sources and asserts the parsed tables equal the
banked ``pilot_tables.json`` element-for-element.  Skips cleanly when offline.
"""

import pytest

from atsc3lib import pilot_tables
from tools import fetch_pilot_tables as F


@pytest.fixture(scope='module')
def fetched():
    try:
        params_h = F._fetch(F.PARAMS_FILE, None)
        pilotgen = F._fetch(F.PILOTGEN_FILE, None)
    except OSError as exc:
        pytest.skip(f'{F.REF_REPO}@{F.REF_COMMIT[:12]} unreachable: {exc}')
    return F.parse_tables(params_h, pilotgen)


def test_fetched_tables_match_banked(fetched):
    assert fetched.noc == pilot_tables.NOC
    assert fetched.avail_data == pilot_tables.AVAIL_DATA
    assert fetched.sbs_total == pilot_tables.SBS_TOTAL
    assert fetched.sbs_active == pilot_tables.SBS_ACTIVE
    assert fetched.additional_cp == pilot_tables.ADDITIONAL_CP
    assert fetched.common_cp == pilot_tables.COMMON_CP


def test_identity_gate_passes(fetched):
    checked = F.verify(fetched)
    assert checked == len(F.FFT_SIZES) * len(F.CRED_VALUES) * len(F.PATTERNS)
