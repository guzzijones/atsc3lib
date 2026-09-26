"""Cross-check the banked A/322 Annex A.1/B.1 tables against a public source.

The specification is ground truth; this test re-derives the same tables from an
independent transcription (``tools/crosscheck_web_tables.py``) and asserts
equality.  It needs network access, so it skips when the fetch fails rather
than making the suite flaky.
"""

import pytest

from tools.crosscheck_web_tables import (
    GROUP_FILE, LDPC_FILE, REF_COMMIT, REF_REPO, check_group, check_ldpc,
)
from tools.crosscheck_web_tables import _fetch, _RAW


@pytest.fixture(scope='module')
def sources():
    try:
        ldpc = _fetch(f'{_RAW}/{LDPC_FILE}', None, LDPC_FILE)
        group = _fetch(f'{_RAW}/{GROUP_FILE}', None, GROUP_FILE)
    except OSError as exc:
        pytest.skip(f'{REF_REPO}@{REF_COMMIT[:12]} unreachable: {exc}')
    return ldpc, group


def test_ldpc_tables_match_reference(sources):
    result = check_ldpc(sources[0])
    assert result.ok, result.mismatches


def test_group_tables_match_reference(sources):
    result = check_group(sources[1])
    assert result.ok, result.mismatches
