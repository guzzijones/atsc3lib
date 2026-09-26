"""Cross-check the banked A/322 Annex A/B tables against an independent source.

WHY
---
The banked JSON under ``atsc3lib/data/`` is extracted from the A/322 PDF by
``extract_ldpc64k.py`` / ``extract_bicm.py``; the specification is the ground
truth.  A mistake in the PDF column split would be invisible to the extractor's
own gates, so this tool re-derives the same tables from an independent public
transcription and asserts element-for-element equality.

The reference is ``drmpeg/gr-atsc3`` (GPL-3.0), a GNU Radio ATSC 3.0
transmitter, pinned to a known commit.  Only the numeric standard tables are
read from it; the numbers are facts of A/322 and no code is copied.  The file
is fetched at run time and never committed.

Usage:
    python tools/crosscheck_web_tables.py            # fetch + compare
    python tools/crosscheck_web_tables.py --src DIR  # use pre-fetched files
Requires network access (or ``--src``).
"""

import argparse
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Dict, List

from atsc3lib.ldpc_exact import (
    NINNER_NORMAL, RATE_MIN, RATE_MAX, GROUP_SIZE,
)
from atsc3lib.nuc import QPSK, QAM16, QAM64, QAM256, QAM1024, QAM4096

#: Pinned upstream revision of the reference transcription.
REF_COMMIT = '000b86a325e2506eb063a25cd182d79c4b57acdd'
REF_REPO = 'drmpeg/gr-atsc3'
_RAW = f'https://raw.githubusercontent.com/{REF_REPO}/{REF_COMMIT}/lib'
#: Source files holding the LDPC and group-interleaver tables.
LDPC_FILE = 'ldpc_bb_impl.cc'
GROUP_FILE = 'interleaver_bb_impl.cc'

RATES = list(range(RATE_MIN, RATE_MAX + 1))
NGROUP = NINNER_NORMAL // GROUP_SIZE

#: Modulation name -> upstream C array suffix.
_MOD_SUFFIX = {QPSK: 'QPSK', QAM16: '16QAM', QAM64: '64QAM',
               QAM256: '256QAM', QAM1024: '1024QAM', QAM4096: '4096QAM'}

_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                     'atsc3lib', 'data')


@dataclass(frozen=True)
class CheckResult:
    """One table family's comparison outcome."""
    name: str
    compared: int
    mismatches: List[str]

    @property
    def ok(self) -> bool:
        return not self.mismatches


def _fetch(url: str, src_dir: str, name: str) -> str:
    if src_dir:
        with open(os.path.join(src_dir, name)) as f:
            return f.read()
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode()


def _c_int_rows(block: str) -> List[List[int]]:
    rows = []
    for line in block.strip().splitlines():
        nums = [int(x) for x in re.findall(r'-?\d+', line)]
        rows.append(nums[1:1 + nums[0]])
    return rows


def _c_int_list(block: str) -> List[int]:
    return [int(x) for x in re.findall(r'-?\d+', block)]


def check_ldpc(text: str) -> CheckResult:
    banked = json.load(open(os.path.join(_DATA, 'ldpc_tables_N64800.json')))
    mismatches = []
    for rate in RATES:
        m = re.search(
            rf'ldpc_tab_{rate}_15N\[\d+\]\[\d+\] = \{{(.*?)\n    \}};',
            text, re.S)
        if not m:
            mismatches.append(f'rate {rate}: array not found')
            continue
        if _c_int_rows(m.group(1)) != banked[str(rate)]['rows']:
            mismatches.append(f'rate {rate}: rows differ')
    return CheckResult('Annex A.1 LDPC', len(RATES), mismatches)


def check_group(text: str) -> CheckResult:
    banked = json.load(open(
        os.path.join(_DATA, 'group_interleaver_64800.json')))['tables']
    mismatches = []
    compared = 0
    for mod, suffix in _MOD_SUFFIX.items():
        for rate in RATES:
            compared += 1
            m = re.search(
                rf'group_tab_{rate}_15N_{suffix}\[180\] = \{{(.*?)\}};',
                text, re.S)
            if not m:
                mismatches.append(f'{mod} rate {rate}: array not found')
                continue
            if _c_int_list(m.group(1)) != banked[mod][str(rate)]:
                mismatches.append(f'{mod} rate {rate}: permutation differs')
    return CheckResult('Annex B.1 group interleaver', compared, mismatches)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=None,
                    help='directory with pre-fetched .cc files')
    args = ap.parse_args(argv)

    ldpc = check_ldpc(_fetch(f'{_RAW}/{LDPC_FILE}', args.src, LDPC_FILE))
    group = check_group(_fetch(f'{_RAW}/{GROUP_FILE}', args.src, GROUP_FILE))

    failed = False
    for result in (ldpc, group):
        status = 'OK' if result.ok else 'FAIL'
        print(f'{result.name}: {result.compared} tables, {status}')
        for msg in result.mismatches:
            print(f'  {msg}')
            failed = True
    if failed:
        raise SystemExit(1)
    print(f'banked tables match {REF_REPO}@{REF_COMMIT[:12]}')


if __name__ == '__main__':
    main()
