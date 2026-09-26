"""Fetch the A/322 pilot and data-cell tables from the pinned reference source.

WHY
---
Subframe demodulation needs, per FFT size and scattered-pilot pattern, the
additional continual pilots (A/322 Table D.1.4/D.1.5) and the data-cell counts
(Tables 7.3-7.6, Annex F).  Extracting these from the PDF is error-prone (the
SP*_4 groups print three rows and the naive read loses the first additional
CP).  The independent GNU Radio transmitter ``drmpeg/gr-atsc3`` carries the same
tables as plain C, pinned to a known commit; this tool fetches that source and
rewrites ``atsc3lib/data/pilot_tables.json``.

SOURCES (never committed)
-------------------------
* ``lib/params.h``            - NoC, available data cells (7.3/7.4), SBS total
                                (7.5/7.6) and SBS active (Annex F).
* ``lib/pilotgenerator_cc_impl.cc`` - common CP tables and the additional CP
                                positions (D.1.4/D.1.5), including the
                                cred_coeff-conditional ones.

THE GATE
--------
Nothing is trusted.  ``verify()`` rebuilds the pilot grid from first principles
- scattered pilots for every lattice phase, the edge pilots, the common CP
set derived from CP32, and the fetched additional CPs - and requires the
constant number of data carriers (A/322 8.1.4.1) to equal Tables 7.3/7.4 for
every FFT size, cred_coeff and allowed pattern.  A one-row-short D.1.4 reading
fails this for every _4 pattern, which is exactly the error it exists to catch.

Usage:
    python -m tools.fetch_pilot_tables                     # fetch + write data/
    python -m tools.fetch_pilot_tables --src DIR           # use pre-fetched .h/.cc
"""

import argparse
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Tuple

from atsc3lib import spec
from tools.spec_sources import REF_RAW, REF_REPO, REF_COMMIT

PARAMS_FILE = 'params.h'
PILOTGEN_FILE = 'pilotgenerator_cc_impl.cc'

FFT_SIZES = (8192, 16384, 32768)
CRED_VALUES = (0, 1, 2, 3, 4)
SPB_VALUES = (0, 1, 2, 3, 4)
PATTERNS = tuple(spec.SP_PATTERN_SIGNALING[k] for k in sorted(spec.SP_PATTERN_SIGNALING))

_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                     'atsc3lib', 'data')
DEFAULT_OUT = os.path.join(_DATA, 'pilot_tables.json')

#: C array name -> (fft, shape).  Pattern-major then cred (then boost).
_TABLES = {
    'data_cells_table_8K': (8192, 2), 'data_cells_table_16K': (16384, 2),
    'data_cells_table_32K': (32768, 2),
    'sbs_cells_table_8K': (8192, 2), 'sbs_cells_table_16K': (16384, 2),
    'sbs_cells_table_32K': (32768, 2),
    'sbs_data_cells_table_8K': (8192, 3), 'sbs_data_cells_table_16K': (16384, 3),
    'sbs_data_cells_table_32K': (32768, 3),
}


def _fetch(name: str, src_dir: str) -> str:
    if src_dir:
        with open(os.path.join(src_dir, name)) as f:
            return f.read()
    url = f'{REF_RAW}/{name}'
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode()


def _c_array(text: str, name: str) -> str:
    """The brace body of ``const int <name>[...] = { ... };``."""
    m = re.search(rf'\b{name}\b[^=]*=\s*\{{(.*?)\n\s*\}};', text, re.S)
    assert m, f'array {name} not found'
    return m.group(1)


def _ints(body: str) -> List[int]:
    return [int(x) for x in re.findall(r'-?\d+', body)]


@dataclass(frozen=True)
class PilotTables:
    noc: Dict[Tuple[int, int], int]
    avail_data: Dict[Tuple[int, int], Dict[str, int]]
    sbs_total: Dict[Tuple[int, int], Dict[str, int]]
    sbs_active: Dict[Tuple[int, int, int], Dict[str, int]]
    additional_cp: Dict[int, Dict[str, Dict[int, List[int]]]]
    common_cp: Dict[int, List[int]]


def _nested(body: str, depth: int) -> List[str]:
    """Split a C initialiser body into its top-level ``{...}`` groups."""
    groups, level, start = [], 0, None
    for i, ch in enumerate(body):
        if ch == '{':
            if level == 0:
                start = i + 1
            level += 1
        elif ch == '}':
            level -= 1
            if level == 0:
                groups.append(body[start:i])
    assert groups, 'no nested groups found'
    return groups


def parse_tables(params_h: str, pilotgen_cc: str) -> PilotTables:
    noc = {}
    carrier_rows = _nested(_c_array(params_h, 'carriers_table'), 1)
    for i, fft_size in enumerate(FFT_SIZES):
        for j, cred in enumerate(CRED_VALUES):
            noc[(fft_size, cred)] = _ints(carrier_rows[i])[j]

    avail_data = {(f, c): {} for f in FFT_SIZES for c in CRED_VALUES}
    sbs_total = {(f, c): {} for f in FFT_SIZES for c in CRED_VALUES}
    sbs_active = {(f, c, b): {} for f in FFT_SIZES
                  for c in CRED_VALUES for b in SPB_VALUES}

    for name, (fft, depth) in _TABLES.items():
        rows = _nested(_c_array(params_h, name), 1)
        assert len(rows) == len(PATTERNS), f'{name}: {len(rows)} rows'
        is_data = name.startswith('data_')
        for p, pat in enumerate(PATTERNS):
            if depth == 2:
                vals = _ints(rows[p])
                target = avail_data if is_data else sbs_total
                for c in CRED_VALUES:
                    target[(fft, c)][pat] = vals[c]
            else:
                boost_groups = _nested(rows[p], 1)
                for c, group in enumerate(boost_groups):
                    for b, val in enumerate(_ints(group)):
                        sbs_active[(fft, c, b)][pat] = val
    return PilotTables(noc, avail_data, sbs_total, sbs_active,
                       parse_additional_cp(pilotgen_cc),
                       parse_common_cp(pilotgen_cc))


def parse_common_cp(pilotgen_cc: str) -> Dict[int, List[int]]:
    out = {}
    for name, fft in (('continual_pilot_table_8K', 8192),
                      ('continual_pilot_table_16K', 16384),
                      ('continual_pilot_table_32K', 32768)):
        out[fft] = _ints(_c_array(pilotgen_cc, name))
    return out


def parse_additional_cp(pilotgen_cc: str) -> Dict[int, Dict[str, Dict[int, List[int]]]]:
    """Resolve the per-(pattern, cred) additional continual pilots.

    Reads only the DATA/SBS additional-CP switch of each FFT block.  Entries
    guarded by ``(cred_coeff & 1) == 0`` apply to even cred; a ``switch
    (cred_coeff)`` block supplies the SP32_4 per-cred sets.
    """
    blocks = _fft_blocks(pilotgen_cc)
    out = {}
    for fft, seg in blocks.items():
        data_seg = _data_section(seg)
        out[fft] = {}
        for pat in PATTERNS:
            per_cred = {c: [] for c in CRED_VALUES}
            body = _case_body(data_seg, f'PILOT_{pat}')
            if body is None:
                out[fft][pat] = per_cred
                continue
            _collect_creds(body, per_cred)
            out[fft][pat] = per_cred
    return out


def _fft_blocks(text: str) -> Dict[int, str]:
    marks = []
    for tok, fft in (('FFTSIZE_8K', 8192), ('FFTSIZE_16K', 16384),
                     ('FFTSIZE_32K', 32768)):
        i = text.find(f'case {tok}:')
        assert i != -1, f'{tok} not found'
        marks.append((i, fft))
    marks.sort()
    out = {}
    for k, (i, fft) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else len(text)
        out[fft] = text[i:end]
    return out


def _data_section(seg: str) -> str:
    """The portion after the last PREAMBLE-only block (the DATA/SBS switch)."""
    idx = seg.rfind('== SBS_SYMBOL) || (frame_symbols[symbol] == DATA_SYMBOL)')
    assert idx != -1, 'DATA/SBS section not found'
    return seg[idx:]


def _case_body(seg: str, label: str) -> str:
    start = seg.find(f'case {label}:')
    if start == -1:
        return None
    i = seg.find('\n', start) + 1
    # Body spans to the next `break;` at the case's brace level (case bodies
    # here have no nested braces except the cred guards; balance them).
    level, j = 0, i
    while j < len(seg):
        if seg[j] == '{':
            level += 1
        elif seg[j] == '}':
            level -= 1
        elif seg.startswith('break;', j) and level == 0:
            return seg[i:j]
        elif seg.startswith('case PILOT_', j) and level == 0:
            return seg[i:j]
        j += 1
    return seg[i:]


def _collect_creds(body: str, per_cred: Dict[int, List[int]]) -> None:
    """Apply no-guard, even-cred and per-cred additional CP assignments.

    Scans left to right so an ``if ((cred_coeff & 1) == 0)`` or ``switch
    (cred_coeff)`` guard changes which creds the *following* assignments
    apply to; the guarded block is consumed so its indices are not also
    counted unguarded.
    """
    pos = 0
    while pos < len(body):
        m = re.compile(
            r'(if\s*\(\(cred_coeff & 0x1\) == 0\))|(switch\s*\(cred_coeff\))|'
            r'data_carrier_map\[(\d+)\]\s*=\s*SCATTERED_CARRIER').search(body, pos)
        if not m:
            break
        if m.group(1):
            inner, end = _brace_block(body, m.end())
            for c in (0, 2, 4):
                per_cred[c] += _ints(inner)
            pos = end
        elif m.group(2):
            sw, end = _brace_block(body, m.end())
            for cm in re.finditer(
                    r'case CRED_(\d+):(.*?)(?=case CRED_|default:|$)', sw, re.S):
                per_cred[int(cm.group(1))] += _ints(cm.group(2))
            pos = end
        else:
            idx = int(m.group(3))
            for c in CRED_VALUES:
                per_cred[c].append(idx)
            pos = m.end()
    for c in per_cred:
        per_cred[c] = sorted(set(per_cred[c]))


def _brace_block(text: str, start: int) -> Tuple[str, int]:
    """The contents of the first ``{...}`` at/after ``start`` and its end index."""
    i = text.find('{', start)
    assert i != -1, 'opening brace not found'
    level, j = 0, i
    while j < len(text):
        if text[j] == '{':
            level += 1
        elif text[j] == '}':
            level -= 1
            if level == 0:
                return text[i + 1:j], j + 1
        j += 1
    raise AssertionError('unbalanced braces')


def _additional_relative(tables: PilotTables, fft: int, pattern: str,
                         cred: int) -> Tuple[int, ...]:
    vals = tables.additional_cp[fft][pattern][cred]
    noc = tables.noc[(fft, cred)]
    return tuple(v for v in vals if 0 <= v < noc)


def verify(tables: PilotTables) -> int:
    """Rebuild the pilot grid; require the constant-data-carrier identity (8.1.4.1).

    For every FFT size, cred_coeff and pattern, the number of data carriers must
    be the SAME for every scattered-pilot lattice phase (a phase is ``l mod dy``).
    The additional continual pilots exist precisely to make that true, so this
    pins Table D.1.4's reading: a one-row-short read fails for every ``_4``
    pattern.  It also independently confirms A/322 Table 8.3: the patterns whose
    data count is constant must be exactly the allowed ones for that FFT size.

    Returns the number of (FFT, cred, pattern) rows checked.
    """
    checked = 0
    for fft in FFT_SIZES:
        per_cred = []
        for cred in CRED_VALUES:
            noc = tables.noc[(fft, cred)]
            cp = _common_cp_relative(tables, fft, cred)
            constant = set()
            for pattern in PATTERNS:
                expect = tables.avail_data[(fft, cred)][pattern]
                dx, dy = spec.SP_DXDY[pattern]
                add = _additional_relative(tables, fft, pattern, cred)
                counts = {
                    noc - len({v for v in
                               (set(range(dx * phase, noc, dx * dy))
                                | {0, noc - 1} | set(cp) | set(add))
                               if 0 <= v < noc})
                    for phase in range(dy)}
                if len(counts) == 1:
                    # A constant count is only legal if it equals the table;
                    # otherwise the additional-CP extraction is wrong.
                    assert counts == {expect}, (
                        f'{fft}/{cred}/{pattern}: data {counts} != table '
                        f'{expect}')
                    constant.add(pattern)
                checked += 1
            per_cred.append(constant)
        # Allowed patterns are a function of (FFT, GI), not cred; the identity
        # must agree across cred, and must be exactly Table 8.3's union.
        assert all(c == per_cred[0] for c in per_cred), (
            f'{fft}: constant set varies with cred: {per_cred}')
        assert per_cred[0] == spec.allowed_patterns(fft), (
            f'{fft}: identity-constant patterns {sorted(per_cred[0])} != '
            f'Table 8.3 allowed {sorted(spec.allowed_patterns(fft))}')
    return checked


def _common_cp_relative(tables: PilotTables, fft: int, cred: int) -> List[int]:
    """Common CPs shifted into the relative carrier grid (A/322 8.1.4.1)."""
    noc = tables.noc[(fft, cred)]
    origin = (spec.NOC_MAX[fft] - noc) // 2
    return [c - origin for c in tables.common_cp[fft]
            if origin <= c < origin + noc]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=None,
                    help='directory with pre-fetched params.h / pilotgen .cc')
    ap.add_argument('--out', default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    params_h = _fetch(PARAMS_FILE, args.src)
    pilotgen = _fetch(PILOTGEN_FILE, args.src)
    tables = parse_tables(params_h, pilotgen)
    checked = verify(tables)
    print(f'pilot tables verified against the constant-data-carrier identity: '
          f'{checked} (FFT, cred, pattern) rows')
    payload = {
        'source': f'{REF_REPO}@{REF_COMMIT}',
        'noc': {f'{f},{c}': v for (f, c), v in tables.noc.items()},
        'avail_data': {f'{f},{c}': v for (f, c), v in tables.avail_data.items()},
        'sbs_total': {f'{f},{c}': v for (f, c), v in tables.sbs_total.items()},
        'sbs_active': {f'{f},{c},{b}': v for (f, c, b), v in tables.sbs_active.items()},
        'additional_cp': {str(f): {p: {str(c): v for c, v in cc.items()}
                                  for p, cc in pp.items()}
                          for f, pp in tables.additional_cp.items()},
        'common_cp': {str(f): v for f, v in tables.common_cp.items()},
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
