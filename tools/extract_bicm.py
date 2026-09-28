"""Extract the A/322 Annex B.1 group-wise interleaver tables (Ninner = 64800).

WHY
---
The A/322 6.2 bit interleaver permutes 360-bit groups per (modulation, code
rate); the permutation is printed in Annex B.1 (Tables B.1.1-B.1.6).  This
tool extracts those permutations once, gates them, and writes
``atsc3lib/group_tables.py`` (which also holds the inlined short-frame tables).

EXTRACTION DISCIPLINE
---------------------
* **PyMuPDF.**  Each table has a printed identity header row (0, 1, ..., 179)
  followed by one Ngroup-wide permutation per code rate, wrapped across visual
  lines.  PyMuPDF keeps the line boxes so the rows reassemble in order; the
  footer page number (below ``FOOTER_Y``) is excluded.
* **Gates that can fail.**  Every table carries a free per-table checksum --
  the identity header row -- and every extracted permutation must be a
  permutation of 0..Ngroup-1.

GATES
-----
G1  the printed identity header row reads exactly 0..Ngroup-1 (per table)
G2  every permutation is a bijection of 0..Ngroup-1
G3  Ngroup values per tabulated rate

Usage:
    python tools/extract_bicm.py [pdf] [out]
The pdf argument is optional: when omitted the official A/322 PDF is fetched
to a local cache (see ``tools/spec_sources.py``) and the inlined
``atsc3lib/group_tables.py`` is overwritten.
Requires PyMuPDF (`pip install pymupdf`).
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import pymupdf

from atsc3lib.ldpc_exact import NINNER_NORMAL, RATE_MIN, RATE_MAX, GROUP_SIZE
from atsc3lib.nuc import QPSK, QAM16, QAM64, QAM256, QAM1024, QAM4096
from tools.pdf_layout import integer_lines, line_values
from tools.spec_sources import A322, pdf_path

#: Annex B.1 table page ranges (0-based PDF page indices), per modulation.
TABLE_PAGES: Dict[str, range] = {
    QPSK: range(160, 163),
    QAM16: range(163, 165),
    QAM64: range(165, 167),
    QAM256: range(167, 169),
    QAM1024: range(169, 171),
    QAM4096: range(171, 173),
}

#: A/322 6.2.2: bit groups are 360 bits; Ngroup = Ninner / 360.
NGROUP = NINNER_NORMAL // GROUP_SIZE
RATES = list(range(RATE_MIN, RATE_MAX + 1))

#: The printed identity header row of a group-interleaver table (a per-table
#: checksum of the column ordering).
IDENTITY = list(range(NGROUP))

#: Default output path: the inlined table module in the package.
DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'atsc3lib', 'group_tables.py')


@dataclass(frozen=True)
class AnnexBlock:
    """One Annex B.1 block: an identity header plus a permutation per rate."""
    tables: Dict[int, List[int]]


def _page_values(page) -> List[int]:
    return [v for line in integer_lines(page) for v in line_values(line)]


def parse_modulation(doc, pages: range) -> Dict[int, List[int]]:
    """Return {rate: permutation} for one modulation's Annex B.1 blocks."""
    values = []
    for page_no in pages:
        page = _page_values(doc[page_no])
        if page and page[:NGROUP] == IDENTITY:
            page = page[NGROUP:]
        values.extend(page)
    assert len(values) == len(RATES) * NGROUP, (
        f"expected {len(RATES) * NGROUP} values, got {len(values)}")
    return {rate: values[(rate - RATE_MIN) * NGROUP:
                         (rate - RATE_MIN + 1) * NGROUP]
            for rate in RATES}


def gate(block: AnnexBlock) -> None:
    for rate, perm in block.tables.items():
        assert len(perm) == NGROUP, (
            f"rate {rate}: {len(perm)} values, expected {NGROUP}")
        assert sorted(perm) == IDENTITY, (
            f"rate {rate}: not a permutation of 0..{NGROUP - 1}")


def _write_group_module(path, short_tables, normal_tables, ninner, ngroup):
    header = (
        '"""A/322 Annex B group-wise bit-interleaver tables (short and normal).\n'
        '\n'
        'Extracted from ATSC A/322:2024-04 Annex B.  Keyed ``[modulation][rate]``\n'
        'with a tuple of Ngroup = Ninner/360 group indices (a permutation of\n'
        '0..Ngroup-1).  Modulation keys are the numeric Annex B.2 order\n'
        '``1``=QPSK, ``2``=16QAM, ``3``=64QAM, ``4``=256QAM for short frames and\n'
        'the named strings for normal frames.\n'
        '"""\n'
        '\n'
    )
    out = [header, 'GROUP_TABLES_16200 = {\n']
    for mod in sorted(short_tables, key=int):
        d = short_tables[mod]
        out.append(f'    {mod}: {{\n')
        for rate in sorted(d, key=int):
            out.append(f'        {rate}: ({", ".join(str(x) for x in d[rate])},),\n')
        out.append('    },\n')
    out.append('}\n\nGROUP_TABLES_64800 = {\n')
    for mod in normal_tables:
        d = normal_tables[mod]
        out.append(f'    {mod!r}: {{\n')
        for rate in sorted(d, key=int):
            out.append(f'        {rate}: ({", ".join(str(x) for x in d[rate])},),\n')
        out.append('    },\n')
    out.append('}\n')
    with open(path, 'w') as f:
        f.write(''.join(out))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default=None,
                    help="A/322 PDF (default: fetch the official copy)")
    ap.add_argument("out", nargs="?", default=DEFAULT_OUT,
                    help="output Python module (default: the inlined table)")
    args = ap.parse_args(argv)

    doc = pymupdf.open(pdf_path(A322, args.pdf))
    out = {}
    for mod, pages in TABLE_PAGES.items():
        block = AnnexBlock(parse_modulation(doc, pages))
        gate(block)
        out[mod] = {rate: tuple(block.tables[rate]) for rate in RATES}
        print(f"{mod}: {len(RATES)} rates x {NGROUP} groups, gated")
    doc.close()

    from atsc3lib.group_tables import GROUP_TABLES_16200
    _write_group_module(args.out, GROUP_TABLES_16200, out, NINNER_NORMAL, NGROUP)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
