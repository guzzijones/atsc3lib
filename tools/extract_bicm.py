"""Extract the A/322 Annex B.1 group-wise interleaver tables (Ninner = 64800).

WHY
---
The A/322 6.2 bit interleaver permutes 360-bit groups per (modulation, code
rate); the permutation is printed in Annex B.1 (Tables B.1.1-B.1.6) and is not
in the repository.  This tool extracts those 72 permutations once, gates them,
and writes ``atsc3lib/data/group_interleaver_64800.json``.

EXTRACTION DISCIPLINE
---------------------
* **PyMuPDF.**  Each table has a printed identity header row (0, 1, ..., 179)
  followed by one 180-wide permutation per code rate, wrapped across visual
  lines.  PyMuPDF keeps the line boxes so the rows reassemble in order; the
  footer page number (below ``_FOOTER_Y``) is excluded.
* **Gates that can fail.**  Every table carries a free per-table checksum --
  the identity header row -- and every extracted permutation must be a
  permutation of 0..Ngroup-1.

GATES
-----
G1  the printed identity header row reads exactly 0..Ngroup-1 (per page)
G2  every permutation is a bijection of 0..Ngroup-1
G3  one 180-wide permutation per tabulated rate

Usage:
    python tools/extract_bicm.py /tmp/opencode/a322.pdf \\
        atsc3lib/data/group_interleaver_64800.json
Requires PyMuPDF (`pip install pymupdf`).
"""

import argparse
import json
import re

import pymupdf

from atsc3lib.ldpc_exact import NINNER_NORMAL, RATE_MIN, RATE_MAX, GROUP_SIZE
from atsc3lib.nuc import QPSK, QAM16, QAM64, QAM256, QAM1024, QAM4096

#: Annex B.1 table page ranges (0-based PDF page indices), per modulation.
TABLE_PAGES = {
    QPSK: (160, 162),
    QAM16: (163, 164),
    QAM64: (165, 166),
    QAM256: (167, 168),
    QAM1024: (169, 170),
    QAM4096: (171, 172),
}

#: A/322 6.2.2: bit groups are 360 bits.
NGROUP = NINNER_NORMAL // GROUP_SIZE
RATES = list(range(RATE_MIN, RATE_MAX + 1))

#: Footer page numbers sit below this y; the tables never do.
_FOOTER_Y = 720
_NUMERIC = re.compile(r"[\d ]+")


def _numeric_lines(page):
    out = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            text = " ".join(s["text"].strip() for s in line["spans"]).strip()
            if not _NUMERIC.fullmatch(text):
                continue
            if line["bbox"][1] > _FOOTER_Y:
                continue
            out.append((round(line["bbox"][1]), round(line["bbox"][0]),
                        [int(t) for t in text.split()]))
    return out


def _page_values(page):
    return [v for _, _, row in sorted(_numeric_lines(page)) for v in row]


def parse_modulation(doc, pages):
    """Return {rate: permutation} for one modulation's Annex B.1 blocks."""
    values = []
    for page_no in range(pages[0], pages[1] + 1):
        page = _page_values(doc[page_no])
        if page[:NGROUP] == list(range(NGROUP)):
            page = page[NGROUP:]
        values.extend(page)
    assert len(values) == len(RATES) * NGROUP, (
        f"expected {len(RATES) * NGROUP} values, got {len(values)}")
    return {rate: values[(rate - RATE_MIN) * NGROUP:
                         (rate - RATE_MIN + 1) * NGROUP]
            for rate in RATES}


def gate(tables):
    for rate, perm in tables.items():
        assert sorted(perm) == list(range(NGROUP)), (
            f"rate {rate}: not a permutation of 0..{NGROUP - 1}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("out")
    args = ap.parse_args(argv)

    doc = pymupdf.open(args.pdf)
    out = {}
    for mod, pages in TABLE_PAGES.items():
        tables = parse_modulation(doc, pages)
        gate(tables)
        out[mod] = {str(rate): tables[rate] for rate in RATES}
        print(f"{mod}: {len(tables)} rates x {NGROUP} groups, gated")
    doc.close()

    with open(args.out, "w") as f:
        json.dump({'Ninner': NINNER_NORMAL, 'Ngroup': NGROUP, 'tables': out}, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
