"""Extract the A/322 Annex A.1 LDPC parity-check address tables (Ninner = 64800).

WHY
---
The receiver's short-frame chain (Ninner = 16200) is validated; the normal
frame (Ninner = 64800) is signalled by PLP-1 and by most other stations.  Its
code matrices live in A/322 Annex A.1 and are not in the repository, so they
are extracted here once, gated, and written to
``atsc3lib/data/ldpc_tables_N64800.json``.  Section 6.1.3's coding parameters
(Table 6.5 Type A sizes, Table 6.7 Type B Qldpc) come from the library's
``TYPE_A_PARAMS_64800`` / ``TYPE_B_QLDPC_64800``.

EXTRACTION DISCIPLINE
---------------------
* **PyMuPDF, not pdftotext.**  The Annex A.1 pages are two-column.  pdftotext
  and pdfplumber interleave the columns and corrupt the row boundaries; PyMuPDF
  keeps line bounding boxes, so the tables are read column-major exactly.
* **Gates that can fail.**  Every table must satisfy the closed arithmetic of
  Section 6.1.3, and every address must land in the parity range.  The
  row-interleaved reading (control C1) is asserted to fail the gates on every
  two-column page, proving the column distinction is load-bearing.

GATES
-----
G1  row count == Kldpc/360 (Type B) or Kldpc/360 + Q1 (Type A, 6.1.3.1 steps
    (vii)-(viii) feed the M1 parity back in)
G2  every address in [0, Ninner - Kinner)
G3  every row strictly increasing, no duplicate inside a row
G4  the 360-degree expansion touches every parity accumulator at least once
C1  the printed two-column tables list row weights in non-increasing order;
    a row-interleaved read breaks that order, so C1 proves the column split
    is load-bearing (G1-G4 alone cannot see row order)

Usage:
    python tools/extract_ldpc64k.py [pdf] [out]
The pdf argument is optional: when omitted the official A/322 PDF is fetched
to a local cache (see ``tools/spec_sources.py``) and the banked
``atsc3lib/data/ldpc_tables_N64800.json`` is overwritten.
Requires PyMuPDF (`pip install pymupdf`).
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List

import pymupdf

from atsc3lib.ldpc_exact import (
    NINNER_NORMAL, RATE_DENOM, RATE_MIN, RATE_MAX, GROUP_SIZE,
    FEC_TYPE_A, FEC_TYPE_B, TYPE_A_PARAMS_64800, TYPE_B_QLDPC_64800,
)
from tools.pdf_layout import (
    CAPTION_GAP, COLUMN_GAP, PdfLine, integer_lines, line_values, page_lines,
)
from tools.spec_sources import A322, pdf_path

#: Table caption form: "Table A.1.<n> Rate = <r>/<denom>".
_RATE_CAPTION = re.compile(
    rf"Table A\.1\.\d+\s+Rate\s*=\s*(\d+)\s*/\s*{RATE_DENOM}")
_CAPTION_PREFIX = "Table A.1."

#: Default output path: the banked normal-frame table in the package.
DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'atsc3lib', 'data', 'ldpc_tables_N64800.json')


@dataclass(frozen=True)
class CodeParams:
    """Coding parameters used to gate one Annex A.1 table (A/322 6.1.3)."""
    k: int
    m: int
    code_type: str
    m1: int = 0
    m2: int = 0
    q1: int = 0
    q2: int = 0


@dataclass(frozen=True)
class TableRead:
    """One table's parsed rows, plus whether its page has two columns."""
    rows: List[List[int]]
    two_column: bool


def code_params(rate: int) -> CodeParams:
    k = NINNER_NORMAL * rate // RATE_DENOM
    m = NINNER_NORMAL - k
    if rate in TYPE_A_PARAMS_64800:
        p = TYPE_A_PARAMS_64800[rate]
        return CodeParams(k, m, FEC_TYPE_A, p.m1, p.m2, p.q1, p.q2)
    return CodeParams(k, m, FEC_TYPE_B, q2=TYPE_B_QLDPC_64800[rate])


def row_count_expected(p: CodeParams) -> int:
    return p.k // GROUP_SIZE + (p.q1 if p.code_type == FEC_TYPE_A else 0)


def expand(row: List[int], p: CodeParams) -> List[int]:
    """All accumulator addresses a row touches (A/322 6.1.3.1 / 6.1.3.2)."""
    out = []
    for m in range(GROUP_SIZE):
        for a in row:
            if p.code_type == FEC_TYPE_B:
                out.append((a + m * p.q2) % p.m)
            elif a < p.m1:
                out.append((a + m * p.q1) % p.m1)
            else:
                out.append(p.m1 + (a - p.m1 + m * p.q2) % (p.m - p.m1))
    return out


def _table_pages(doc) -> Dict[int, int]:
    pages = {}
    for i in range(doc.page_count):
        for line in page_lines(doc[i]):
            m = _RATE_CAPTION.match(line.text.replace("\u2013", "-"))
            if m:
                pages[int(m.group(1))] = i
    return pages


def _split_columns(body: List[PdfLine]):
    xs = sorted({line.x for line in body})
    left_x = xs[0]
    right_xs = [x for x in xs if x > left_x + COLUMN_GAP]
    if not right_xs:
        return sorted(body, key=lambda l: l.y), []
    left = sorted((l for l in body if l.x < left_x + COLUMN_GAP),
                  key=lambda l: l.y)
    right = sorted((l for l in body if l.x >= min(right_xs)), key=lambda l: l.y)
    return left, right


def parse_table(doc, page_no, interleave=False) -> TableRead:
    """Extract one Annex A.1 table (column-major unless ``interleave``)."""
    lines = page_lines(doc[page_no])
    cap_y = min(l.y for l in lines if l.text.startswith(_CAPTION_PREFIX))
    body = [l for l in integer_lines(doc[page_no]) if l.y > cap_y + CAPTION_GAP]
    left, right = _split_columns(body)
    if not right:
        return TableRead([line_values(l) for l in left], False)
    if interleave:
        pairs = zip(left, right)
        rows = [row for pair in pairs for row in pair]
        rows += left[len(right):] + right[len(left):]
    else:
        rows = left + right
    return TableRead([line_values(l) for l in rows], True)


def gate_rate(read: TableRead, rate: int) -> None:
    p = code_params(rate)
    rows = read.rows
    assert len(rows) == row_count_expected(p), (
        f"rate {rate}: {len(rows)} rows, expected {row_count_expected(p)}")
    if read.two_column:
        assert _row_weights_nonincreasing(rows), (
            f"rate {rate}: printed two-column rows are not weight-ordered")
    reached = bytearray(p.m)
    for row in rows:
        assert row == sorted(row) and len(set(row)) == len(row), (
            f"rate {rate}: row not strictly increasing")
        for a in row:
            assert 0 <= a < p.m, f"rate {rate}: address {a} out of range"
        for acc in expand(row, p):
            reached[acc] = 1
    assert all(reached), (
        f"rate {rate}: {reached.count(0)} accumulators unreached")


def _row_weights_nonincreasing(rows: List[List[int]]) -> bool:
    weights = [len(row) for row in rows]
    return all(weights[i] >= weights[i + 1] for i in range(len(weights) - 1))


def _control_row_interleave(doc, pages) -> int:
    """C1: a row-interleaved read of a two-column page must fail the gates.

    G1-G4 cannot see row *order*, so this control uses the layout fact that the
    printed (column-major) rows run in non-increasing weight; interleaving the
    columns breaks that order.  If both readings were weight-ordered the column
    split would be unproven and the control fails loudly.
    """
    checked = 0
    for rate, page_no in pages.items():
        column_major = parse_table(doc, page_no)
        if not column_major.two_column:
            continue
        checked += 1
        interleaved = parse_table(doc, page_no, interleave=True)
        assert column_major.rows != interleaved.rows, (
            f"rate {rate}: readings identical, page is not truly two-column")
        assert _row_weights_nonincreasing(column_major.rows), (
            f"rate {rate}: printed rows are not weight-ordered")
        assert not _row_weights_nonincreasing(interleaved.rows), (
            f"rate {rate}: row-interleaved read is also weight-ordered, so "
            f"the column split is not proven necessary")
    return checked


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default=None,
                    help="A/322 PDF (default: fetch the official copy)")
    ap.add_argument("out", nargs="?", default=DEFAULT_OUT,
                    help="output JSON (default: the banked table)")
    args = ap.parse_args(argv)

    doc = pymupdf.open(pdf_path(A322, args.pdf))
    pages = _table_pages(doc)
    tables = {}
    for rate in range(RATE_MIN, RATE_MAX + 1):
        read = parse_table(doc, pages[rate])
        gate_rate(read, rate)
        p = code_params(rate)
        tables[rate] = dict(rate=rate, Ninner=NINNER_NORMAL, Kldpc=p.k,
                            code_type=p.code_type, rows=read.rows)
        print(f"rate {rate}/{RATE_DENOM}: {len(read.rows)} rows, "
              f"{sum(len(r) for r in read.rows)} addresses, gated")
    checked = _control_row_interleave(doc, pages)
    print(f"control C1: row-interleaved read failed gates on {checked} pages")
    doc.close()

    with open(args.out, "w") as f:
        json.dump({str(k): v for k, v in tables.items()}, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
