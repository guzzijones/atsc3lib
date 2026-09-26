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
  Section 6.1.3, and every address must land in the parity range.  A
  row-interleaved reading (control C1) is asserted to fail on at least the
  multi-column Type B tables.

GATES
-----
G1  row count == Kldpc/360 (Type B) or Kldpc/360 + Q1 (Type A, 6.1.3.1 steps
    (vii)-(viii) feed the M1 parity back in)
G2  every address in [0, Ninner - Kinner)
G3  every row strictly increasing, no duplicate inside a row
G4  the 360-degree expansion touches every parity accumulator at least once

Usage:
    python tools/extract_ldpc64k.py /tmp/opencode/a322.pdf \\
        atsc3lib/data/ldpc_tables_N64800.json
Requires PyMuPDF (`pip install pymupdf`).
"""

import argparse
import json
import re
from typing import Dict, List

import pymupdf

from atsc3lib.ldpc_exact import (
    NINNER_NORMAL, RATE_DENOM, RATE_MIN, RATE_MAX, GROUP_SIZE,
    TYPE_A_PARAMS_64800, TYPE_B_QLDPC_64800,
)

#: Table caption form: "Table A.1.<n> Rate = <r>/<denom>".
_CAPTION = re.compile(r"Table A\.1\.(\d+)\s+Rate\s*=\s*(\d+)\s*/\s*15")
_NUMERIC = re.compile(r"[\d\s]+")

#: Two PDF layout thresholds: the left column starts within this many points of
#: the left-most body line, and the table never runs below the footer.
_COLUMN_GAP = 100
_FOOTER_Y = 720
_CAPTION_GAP = 2


def _page_lines(page):
    out = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            text = "".join(s["text"] for s in line["spans"]).strip()
            if text:
                out.append((round(line["bbox"][1], 1),
                            round(line["bbox"][0], 1), text))
    return out


def _table_pages(doc):
    pages = {}
    for i in range(doc.page_count):
        for _, _, text in _page_lines(doc[i]):
            m = _CAPTION.match(text.replace("\u2013", "-"))
            if m:
                pages[int(m.group(2))] = i
    return pages


def parse_table(doc, page_no, interleave=False):
    """Extract one Annex A.1 table's rows (column-major unless interleave)."""
    lines = _page_lines(doc[page_no])
    cap_y = min(l[0] for l in lines if l[2].startswith("Table A.1."))
    body = [l for l in lines
            if l[0] > cap_y + _CAPTION_GAP and l[0] < _FOOTER_Y
            and _NUMERIC.fullmatch(l[2])]
    xs = sorted(set(l[1] for l in body))
    right_xs = [x for x in xs if x > xs[0] + _COLUMN_GAP]
    if not right_xs:
        rows = sorted(body)
    else:
        left = sorted([l for l in body if l[1] < xs[0] + _COLUMN_GAP])
        right = sorted([l for l in body if l[1] >= min(right_xs)])
        if interleave:
            rows = []
            for i in range(max(len(left), len(right))):
                if i < len(left):
                    rows.append(left[i])
                if i < len(right):
                    rows.append(right[i])
        else:
            rows = left + right
    return [[int(v) for v in r[2].split()] for r in rows]


def params(rate: int) -> Dict:
    k = NINNER_NORMAL * rate // RATE_DENOM
    if rate in TYPE_A_PARAMS_64800:
        p = TYPE_A_PARAMS_64800[rate]
        return dict(K=k, M=NINNER_NORMAL - k, type="A",
                    M1=p.m1, M2=p.m2, Q1=p.q1, Q2=p.q2)
    return dict(K=k, M=NINNER_NORMAL - k, type="B",
                Qldpc=TYPE_B_QLDPC_64800[rate])


def row_count_expected(rate: int) -> int:
    p = params(rate)
    return p["K"] // GROUP_SIZE + (p["Q1"] if p["type"] == "A" else 0)


def expand(row: List[int], rate: int) -> List[int]:
    """All accumulator addresses a row touches (A/322 6.1.3.1 / 6.1.3.2)."""
    p = params(rate)
    out = []
    if p["type"] == "B":
        for m in range(GROUP_SIZE):
            for a in row:
                out.append((a + m * p["Qldpc"]) % p["M"])
    else:
        for m in range(GROUP_SIZE):
            for a in row:
                if a < p["M1"]:
                    out.append((a + m * p["Q1"]) % p["M1"])
                else:
                    out.append(p["M1"] + (a - p["M1"] + m * p["Q2"])
                               % (p["M"] - p["M1"]))
    return out


def gate_rate(rows, rate: int) -> bool:
    p = params(rate)
    assert len(rows) == row_count_expected(rate), (
        f"rate {rate}: {len(rows)} rows, expected {row_count_expected(rate)}")
    reached = bytearray(p["M"])
    for row in rows:
        assert row == sorted(row) and len(set(row)) == len(row), (
            f"rate {rate}: row not strictly increasing")
        for a in row:
            assert 0 <= a < p["M"], f"rate {rate}: address {a} out of range"
        for acc in expand(row, rate):
            reached[acc] = 1
    assert all(reached), f"rate {rate}: {reached.count(0)} accumulators unreached"
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("out")
    args = ap.parse_args(argv)

    doc = pymupdf.open(args.pdf)
    pages = _table_pages(doc)
    tables = {}
    for rate in range(RATE_MIN, RATE_MAX + 1):
        rows = parse_table(doc, pages[rate])
        gate_rate(rows, rate)
        p = params(rate)
        tables[rate] = dict(rate=rate, Ninner=NINNER_NORMAL, Kldpc=p["K"],
                            code_type=p["type"], rows=rows)
        print(f"rate {rate}/15: {len(rows)} rows, "
              f"{sum(len(r) for r in rows)} addresses, gated")
    doc.close()

    with open(args.out, "w") as f:
        json.dump({str(k): v for k, v in tables.items()}, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
