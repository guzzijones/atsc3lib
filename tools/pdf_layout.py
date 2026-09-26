"""Shared PyMuPDF helpers for the A/322 Annex table extractors.

WHY
---
The Annex A.1 (parity) and Annex B.1 (interleaver) pages are two-column body
text with a page number in the footer.  Naive text extraction interleaves the
columns and corrupts row boundaries, so both extractors work from PyMuPDF line
bounding boxes.  This module holds the layout thresholds, the line record, and
the two line readers shared by ``extract_ldpc64k`` and ``extract_bicm``.

All thresholds are page geometry (PDF user-space points), not spec values;
they are named here so neither extractor inlines them.
"""

import re
from dataclasses import dataclass
from typing import List

#: Body text never descends below this y (a US-Letter footer band); the
#: running page number sits past it and must be excluded from the tables.
FOOTER_Y = 720

#: Horizontal distance beyond the left-most body line that marks the start of
#: the right-hand column of a two-column Annex page.
COLUMN_GAP = 100

#: Minimum vertical gap between the table caption line and the first data row.
CAPTION_GAP = 2

#: A body row is a whitespace-separated run of integers only.
_INTEGERS = re.compile(r"[\d\s]+")


@dataclass(frozen=True)
class PdfLine:
    """One extracted PDF text line (A/322 Annex tables are laid out as lines)."""
    y: float
    x: float
    text: str


def page_lines(page) -> List[PdfLine]:
    """All non-empty text lines on a page, with rounded bounding-box origin."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            text = "".join(s["text"] for s in line["spans"]).strip()
            if text:
                x, y = line["bbox"][0], line["bbox"][1]
                out.append(PdfLine(round(y, 1), round(x, 1), text))
    return out


def integer_lines(page) -> List[PdfLine]:
    """Body lines above the footer whose text is nothing but integers."""
    return [line for line in page_lines(page)
            if line.y < FOOTER_Y and _INTEGERS.fullmatch(line.text)]


def line_values(line: PdfLine) -> List[int]:
    """The integers printed on one line, in reading order."""
    return [int(t) for t in line.text.split()]
