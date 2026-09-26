"""Canonical sources for the A/322/A/330 spec tables and the reference witness.

WHY
---
The extractors and the cross-check need to know *where* their inputs come from;
those URLs and pins must live in one named place, not be scattered through the
tools.  The ATSC PDFs are the ground truth for the banked tables; the GNU Radio
transmitter is an independent witness used only to cross-check them.

The PDFs are large and are never committed.  ``pdf_path`` returns a per-file
cache path and downloads on demand, so ``tools/extract_*`` can run with no
arguments.
"""

import os
import tempfile
import urllib.request
from dataclasses import dataclass

#: Official ATSC A/322 "Physical Layer Protocol" (2024-04) PDF.
A322_PDF_URL = ("https://www.atsc.org/wp-content/uploads/2024/04/"
                "A322-2024-04-Physical-Layer-Protocol.pdf")
#: Official ATSC A/330 "Link-Layer Protocol" (2019) PDF.
A330_PDF_URL = ("https://www.atsc.org/wp-content/uploads/2016/10/"
                "A330-2019-Link-Layer-Protocol.pdf")

#: Default local cache directory for downloaded specs.
CACHE_DIR = os.path.join(tempfile.gettempdir(), 'atsc3lib-spec')

#: Independent public transcription of the A/322 tables, used only as a
#: witness by ``crosscheck_web_tables``.  Pinned so the comparison is stable.
REF_REPO = 'drmpeg/gr-atsc3'
REF_COMMIT = '000b86a325e2506eb063a25cd182d79c4b57acdd'
#: Raw-file base for the pinned revision.
REF_RAW = f'https://raw.githubusercontent.com/{REF_REPO}/{REF_COMMIT}/lib'


@dataclass(frozen=True)
class SpecPdf:
    """One downloadable specification PDF."""
    label: str
    url: str
    filename: str


#: The spec PDFs the tools can fetch on demand.
A322 = SpecPdf('A/322', A322_PDF_URL, 'A322-2024-04-Physical-Layer-Protocol.pdf')
A330 = SpecPdf('A/330', A330_PDF_URL, 'A330-2019-Link-Layer-Protocol.pdf')


def _download(url: str, dest: str) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(dest, 'wb') as f:
        f.write(resp.read())
    return dest


def pdf_path(spec: SpecPdf, override: str = None) -> str:
    """Return a local path to ``spec``, downloading once if not cached."""
    if override:
        return override
    dest = os.path.join(CACHE_DIR, spec.filename)
    if not os.path.exists(dest):
        _download(spec.url, dest)
    return dest
