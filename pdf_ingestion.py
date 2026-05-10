"""PDF ingestion utilities.

- `get_pdf_text(file_path)` keeps the original "all pages as one string" interface.
- `get_pdf_text_by_page(file_path)` is used by the Chroma-backed pipeline to attach
  page-level metadata for citations.
"""

from __future__ import annotations

import os
from typing import List, Tuple

from pypdf import PdfReader


def get_pdf_text(file_path: str) -> str:
    if not os.path.exists(file_path):
        return f"❌ ERROR: File not found - {file_path}"
    if not file_path.lower().endswith(".pdf"):
        return f"❌ ERROR: File is not a PDF - {file_path}"

    pages = get_pdf_text_by_page(file_path)
    if isinstance(pages, str):
        return pages

    full = "\n\n".join([t for _, t in pages if t])
    return " ".join(full.split())


def get_pdf_text_by_page(file_path: str) -> str | List[Tuple[int, str]]:
    """Extract text per page.

    Returns either:
    - list of (page_number, page_text)
    - or an error string starting with "❌"
    """

    if not os.path.exists(file_path):
        return f"❌ ERROR: File not found - {file_path}"
    if not file_path.lower().endswith(".pdf"):
        return f"❌ ERROR: File is not a PDF - {file_path}"

    try:
        reader = PdfReader(file_path)
        out: List[Tuple[int, str]] = []
        for i, page in enumerate(reader.pages, start=1):
            txt = page.extract_text() or ""
            txt = " ".join(txt.split())
            out.append((i, txt))
        return out
    except Exception as e:
        return f"❌ ERROR: Failed to read PDF - {e}"


if __name__ == "__main__":
    # minimal smoke check
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not path:
        print("Usage: python pdf_ingestion.py <file.pdf>")
        raise SystemExit(2)

    pages = get_pdf_text_by_page(path)
    if isinstance(pages, str):
        print(pages)
    else:
        print(f"Pages extracted: {len(pages)}")
        print(pages[0][1][:200])
