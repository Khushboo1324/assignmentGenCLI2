from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Chunk:
    text: str
    metadata: Dict


def fixed_chunk(text: str, *, chunk_size: int, overlap: int, source: str, page: int | None = None) -> List[Chunk]:
    chunks: List[Chunk] = []
    start = 0
    n = len(text)

    step = max(1, chunk_size - overlap)
    while start < n:
        end = min(n, start + chunk_size)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(text=chunk_text, metadata={"source": source, "page": page}))
        start += step

    return chunks


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def semantic_chunk(
    text: str,
    *,
    max_chars: int,
    source: str,
    page: int | None = None,
) -> List[Chunk]:
    """Heuristic 'semantic' chunking.

    Splits by sentence boundaries and groups sentences until max_chars.
    This is not embedding-based segmentation, but behaves much better than raw fixed windows.
    """

    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    chunks: List[Chunk] = []
    buf: List[str] = []
    buf_len = 0

    for sent in sentences:
        if buf_len + len(sent) + 1 > max_chars and buf:
            chunks.append(Chunk(text=" ".join(buf).strip(), metadata={"source": source, "page": page}))
            buf = [sent]
            buf_len = len(sent)
        else:
            buf.append(sent)
            buf_len += len(sent) + 1

    if buf:
        chunks.append(Chunk(text=" ".join(buf).strip(), metadata={"source": source, "page": page}))

    return chunks
