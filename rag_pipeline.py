from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer

try:
    # Package imports (preferred when `RAG/` is imported)
    from .chunking_strategies import Chunk, fixed_chunk, semantic_chunk
    from .generation import generate_answer
    from .pdf_ingestion import get_pdf_text_by_page
    from .vector_store import ChromaVectorStore
except Exception:  # pragma: no cover
    # Script-style imports (when running inside the folder)
    from chunking_strategies import Chunk, fixed_chunk, semantic_chunk
    from generation import generate_answer
    from pdf_ingestion import get_pdf_text_by_page
    from vector_store import ChromaVectorStore


@dataclass
class RetrievalHit:
    text: str
    metadata: Dict[str, Any]
    distance: float | None

    @property
    def similarity(self) -> float | None:
        if self.distance is None:
            return None
        try:
            return max(0.0, min(1.0, 1.0 - float(self.distance)))
        except Exception:
            return None


class RAGPipeline:
    def __init__(self, *, vector_store: Optional[ChromaVectorStore] = None, embedder: Optional[SentenceTransformer] = None):
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedder = embedder or SentenceTransformer("all-MiniLM-L6-v2")

    def index_pdf(
        self,
        *,
        file_path: str,
        filename: str,
        document_id: str,
        chunk_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> Dict[str, Any]:
        started = datetime.now(timezone.utc).isoformat()

        pages = get_pdf_text_by_page(file_path)
        if isinstance(pages, str) and pages.startswith("❌"):
            raise ValueError(pages)

        all_chunks: List[str] = []
        all_mds: List[Dict[str, Any]] = []

        for page_num, page_text in pages:
            if not page_text.strip():
                continue

            if chunk_strategy == "semantic":
                chunks: List[Chunk] = semantic_chunk(page_text, max_chars=chunk_size, source="pdf", page=page_num)
            else:
                chunks = fixed_chunk(page_text, chunk_size=chunk_size, overlap=chunk_overlap, source="pdf", page=page_num)

            for c in chunks:
                all_chunks.append(c.text)
                all_mds.append({**c.metadata, "timestamp": started})

        embeddings = self.embedder.encode(all_chunks).tolist()

        self.vector_store.upsert_chunks(
            document_id=document_id,
            filename=filename,
            chunks=all_chunks,
            metadatas=all_mds,
            embeddings=embeddings,
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "chunks": len(all_chunks),
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }

    def retrieve(self, *, query: str, top_k: int, active_document_id: str | None = None) -> List[RetrievalHit]:
        q_emb = self.embedder.encode(query).tolist()
        where = {"document_id": active_document_id} if active_document_id else None
        hits = self.vector_store.query(query_text=query, query_embedding=q_emb, top_k=top_k, where=where)
        return [RetrievalHit(text=h.text, metadata=h.metadata, distance=h.distance) for h in hits]

    def answer(self, *, query: str, top_k: int, active_document_id: str | None = None) -> Tuple[str, List[RetrievalHit]]:
        hits = self.retrieve(query=query, top_k=top_k, active_document_id=active_document_id)
        if not hits:
            return "I could not find that in uploaded documents.", []

        context = "\n\n---\n\n".join([h.text for h in hits])
        ans = generate_answer(query=query, context=context)
        return ans, hits
