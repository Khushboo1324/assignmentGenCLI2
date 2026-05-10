from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

try:
    from .config import settings
    from .logger import log
except Exception:  # pragma: no cover
    from config import settings
    from logger import log


@dataclass
class StoredChunk:
    id: str
    document_id: str
    text: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()


class ChromaVectorStore:
    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def list_documents(self) -> List[Dict[str, Any]]:
        """Back-compat document listing (no stats)."""
        try:
            peek = self._collection.peek(limit=10000)
        except Exception:
            return []

        doc_map: Dict[str, Dict[str, Any]] = {}
        for md in (peek.get("metadatas") or []):
            if not md:
                continue
            doc_id = md.get("document_id")
            if not doc_id:
                continue
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "document_id": doc_id,
                    "filename": md.get("filename", doc_id),
                    "source": md.get("source", "pdf"),
                    "timestamp": md.get("timestamp"),
                }

        return sorted(doc_map.values(), key=lambda x: x.get("timestamp") or "")

    def list_documents_with_stats(self) -> List[Dict[str, Any]]:
        """List documents enriched with computed `pages` and `chunks` counts."""
        try:
            peek = self._collection.peek(limit=10000)
        except Exception:
            return []

        stats: Dict[str, Dict[str, Any]] = {}
        for md in (peek.get("metadatas") or []):
            if not md:
                continue
            doc_id = md.get("document_id")
            if not doc_id:
                continue

            s = stats.setdefault(
                doc_id,
                {
                    "document_id": doc_id,
                    "filename": md.get("filename", doc_id),
                    "source": md.get("source", "pdf"),
                    "timestamp": md.get("timestamp"),
                    "chunks": 0,
                    "_pages": set(),
                },
            )
            s["chunks"] += 1
            p = md.get("page")
            if p is not None:
                s["_pages"].add(p)

        out: List[Dict[str, Any]] = []
        for s in stats.values():
            pages = s.pop("_pages", set())
            s["pages"] = len(pages)
            out.append(s)

        return sorted(out, key=lambda x: x.get("timestamp") or "")

    def total_chunks(self) -> int:
        """Best-effort total number of chunks stored in the collection."""
        try:
            return int(self._collection.count())
        except Exception:
            return 0

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})
        log.info("Deleted document_id={} from Chroma", document_id)

    def clear_all_documents(self) -> None:
        """Delete *all* stored chunks/documents in the collection."""
        self._collection.delete()
        log.info("Cleared all documents from Chroma collection={}", self.collection_name)

    def upsert_chunks(
        self,
        *,
        document_id: str,
        filename: str,
        chunks: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]] | Any,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()

        ids: List[str] = []
        final_mds: List[Dict[str, Any]] = []
        for i, (text, md) in enumerate(zip(chunks, metadatas)):
            chunk_id = _stable_id(document_id, str(i), str(md.get("page", "")), text[:64])
            ids.append(chunk_id)
            final_mds.append(
                {
                    "document_id": document_id,
                    "chunk_index": i,
                    "filename": filename,
                    "source": md.get("source", "pdf"),
                    "page": md.get("page"),
                    "timestamp": md.get("timestamp", now),
                    **{k: v for k, v in md.items() if k not in {"timestamp"}},
                }
            )

        self._collection.upsert(ids=ids, documents=chunks, metadatas=final_mds, embeddings=embeddings)

    def query(
        self,
        *,
        query_text: str,
        query_embedding: List[float],
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[StoredChunk]:
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs = (res.get("documents") or [[]])[0]
        mds = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]

        out: List[StoredChunk] = []
        for _id, doc, md, dist in zip(ids, docs, mds, dists):
            md = md or {}
            md.setdefault("chunk_id", _id)
            out.append(
                StoredChunk(
                    id=_id,
                    document_id=md.get("document_id", ""),
                    text=doc,
                    metadata=md,
                    distance=dist,
                )
            )
        return out