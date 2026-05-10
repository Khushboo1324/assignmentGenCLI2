"""End-to-end smoke test for the *new* Chroma-backed RAG architecture.

This test is intentionally isolated:
- temp Chroma persist dir
- unique collection name
- cleanup teardown

Run:
  python -m RAG.tests.test_real_rag
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid

from RAG.rag_pipeline import RAGPipeline
from RAG.vector_store import ChromaVectorStore


def _make_isolated_pipe() -> RAGPipeline:
    tmp_dir = tempfile.mkdtemp(prefix="notebookrag_test_chroma_")
    collection = f"test_{uuid.uuid4().hex[:10]}"

    # Make sure config/env doesn't leak into the test run
    os.environ["CHROMA_PERSIST_DIR"] = tmp_dir
    os.environ["CHROMA_COLLECTION"] = collection

    store = ChromaVectorStore(persist_dir=tmp_dir, collection_name=collection)
    pipe = RAGPipeline(vector_store=store)

    # attach cleanup hook
    pipe._test_tmp_dir = tmp_dir  # type: ignore[attr-defined]
    return pipe


def _cleanup_pipe(pipe: RAGPipeline) -> None:
    tmp_dir = getattr(pipe, "_test_tmp_dir", None)
    if tmp_dir and os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_fixed_and_semantic_index_and_retrieve() -> None:
    pipe = _make_isolated_pipe()
    try:
        pdf_path = "RAG/docs/rag_info.pdf"
        query = "What is RAG?"

        # Fixed chunking
        doc_id_fixed = f"test-fixed-{uuid.uuid4().hex[:8]}"
        res_fixed = pipe.index_pdf(
            file_path=pdf_path,
            filename="rag_info.pdf",
            document_id=doc_id_fixed,
            chunk_strategy="fixed",
            chunk_size=600,
            chunk_overlap=120,
        )
        assert res_fixed["chunks"] > 0

        hits_fixed = pipe.retrieve(query=query, top_k=5, active_document_id=doc_id_fixed)
        assert len(hits_fixed) > 0

        h0 = hits_fixed[0]
        md = h0.metadata
        assert md.get("filename") == "rag_info.pdf"
        assert md.get("document_id") == doc_id_fixed
        assert md.get("page") is not None
        assert md.get("chunk_id")
        assert md.get("chunk_index") is not None
        assert h0.similarity is not None
        assert 0.0 <= float(h0.similarity) <= 1.0

        # Semantic chunking
        doc_id_sem = f"test-sem-{uuid.uuid4().hex[:8]}"
        res_sem = pipe.index_pdf(
            file_path=pdf_path,
            filename="rag_info.pdf",
            document_id=doc_id_sem,
            chunk_strategy="semantic",
            chunk_size=600,
            chunk_overlap=120,
        )
        assert res_sem["chunks"] > 0

        hits_sem = pipe.retrieve(query=query, top_k=5, active_document_id=doc_id_sem)
        assert len(hits_sem) > 0
        assert all((h.metadata or {}).get("document_id") == doc_id_sem for h in hits_sem)

        # Answer should be grounded (or graceful fallback if generator unavailable)
        answer, hits = pipe.answer(query=query, top_k=3, active_document_id=doc_id_fixed)
        assert hits, "expected hits for grounded answering"
        assert isinstance(answer, str) and answer.strip()

        fallback = "I could not find that in uploaded documents."
        assert answer == fallback or len(answer) >= 10
    finally:
        _cleanup_pipe(pipe)


def main() -> None:
    test_fixed_and_semantic_index_and_retrieve()
    print("✅ New-architecture RAG smoke test passed (isolated Chroma)")


if __name__ == "__main__":
    main()