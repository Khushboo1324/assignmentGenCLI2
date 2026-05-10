# NotebookRAG — “NotebookLM meets ChatGPT” (RAG Assignment)

A submission-ready, local-first RAG app that lets you upload PDFs, index them into a persistent **ChromaDB** vector store, and chat with **grounded, cited** answers in a premium **Streamlit** UI.

## Highlights
- **Premium Streamlit UI/UX**: document library sidebar, scoped retrieval, dashboard cards, chat bubbles, citations panel, retrieved-context expander, theme toggle.
- **Persistent ChromaDB**: local vector DB on disk with per-chunk metadata (filename, page, chunk index, timestamp, chunk_id).
- **Chunking strategies**: `fixed` or `semantic` chunking at index time.
- **Grounded answering**: retrieves top-k chunks and generates answers using (optionally) **Ollama**; falls back safely if generation is unavailable.
- **Scoped retrieval**: select a specific document or “All documents”.

---

## Project structure (relevant)