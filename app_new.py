from __future__ import annotations

import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from config import settings
from logger import log
from rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="NotebookRAG — Study workspace",
    page_icon="📓",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def ensure_state() -> None:
    st.session_state.setdefault("theme", "dark")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("active_doc", None)
    st.session_state.setdefault("last_hits", [])
    st.session_state.setdefault("last_context", "")


def toast(msg: str, kind: str = "info") -> None:
    try:
        st.toast(msg, icon={"info": "ℹ️", "success": "✅", "error": "⚠️", "warning": "⚠️"}.get(kind, "ℹ️"))
    except Exception:
        if kind == "error":
            st.error(msg)
        elif kind == "success":
            st.success(msg)
        elif kind == "warning":
            st.warning(msg)
        else:
            st.info(msg)


def inject_css(theme: str) -> None:
    if theme == "dark":
        bg, bg2 = "#0b1220", "#0f1a2f"
        panel, border = "rgba(255,255,255,0.06)", "rgba(255,255,255,0.10)"
        text, muted = "#e7eefc", "rgba(231,238,252,0.72)"
        bubble_user = "linear-gradient(135deg, rgba(124,156,255,0.22), rgba(124,156,255,0.08))"
        bubble_ai = "rgba(255,255,255,0.06)"
        shadow = "0 16px 40px rgba(0,0,0,0.40)"
    else:
        bg, bg2 = "#f7f8fb", "#eef2ff"
        panel, border = "rgba(255,255,255,0.72)", "rgba(17,24,39,0.10)"
        text, muted = "#0b1220", "rgba(11,18,32,0.70)"
        bubble_user = "linear-gradient(135deg, rgba(54,92,255,0.18), rgba(54,92,255,0.06))"
        bubble_ai = "rgba(255,255,255,0.78)"
        shadow = "0 16px 40px rgba(17,24,39,0.12)"

    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
  --bg: {bg};
  --bg2: {bg2};
  --panel: {panel};
  --border: {border};
  --text: {text};
  --muted: {muted};
  --bubble_user: {bubble_user};
  --bubble_ai: {bubble_ai};
  --shadow: {shadow};
}}

html, body, [data-testid="stAppViewContainer"] {{
  background: radial-gradient(1000px 500px at 20% 0%, var(--bg2), transparent 60%),
              radial-gradient(900px 500px at 90% 10%, rgba(20,184,166,0.18), transparent 60%),
              var(--bg);
  color: var(--text);
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}}

#MainMenu, footer, header {{ visibility: hidden; }}

section[data-testid="stMain"] .block-container {{
  max-width: 1100px;
  padding-top: 1.5rem;
  padding-bottom: 2.0rem;
}}

[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
  border-right: 1px solid var(--border);
}}

.nr-panel {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
}}

.nr-header {{ padding: 18px 18px 12px 18px; }}
.nr-title {{ font-weight: 700; letter-spacing: -0.02em; font-size: 1.42rem; margin: 0; }}
.nr-subtitle {{ margin: 6px 0 0 0; color: var(--muted); font-size: 0.95rem; }}

[data-testid="stFileUploader"] {{
  background: rgba(255,255,255,0.04);
  border: 1px dashed var(--border);
  border-radius: 14px;
  padding: 16px 14px;
}}

.nr-chat-wrap {{ padding: 14px 14px 2px 14px; }}
.nr-bubble {{
  border-radius: 18px;
  padding: 14px 14px;
  border: 1px solid var(--border);
  margin-bottom: 10px;
  line-height: 1.45;
}}
.nr-bubble-user {{ background: var(--bubble_user); }}
.nr-bubble-ai {{ background: var(--bubble_ai); }}
.nr-meta {{ font-size: 0.78rem; color: var(--muted); margin-top: 8px; }}

.nr-cite {{
  padding: 10px 10px;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: rgba(255,255,255,0.03);
  margin-bottom: 8px;
}}

.nr-pill {{
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid var(--border);
  color: var(--muted);
  font-size: 0.75rem;
  margin-right: 6px;
}}

.stButton>button {{
  border-radius: 12px;
  border: 1px solid var(--border) !important;
  background: rgba(255,255,255,0.04) !important;
}}

.stChatInput {{ border-radius: 14px; border: 1px solid var(--border); }}
</style>
""",
        unsafe_allow_html=True,
    )


def export_chat_md(messages: List[Dict[str, Any]]) -> str:
    out = [f"# Chat Export — {datetime.now().isoformat(timespec='seconds')}\n"]
    for m in messages:
        role = m.get("role", "")
        ts = m.get("ts", "")
        out.append(f"## {role.title()} ({ts})\n")
        out.append(m.get("content", ""))
        hits = m.get("citations") or []
        if hits:
            out.append("\n**Sources**\n")
            for h in hits:
                out.append(f"- {h.get('filename','')} — page {h.get('page','?')} — similarity {h.get('similarity_pct','?')}%")
        out.append("\n")
    return "\n".join(out)


def save_upload_to_tmp(upload) -> str:
    tmp_dir = Path(".uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / f"{uuid.uuid4().hex}_{upload.name.replace('/', '_')}"
    out_path.write_bytes(upload.getvalue())
    return str(out_path)


def append_message(role: str, content: str, citations: List[Dict[str, Any]] | None = None, context: str | None = None):
    st.session_state["messages"].append(
        {
            "role": role,
            "content": content,
            "ts": datetime.now().strftime("%H:%M"),
            "citations": citations or [],
            "context": context or "",
        }
    )


ensure_state()
inject_css(st.session_state["theme"])
pipe = get_pipeline()

docs = pipe.vector_store.list_documents()

with st.sidebar:
    st.markdown(
        """<div style="padding:16px 14px 4px 14px;"><div style="font-weight:700;font-size:1.05rem;">📓 NotebookRAG</div><div style="color:var(--muted);font-size:0.85rem;">NotebookLM-style study workspace</div></div>""",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🌗 Theme", use_container_width=True):
            st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"
            st.rerun()
    with c2:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["last_hits"] = []
            st.session_state["last_context"] = ""
            toast("Chat cleared", kind="success")
            st.rerun()

    st.divider()
    st.markdown("### Document library")
    if not docs:
        st.caption("No documents indexed yet.")
    else:
        labels = [d.get("filename", d.get("document_id", "(unknown)")) for d in docs]
        ids = [d.get("document_id") for d in docs]
        cur = st.session_state.get("active_doc")
        idx = ids.index(cur) if cur in ids else 0
        selected = st.selectbox("Active scope", options=list(range(len(docs))), format_func=lambda i: labels[i], index=idx)
        st.session_state["active_doc"] = ids[selected]
        d1, d2 = st.columns(2)
        with d1:
            if st.button("🗑️ Delete", use_container_width=True):
                try:
                    pipe.vector_store.delete_document(st.session_state["active_doc"])
                    st.session_state["active_doc"] = None
                    toast("Document deleted", kind="success")
                except Exception as e:
                    log.exception("delete_document failed")
                    toast(f"Delete failed: {e}", kind="error")
                st.rerun()
        with d2:
            st.caption(" ")

    st.divider()
    st.markdown("### Add PDFs")
    uploaded = st.file_uploader("Drag & drop PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

    st.markdown("### Retrieval controls")
    top_k = st.slider("Top‑k", 1, 12, int(settings.top_k_default))
    chunk_strategy = st.radio(
        "Chunking",
        options=["fixed", "semantic"],
        index=0 if settings.chunk_strategy_default == "fixed" else 1,
        horizontal=True,
    )
    chunk_size = st.slider("Chunk size (chars)", 250, 1200, int(settings.chunk_size), step=50)
    chunk_overlap = st.slider("Overlap (fixed)", 0, 300, int(settings.chunk_overlap), step=10)

    st.divider()
    st.download_button(
        "⬇️ Export chat (MD)",
        data=export_chat_md(st.session_state["messages"]).encode("utf-8"),
        file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True,
    )


st.markdown(
    """
<div class="nr-panel">
  <div class="nr-header">
    <div class="nr-title">Your study notebook</div>
    <div class="nr-subtitle">Upload PDFs, index them into a local vector database, and chat with grounded citations.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


if uploaded:
    st.markdown("""<div style="height:12px"></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="nr-panel" style="padding:14px 14px;">""", unsafe_allow_html=True)
    st.markdown("**Indexing queue**")

    prog = st.progress(0, text="Preparing…")
    status = st.empty()

    try:
        for i, up in enumerate(uploaded, start=1):
            frac = int((i - 1) / max(1, len(uploaded)) * 100)
            prog.progress(frac, text=f"Queued {i-1}/{len(uploaded)}")
            status.markdown(f"- **{up.name}** — Processing")

            tmp_path = save_upload_to_tmp(up)
            res = pipe.index_pdf(
                file_path=tmp_path,
                filename=up.name,
                document_id=uuid.uuid4().hex,
                chunk_strategy=chunk_strategy,
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
            )

            try:
                os.remove(tmp_path)
            except Exception:
                pass

            status.markdown(f"- **{up.name}** — Indexed • **{res['chunks']}** chunks")

        prog.progress(100, text="Done")
        st.markdown("</div>", unsafe_allow_html=True)
        toast("Indexing complete", kind="success")
        st.rerun()
    except Exception as e:
        log.exception("indexing failed")
        prog.progress(100, text="Failed")
        st.markdown("</div>", unsafe_allow_html=True)
        toast(f"Indexing failed: {e}", kind="error")


docs = pipe.vector_store.list_documents()

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Indexed docs", len(docs))
with m2:
    st.metric("Vector store", "ChromaDB")
with m3:
    st.metric("Chunking", chunk_strategy)
with m4:
    st.metric("Top‑k", top_k)

st.markdown("""<div style="height:10px"></div>""", unsafe_allow_html=True)

chat_col, cite_col = st.columns([1.6, 1.0], gap="large")

with chat_col:
    st.markdown("""<div class="nr-panel nr-chat-wrap">""", unsafe_allow_html=True)
    for m in st.session_state["messages"]:
        bubble = "nr-bubble-user" if m.get("role") == "user" else "nr-bubble-ai"
        st.markdown(
            f"""<div class="nr-bubble {bubble}">{m.get('content','')}<div class="nr-meta">{m.get('role','')} • {m.get('ts','')}</div></div>""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with cite_col:
    st.markdown("""<div class="nr-panel" style="padding:14px 14px;">""", unsafe_allow_html=True)
    st.markdown("**Sources & citations**")

    hits = st.session_state.get("last_hits") or []
    if not hits:
        st.caption("Ask a question to see citations.")
    else:
        for h in hits:
            st.markdown(
                """
<div class="nr-cite">
  <div>
    <span class="nr-pill">{filename}</span>
    <span class="nr-pill">page {page}</span>
    <span class="nr-pill">sim {sim}%</span>
  </div>
  <div style="margin-top:8px;color:var(--muted);font-size:0.86rem;">{preview}</div>
</div>
""".format(
                    filename=h.get("filename", ""),
                    page=h.get("page", "?"),
                    sim=h.get("similarity_pct", "?"),
                    preview=(h.get("text", "")[:180].replace("\n", " ") + "…"),
                ),
                unsafe_allow_html=True,
            )

    with st.expander("Retrieved context", expanded=False):
        ctx = st.session_state.get("last_context") or ""
        if not ctx:
            st.caption("No retrieved context yet.")
        else:
            st.text(ctx[:4000])

    st.markdown("</div>", unsafe_allow_html=True)


prefill = st.session_state.pop("prefill", "") if "prefill" in st.session_state else ""
q_in = st.chat_input("Ask a question about your uploaded PDFs…")
if prefill:
    q_in = prefill

if q_in:
    q = q_in.strip()
    if not q:
        st.stop()

    append_message("user", q)

    with chat_col:
        typing = st.empty()
        typing.markdown(
            """<div class="nr-panel nr-chat-wrap"><div class="nr-bubble nr-bubble-ai">Thinking…</div></div>""",
            unsafe_allow_html=True,
        )

    try:
        answer, hits = pipe.answer(query=q, top_k=int(top_k))

        cites: List[Dict[str, Any]] = []
        ctx_chunks: List[str] = []
        for h in hits:
            md = h.metadata or {}
            sim = h.similarity
            cites.append(
                {
                    "filename": md.get("filename", ""),
                    "page": md.get("page", "?"),
                    "similarity_pct": "?" if sim is None else int(round(sim * 100)),
                    "text": h.text,
                }
            )
            ctx_chunks.append(f"[{md.get('filename','')}] page {md.get('page','?')}:\n{h.text}")

        retrieved_context = "\n\n---\n\n".join(ctx_chunks)
        st.session_state["last_hits"] = cites
        st.session_state["last_context"] = retrieved_context

        if answer.strip().lower().startswith("i don't know"):
            answer = "I could not find that in uploaded documents."

        typing.empty()
        append_message("assistant", answer, citations=cites, context=retrieved_context)
        st.rerun()

    except Exception as e:
        log.exception("answering failed")
        typing.empty()
        append_message("assistant", "I ran into an error while answering. Please try again.")
        toast(f"Error: {e}", kind="error")
        st.rerun()
