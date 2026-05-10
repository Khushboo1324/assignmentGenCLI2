from __future__ import annotations

import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from config import settings
from logger import log
from rag_pipeline import RAGPipeline


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="NotebookRAG — Study workspace",
    page_icon="📓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------
# Pipeline (cached)
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


# ----------------------------
# Session state
# ----------------------------
def ensure_state() -> None:
    st.session_state.setdefault("theme", "dark")
    st.session_state.setdefault("messages", [])  # list[dict]
    st.session_state.setdefault("active_doc", None)  # document_id | None
    st.session_state.setdefault("last_hits", [])  # list[dict]
    st.session_state.setdefault("last_context", "")
    st.session_state.setdefault("prefill", "")
    st.session_state.setdefault("ui_top_k", int(getattr(settings, "top_k_default", 6)))
    st.session_state.setdefault("ui_chunk_strategy", str(getattr(settings, "chunk_strategy_default", "fixed")))
    st.session_state.setdefault("ui_chunk_size", int(getattr(settings, "chunk_size", 600)))
    st.session_state.setdefault("ui_chunk_overlap", int(getattr(settings, "chunk_overlap", 120)))


def toast(msg: str, kind: str = "info") -> None:
    icons = {"info": "ℹ️", "success": "✅", "error": "⚠️", "warning": "⚠️"}
    try:
        st.toast(msg, icon=icons.get(kind, "ℹ️"))
    except Exception:
        if kind == "error":
            st.error(msg)
        elif kind == "success":
            st.success(msg)
        elif kind == "warning":
            st.warning(msg)
        else:
            st.info(msg)


# ----------------------------
# UI helpers
# ----------------------------
def _fmt_ts(ts: Optional[str]) -> str:
    if not ts:
        return ""
    try:
        return ts.replace("T", " ")[:19]
    except Exception:
        return str(ts)


def _confidence_badge(similarity: Optional[float]) -> tuple[str, str]:
    # (label, css background)
    if similarity is None:
        return "Unknown", "rgba(148,163,184,0.22)"
    if similarity >= 0.80:
        return "High", "rgba(34,197,94,0.18)"
    if similarity >= 0.65:
        return "Medium", "rgba(234,179,8,0.18)"
    return "Low", "rgba(239,68,68,0.16)"


def export_chat_md(messages: List[Dict[str, Any]]) -> str:
    out = [f"# Chat Export — {datetime.now().isoformat(timespec='seconds')}\n"]
    for m in messages:
        role = (m.get("role") or "").title()
        ts = m.get("ts") or ""
        out.append(f"## {role} ({ts})\n")
        out.append(m.get("content", ""))

        hits = m.get("citations") or []
        if hits:
            out.append("\n**Sources**\n")
            for h in hits:
                out.append(
                    f"- {h.get('filename','')} — page {h.get('page','?')} — chunk {h.get('chunk_index','?')} — sim {h.get('similarity_pct','?')}%"
                )
        out.append("\n")
    return "\n".join(out)


def save_upload_to_tmp(upload) -> str:
    tmp_dir = Path(".uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / f"{uuid.uuid4().hex}_{upload.name.replace('/', '_')}"
    out_path.write_bytes(upload.getvalue())
    return str(out_path)


def append_message(
    role: str,
    content: str,
    *,
    citations: Optional[List[Dict[str, Any]]] = None,
    context: Optional[str] = None,
    answer_meta: Optional[Dict[str, Any]] = None,
) -> None:
    st.session_state["messages"].append(
        {
            "role": role,
            "content": content,
            "ts": datetime.now().strftime("%H:%M"),
            "citations": citations or [],
            "context": context or "",
            "answer_meta": answer_meta or {},
        }
    )


def inject_css(theme: str) -> None:
    # Premium tokens
    if theme == "dark":
        bg0 = "#070B14"
        bg1 = "#0b1220"
        bg2 = "#0f1a2f"
        panel = "rgba(255,255,255,0.06)"
        panel2 = "rgba(255,255,255,0.04)"
        border = "rgba(255,255,255,0.10)"
        text = "#e7eefc"
        muted = "rgba(231,238,252,0.72)"
        accent = "#7C9CFF"
        accent2 = "#14B8A6"
        bubble_user = "linear-gradient(135deg, rgba(124,156,255,0.30), rgba(124,156,255,0.10))"
        bubble_ai = "linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.05))"
        shadow = "0 18px 50px rgba(0,0,0,0.45)"
        code_bg = "rgba(2,6,23,0.45)"
        chip_bg = "rgba(255,255,255,0.06)"
    else:
        bg0 = "#F6F8FF"
        bg1 = "#F7F8FB"
        bg2 = "#ECF2FF"
        panel = "rgba(255,255,255,0.78)"
        panel2 = "rgba(255,255,255,0.70)"
        border = "rgba(17,24,39,0.10)"
        text = "#0b1220"
        muted = "rgba(11,18,32,0.70)"
        accent = "#365CFF"
        accent2 = "#0EA5A0"
        bubble_user = "linear-gradient(135deg, rgba(54,92,255,0.18), rgba(54,92,255,0.06))"
        bubble_ai = "linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,255,255,0.76))"
        shadow = "0 18px 50px rgba(17,24,39,0.12)"
        code_bg = "rgba(2,6,23,0.06)"
        chip_bg = "rgba(54,92,255,0.08)"

    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
  --bg0: {bg0};
  --bg1: {bg1};
  --bg2: {bg2};
  --panel: {panel};
  --panel2: {panel2};
  --border: {border};
  --text: {text};
  --muted: {muted};
  --accent: {accent};
  --accent2: {accent2};
  --bubble_user: {bubble_user};
  --bubble_ai: {bubble_ai};
  --shadow: {shadow};
  --code_bg: {code_bg};
  --chip_bg: {chip_bg};
}}

html, body, [data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(1100px 680px at 18% 0%, var(--bg2), transparent 62%),
    radial-gradient(900px 520px at 92% 10%, rgba(20,184,166,0.18), transparent 62%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
  color: var(--text);
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}}

#MainMenu, footer, header {{ visibility: hidden; }}

section[data-testid="stMain"] .block-container {{
  max-width: 1120px;
  padding-top: 1.25rem;
  padding-bottom: 2.0rem;
}}

[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
  border-right: 1px solid var(--border);
}}

.stMarkdown, .stText, .stCaption, .stMetric, label, p, li {{
  color: var(--text);
}}

.small-muted {{ color: var(--muted); font-size: 0.92rem; }}

.nr-panel {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
}}

.nr-header {{
  padding: 18px 18px 12px 18px;
}}

.nr-title {{
  font-weight: 700;
  letter-spacing: -0.02em;
  font-size: 1.52rem;
  margin: 0;
}}

.nr-subtitle {{
  margin: 6px 0 0 0;
  color: var(--muted);
  font-size: 0.98rem;
}}

.nr-card-row {{
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 10px;
  margin-top: 12px;
}}

@media (max-width: 980px) {{
  .nr-card-row {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}

.nr-metric {{
  padding: 12px 12px;
  border-radius: 16px;
  border: 1px solid var(--border);
  background: var(--panel2);
}}

.nr-metric .k {{
  color: var(--muted);
  font-size: 0.78rem;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}}

.nr-metric .v {{
  font-size: 1.12rem;
  font-weight: 700;
  margin-top: 6px;
}}

.nr-metric .s {{
  font-size: 0.82rem;
  color: var(--muted);
  margin-top: 4px;
}}

[data-testid="stFileUploader"] {{
  background: rgba(255,255,255,0.04);
  border: 1px dashed var(--border);
  border-radius: 14px;
  padding: 14px 12px;
}}

.stButton>button {{
  border-radius: 12px !important;
  border: 1px solid var(--border) !important;
  background: rgba(255,255,255,0.05) !important;
}}

.stButton>button:hover {{
  border-color: rgba(124,156,255,0.40) !important;
  box-shadow: 0 10px 28px rgba(124,156,255,0.12);
}}

.stChatInput {{
  border-radius: 16px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.03);
}}

.nr-chat-shell {{
  padding: 14px 14px 10px 14px;
}}

.nr-row {{
  display:flex;
  gap: 10px;
  margin-bottom: 12px;
  align-items:flex-start;
}}

.nr-row.user {{
  justify-content:flex-end;
}}

.nr-avatar {{
  width: 34px;
  height: 34px;
  border-radius: 999px;
  display:flex;
  align-items:center;
  justify-content:center;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.06);
  flex: 0 0 auto;
  font-weight: 700;
  letter-spacing: -0.02em;
}}

.nr-avatar.user {{
  background: rgba(124,156,255,0.18);
}}

.nr-avatar.ai {{
  background: rgba(20,184,166,0.16);
}}

.nr-bubble {{
  max-width: 82%;
  border-radius: 18px;
  padding: 13px 14px;
  border: 1px solid var(--border);
  line-height: 1.50;
}}

.nr-bubble.user {{
  background: var(--bubble_user);
}}

.nr-bubble.ai {{
  background: var(--bubble_ai);
}}

.nr-meta {{
  font-size: 0.78rem;
  color: var(--muted);
  margin-top: 8px;
}}

.nr-bubble pre {{
  background: var(--code_bg);
  padding: 12px;
  border-radius: 14px;
  overflow-x: auto;
  border: 1px solid var(--border);
}}

.nr-bubble code {{
  background: rgba(148,163,184,0.14);
  padding: 2px 6px;
  border-radius: 8px;
}}

.nr-typing {{
  display: inline-flex;
  gap: 6px;
  align-items: center;
}}

.nr-dot {{
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: rgba(148,163,184,0.8);
  animation: nrBounce 1.2s infinite ease-in-out;
}}

.nr-dot:nth-child(2) {{ animation-delay: 0.15s; }}
.nr-dot:nth-child(3) {{ animation-delay: 0.30s; }}

@keyframes nrBounce {{
  0%, 80%, 100% {{ transform: translateY(0); opacity: 0.55; }}
  40% {{ transform: translateY(-4px); opacity: 1.0; }}
}}

.nr-inputbar {{
  position: sticky;
  bottom: 0;
  padding-top: 10px;
  background: linear-gradient(180deg, transparent, var(--bg1) 35%);
  z-index: 5;
}}

.nr-answer-card {{
  padding: 14px 14px;
  border-radius: 18px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.03);
}}

.nr-answer-h {{
  font-weight: 800;
  font-size: 0.92rem;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 6px;
}}

.nr-sources {{
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px solid var(--border);
}}

.nr-source-item {{
  padding: 10px 10px;
  border-radius: 14px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.03);
  margin-bottom: 8px;
}}

.nr-pill {{
  display: inline-block;
  padding: 3px 8px;
  border-radius: 999px;
  border: 1px solid var(--border);
  color: var(--muted);
  font-size: 0.75rem;
  margin-right: 6px;
  background: rgba(255,255,255,0.02);
}}

.nr-pill.accent {{
  color: var(--text);
  border-color: rgba(124,156,255,0.35);
  background: rgba(124,156,255,0.12);
}}

.nr-chip-row {{
  display:flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 12px;
}}

.nr-chip {{
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--chip_bg);
  color: var(--text);
  font-size: 0.90rem;
}}

.nr-hero {{
  padding: 18px 18px;
  margin-top: 14px;
}}

.nr-hero h2 {{
  margin: 0;
  font-size: 1.25rem;
  letter-spacing: -0.02em;
}}

.nr-hero p {{
  margin: 8px 0 0 0;
  color: var(--muted);
  font-size: 0.98rem;
}}

html {{ scroll-behavior: smooth; }}
</style>
""",
        unsafe_allow_html=True,
    )


# ----------------------------
# Render helpers
# ----------------------------
def render_chat_message(m: Dict[str, Any]) -> None:
    role = m.get("role", "")
    ts = m.get("ts", "")
    content = m.get("content", "") or ""

    if role == "user":
        st.markdown(
            f"""
<div class="nr-row user">
  <div class="nr-bubble user">
    <div>{st._utils._escape_markdown(content) if False else ""}</div>
  </div>
  <div class="nr-avatar user">You</div>
</div>
""",
            unsafe_allow_html=True,
        )
        # Use markdown below to preserve formatting; keep bubble above for layout
        st.markdown(f"<div style='margin-top:-52px; margin-right:44px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
<div class="nr-row user" style="margin-top:-2px;">
  <div class="nr-bubble user">
    <div class="nr-md">{content}</div>
    <div class="nr-meta">you • {ts}</div>
  </div>
  <div class="nr-avatar user">Y</div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    # assistant
    answer_meta = m.get("answer_meta") or {}
    conf_label = answer_meta.get("confidence_label")
    conf_bg = answer_meta.get("confidence_bg")

    # Render answer card inside assistant bubble
    st.markdown(
        f"""
<div class="nr-row">
  <div class="nr-avatar ai">AI</div>
  <div class="nr-bubble ai" style="width:100%;">
    <div class="nr-answer-card">
      <div class="nr-answer-h">Answer</div>
      <div class="nr-md">{content}</div>
      <div class="nr-sources">
        <div class="nr-answer-h">Confidence</div>
        <span class="nr-pill accent" style="background:{conf_bg};">{conf_label}</span>
      </div>
    </div>
    <div class="nr-meta">assistant • {ts}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sources_panel(hits: List[Dict[str, Any]]) -> None:
    st.markdown("**Sources used**")
    if not hits:
        st.caption("Ask a question to see sources.")
        return

    for h in hits:
        filename = h.get("filename", "")
        page = h.get("page", "?")
        chunk_index = h.get("chunk_index", "?")
        sim = h.get("similarity_pct", "?")
        preview = (h.get("text", "") or "").replace("\n", " ")[:220] + "…"

        st.markdown(
            f"""
<div class="nr-source-item">
  <div>
    <span class="nr-pill">{filename}</span>
    <span class="nr-pill">page {page}</span>
    <span class="nr-pill">chunk #{chunk_index}</span>
    <span class="nr-pill">sim {sim}%</span>
  </div>
  <div style="margin-top:8px;color:var(--muted);font-size:0.90rem;">{preview}</div>
</div>
""",
            unsafe_allow_html=True,
        )


# ----------------------------
# App start
# ----------------------------
ensure_state()
inject_css(st.session_state["theme"])
pipe = get_pipeline()

# Use stats-aware listing when available
try:
    docs = pipe.vector_store.list_documents_with_stats()
except Exception:
    docs = pipe.vector_store.list_documents()


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown(
        """
<div style="padding:16px 14px 8px 14px;">
  <div style="font-weight:800;font-size:1.05rem; letter-spacing:-0.01em;">📓 NotebookRAG</div>
  <div style="color:var(--muted);font-size:0.86rem;">NotebookLM-style workspace • grounded citations</div>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🌗 Theme", use_container_width=True):
            st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"
            st.rerun()
    with c2:
        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["last_hits"] = []
            st.session_state["last_context"] = ""
            toast("Chat cleared", kind="success")
            st.rerun()

    st.divider()
    st.markdown("### Document library")

    if not docs:
        st.caption("No documents indexed yet.")
        st.session_state["active_doc"] = None
    else:
        # All Documents selector (top)
        scopes = ["All documents"] + [d.get("document_id") for d in docs]

        def _label(scope_id: str) -> str:
            if scope_id == "All documents":
                return "All documents"
            d = next((x for x in docs if x.get("document_id") == scope_id), None) or {}
            return d.get("filename", scope_id)

        current = st.session_state.get("active_doc")
        current_scope = current if current else "All documents"
        if current_scope not in scopes:
            current_scope = "All documents"

        selected_scope = st.selectbox(
            "Active scope",
            options=scopes,
            index=scopes.index(current_scope),
            format_func=_label,
        )
        st.session_state["active_doc"] = None if selected_scope == "All documents" else selected_scope

        # Document cards
        st.markdown("#### Files")
        for d in docs:
            did = d.get("document_id")
            name = d.get("filename", did)
            ts = _fmt_ts(d.get("timestamp"))
            pages = d.get("pages")
            chunks = d.get("chunks")

            active = (st.session_state.get("active_doc") == did)
            active_tag = " <span class='nr-pill accent'>active</span>" if active else ""

            st.markdown(
                f"""
<div class="nr-panel" style="padding:12px 12px; margin-bottom:10px;">
  <div style="font-weight:700; letter-spacing:-0.01em;">{name}{active_tag}</div>
  <div style="margin-top:6px;">
    <span class="nr-pill">pages {pages if pages is not None else "—"}</span>
    <span class="nr-pill">chunks {chunks if chunks is not None else "—"}</span>
  </div>
  <div style="color:var(--muted); font-size:0.86rem; margin-top:6px;">uploaded: {ts if ts else "—"}</div>
  <div style="color:var(--muted); font-size:0.78rem; margin-top:2px;">document_id: {did}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        d1, d2 = st.columns(2)
        with d1:
            if st.button(
                "🗑️ Delete document",
                use_container_width=True,
                disabled=st.session_state.get("active_doc") is None,
            ):
                try:
                    pipe.vector_store.delete_document(st.session_state["active_doc"])
                    st.session_state["active_doc"] = None
                    toast("Document deleted", kind="success")
                except Exception as e:
                    log.exception("delete_document failed")
                    toast(f"Delete failed: {e}", kind="error")
                st.rerun()
        with d2:
            if st.button("🧨 Clear all", use_container_width=True):
                try:
                    pipe.vector_store.clear_all_documents()
                    st.session_state["active_doc"] = None
                    toast("All documents cleared", kind="success")
                except Exception as e:
                    log.exception("clear_all_documents failed")
                    toast(f"Clear failed: {e}", kind="error")
                st.rerun()

    st.divider()
    st.markdown("### Add PDFs")
    uploaded = st.file_uploader(
        "Drag & drop PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown("### Retrieval controls")
    st.session_state["ui_top_k"] = st.slider("Top‑k", 1, 12, int(st.session_state["ui_top_k"]))
    st.session_state["ui_chunk_strategy"] = st.radio(
        "Chunk strategy",
        options=["fixed", "semantic"],
        index=0 if st.session_state["ui_chunk_strategy"] == "fixed" else 1,
        horizontal=True,
    )
    st.session_state["ui_chunk_size"] = st.slider("Chunk size (chars)", 250, 1200, int(st.session_state["ui_chunk_size"]), step=50)
    st.session_state["ui_chunk_overlap"] = st.slider("Overlap (fixed)", 0, 300, int(st.session_state["ui_chunk_overlap"]), step=10)

    st.divider()
    st.download_button(
        "⬇️ Export chat (MD)",
        data=export_chat_md(st.session_state["messages"]).encode("utf-8"),
        file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True,
    )


# ----------------------------
# Main header + dashboard cards
# ----------------------------
active_scope_label = "All documents" if not st.session_state.get("active_doc") else _label(st.session_state["active_doc"])
embedding_model_name = getattr(pipe.embedder, "name_or_path", None) or getattr(pipe.embedder, "model_name", None) or "all-MiniLM-L6-v2"

try:
    total_chunks = pipe.vector_store.total_chunks()
except Exception:
    # fallback: sum if docs has stats
    total_chunks = sum(int(d.get("chunks") or 0) for d in docs) if docs else 0

st.markdown(
    """
<div class="nr-panel">
  <div class="nr-header">
    <div class="nr-title">Your study notebook</div>
    <div class="nr-subtitle">Upload PDFs, build a private local index, and chat with grounded citations — NotebookLM meets ChatGPT.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="nr-card-row">
  <div class="nr-metric">
    <div class="k">Indexed Docs</div>
    <div class="v">{len(docs)}</div>
    <div class="s">Documents in library</div>
  </div>
  <div class="nr-metric">
    <div class="k">Total Chunks</div>
    <div class="v">{total_chunks}</div>
    <div class="s">Chunks in vector store</div>
  </div>
  <div class="nr-metric">
    <div class="k">Embedding Model</div>
    <div class="v">{embedding_model_name}</div>
    <div class="s">SentenceTransformer</div>
  </div>
  <div class="nr-metric">
    <div class="k">Chunk Strategy</div>
    <div class="v">{st.session_state["ui_chunk_strategy"]}</div>
    <div class="s">Index-time chunking</div>
  </div>
  <div class="nr-metric">
    <div class="k">Active Scope</div>
    <div class="v">{active_scope_label}</div>
    <div class="s">Retrieval filter</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("""<div style="height:10px"></div>""", unsafe_allow_html=True)

# ----------------------------
# Empty state hero
# ----------------------------
if not docs and not st.session_state.get("messages"):
    st.markdown(
        """
<div class="nr-panel nr-hero">
  <h2>Upload a PDF and start chatting with your documents</h2>
  <p>NotebookRAG indexes your PDFs locally into ChromaDB and answers only from retrieved context — with page + chunk citations.</p>
  <div class="nr-chip-row">
    <div class="nr-chip">Summarize in 5 bullets</div>
    <div class="nr-chip">Create flashcards</div>
    <div class="nr-chip">Extract key definitions</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    if c1.button("Summarize in 5 bullets", use_container_width=True):
        st.session_state["prefill"] = "Summarize this document in 5 bullet points with page references."
        st.rerun()
    if c2.button("Make flashcards", use_container_width=True):
        st.session_state["prefill"] = "Create 10 Q/A flashcards from the document with page citations."
        st.rerun()
    if c3.button("Find key definitions", use_container_width=True):
        st.session_state["prefill"] = "List key terms and their definitions, and cite the pages where they appear."
        st.rerun()


# ----------------------------
# Indexing upload queue
# ----------------------------
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
                chunk_strategy=st.session_state["ui_chunk_strategy"],
                chunk_size=int(st.session_state["ui_chunk_size"]),
                chunk_overlap=int(st.session_state["ui_chunk_overlap"]),
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


# Refresh docs/stats after indexing
try:
    docs = pipe.vector_store.list_documents_with_stats()
except Exception:
    docs = pipe.vector_store.list_documents()


# ----------------------------
# Main layout: chat + sources
# ----------------------------
chat_col, cite_col = st.columns([1.65, 1.0], gap="large")

with chat_col:
    st.markdown("""<div class="nr-panel nr-chat-shell">""", unsafe_allow_html=True)

    # Render chat history
    for m in st.session_state["messages"]:
        role = m.get("role")
        if role == "assistant":
            # assistant answer card; sources and context shown on right + expander
            render_chat_message(m)
        else:
            # user bubble with markdown
            ts = m.get("ts", "")
            content = m.get("content", "") or ""
            st.markdown(
                f"""
<div class="nr-row user">
  <div class="nr-bubble user">
    <div class="nr-md">{content}</div>
    <div class="nr-meta">you • {ts}</div>
  </div>
  <div class="nr-avatar user">Y</div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # Sticky input bar
    st.markdown("""<div class="nr-inputbar">""", unsafe_allow_html=True)

    prefill = st.session_state.pop("prefill", "") if "prefill" in st.session_state else ""
    q_in = st.chat_input("Ask a question about your PDFs…")
    if prefill:
        q_in = prefill

    st.markdown("</div>", unsafe_allow_html=True)

with cite_col:
    st.markdown("""<div class="nr-panel" style="padding:14px 14px;">""", unsafe_allow_html=True)
    render_sources_panel(st.session_state.get("last_hits") or [])

    with st.expander("Retrieved context", expanded=False):
        ctx = st.session_state.get("last_context") or ""
        if not ctx:
            st.caption("No retrieved context yet.")
        else:
            st.text(ctx[:8000])

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# Answering logic
# ----------------------------
def _typing_html() -> str:
    return """
<div class="nr-row">
  <div class="nr-avatar ai">AI</div>
  <div class="nr-bubble ai">
    <div class="nr-typing">
      <span style="color:var(--muted); font-weight:600;">Thinking</span>
      <span class="nr-dot"></span><span class="nr-dot"></span><span class="nr-dot"></span>
    </div>
  </div>
</div>
"""


if q_in:
    q = (q_in or "").strip()
    if not q:
        st.stop()

    append_message("user", q)

    # Typing animation
    with chat_col:
        typing = st.empty()
        typing.markdown(_typing_html(), unsafe_allow_html=True)

    try:
        top_k = int(st.session_state["ui_top_k"])
        active_doc = st.session_state.get("active_doc")

        answer, hits = pipe.answer(query=q, top_k=top_k, active_document_id=active_doc)

        cites: List[Dict[str, Any]] = []
        ctx_chunks: List[str] = []
        best_sim: Optional[float] = None

        for h in hits:
            md = h.metadata or {}
            sim = h.similarity
            if sim is not None:
                best_sim = sim if best_sim is None else max(best_sim, sim)

            cites.append(
                {
                    "filename": md.get("filename", ""),
                    "page": md.get("page", "?"),
                    "chunk_index": md.get("chunk_index", md.get("chunk", md.get("chunk_idx", "?"))),
                    "chunk_id": md.get("chunk_id", ""),
                    "similarity_pct": "?" if sim is None else int(round(sim * 100)),
                    "text": h.text,
                }
            )
            ctx_chunks.append(
                f"[{md.get('filename','')}] page {md.get('page','?')} • chunk {md.get('chunk_index', '?')}:\n{h.text}"
            )

        retrieved_context = "\n\n---\n\n".join(ctx_chunks)
        st.session_state["last_hits"] = cites
        st.session_state["last_context"] = retrieved_context

        # Normalize answer fallback
        fallback = "I could not find that in uploaded documents."
        if not isinstance(answer, str) or not answer.strip():
            answer = fallback
        if answer.strip().lower().startswith("i don't know"):
            answer = fallback

        conf_label, conf_bg = _confidence_badge(best_sim)

        typing.empty()
        append_message(
            "assistant",
            answer,
            citations=cites,
            context=retrieved_context,
            answer_meta={"confidence_label": conf_label, "confidence_bg": conf_bg},
        )
        st.rerun()

    except Exception as e:
        log.exception("answering failed")
        typing.empty()
        append_message(
            "assistant",
            "I ran into an error while answering. Please try again.",
            answer_meta={"confidence_label": "Error", "confidence_bg": "rgba(239,68,68,0.16)"},
        )
        toast(f"Error: {e}", kind="error")
        st.rerun()