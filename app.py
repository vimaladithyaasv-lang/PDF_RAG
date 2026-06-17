"""
app.py — PDF RAG Chat
Streamlit front-end that wires the file upload, chat UI, and source display
directly to RAGEngine (rag_engine.py). This replaces the previous app.py,
which was an unrelated Excel→MySQL dashboard that never called RAGEngine.

Run:
    export GROQ_API_KEY=your_key_here
    streamlit run app.py
"""

import os
import streamlit as st

from rag_engine import (
    RAGEngine,
    PDFTooLargeError,
    PDFParseError,
    MAX_PDF_SIZE_BYTES,
)

INDEX_DIR = os.environ.get("RAG_INDEX_DIR", ".rag_index")

st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="wide")

# ─────────────────────────────────────────────────────────────────────────
# ENGINE INITIALIZATION (cached across reruns, persisted across restarts)
# ─────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_engine():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    eng = RAGEngine(groq_api_key=api_key)
    eng.load_index(INDEX_DIR)  # no-op if nothing saved yet
    return eng


engine = get_engine()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [{"role": ..., "content": ...}, ...]

# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR — API key check, upload, indexed files
# ─────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📄 PDF RAG Chat")

    if engine is None:
        st.error(
            "GROQ_API_KEY is not set. Set it as an environment variable "
            "before starting the app, e.g.:\n\n"
            "`export GROQ_API_KEY=your_key_here`"
        )
        st.stop()

    st.caption(
        f"Max file size: {MAX_PDF_SIZE_BYTES // (1024*1024)} MB per PDF."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        for uf in uploaded_files:
            with st.spinner(f"Indexing {uf.name}…"):
                try:
                    pdf_bytes = uf.read()
                    already_had = uf.name in engine.indexed_files()
                    result = engine.add_pdf(
                        pdf_bytes, uf.name, index_images=False, replace=True
                    )
                    if result["text_chunks"] == 0:
                        st.warning(
                            f"'{uf.name}': no extractable text found "
                            "(it may be a scanned/image-only PDF)."
                        )
                    else:
                        verb = "Re-indexed" if already_had else "Indexed"
                        st.success(
                            f"{verb} '{uf.name}': {result['text_chunks']} chunk(s)."
                        )
                        engine.save_index(INDEX_DIR)
                except PDFTooLargeError as e:
                    st.error(str(e))
                except PDFParseError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Unexpected error indexing '{uf.name}': {e}")

    st.divider()
    st.subheader("Indexed documents")
    if engine.is_ready:
        for fname in engine.indexed_files():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"- {fname}")
            with col2:
                if st.button("🗑️", key=f"remove_{fname}", help=f"Remove {fname}"):
                    engine.remove_document(fname)
                    engine.save_index(INDEX_DIR)
                    st.rerun()
        st.caption(f"{engine.total_chunks} chunk(s) total")
    else:
        st.caption("No documents indexed yet.")

    if st.button("🗑️ Clear all documents"):
        st.session_state.chat_history = []
        get_engine.clear()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────
# MAIN — chat interface
# ─────────────────────────────────────────────────────────────────────────

st.title("Ask your documents")

if not engine.is_ready:
    st.info("Upload one or more PDFs in the sidebar to get started.")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input(
    "Ask a question about your documents…",
    disabled=not engine.is_ready,
)

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer_text, sources = engine.answer(
                query,
                chat_history=st.session_state.chat_history[:-1],
            )
        st.markdown(answer_text)

        if sources:
            with st.expander(f"📑 {len(sources)} source(s) used"):
                for i, (chunk, fname, pnum, dist, ctype) in enumerate(sources, 1):
                    st.markdown(f"**{i}. {fname} — page {pnum}** ({ctype})")
                    preview = chunk if len(chunk) <= 400 else chunk[:400] + "…"
                    st.caption(preview)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer_text}
    )
