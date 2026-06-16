# PDF RAG Chat

Upload PDFs and ask questions about them. Retrieval-augmented generation
pipeline: PDF text extraction → chunking → embedding (Sentence-Transformers)
→ FAISS vector search (cosine similarity) → answer generation via Groq.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and set GROQ_API_KEY
export $(grep -v '^#' .env | xargs)   # or use a tool like python-dotenv / direnv
streamlit run app.py
```

## Project layout

- `rag_engine.py` — core RAG pipeline (PDF parsing, chunking, embedding, FAISS index, Groq call). No UI dependencies; fully unit-testable.
- `app.py` — Streamlit chat UI built on top of `RAGEngine`.
- `test_rag.py` — pytest suite (42 tests) covering chunking, indexing, retrieval, PDF extraction, and the answer pipeline with mocked Groq/embedding calls.

Run tests with:

```bash
pytest test_rag.py -v
```

## Configuration

All tunables are environment variables (see `.env.example`):
`GROQ_API_KEY` (required), `GROQ_MODEL`, `GROQ_TIMEOUT_SECONDS`,
`GROQ_MAX_RETRIES`, `MAX_PDF_SIZE_BYTES`, `MAX_PDF_PAGES`,
`RELEVANCE_THRESHOLD`, `RAG_INDEX_DIR`.

## Persistence

The FAISS index and chunk metadata are saved to `RAG_INDEX_DIR` (default
`.rag_index/`) after each successful upload, and reloaded automatically on
startup, so indexed documents survive an app restart.

## Security note

An earlier version of this repository contained a previous, unrelated
Excel-dashboard `app.py` with a **hardcoded MySQL password committed to git
history**. That dashboard has been removed entirely — the project no longer
uses MySQL. If that password is still in use anywhere, rotate it; it should
be treated as compromised since it was pushed to a public repo. Going
forward, all credentials are sourced from environment variables only and
`.env` is git-ignored.

## Known limitations

- In-memory/single-process FAISS index — fine for a single user or small
  team; for multi-user production use, separate indexes per user/tenant and
  consider a managed vector DB for scale beyond a few hundred thousand chunks.
- No OCR fallback for scanned/image-only PDFs (text extraction only).
- No query rewriting from chat history — very short follow-ups like "tell me
  more" are embedded literally.
