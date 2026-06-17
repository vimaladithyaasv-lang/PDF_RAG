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
- `eval_rag.py` — evaluation harness (retrieval recall@k + LLM-judged faithfulness/relevance); see Evaluation section below.
- `eval_testset.example.json` — example test set for `eval_rag.py`.

Run tests with:

```bash
pytest test_rag.py -v
```

## Configuration

All tunables are environment variables (see `.env.example`):
`GROQ_API_KEY` (required), `GROQ_MODEL`, `GROQ_TIMEOUT_SECONDS`,
`GROQ_MAX_RETRIES`, `MAX_PDF_SIZE_BYTES`, `MAX_PDF_PAGES`,
`RELEVANCE_THRESHOLD`, `MAX_CONTEXT_TOKENS`, `HYBRID_DENSE_WEIGHT`, `RAG_INDEX_DIR`.

## Retrieval

Retrieval is hybrid: dense cosine similarity (sentence embeddings) combined
with BM25 lexical scoring, reranked together. This catches both paraphrased
queries (dense) and exact keyword/ID/code matches (BM25) that pure dense
search tends to miss. Set `use_hybrid=False` on `retrieve()` to fall back to
pure dense search. Context passed to the LLM is trimmed to fit
`MAX_CONTEXT_TOKENS` (estimated at ~4 chars/token), so a large `top_k` won't
silently overflow the model's context window.

## Document management

Re-uploading a PDF with the same filename replaces its existing chunks
in-place (`add_pdf(..., replace=True)`, the default) rather than duplicating
them. Call `engine.remove_document(filename)` directly to delete a document's
chunks without replacing them. Both rebuild the FAISS index from the
surviving chunks, since flat FAISS indexes don't support row deletion —
fine up to a few hundred thousand chunks; swap to an ID-mapped index type
for larger scale.

## Evaluation

`eval_rag.py` is a lightweight harness: retrieval recall@k against a labeled
testset, plus an LLM-judged faithfulness/relevance check (does the generated
answer stay within what the retrieved context actually supports, and does it
address the question asked). Not a substitute for a full framework like
RAGAS once the project's scale justifies one.

```bash
python eval_rag.py --testset eval_testset.example.json
python eval_rag.py --testset eval_testset.example.json --no-judge   # retrieval-only, no extra LLM calls
python eval_rag.py --testset eval_testset.example.json --out report.json
```

Test set format (see `eval_testset.example.json`); `expected_source` and
`top_k` are both optional per-case — omit `expected_source` to evaluate
answer quality only, without checking retrieval recall:
```json
[
  {"query": "What is the warranty period?", "expected_source": "manual.pdf"},
  {"query": "What is the capital of France?"}
]
```

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

- In-memory FAISS index, rebuilt from scratch on delete — fine for a single
  user or small team and up to a few hundred thousand chunks; for true
  multi-user production use, separate indexes per user/tenant and consider a
  managed vector DB or an ID-mapped FAISS index for larger-scale deletes.
- No OCR fallback for scanned/image-only PDFs (text extraction only).
- No query rewriting from chat history — very short follow-ups like "tell me
  more" are embedded literally.
- Image indexing (`index_images=True`) stores a text placeholder per image
  ("Image on page N"), not the image's actual visual content — there's no
  multimodal embedding/captioning step, so images aren't searchable by what
  they depict.
