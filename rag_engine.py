"""
rag_engine.py — Core RAG pipeline
PDF ingestion → chunking → embedding → FAISS index → Groq LLM answer
"""

import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from typing import List, Tuple
import re

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL       = "llama3-8b-8192"
TOP_K            = 5
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 80


class RAGEngine:
    def __init__(self, groq_api_key: str):
        self.groq_client   = Groq(api_key=groq_api_key)
        self.embed_model   = SentenceTransformer(EMBED_MODEL_NAME)
        self.index         = None
        self.chunks        = []
        self.chunk_sources = []

    def extract_text_from_pdf(self, pdf_bytes, filename):
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                pages.append((text, page_num))
        return pages

    def chunk_text(self, text, page_num, filename):
        chunks = []
        start  = 0
        while start < len(text):
            end   = start + CHUNK_SIZE
            chunk = text[start:end].strip()
            if chunk:
                chunks.append((chunk, filename, page_num))
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def add_pdf(self, pdf_bytes, filename):
        new_chunks  = []
        new_sources = []

        pages = self.extract_text_from_pdf(pdf_bytes, filename)
        for page_text, page_num in pages:
            for chunk, fname, pnum in self.chunk_text(page_text, page_num, filename):
                new_chunks.append(chunk)
                new_sources.append((fname, pnum))

        if not new_chunks:
            return 0

        embeddings = self.embed_model.encode(
            new_chunks, show_progress_bar=False, convert_to_numpy=True
        ).astype("float32")

        if self.index is None:
            dim        = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        self.chunk_sources.extend(new_sources)
        return len(new_chunks)

    def retrieve(self, query, top_k=TOP_K):
        if self.index is None or self.index.ntotal == 0:
            return []

        q_emb = self.embed_model.encode([query], convert_to_numpy=True).astype("float32")
        k     = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(q_emb, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            fname, pnum = self.chunk_sources[idx]
            results.append((self.chunks[idx], fname, pnum, float(dist)))
        return results

    def answer(self, query, chat_history=None, top_k=TOP_K):
        retrieved = self.retrieve(query, top_k)

        if not retrieved:
            return "I couldn't find relevant information in the uploaded document(s).", []

        context_parts = []
        for i, (chunk, fname, pnum, _dist) in enumerate(retrieved, 1):
            context_parts.append(f"[Chunk {i} | {fname} p.{pnum}]\n{chunk}")
        context = "\n\n---\n\n".join(context_parts)

        system_prompt = (
            "You are a precise, helpful document assistant. "
            "Answer the user's question using ONLY the context provided below. "
            "If the answer is not in the context, say so clearly.\n\n"
            f"CONTEXT:\n{context}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            messages.extend(chat_history[-6:])
        messages.append({"role": "user", "content": query})

        response = self.groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip(), retrieved

    @property
    def is_ready(self):
        return self.index is not None and self.index.ntotal > 0

    @property
    def total_chunks(self):
        return len(self.chunks)

    def indexed_files(self):
        return sorted(set(fname for fname, _ in self.chunk_sources))
