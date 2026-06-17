"""
test_rag.py — Pytest test cases for RAG PDF Q&A project
Tests: chunking, embedding, FAISS indexing, retrieval, PDF extraction, stats

Run:
    pip install pytest
    pytest test_rag.py -v

Note: These tests do NOT require a Groq API key.
      Groq API calls are mocked using unittest.mock.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Patch Groq and TextEmbedding before importing rag_engine
with patch("rag_engine.Groq"), patch("rag_engine.TextEmbedding"):
    from rag_engine import RAGEngine, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, MIN_IMAGE_SIZE


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    """RAGEngine with mocked Groq and embed model. No real API calls."""
    with patch("rag_engine.Groq"), patch("rag_engine.TextEmbedding") as mock_embed_cls:
        instance = mock_embed_cls.return_value
        # Returns deterministic 384-dim float32 vectors
        instance.embed.side_effect = lambda texts: [
            np.random.rand(384).astype("float32") for _ in texts
        ]
        eng = RAGEngine(groq_api_key="fake_key_for_testing")
        eng.embed_model = instance
        return eng


@pytest.fixture
def sample_pdf_bytes():
    """Minimal valid single-page PDF with text content."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
        b"   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello World Test Page) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000360 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n441\n%%EOF\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Chunking Tests (TC-001 to TC-007)
# ══════════════════════════════════════════════════════════════════════════════

class TestChunking:

    def test_TC001_long_text_produces_multiple_chunks(self, engine):
        """TC-001: Text longer than CHUNK_SIZE must be split into multiple chunks."""
        text = "A" * 1200
        chunks = engine.chunk_text(text, page_num=1, filename="test.pdf")
        assert len(chunks) > 1

    def test_TC002_chunk_does_not_exceed_max_size(self, engine):
        """TC-002: No single chunk should exceed CHUNK_SIZE characters."""
        text = "Word " * 300
        chunks = engine.chunk_text(text, page_num=1, filename="test.pdf")
        for chunk_text, _, _ in chunks:
            assert len(chunk_text) <= CHUNK_SIZE

    def test_TC003_adjacent_chunks_overlap(self, engine):
        """TC-003: Adjacent chunks must share CHUNK_OVERLAP characters."""
        text = "X" * 700
        chunks = engine.chunk_text(text, page_num=1, filename="test.pdf")
        if len(chunks) >= 2:
            assert chunks[0][0][-CHUNK_OVERLAP:] == chunks[1][0][:CHUNK_OVERLAP]

    def test_TC004_short_text_produces_one_chunk(self, engine):
        """TC-004: Text shorter than CHUNK_SIZE should produce exactly 1 chunk."""
        text = "Short text."
        chunks = engine.chunk_text(text, page_num=1, filename="test.pdf")
        assert len(chunks) == 1

    def test_TC005_chunk_stores_filename(self, engine):
        """TC-005: Chunk metadata must store the correct filename."""
        chunks = engine.chunk_text("Some text.", page_num=1, filename="myfile.pdf")
        for _, fname, _ in chunks:
            assert fname == "myfile.pdf"

    def test_TC006_chunk_stores_page_number(self, engine):
        """TC-006: Chunk metadata must store the correct page number."""
        chunks = engine.chunk_text("Some text.", page_num=7, filename="test.pdf")
        for _, _, pnum in chunks:
            assert pnum == 7

    def test_TC007_empty_text_produces_no_chunks(self, engine):
        """TC-007: Empty or whitespace-only text should produce zero chunks."""
        assert engine.chunk_text("", 1, "f.pdf") == []
        assert engine.chunk_text("   \n\t  ", 1, "f.pdf") == []


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: FAISS Indexing Tests (TC-008 to TC-015)
# ══════════════════════════════════════════════════════════════════════════════

class TestFAISSIndexing:

    def test_TC008_index_is_none_initially(self, engine):
        """TC-008: Before any indexing, index must be None and is_ready False."""
        assert engine.index is None
        assert engine.total_chunks == 0
        assert engine.is_ready is False

    def test_TC009_index_created_after_add(self, engine):
        """TC-009: _add_to_index must create FAISS index and store vectors."""
        engine._add_to_index(["hello", "world"], [("f.pdf", 1, "text")] * 2)
        assert engine.index is not None
        assert engine.index.ntotal == 2

    def test_TC010_is_ready_true_after_indexing(self, engine):
        """TC-010: is_ready must return True after successful indexing."""
        engine._add_to_index(["text"], [("f.pdf", 1, "text")])
        assert engine.is_ready is True

    def test_TC011_total_chunks_correct_count(self, engine):
        """TC-011: total_chunks must equal the number of indexed texts."""
        texts   = ["a", "b", "c", "d"]
        sources = [("f.pdf", 1, "text")] * 4
        engine._add_to_index(texts, sources)
        assert engine.total_chunks == 4

    def test_TC012_batch_indexing_handles_large_input(self, engine):
        """TC-012: Batch indexing with 200 chunks must not raise MemoryError."""
        texts   = [f"chunk {i}" for i in range(200)]
        sources = [("big.pdf", i % 10 + 1, "text") for i in range(200)]
        engine._add_to_index(texts, sources, batch_size=32)
        assert engine.total_chunks == 200

    def test_TC013_indexed_files_sorted_unique(self, engine):
        """TC-013: indexed_files() must return sorted, unique filenames."""
        engine._add_to_index(
            ["t1", "t2", "t3"],
            [("b.pdf", 1, "text"), ("a.pdf", 1, "text"), ("b.pdf", 2, "text")]
        )
        assert engine.indexed_files() == ["a.pdf", "b.pdf"]

    def test_TC014_stats_text_and_image_counts(self, engine):
        """TC-014: stats() must accurately count text vs image chunks."""
        engine._add_to_index(
            ["text1", "text2", "img_desc"],
            [("f.pdf", 1, "text"), ("f.pdf", 2, "text"), ("f.pdf", 3, "image")]
        )
        s = engine.stats()
        assert s["text"] == 2
        assert s["images"] == 1

    def test_TC015_multiple_adds_accumulate(self, engine):
        """TC-015: Multiple _add_to_index calls must accumulate all chunks."""
        engine._add_to_index(["batch1"], [("a.pdf", 1, "text")])
        engine._add_to_index(["batch2"], [("b.pdf", 1, "text")])
        engine._add_to_index(["batch3"], [("c.pdf", 1, "text")])
        assert engine.total_chunks == 3
        assert len(engine.indexed_files()) == 3


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Retrieval Tests (TC-016 to TC-021)
# ══════════════════════════════════════════════════════════════════════════════

class TestRetrieval:

    def test_TC016_retrieve_empty_index_returns_empty_list(self, engine):
        """TC-016: retrieve() on empty index must return []."""
        assert engine.retrieve("anything") == []

    def test_TC017_retrieve_returns_at_most_top_k(self, engine):
        """TC-017: retrieve() must return at most TOP_K results."""
        texts   = [f"doc {i}" for i in range(20)]
        sources = [("f.pdf", i+1, "text") for i in range(20)]
        engine._add_to_index(texts, sources)
        results = engine.retrieve("query", top_k=5)
        assert len(results) <= 5

    def test_TC018_retrieve_result_is_5_tuple(self, engine):
        """TC-018: Each result must be a 5-tuple: (chunk, fname, pnum, dist, ctype)."""
        engine._add_to_index(["hello world"], [("test.pdf", 1, "text")])
        results = engine.retrieve("hello")
        chunk, fname, pnum, dist, ctype = results[0]
        assert isinstance(chunk, str)
        assert isinstance(fname, str)
        assert isinstance(pnum, int)
        assert isinstance(dist, float)
        assert ctype in ("text", "image")

    def test_TC019_retrieve_fewer_chunks_than_top_k(self, engine):
        """TC-019: retrieve() must handle index smaller than top_k gracefully."""
        engine._add_to_index(["only one"], [("f.pdf", 1, "text")])
        results = engine.retrieve("query", top_k=10)
        assert len(results) == 1

    def test_TC020_retrieve_distances_non_negative(self, engine):
        """TC-020: All L2 distances must be >= 0."""
        texts   = ["cat", "dog", "bird"]
        sources = [("f.pdf", i+1, "text") for i in range(3)]
        engine._add_to_index(texts, sources)
        for _, _, _, dist, _ in engine.retrieve("animal"):
            assert dist >= 0

    def test_TC021_retrieve_ctype_valid_values(self, engine):
        """TC-021: ctype in results must only be 'text' or 'image'."""
        engine._add_to_index(
            ["text chunk", "image desc"],
            [("f.pdf", 1, "text"), ("f.pdf", 2, "image")]
        )
        for _, _, _, _, ctype in engine.retrieve("query"):
            assert ctype in ("text", "image")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PDF Extraction Tests (TC-022 to TC-026)
# ══════════════════════════════════════════════════════════════════════════════

class TestPDFExtraction:

    def test_TC022_extract_text_returns_list(self, engine, sample_pdf_bytes):
        """TC-022: extract_text_from_pdf must return a list."""
        pages = engine.extract_text_from_pdf(sample_pdf_bytes, "test.pdf")
        assert isinstance(pages, list)

    def test_TC023_page_numbers_start_from_1(self, engine, sample_pdf_bytes):
        """TC-023: All page numbers in extracted text must be >= 1."""
        pages = engine.extract_text_from_pdf(sample_pdf_bytes, "test.pdf")
        for _, page_num in pages:
            assert page_num >= 1

    def test_TC024_no_empty_text_pages(self, engine, sample_pdf_bytes):
        """TC-024: Extracted pages must not contain empty strings."""
        pages = engine.extract_text_from_pdf(sample_pdf_bytes, "test.pdf")
        for text, _ in pages:
            assert text.strip() != ""

    def test_TC025_extract_images_returns_list(self, engine, sample_pdf_bytes):
        """TC-025: extract_images_from_pdf must return a list."""
        images = engine.extract_images_from_pdf(sample_pdf_bytes, "test.pdf")
        assert isinstance(images, list)

    def test_TC026_image_tuples_have_correct_types(self, engine, sample_pdf_bytes):
        """TC-026: Image tuples must be (bytes, str starting with image/, int)."""
        images = engine.extract_images_from_pdf(sample_pdf_bytes, "test.pdf")
        for img_bytes, media_type, page_num in images:
            assert isinstance(img_bytes, bytes)
            assert media_type.startswith("image/")
            assert isinstance(page_num, int) and page_num >= 1


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: add_pdf Integration Tests (TC-027 to TC-031)
# ══════════════════════════════════════════════════════════════════════════════

class TestAddPDF:

    def test_TC027_add_pdf_returns_dict_with_correct_keys(self, engine, sample_pdf_bytes):
        """TC-027: add_pdf() must return dict with text_chunks and images keys."""
        result = engine.add_pdf(sample_pdf_bytes, "test.pdf", index_images=False)
        assert "text_chunks" in result
        assert "images" in result

    def test_TC028_images_zero_when_disabled(self, engine, sample_pdf_bytes):
        """TC-028: add_pdf with index_images=False must return images=0."""
        result = engine.add_pdf(sample_pdf_bytes, "test.pdf", index_images=False)
        assert result["images"] == 0

    def test_TC029_text_chunks_non_negative(self, engine, sample_pdf_bytes):
        """TC-029: text_chunks count must be a non-negative integer."""
        result = engine.add_pdf(sample_pdf_bytes, "test.pdf", index_images=False)
        assert isinstance(result["text_chunks"], int)
        assert result["text_chunks"] >= 0

    def test_TC030_is_ready_boolean(self, engine, sample_pdf_bytes):
        """TC-030: is_ready must be a boolean after add_pdf."""
        engine.add_pdf(sample_pdf_bytes, "test.pdf", index_images=False)
        assert isinstance(engine.is_ready, bool)

    def test_TC031_filename_in_indexed_files_if_text_found(self, engine, sample_pdf_bytes):
        """TC-031: If text was extracted, filename must appear in indexed_files()."""
        engine.add_pdf(sample_pdf_bytes, "myfile.pdf", index_images=False)
        if engine.is_ready:
            assert "myfile.pdf" in engine.indexed_files()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Answer / Groq API Tests (TC-032 to TC-036)
# ══════════════════════════════════════════════════════════════════════════════

class TestAnswer:

    def _add_wagoneer_chunks(self, engine):
        texts = [
            "The Wagoneer has a V8 engine with 392 horsepower.",
            "Keyless Enter n Go allows engine start without inserting the key.",
            "The fuel tank capacity is 26 gallons.",
            "Adaptive Cruise Control maintains speed and following distance.",
            "Towing capacity is up to 10000 pounds.",
        ]
        engine._add_to_index(texts, [("wagoneer.pdf", i+1, "text") for i in range(5)])

    def test_TC032_answer_empty_index_returns_not_found(self, engine):
        """TC-032: answer() on empty index must return not-found message and empty sources."""
        answer, sources = engine.answer("What is horsepower?")
        assert isinstance(answer, str) and len(answer) > 0
        assert sources == []

    def test_TC033_answer_calls_groq_when_chunks_exist(self, engine):
        """TC-033: answer() must call Groq API when chunks are retrieved."""
        self._add_wagoneer_chunks(engine)
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "392 horsepower."
        engine.groq_client.chat.completions.create.return_value = mock_resp
        engine.answer("What is the horsepower?")
        assert engine.groq_client.chat.completions.create.called

    def test_TC034_answer_returns_str_and_list(self, engine):
        """TC-034: answer() must return a (str, list) tuple."""
        self._add_wagoneer_chunks(engine)
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Some answer."
        engine.groq_client.chat.completions.create.return_value = mock_resp
        answer, sources = engine.answer("Tell me about the engine.")
        assert isinstance(answer, str)
        assert isinstance(sources, list)

    def test_TC035_sources_are_5_tuples(self, engine):
        """TC-035: Each source in answer() result must be a 5-tuple."""
        self._add_wagoneer_chunks(engine)
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Towing answer."
        engine.groq_client.chat.completions.create.return_value = mock_resp
        _, sources = engine.answer("What is towing capacity?")
        for item in sources:
            assert len(item) == 5
            _, _, _, dist, ctype = item
            assert dist >= 0
            assert ctype in ("text", "image")

    def test_TC036_answer_with_chat_history(self, engine):
        """TC-036: answer() must accept and use chat_history without error."""
        self._add_wagoneer_chunks(engine)
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Follow-up answer."
        engine.groq_client.chat.completions.create.return_value = mock_resp
        history = [
            {"role": "user", "content": "What is horsepower?"},
            {"role": "assistant", "content": "392 hp."}
        ]
        answer, _ = engine.answer("Tell me more.", chat_history=history)
        assert isinstance(answer, str)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Constants & Edge Case Tests (TC-037 to TC-042)
# ══════════════════════════════════════════════════════════════════════════════

class TestConstantsAndEdgeCases:

    def test_TC037_chunk_size_positive_int(self):
        """TC-037: CHUNK_SIZE must be a positive integer."""
        assert isinstance(CHUNK_SIZE, int) and CHUNK_SIZE > 0

    def test_TC038_overlap_less_than_chunk_size(self):
        """TC-038: CHUNK_OVERLAP must be less than CHUNK_SIZE to avoid infinite loop."""
        assert CHUNK_OVERLAP < CHUNK_SIZE

    def test_TC039_top_k_positive_int(self):
        """TC-039: TOP_K must be a positive integer."""
        assert isinstance(TOP_K, int) and TOP_K > 0

    def test_TC040_min_image_size_positive(self):
        """TC-040: MIN_IMAGE_SIZE must be a positive integer."""
        assert isinstance(MIN_IMAGE_SIZE, int) and MIN_IMAGE_SIZE > 0

    def test_TC041_retrieve_top_k_larger_than_index(self, engine):
        """TC-041: retrieve() with top_k > index size must not crash."""
        engine._add_to_index(["one chunk"], [("f.pdf", 1, "text")])
        results = engine.retrieve("query", top_k=999)
        assert len(results) == 1

    def test_TC042_stats_empty_engine(self, engine):
        """TC-042: stats() on empty engine must return zeros."""
        s = engine.stats()
        assert s["text"] == 0
        assert s["images"] == 0
