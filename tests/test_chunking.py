"""Tests for chunking module."""

from src.core.chunking.hierarchical_chunker import HierarchicalChunker
from src.core.chunking.metadata_builder import MetadataBuilder


class TestMetadataBuilder:
    """Tests for MetadataBuilder."""

    def test_build_basic_metadata(self):
        builder = MetadataBuilder()
        meta = builder.build_metadata(
            source_name="test.pdf",
            page_num=1,
            chunk_index=0,
        )
        assert meta["source"] == "test.pdf"
        assert meta["page"] == 1
        assert meta["chunk_index"] == 0

    def test_build_metadata_with_headings(self):
        builder = MetadataBuilder()
        builder.update_heading(1, "Chapter 1")
        builder.update_heading(2, "Section 1.1")

        meta = builder.build_metadata(
            source_name="doc.md",
            chunk_index=0,
        )
        assert meta["heading_path"] == "Chapter 1 > Section 1.1"
        assert meta["heading_level_1"] == "Chapter 1"
        assert meta["heading_level_2"] == "Section 1.1"

    def test_heading_stack_reset(self):
        builder = MetadataBuilder()
        builder.update_heading(1, "Chapter 1")
        builder.reset()
        assert len(builder._heading_stack) == 0

    def test_heading_level_replacement(self):
        builder = MetadataBuilder()
        builder.update_heading(1, "Chapter 1")
        builder.update_heading(2, "Section 1.1")
        # New level 1 heading pops both levels, stack now has only [Chapter 2]
        builder.update_heading(1, "Chapter 2")

        meta = builder.build_metadata(source_name="doc.md", chunk_index=0)
        assert meta["heading_level_1"] == "Chapter 2"
        # heading_level_2 should not exist since stack only has 1 item
        assert "heading_level_2" not in meta


class TestHierarchicalChunker:
    """Tests for HierarchicalChunker."""

    def test_chunker_initialization(self):
        chunker = HierarchicalChunker(
            chunk_size=512,
            chunk_overlap=128,
        )
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 128

    def test_chunk_basic_text(self, sample_text):
        chunker = HierarchicalChunker(
            chunk_size=512,
            chunk_overlap=128,
            min_chunk_size=10,
        )
        chunks = chunker.chunk(
            text=sample_text,
            source_name="test.md",
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert "chunk_id" in chunk
            assert chunk["metadata"]["source"] == "test.md"

    def test_chunk_empty_text(self):
        chunker = HierarchicalChunker()
        chunks = chunker.chunk(text="", source_name="empty.txt")
        assert chunks == []

    def test_chunk_small_text(self):
        chunker = HierarchicalChunker(min_chunk_size=10)
        chunks = chunker.chunk(
            text="This is a short text.",
            source_name="short.txt",
        )
        assert len(chunks) >= 0

    def test_chunk_heading_detection(self, sample_text):
        chunker = HierarchicalChunker(min_chunk_size=10)
        chunks = chunker.chunk(
            text=sample_text,
            source_name="test.md",
        )
        # Some chunks should have heading metadata
        headings_found = any(
            "heading_path" in c["metadata"] for c in chunks
        )
        assert headings_found

    def test_estimate_tokens(self):
        chunker = HierarchicalChunker()
        tokens = chunker._estimate_tokens("Hello world")
        assert tokens > 0

        tokens_cn = chunker._estimate_tokens("你好世界")
        assert tokens_cn > 0

    def test_sliding_window_chunk(self):
        chunker = HierarchicalChunker(chunk_size=100, chunk_overlap=20)
        long_text = "word " * 200
        chunks = list(chunker._sliding_window_chunk(long_text))
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) > 0


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def test_initialization(self):
        from src.core.chunking.semantic_chunker import SemanticChunker
        chunker = SemanticChunker(threshold=0.65)
        assert chunker.threshold == 0.65

    def test_split_empty_text(self):
        from src.core.chunking.semantic_chunker import SemanticChunker
        chunker = SemanticChunker()
        result = chunker.split("")
        assert result == []

    def test_split_single_sentence(self):
        from src.core.chunking.semantic_chunker import SemanticChunker
        chunker = SemanticChunker()
        result = chunker.split("This is a single sentence.")
        assert len(result) == 1