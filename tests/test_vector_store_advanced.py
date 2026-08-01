"""Advanced vector store tests for VectorStore.

Tests chunk removal, single chunk retrieval, statistics, pre-computed
embedding search, batch operations, metadata sanitization, corrupted
database handling, HNSW parameters, clear/reset, and counting.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.indexing.vector_store import _CHROMA_METADATA_TYPES, VectorStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield tmpdir


@pytest.fixture
def empty_store(temp_dir):
    """Create an empty VectorStore."""
    store = VectorStore(
        persist_directory=os.path.join(temp_dir, "chroma_vs"),
        collection_name="test_vs",
        embedding_model="BAAI/bge-small-zh-v1.5",
    )
    store.clear()
    return store


@pytest.fixture
def populated_store(temp_dir):
    """Create a VectorStore with pre-loaded chunks."""
    store = VectorStore(
        persist_directory=os.path.join(temp_dir, "chroma_vs_populated"),
        collection_name="test_vs_populated",
        embedding_model="BAAI/bge-small-zh-v1.5",
    )
    store.clear()
    chunks = [
        {
            "chunk_id": f"chunk_{i}",
            "text": f"This is test chunk {i} about topic {i % 5}. "
                    f"It contains content for vector store testing.",
            "metadata": {"source": f"doc_{i//3}.txt", "chunk_idx": i, "topic": str(i % 5)},
        }
        for i in range(10)
    ]
    store.add_chunks(chunks)
    return store


# ---------------------------------------------------------------------------
# Remove chunks tests
# ---------------------------------------------------------------------------

class TestRemoveChunks:
    """Test remove_chunks for specific chunks."""

    def test_remove_existing_chunks(self, populated_store):
        """Test removing specific existing chunks."""
        initial_count = populated_store.count()
        assert initial_count == 10, f"Should have 10 chunks initially, got {initial_count}"

        removed = populated_store.remove_chunks(["chunk_0", "chunk_1", "chunk_2"])
        assert removed == 3, f"Should remove 3 chunks, got {removed}"
        assert populated_store.count() == 7, "Count should be 7 after removal"

    def test_remove_nonexistent_chunks(self, populated_store):
        """Test removing chunk IDs that don't exist."""
        removed = populated_store.remove_chunks(["nonexistent_1", "nonexistent_2"])
        assert removed == 0, "Should remove 0 nonexistent chunks"

    def test_remove_empty_list(self, populated_store):
        """Test removing with empty chunk_id list."""
        removed = populated_store.remove_chunks([])
        assert removed == 0, "Empty list should remove 0 chunks"

    def test_remove_mixed_existing_and_nonexistent(self, populated_store):
        """Test removing mix of existing and nonexistent chunk IDs."""
        removed = populated_store.remove_chunks(["chunk_0", "nonexistent", "chunk_5"])
        assert removed == 2, "Should remove only the 2 existing chunks"

    def test_remove_all_chunks(self, populated_store):
        """Test removing all chunks."""
        all_ids = [f"chunk_{i}" for i in range(10)]
        removed = populated_store.remove_chunks(all_ids)
        assert removed == 10, "Should remove all 10 chunks"
        assert populated_store.count() == 0, "Store should be empty"

    def test_remove_from_empty_store(self, empty_store):
        """Test removing chunks from an empty store."""
        removed = empty_store.remove_chunks(["chunk_0"])
        assert removed == 0, "Should remove 0 from empty store"


# ---------------------------------------------------------------------------
# Get chunk tests
# ---------------------------------------------------------------------------

class TestGetChunk:
    """Test get_chunk for single chunk retrieval."""

    def test_get_existing_chunk(self, populated_store):
        """Test retrieving an existing chunk by ID."""
        chunk = populated_store.get_chunk("chunk_5")
        assert chunk is not None, "Should find existing chunk"
        assert chunk["chunk_id"] == "chunk_5", "Should have correct chunk_id"
        assert "text" in chunk, "Should have text"
        assert "metadata" in chunk, "Should have metadata"
        assert "embedding" in chunk, "Should have embedding"
        assert len(chunk["text"]) > 0, "Text should not be empty"

    def test_get_nonexistent_chunk(self, populated_store):
        """Test retrieving a nonexistent chunk."""
        chunk = populated_store.get_chunk("nonexistent_chunk")
        assert chunk is None, "Should return None for nonexistent chunk"

    def test_get_chunk_from_empty_store(self, empty_store):
        """Test get_chunk on empty store."""
        chunk = empty_store.get_chunk("any_id")
        assert chunk is None, "Should return None on empty store"

    def test_get_chunk_returns_embedding(self, populated_store):
        """Test that get_chunk returns embedding data."""
        chunk = populated_store.get_chunk("chunk_0")
        assert chunk is not None, "Should find chunk"
        embedding = chunk.get("embedding")
        assert embedding is not None, "Should have embedding"
        assert isinstance(embedding, list), "Embedding should be a list"
        assert len(embedding) > 0, "Embedding should not be empty"


# ---------------------------------------------------------------------------
# Get stats tests
# ---------------------------------------------------------------------------

class TestGetStats:
    """Test get_stats returns correct information."""

    def test_get_stats_populated(self, populated_store):
        """Test get_stats on populated store."""
        stats = populated_store.get_stats()
        assert stats["collection_name"] == "test_vs_populated", "Correct collection name"
        assert stats["total_chunks"] == 10, "Should have 10 chunks"
        assert stats["embedding_dimension"] > 0, "Should have positive embedding dimension"
        assert "persist_directory" in stats, "Should have persist_directory"
        assert "hnsw_config" in stats, "Should have hnsw_config"

    def test_get_stats_empty(self, empty_store):
        """Test get_stats on empty store."""
        stats = empty_store.get_stats()
        assert stats["total_chunks"] == 0, "Empty store should have 0 chunks"
        assert stats["embedding_dimension"] == 0, "Empty store should have 0 embedding dimension"
        assert stats["collection_name"] == "test_vs", "Correct collection name"

    def test_get_stats_hnsw_config(self, temp_dir):
        """Test get_stats returns correct HNSW configuration."""
        store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_hnsw"),
            collection_name="test_hnsw",
            hnsw_ef_construction=200,
            hnsw_M=16,
            hnsw_ef_search=100,
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        stats = store.get_stats()
        assert stats["hnsw_config"]["ef_construction"] == 200, "Correct ef_construction"
        assert stats["hnsw_config"]["M"] == 16, "Correct M"
        assert stats["hnsw_config"]["ef_search"] == 100, "Correct ef_search"

    def test_get_stats_after_add(self, populated_store):
        """Test get_stats after adding more chunks."""
        initial = populated_store.get_stats()
        new_chunks = [
            {
                "chunk_id": "extra_1",
                "text": "Extra chunk for testing.",
                "metadata": {"source": "extra.txt"},
            }
        ]
        populated_store.add_chunks(new_chunks)
        after = populated_store.get_stats()
        assert after["total_chunks"] == initial["total_chunks"] + 1, "Count should increase by 1"


# ---------------------------------------------------------------------------
# Search by embedding tests
# ---------------------------------------------------------------------------

class TestSearchByEmbedding:
    """Test search_by_embedding with pre-computed embeddings."""

    def test_search_by_embedding_basic(self, populated_store):
        """Test search using a pre-computed embedding."""
        # Get embedding from a known chunk
        chunk = populated_store.get_chunk("chunk_0")
        assert chunk is not None, "Should have chunk_0"
        embedding = chunk["embedding"]

        results = populated_store.search_by_embedding(embedding, top_k=5)
        assert len(results) > 0, "Should return results"
        assert len(results) <= 5, "Should respect top_k"
        # The first result should be chunk_0 (most similar)
        assert results[0]["chunk_id"] == "chunk_0", \
            "Same embedding should return itself as top result"

    def test_search_by_embedding_with_filter(self, populated_store):
        """Test search_by_embedding with metadata filter."""
        chunk = populated_store.get_chunk("chunk_0")
        embedding = chunk["embedding"]

        results = populated_store.search_by_embedding(
            embedding,
            top_k=5,
            filter_metadata={"source": "doc_0.txt"},
        )
        assert len(results) > 0, "Should return filtered results"
        for r in results:
            assert r.get("metadata", {}).get("source") == "doc_0.txt", \
                "All results should match the filter"

    def test_search_by_embedding_empty_store(self, empty_store):
        """Test search_by_embedding on empty store."""
        dummy_embedding = [0.0] * 384  # Common embedding dimension
        results = empty_store.search_by_embedding(dummy_embedding, top_k=5)
        assert results == [], "Empty store should return empty results"

    def test_search_by_embedding_large_top_k(self, populated_store):
        """Test search_by_embedding with top_k larger than store size."""
        chunk = populated_store.get_chunk("chunk_0")
        embedding = chunk["embedding"]

        results = populated_store.search_by_embedding(embedding, top_k=100)
        assert len(results) <= 10, "Should return at most all available chunks"


# ---------------------------------------------------------------------------
# Batch add chunks tests
# ---------------------------------------------------------------------------

class TestBatchAddChunks:
    """Test batch add_chunks with large batches."""

    def test_batch_add_large_batch(self, temp_dir):
        """Test adding a large batch of chunks."""
        store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_batch"),
            collection_name="test_batch",
            batch_size=10,
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        store.clear()

        chunks = [
            {
                "chunk_id": f"batch_{i}",
                "text": f"Batch test chunk {i} with enough content to be meaningful. "
                        f"This is about topic {i % 10} for testing purposes.",
                "metadata": {"source": f"batch_{i//10}.txt", "idx": i},
            }
            for i in range(50)
        ]

        store.add_chunks(chunks)
        assert store.count() == 50, "Should have 50 chunks after batch add"

    def test_batch_add_empty_list(self, empty_store):
        """Test adding empty list of chunks."""
        empty_store.add_chunks([])
        assert empty_store.count() == 0, "Should still have 0 chunks"

    def test_batch_add_single_chunk(self, empty_store):
        """Test adding a single chunk."""
        chunks = [
            {"chunk_id": "single", "text": "Single chunk text.", "metadata": {"source": "single.txt"}},
        ]
        empty_store.add_chunks(chunks)
        assert empty_store.count() == 1, "Should have 1 chunk"

    def test_batch_add_duplicate_ids(self, populated_store):
        """Test adding chunks with duplicate IDs (should update/overwrite)."""
        new_chunks = [
            {"chunk_id": "chunk_0", "text": "Updated content for chunk_0.", "metadata": {"source": "updated.txt"}},
        ]
        populated_store.add_chunks(new_chunks)
        # ChromaDB upserts by ID, so count should stay the same
        assert populated_store.count() == 10, "Count should remain the same after upsert"

        # Verify the chunk was updated
        chunk = populated_store.get_chunk("chunk_0")
        assert chunk is not None, "Should find chunk_0"
        assert "Updated content" in chunk["text"], "Content should be updated"


# ---------------------------------------------------------------------------
# Metadata sanitization tests
# ---------------------------------------------------------------------------

class TestMetadataSanitization:
    """Test metadata sanitization for ChromaDB compatibility."""

    def test_sanitize_list_value(self):
        """Test sanitizing list values in metadata."""
        metadata = {"tags": ["ml", "ai", "dl"], "count": 5}
        sanitized = VectorStore._sanitize_metadata(metadata)
        assert sanitized["count"] == 5, "int should remain int"
        assert isinstance(sanitized["tags"], str), "list should become string"

    def test_sanitize_dict_value(self):
        """Test sanitizing dict values in metadata."""
        metadata = {"config": {"key": "value"}, "name": "test"}
        sanitized = VectorStore._sanitize_metadata(metadata)
        assert sanitized["name"] == "test", "str should remain str"
        assert isinstance(sanitized["config"], str), "dict should become string"

    def test_sanitize_none_value(self):
        """Test sanitizing None values in metadata."""
        metadata = {"optional_field": None, "name": "test", "count": 3}
        sanitized = VectorStore._sanitize_metadata(metadata)
        assert sanitized["optional_field"] == "", "None should become empty string"
        assert sanitized["name"] == "test", "str should remain unchanged"
        assert sanitized["count"] == 3, "int should remain unchanged"

    def test_sanitize_bool_value(self):
        """Test sanitizing bool values in metadata."""
        metadata = {"is_active": True, "is_deleted": False}
        sanitized = VectorStore._sanitize_metadata(metadata)
        assert sanitized["is_active"] is True, "bool should remain bool"
        assert sanitized["is_deleted"] is False, "bool should remain bool"

    def test_sanitize_float_value(self):
        """Test sanitizing float values in metadata."""
        metadata = {"score": 0.95, "weight": 3.14}
        sanitized = VectorStore._sanitize_metadata(metadata)
        assert sanitized["score"] == 0.95, "float should remain float"
        assert sanitized["weight"] == 3.14, "float should remain float"

    def test_sanitize_mixed_values(self):
        """Test sanitizing a mix of compatible and incompatible types."""
        metadata = {
            "name": "test",
            "count": 10,
            "score": 0.85,
            "tags": ["a", "b"],
            "extra": None,
            "nested": {"key": "val"},
            "flag": True,
        }
        sanitized = VectorStore._sanitize_metadata(metadata)
        assert sanitized["name"] == "test", "str should remain"
        assert sanitized["count"] == 10, "int should remain"
        assert sanitized["score"] == 0.85, "float should remain"
        assert isinstance(sanitized["tags"], str), "list should become str"
        assert sanitized["extra"] == "", "None should become empty str"
        assert isinstance(sanitized["nested"], str), "dict should become str"
        assert sanitized["flag"] is True, "bool should remain"

    def test_sanitize_empty_metadata(self):
        """Test sanitizing empty metadata."""
        sanitized = VectorStore._sanitize_metadata({})
        assert sanitized == {}, "Empty metadata should remain empty"

    def test_chroma_metadata_types(self):
        """Test that _CHROMA_METADATA_TYPES contains expected types."""
        assert str in _CHROMA_METADATA_TYPES, "str should be in allowed types"
        assert int in _CHROMA_METADATA_TYPES, "int should be in allowed types"
        assert float in _CHROMA_METADATA_TYPES, "float should be in allowed types"
        assert bool in _CHROMA_METADATA_TYPES, "bool should be in allowed types"


# ---------------------------------------------------------------------------
# Corrupted database error handling tests
# ---------------------------------------------------------------------------

class TestCorruptedDatabase:
    """Test error handling for corrupted database scenarios."""

    def test_corrupted_database_raises_runtime_error(self, temp_dir):
        """Test that corrupted database raises RuntimeError."""
        store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_corrupt"),
            collection_name="test_corrupt",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )

        # Simulate ChromaDB client initialization failure.
        # chromadb is imported inside _get_client(), so we patch the
        # module-level reference that will be used at runtime.
        with patch('chromadb.PersistentClient',
                   side_effect=Exception("Database corruption detected")):
            with pytest.raises(Exception, match="Database corruption detected"):
                store._get_client()

    def test_get_client_resilience(self, temp_dir):
        """Test that _get_client can handle transient errors."""
        store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_resilient"),
            collection_name="test_resilient",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        # First call should succeed
        client = store._get_client()
        assert client is not None, "Should get a valid client"

    def test_count_handles_error(self, empty_store):
        """Test that count() handles errors gracefully."""
        # Simulate collection error
        with patch.object(empty_store, '_get_collection', side_effect=RuntimeError("Collection error")):
            count = empty_store.count()
            assert count == 0, "Should return 0 on error"

    def test_clear_handles_error(self, empty_store):
        """Test that clear() handles errors gracefully."""
        with patch.object(empty_store, '_get_client', side_effect=RuntimeError("Client error")):
            # Should not raise
            empty_store.clear()

    def test_remove_chunks_handles_error(self, empty_store):
        """Test that remove_chunks handles errors gracefully."""
        with patch.object(empty_store, '_get_collection', side_effect=RuntimeError("Collection error")):
            removed = empty_store.remove_chunks(["id1"])
            assert removed == 0, "Should return 0 on error"

    def test_get_chunk_handles_error(self, empty_store):
        """Test that get_chunk handles errors gracefully."""
        with patch.object(empty_store, '_get_collection', side_effect=RuntimeError("Collection error")):
            chunk = empty_store.get_chunk("id1")
            assert chunk is None, "Should return None on error"


# ---------------------------------------------------------------------------
# HNSW parameter tests
# ---------------------------------------------------------------------------

class TestHNSWParameters:
    """Test collection creation with custom HNSW parameters."""

    def test_default_hnsw_parameters(self, temp_dir):
        """Test default HNSW parameters."""
        store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_hnsw_default"),
            collection_name="test_hnsw_default",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        assert store.hnsw_ef_construction == 100, "Default ef_construction"
        assert store.hnsw_M == 8, "Default M"
        assert store.hnsw_ef_search == 50, "Default ef_search"

    def test_custom_hnsw_parameters(self, temp_dir):
        """Test custom HNSW parameters."""
        store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_hnsw_custom"),
            collection_name="test_hnsw_custom",
            hnsw_ef_construction=200,
            hnsw_M=32,
            hnsw_ef_search=100,
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        assert store.hnsw_ef_construction == 200, "Custom ef_construction"
        assert store.hnsw_M == 32, "Custom M"
        assert store.hnsw_ef_search == 100, "Custom ef_search"

    def test_hnsw_parameters_persist(self, temp_dir):
        """Test that HNSW parameters are reflected in stats."""
        store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_hnsw_persist"),
            collection_name="test_hnsw_persist",
            hnsw_ef_construction=150,
            hnsw_M=24,
            hnsw_ef_search=75,
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        stats = store.get_stats()
        assert stats["hnsw_config"]["ef_construction"] == 150
        assert stats["hnsw_config"]["M"] == 24
        assert stats["hnsw_config"]["ef_search"] == 75


# ---------------------------------------------------------------------------
# Clear and reset tests
# ---------------------------------------------------------------------------

class TestClearAndReset:
    """Test clear and reset operations."""

    def test_clear_removes_all_chunks(self, populated_store):
        """Test that clear removes all chunks."""
        assert populated_store.count() == 10, "Should have 10 chunks initially"
        populated_store.clear()
        assert populated_store.count() == 0, "Should have 0 chunks after clear"

    def test_reset_calls_clear(self, populated_store):
        """Test that reset calls clear."""
        assert populated_store.count() == 10, "Should have chunks initially"
        populated_store.reset()
        assert populated_store.count() == 0, "Should have 0 chunks after reset"

    def test_clear_idempotent(self, empty_store):
        """Test that clearing an empty store is idempotent."""
        empty_store.clear()
        assert empty_store.count() == 0, "Should still be 0 after clearing empty store"

    def test_clear_then_reuse(self, populated_store):
        """Test that store can be reused after clear."""
        populated_store.clear()
        assert populated_store.count() == 0, "Should be empty after clear"

        new_chunks = [
            {"chunk_id": "new_1", "text": "New content after clear.", "metadata": {"source": "new.txt"}},
        ]
        populated_store.add_chunks(new_chunks)
        assert populated_store.count() == 1, "Should be able to add chunks after clear"

    def test_multiple_clear_cycles(self, temp_dir):
        """Test multiple clear and re-populate cycles."""
        store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_cycle"),
            collection_name="test_cycle",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        store.clear()

        for cycle in range(3):
            chunks = [
                {"chunk_id": f"cycle{cycle}_{i}", "text": f"Cycle {cycle} chunk {i}.",
                 "metadata": {"source": f"cycle_{cycle}.txt"}}
                for i in range(5)
            ]
            store.add_chunks(chunks)
            assert store.count() == 5, f"Cycle {cycle}: should have 5 chunks"
            store.clear()
            assert store.count() == 0, f"Cycle {cycle}: should be empty after clear"


# ---------------------------------------------------------------------------
# Count tests
# ---------------------------------------------------------------------------

class TestCount:
    """Test count after various operations."""

    def test_count_initial(self, empty_store):
        """Test initial count is 0."""
        assert empty_store.count() == 0, "Initial count should be 0"

    def test_count_after_add(self, empty_store):
        """Test count after adding chunks."""
        chunks = [
            {"chunk_id": f"cnt_{i}", "text": f"Count test {i}.", "metadata": {"source": "count.txt"}}
            for i in range(5)
        ]
        empty_store.add_chunks(chunks)
        assert empty_store.count() == 5, "Count should be 5 after adding"

    def test_count_after_remove(self, populated_store):
        """Test count after removing chunks."""
        populated_store.remove_chunks(["chunk_0", "chunk_1"])
        assert populated_store.count() == 8, "Count should be 8 after removing 2 chunks"

    def test_count_after_clear(self, populated_store):
        """Test count after clear."""
        populated_store.clear()
        assert populated_store.count() == 0, "Count should be 0 after clear"

    def test_count_consistency(self, populated_store):
        """Test that count is consistent with search results."""
        count = populated_store.count()
        results = populated_store.search("test", top_k=count + 10)
        assert len(results) <= count, "Search results should not exceed count"


# ---------------------------------------------------------------------------
# Embedding model tests
# ---------------------------------------------------------------------------

class TestEmbeddingModel:
    """Test embedding model access and configuration."""

    def test_get_embedder_returns_model(self, empty_store):
        """Test that get_embedder returns the embedding model."""
        embedder = empty_store.get_embedder()
        assert embedder is not None, "Should return an embedder"

    def test_embedding_model_config(self, temp_dir):
        """Test custom embedding model configuration."""
        store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_emb"),
            collection_name="test_emb",
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_device="cpu",
        )
        assert store.embedding_model == "BAAI/bge-small-en-v1.5", "Custom embedding model"
        assert store.embedding_device == "cpu", "Should use CPU"

    def test_embedder_is_singleton(self, empty_store):
        """Test that get_embedder returns the same instance."""
        embedder1 = empty_store.get_embedder()
        embedder2 = empty_store.get_embedder()
        assert embedder1 is embedder2, "Should return the same embedder instance"


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------

class TestSearchAdvanced:
    """Test search with various parameters."""

    def test_search_with_metadata_filter(self, populated_store):
        """Test search with metadata filter."""
        results = populated_store.search(
            "test topic",
            top_k=10,
            filter_metadata={"source": "doc_0.txt"},
        )
        assert len(results) > 0, "Should return filtered results"
        for r in results:
            src = r.get("metadata", {}).get("source", "")
            assert src == "doc_0.txt", f"All results should be from doc_0.txt, got {src}"

    def test_search_result_structure(self, populated_store):
        """Test that search results have correct structure."""
        results = populated_store.search("test", top_k=3)
        for r in results:
            assert "chunk_id" in r, "Should have chunk_id"
            assert "text" in r, "Should have text"
            assert "metadata" in r, "Should have metadata"
            assert "score" in r, "Should have score"
            assert 0.0 <= r["score"] <= 1.0, f"Score {r['score']} should be in [0, 1]"

    def test_search_scores_descending(self, populated_store):
        """Test that search results are sorted by descending score."""
        results = populated_store.search("test topic", top_k=5)
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i]["score"] >= results[i + 1]["score"], \
                    "Scores should be in descending order"

    def test_search_top_k_limit(self, populated_store):
        """Test that search respects top_k."""
        for k in [1, 3, 5, 10]:
            results = populated_store.search("test", top_k=k)
            assert len(results) <= k, f"Should return at most {k} results"

    def test_search_empty_query(self, populated_store):
        """Test search with empty query."""
        results = populated_store.search("", top_k=5)
        assert isinstance(results, list), "Should return a list, not crash"