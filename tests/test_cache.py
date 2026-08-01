"""Tests for query cache."""

import pytest
import numpy as np
from src.utils.cache import QueryCache


class TestQueryCache:
    """Tests for QueryCache."""

    def test_cache_miss_on_empty(self):
        cache = QueryCache(max_size=100)
        result = cache.get("test query")
        assert result is None

    def test_cache_set_and_get(self):
        cache = QueryCache(max_size=100)
        results = [{"chunk_id": "1", "text": "test"}]
        cache.set("test query", results)
        cached = cache.get("test query")
        assert cached == results

    def test_cache_size_limit(self):
        cache = QueryCache(max_size=3)
        for i in range(5):
            cache.set(f"query_{i}", [{"id": i}])
        assert cache.size() <= 3

    def test_cache_clear(self):
        cache = QueryCache(max_size=100)
        cache.set("q1", [{"id": 1}])
        cache.set("q2", [{"id": 2}])
        cache.clear()
        assert cache.size() == 0

    def test_cache_ttl_expiry(self):
        cache = QueryCache(max_size=100, ttl_seconds=0)  # Immediate expiry
        cache.set("q1", [{"id": 1}])
        result = cache.get("q1")
        assert result is None

    def test_semantic_similarity_cache_hit(self):
        cache = QueryCache(max_size=100, similarity_threshold=0.9)
        results = [{"chunk_id": "1", "text": "semantic"}]
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        cache.set("machine learning", results, emb)

        # Similar query with same embedding should hit
        similar_emb = emb + np.random.randn(384).astype(np.float32) * 0.01
        similar_emb = similar_emb / np.linalg.norm(similar_emb)
        cached = cache.get("what is machine learning", similar_emb)
        assert cached == results

    def test_hit_rate_tracking(self):
        cache = QueryCache(max_size=100)
        assert cache.hit_rate() == 0.0  # No accesses yet

        cache.set("q1", [{"id": 1}])
        cache.get("q1")  # hit
        assert cache.hit_rate() == 1.0

        cache.get("q2")  # miss
        assert cache.hit_rate() == 0.5

    def test_hit_rate_multiple_hits(self):
        cache = QueryCache(max_size=100)
        cache.set("q1", [{"id": 1}])
        cache.set("q2", [{"id": 2}])

        cache.get("q1")  # hit
        cache.get("q2")  # hit
        cache.get("q3")  # miss
        cache.get("q4")  # miss
        assert cache.hit_rate() == 0.5

    def test_cache_size_after_eviction(self):
        cache = QueryCache(max_size=2)
        cache.set("q1", [{"id": 1}])
        cache.set("q2", [{"id": 2}])
        cache.set("q3", [{"id": 3}])
        assert cache.size() == 2
        # q1 should be evicted (oldest)
        assert cache.get("q1") is None

    def test_cache_ttl_partial_expiry(self):
        cache = QueryCache(max_size=100, ttl_seconds=3600)
        cache.set("q1", [{"id": 1}])
        # Set q2 with 0 TTL (immediate expiry)
        cache2 = QueryCache(max_size=100, ttl_seconds=0)
        cache2.set("q1", [{"id": 1}])
        assert cache2.get("q1") is None
        # q1 in original cache should still be valid
        assert cache.get("q1") == [{"id": 1}]