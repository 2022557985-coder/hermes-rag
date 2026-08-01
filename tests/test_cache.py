"""Comprehensive tests for QueryCache."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.cache import QueryCache


class TestQueryCache:
    """Test semantic cache correctness and edge cases."""

    def test_exact_match_hit(self):
        cache = QueryCache()
        cache.set("test query", [{"id": 1}])
        result = cache.get("test query")
        assert result is not None
        assert result[0]["id"] == 1

    def test_exact_match_miss(self):
        cache = QueryCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_semantic_match(self):
        cache = QueryCache(similarity_threshold=0.8)
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.99, 0.01, 0.0])  # Very similar
        cache.set("query1", [{"id": 1}], query_embedding=emb1)
        result = cache.get("query2", query_embedding=emb2)
        assert result is not None

    def test_semantic_mismatch(self):
        cache = QueryCache(similarity_threshold=0.95)
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])  # Orthogonal
        cache.set("query1", [{"id": 1}], query_embedding=emb1)
        result = cache.get("query2", query_embedding=emb2)
        assert result is None

    def test_ttl_expiry(self):
        cache = QueryCache(ttl_seconds=0)  # Immediate expiry
        cache.set("query", [{"id": 1}])
        time.sleep(0.01)
        result = cache.get("query")
        assert result is None

    def test_lru_eviction(self):
        cache = QueryCache(max_size=2)
        cache.set("q1", [{"id": 1}])
        cache.set("q2", [{"id": 2}])
        cache.set("q3", [{"id": 3}])
        assert cache.get("q1") is None  # Should be evicted
        assert cache.get("q2") is not None
        assert cache.get("q3") is not None

    def test_lru_access_updates_order(self):
        cache = QueryCache(max_size=2)
        cache.set("q1", [{"id": 1}])
        cache.set("q2", [{"id": 2}])
        cache.get("q1")  # Access q1, making it recently used
        cache.set("q3", [{"id": 3}])
        assert cache.get("q1") is not None  # Should NOT be evicted
        assert cache.get("q2") is None  # Should be evicted

    def test_hit_rate_initial(self):
        cache = QueryCache()
        assert cache.hit_rate() == 0.0

    def test_hit_rate_after_hits(self):
        cache = QueryCache()
        cache.set("q1", [{"id": 1}])
        cache.get("q1")  # Hit
        cache.get("q2")  # Miss
        assert cache.hit_rate() == 0.5

    def test_size_tracking(self):
        cache = QueryCache()
        assert cache.size() == 0
        cache.set("q1", [{"id": 1}])
        assert cache.size() == 1
        cache.set("q2", [{"id": 2}])
        assert cache.size() == 2

    def test_clear(self):
        cache = QueryCache()
        cache.set("q1", [{"id": 1}])
        cache.set("q2", [{"id": 2}])
        cache.clear()
        assert cache.size() == 0
        assert cache.get("q1") is None

    def test_cosine_similarity_identical(self):
        cache = QueryCache()
        a = np.array([1.0, 2.0, 3.0])
        sim = cache._cosine_similarity(a, a)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        cache = QueryCache()
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        sim = cache._cosine_similarity(a, b)
        assert abs(sim) < 1e-6

    def test_cosine_similarity_none(self):
        cache = QueryCache()
        sim = cache._cosine_similarity(None, np.array([1.0]))
        assert sim == 0.0

    def test_overwrite_existing(self):
        cache = QueryCache()
        cache.set("q1", [{"id": 1}])
        cache.set("q1", [{"id": 2}])
        result = cache.get("q1")
        assert result[0]["id"] == 2

    def test_hash_deterministic(self):
        cache = QueryCache()
        h1 = cache._hash_query("test")
        h2 = cache._hash_query("test")
        assert h1 == h2

    def test_hash_case_sensitive(self):
        cache = QueryCache()
        h1 = cache._hash_query("Test")
        h2 = cache._hash_query("test")
        assert h1 != h2