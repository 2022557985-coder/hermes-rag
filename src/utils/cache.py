"""Semantic similarity-based query cache for Hermes-RAG."""

import hashlib
import time
from typing import Optional

import numpy as np


class QueryCache:
    """LRU cache with semantic similarity-based lookup.

    For high-frequency or similar queries, returns cached results
    to reduce latency and CPU usage.
    """

    def __init__(
        self,
        max_size: int = 1000,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600,
    ):
        self._max_size = max_size
        self._similarity_threshold = similarity_threshold
        self._ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[list, np.ndarray, float]] = {}  # hash -> (results, embedding, timestamp)
        self._access_order: list[str] = []
        self._hits: int = 0
        self._total_accesses: int = 0

    def _hash_query(self, query: str) -> str:
        """Hash a query string for cache key."""
        return hashlib.md5(query.encode("utf-8")).hexdigest()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def get(self, query: str, query_embedding: Optional[np.ndarray] = None) -> Optional[list]:
        """Look up cached results by query or semantic similarity.

        Args:
            query: The query string.
            query_embedding: The query embedding vector (for semantic matching).

        Returns:
            Cached results list if found, None otherwise.
        """
        now = time.time()
        self._total_accesses += 1

        # Exact match
        key = self._hash_query(query)
        if key in self._cache:
            results, _, timestamp = self._cache[key]
            if now - timestamp < self._ttl_seconds:
                self._access_order.remove(key)
                self._access_order.append(key)
                self._hits += 1
                return results
            else:
                del self._cache[key]
                self._access_order.remove(key)

        # Semantic similarity match
        if query_embedding is not None:
            for cached_key in list(self._access_order):
                if cached_key in self._cache:
                    results, cached_emb, timestamp = self._cache[cached_key]
                    if now - timestamp < self._ttl_seconds:
                        sim = self._cosine_similarity(query_embedding, cached_emb)
                        if sim >= self._similarity_threshold:
                            self._access_order.remove(cached_key)
                            self._access_order.append(cached_key)
                            self._hits += 1
                            return results
                    else:
                        del self._cache[cached_key]
                        self._access_order.remove(cached_key)

        return None

    def set(self, query: str, results: list, query_embedding: Optional[np.ndarray] = None) -> None:
        """Cache results for a query.

        Args:
            query: The query string.
            results: The retrieval results to cache.
            query_embedding: The query embedding vector.
        """
        key = self._hash_query(query)

        if key in self._cache:
            self._access_order.remove(key)

        # Evict oldest if at capacity
        while len(self._cache) >= self._max_size and self._access_order:
            oldest = self._access_order.pop(0)
            if oldest in self._cache:
                del self._cache[oldest]

        self._cache[key] = (results, query_embedding, time.time())
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()

    def size(self) -> int:
        """Return the number of cached entries."""
        return len(self._cache)

    def hit_rate(self) -> float:
        """Return the cache hit rate as a fraction (0.0 to 1.0)."""
        if self._total_accesses == 0:
            return 0.0
        return self._hits / self._total_accesses