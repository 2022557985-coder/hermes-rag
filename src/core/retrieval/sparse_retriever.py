"""Sparse BM25 retrieval."""

from typing import Any


class SparseRetriever:
    """Sparse retrieval via BM25."""

    def __init__(self, index_manager):
        self.index_manager = index_manager

    def retrieve(self, query: str, top_k: int = 100) -> list[dict[str, Any]]:
        """Retrieve chunks using BM25 sparse search.

        Args:
            query: Query text.
            top_k: Number of results.

        Returns:
            List of result dicts.
        """
        results = self.index_manager.search_sparse(query, top_k)
        for r in results:
            r["source"] = "sparse"
        return results