"""Dense vector retrieval using ChromaDB."""

from typing import Any


class DenseRetriever:
    """Dense retrieval via vector similarity search."""

    def __init__(self, index_manager):
        self.index_manager = index_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 100,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve chunks using dense vector search.

        Args:
            query: Query text.
            top_k: Number of results.
            filter_metadata: Metadata filter for ChromaDB.

        Returns:
            List of result dicts.
        """
        results = self.index_manager.search_dense(query, top_k, filter_metadata)
        for r in results:
            r["source"] = "dense"
        return results