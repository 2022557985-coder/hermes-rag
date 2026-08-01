"""Index manager that coordinates dual-index (vector + BM25) operations."""

import gc
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("hermes_rag")


class IndexManager:
    """Manages dual-index ingestion and retrieval.

    Uses ThreadPoolExecutor for parallel ingestion. While the GIL prevents true
    CPU parallelism, the underlying I/O operations (ChromaDB writes, SQLite inserts)
    typically release the GIL, allowing effective concurrency for dual-index writes.
    """

    def __init__(
        self,
        vector_store: Any = None,
        bm25_index: Any = None,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def ingest_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Ingest chunks into both indexes in parallel.

        Args:
            chunks: List of chunk dicts.

        Returns:
            dict with counts for each index.
        """
        if not chunks:
            return {"vector_count": 0, "bm25_count": 0}

        results = {}

        # Parallel ingestion via ThreadPoolExecutor
        # Note: ChromaDB and SQLite operations release the GIL during I/O,
        # so threading provides real concurrency benefits here.
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}

            if self.vector_store:
                futures["vector"] = executor.submit(
                    self.vector_store.add_chunks, chunks
                )

            if self.bm25_index:
                futures["bm25"] = executor.submit(
                    self.bm25_index.add_chunks, chunks
                )

            for name, future in futures.items():
                try:
                    future.result()
                except MemoryError as e:
                    logger.error(f"Memory error ingesting to {name} index: {e}")
                    gc.collect()
                except Exception as e:
                    logger.warning(f"Error ingesting to {name} index: {e}")

        results["vector_count"] = self.vector_store.count() if self.vector_store else 0
        results["bm25_count"] = self.bm25_index.count() if self.bm25_index else 0

        # Force garbage collection after ingestion
        gc.collect()

        return results

    def search_dense(
        self,
        query: str,
        top_k: int = 100,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Dense vector search."""
        if self.vector_store is None:
            return []
        return self.vector_store.search(query, top_k, filter_metadata)

    def search_sparse(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Sparse BM25 search."""
        if self.bm25_index is None:
            return []
        return self.bm25_index.search(query, top_k)

    def clear(self) -> None:
        """Clear both indexes."""
        if self.vector_store:
            self.vector_store.clear()
        if self.bm25_index:
            self.bm25_index.clear()
        gc.collect()