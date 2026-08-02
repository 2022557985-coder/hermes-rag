"""Index manager that coordinates vector, BM25, and document-store operations."""

import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.core.indexing.vector_store import IndexDimensionMismatch

logger = logging.getLogger("hermes_rag")


class IndexManager:
    """Manages ingestion and retrieval across all three stores.

    The document store is the source of truth. If the vector collection must be
    rebuilt (for example after changing the embedding model), chunks are
    re-embedded from the document store instead of being lost.
    """

    # Bump when chunk text / metadata layout changes so stale persisted
    # vector and BM25 indexes are rebuilt from the document store once.
    INDEX_VERSION = "2026.08.01-chunker-v2"

    def __init__(
        self,
        vector_store: Any = None,
        bm25_index: Any = None,
        document_store: Any = None,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.document_store = document_store

    @staticmethod
    def compute_layout_signature(embedding_dimension: int | None = None) -> str:
        """Return the layout signature persisted alongside ingested chunks.

        The signature covers the chunk layout and the embedding schema so a
        stale document store can be re-chunked from source after either one
        changes, instead of only being detected at vector-query time.
        """
        dim = embedding_dimension if embedding_dimension is not None else 0
        return f"{IndexManager.INDEX_VERSION}|dim:{dim}"

    def ingest_chunks(self, chunks: list[dict[str, Any]]) -> dict[str, int]:
        """Ingest chunks into all stores in parallel.

        Args:
            chunks: List of chunk dicts.

        Returns:
            dict with counts for each store.
        """
        if not chunks:
            return {"vector_count": 0, "bm25_count": 0, "document_count": 0}

        results = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}

            if self.vector_store:
                futures["vector"] = executor.submit(
                    self.vector_store.add_chunks, chunks
                )
            if self.bm25_index:
                futures["bm25"] = executor.submit(
                    self.bm25_index.add_chunks, chunks
                )
            if self.document_store:
                futures["document"] = executor.submit(
                    self.document_store.add_chunks, chunks
                )

            for name, future in futures.items():
                try:
                    future.result()
                except MemoryError as e:
                    logger.error(f"Memory error ingesting to {name} store: {e}")
                    gc.collect()
                except Exception as e:
                    logger.warning(f"Error ingesting to {name} store: {e}")

        results["vector_count"] = self.vector_store.count() if self.vector_store else 0
        results["bm25_count"] = self.bm25_index.count() if self.bm25_index else 0
        results["document_count"] = self.document_store.count() if self.document_store else 0

        if self.document_store:
            self.document_store.set_meta("index_version", self.INDEX_VERSION)
            try:
                self.document_store.set_meta(
                    "layout_signature",
                    self.compute_layout_signature(
                        self.vector_store._get_embedder_dimension()
                        if self.vector_store else None
                    ),
                )
            except Exception:
                logger.warning("Failed to persist layout signature")

        gc.collect()
        return results

    def ensure_indexes(self) -> dict[str, Any]:
        """Verify index health and rebuild from the document store if needed.

        Returns:
            dict with status and list of rebuild actions performed.
        """
        result: dict[str, Any] = {"status": "ok", "actions": []}

        # Vector collection: rebuild when the embedding dimension changed.
        try:
            if self.vector_store:
                self.vector_store.count()
        except IndexDimensionMismatch as e:
            logger.warning("Vector dimension mismatch: %s", e)
            if self.document_store and self.document_store.count() > 0:
                logger.info("Rebuilding vector collection from document store...")
                chunks = self.document_store.get_all_chunks()
                count = self.vector_store.rebuild_from_chunks(chunks)
                result["actions"].append(f"vector_rebuilt:{count}")
            else:
                result["status"] = "empty_index"
                result["actions"].append("vector_dimension_mismatch_no_documents")
        except Exception as e:
            logger.warning(f"Vector store health check failed: {e}")

        # Sparse index: rebuild from the document store when it is empty.
        if self.bm25_index is not None:
            try:
                if self.bm25_index.count() == 0 and self.document_store and self.document_store.count() > 0:
                    logger.info("Rebuilding BM25 index from document store...")
                    self.bm25_index.add_chunks(self.document_store.get_all_chunks())
                    result["actions"].append("bm25_rebuilt")
            except Exception as e:
                logger.warning(f"BM25 health check failed: {e}")

        # Versioned rebuild: chunk text/metadata layout changed since the
        # persisted indexes were built, so re-index from the source of truth.
        if self.document_store and self.document_store.count() > 0:
            try:
                stored_version = self.document_store.get_meta("index_version")
                if stored_version != self.INDEX_VERSION:
                    logger.info(
                        "Index version changed (%s -> %s), rebuilding from document store...",
                        stored_version,
                        self.INDEX_VERSION,
                    )
                    rebuilt = self.rebuild_from_document_store()
                    self.document_store.set_meta("index_version", self.INDEX_VERSION)
                    result["actions"].append(
                        f"version_rebuild:vector={rebuilt.get('vector_count')},bm25={rebuilt.get('bm25_count')}"
                    )
            except Exception as e:
                logger.warning(f"Versioned index rebuild failed: {e}")

        # Cross-store consistency: vector and BM25 must index exactly the
        # chunks persisted in the document store (the source of truth). A
        # stale or partial store (e.g. a chunk missing from BM25, or leftover
        # vector entries from an older ingest) silently degrades retrieval
        # quality, so rebuild the divergent store(s) from the document store.
        if self.document_store and self.document_store.count() > 0:
            try:
                doc_ids = {c["chunk_id"] for c in self.document_store.get_all_chunks()}
                vector_ids = set(self.vector_store.get_chunk_ids()) if self.vector_store else set()
                bm25_ids = set(self.bm25_index.get_chunk_ids()) if self.bm25_index else set()
                if vector_ids != doc_ids or bm25_ids != doc_ids:
                    logger.warning(
                        "Index consistency check failed (docstore=%d, vector=%d, bm25=%d); "
                        "rebuilding from document store...",
                        len(doc_ids), len(vector_ids), len(bm25_ids),
                    )
                    rebuilt = self.rebuild_from_document_store()
                    result["actions"].append(
                        f"consistency_rebuild:vector={rebuilt.get('vector_count')},bm25={rebuilt.get('bm25_count')}"
                    )
            except Exception as e:
                logger.warning(f"Index consistency check failed: {e}")

        return result

    def rebuild_from_document_store(self) -> dict[str, int]:
        """Clear vector/BM25 indexes and rebuild them from persisted chunks."""
        if self.document_store is None:
            return {"vector_count": 0, "bm25_count": 0}

        chunks = self.document_store.get_all_chunks()
        if self.vector_store:
            if chunks:
                self.vector_store.rebuild_from_chunks(chunks)
            else:
                self.vector_store.clear()
        if self.bm25_index:
            self.bm25_index.clear()
            if chunks:
                self.bm25_index.add_chunks(chunks)
        return {
            "vector_count": self.vector_store.count() if self.vector_store else 0,
            "bm25_count": self.bm25_index.count() if self.bm25_index else 0,
        }

    def search_dense(
        self,
        query: str,
        top_k: int = 100,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Dense vector search."""
        if self.vector_store is None:
            return []
        return self.vector_store.search(query, top_k, filter_metadata)

    def search_sparse(self, query: str, top_k: int = 100) -> list[dict[str, Any]]:
        """Sparse BM25 search."""
        if self.bm25_index is None:
            return []
        return self.bm25_index.search(query, top_k)

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Retrieve a single chunk, preferring the vector store."""
        if self.vector_store is not None:
            try:
                chunk = self.vector_store.get_chunk(chunk_id)
                if chunk is not None:
                    return chunk
            except Exception as e:
                logger.debug(f"Vector get_chunk failed, falling back to document store: {e}")
        if self.document_store is not None:
            return self.document_store.get_chunk(chunk_id)
        return None

    def get_neighbor_chunks(
        self, chunk_id: str, window: int = 1
    ) -> list[dict[str, Any]]:
        """Retrieve neighboring chunks around a given chunk_id."""
        neighbors = []
        try:
            parts = chunk_id.rsplit("_", 1)
            if len(parts) != 2:
                return neighbors
            prefix, idx_str = parts
            idx = int(idx_str)

            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                neighbor_id = f"{prefix}_{idx + offset}"
                chunk = self.get_chunk(neighbor_id)
                if chunk is not None:
                    neighbors.append(chunk)
        except (ValueError, IndexError):
            pass
        return neighbors

    def clear(self) -> None:
        """Clear all stores."""
        if self.vector_store:
            self.vector_store.clear()
        if self.bm25_index:
            self.bm25_index.clear()
        if self.document_store:
            self.document_store.clear()
        gc.collect()

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate index statistics."""
        stats: dict[str, Any] = {}
        if self.vector_store:
            try:
                stats["vector"] = self.vector_store.get_stats()
            except Exception as e:
                stats["vector"] = {"error": str(e)}
        if self.bm25_index:
            stats["bm25"] = self.bm25_index.get_stats()
        if self.document_store:
            stats["document_store"] = self.document_store.get_stats()
        return stats