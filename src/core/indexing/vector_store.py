"""ChromaDB vector store wrapper for Hermes-RAG."""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("hermes_rag")

# Allowed metadata value types for ChromaDB compatibility
_CHROMA_METADATA_TYPES = (str, int, float, bool)


class IndexDimensionMismatch(RuntimeError):
    """Raised when the persisted vector collection has an incompatible dimension.

    Callers should rebuild the collection from the document store instead of
    deleting data silently.
    """

    def __init__(self, expected: int | None, actual: int | None, collection: str):
        self.expected = expected
        self.actual = actual
        self.collection = collection
        super().__init__(
            f"Collection '{collection}' has embedding dimension {expected}, "
            f"but the current model produces {actual}. Rebuild the collection "
            "from the document store before querying."
        )


class VectorStore:
    """ChromaDB-based vector store with optimized HNSW parameters."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "hermes_rag",
        hnsw_ef_construction: int = 100,
        hnsw_M: int = 8,
        hnsw_ef_search: int = 50,
        embedding_model: str = "BAAI/bge-m3",
        embedding_device: str = "cpu",
        batch_size: int = 500,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_M = hnsw_M
        self.hnsw_ef_search = hnsw_ef_search
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.batch_size = batch_size
        self._client = None
        self._collection = None
        self._embedder = None

    def _get_client(self):
        """Lazy-load ChromaDB client."""
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
        return self._client

    def _get_collection(self):
        """Get or create the ChromaDB collection.

        Raises:
            IndexDimensionMismatch: If the persisted collection was built with a
                different embedding dimension. The caller is expected to rebuild
                from the document store.
        """
        if self._collection is None:
            client = self._get_client()
            try:
                self._collection = client.get_collection(
                    name=self.collection_name,
                )
            except Exception:
                current_dim = self._get_embedder_dimension()
                self._collection = client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "hnsw:construction_ef": self.hnsw_ef_construction,
                        "hnsw:M": self.hnsw_M,
                        "hnsw:search_ef": self.hnsw_ef_search,
                        "embedding_dimension": current_dim,
                    },
                )
                return self._collection

            existing_dim = self._collection.metadata.get("embedding_dimension")
            if existing_dim is not None:
                current_dim = self._get_embedder_dimension()
                if current_dim is not None and existing_dim != current_dim:
                    raise IndexDimensionMismatch(
                        expected=existing_dim,
                        actual=current_dim,
                        collection=self.collection_name,
                    )
        return self._collection

    def get_collection_metadata(self) -> dict[str, Any] | None:
        """Return collection metadata, or None if the collection is unavailable."""
        try:
            return self._get_collection().metadata
        except Exception:
            return None

    def rebuild_from_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Delete and recreate the collection, then re-embed all chunks.

        Args:
            chunks: Chunks loaded from the document store.

        Returns:
            Number of chunks successfully embedded.
        """
        client = self._get_client()
        try:
            client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self._collection = None

        current_dim = self._get_embedder_dimension()
        self._collection = client.create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": self.hnsw_ef_construction,
                "hnsw:M": self.hnsw_M,
                "hnsw:search_ef": self.hnsw_ef_search,
                "embedding_dimension": current_dim,
            },
        )
        self.add_chunks(chunks)
        return self.count()

    def _get_embedder_dimension(self):
        """Get the embedding dimension of the current model."""
        try:
            embedder = self._get_embedder()
            return embedder.get_embedding_dimension()
        except Exception:
            return None

    def _get_embedder(self):
        """Lazy-load the embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                self.embedding_model,
                device=self.embedding_device,
                local_files_only=True,
            )
        return self._embedder

    def get_embedder(self):
        """Public accessor for the embedding model."""
        return self._get_embedder()

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Sanitize metadata values to ChromaDB-compatible types."""
        sanitized: dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, _CHROMA_METADATA_TYPES):
                sanitized[k] = v
            elif v is None:
                sanitized[k] = ""
            elif isinstance(v, (list, dict)):
                sanitized[k] = str(v)
            else:
                sanitized[k] = str(v)
        return sanitized

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Add chunks to the vector store with batch processing."""
        if not chunks:
            return

        embedder = self._get_embedder()
        collection = self._get_collection()

        for batch_start in range(0, len(chunks), self.batch_size):
            batch = chunks[batch_start : batch_start + self.batch_size]

            ids = [c["chunk_id"] for c in batch]
            texts = [c["text"] for c in batch]
            metadatas = [self._sanitize_metadata(c.get("metadata", {})) for c in batch]

            embeddings = embedder.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

    def search(
        self,
        query: str,
        top_k: int = 100,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks."""
        embedder = self._get_embedder()
        collection = self._get_collection()

        query_embedding = embedder.encode(
            [query],
            normalize_embeddings=True,
        ).tolist()

        where_filter = None
        if filter_metadata:
            where_filter = filter_metadata

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                output.append({
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1.0 - results["distances"][0][i],
                })

        return output

    def search_by_embedding(
        self,
        embedding: np.ndarray,
        top_k: int = 100,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search using a pre-computed embedding vector."""
        collection = self._get_collection()

        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        where_filter = None
        if filter_metadata:
            where_filter = filter_metadata

        results = collection.query(
            query_embeddings=embedding.tolist(),
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                output.append({
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1.0 - results["distances"][0][i],
                })

        return output

    def count(self) -> int:
        """Return the number of chunks in the store."""
        try:
            collection = self._get_collection()
            return collection.count()
        except IndexDimensionMismatch:
            # Dimension changes must propagate so IndexManager can rebuild the
            # collection from the document store instead of silently serving 0.
            raise
        except (AttributeError, ValueError, RuntimeError):
            return 0

    def clear(self) -> None:
        """Delete all chunks from the store."""
        try:
            client = self._get_client()
            client.delete_collection(self.collection_name)
            self._collection = None
        except (ValueError, AttributeError, RuntimeError):
            pass
        except Exception:
            self._collection = None

    def reset(self) -> None:
        """Reset the store (clear all data)."""
        self.clear()

    def remove_chunks(self, chunk_ids: list[str]) -> int:
        """Remove specific chunks by their IDs."""
        if not chunk_ids:
            return 0
        try:
            collection = self._get_collection()
            existing = collection.get(ids=chunk_ids)
            existing_ids = existing.get("ids", [])
            if not existing_ids:
                return 0
            collection.delete(ids=existing_ids)
            return len(existing_ids)
        except (AttributeError, ValueError, RuntimeError):
            return 0
        except Exception:
            logger.warning("Failed to remove chunks from vector store")
            return 0

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Retrieve a single chunk by its ID."""
        try:
            collection = self._get_collection()
            result = collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas", "embeddings"],
            )
            ids = result.get("ids", [])
            # collection.get() returns flat lists, not nested like query()
            if ids and len(ids) > 0:
                return {
                    "chunk_id": ids[0],
                    "text": result["documents"][0] if result.get("documents") else "",
                    "metadata": result["metadatas"][0] if result.get("metadatas") else {},
                    "embedding": result["embeddings"][0].tolist() if (
                        result.get("embeddings") is not None
                        and len(result["embeddings"]) > 0
                        and result["embeddings"][0] is not None
                    ) else None,
                }
            return None
        except Exception as e:
            logger.warning("Failed to retrieve chunk %s: %s", chunk_id, e)
            return None

    def get_stats(self) -> dict[str, Any]:
        """Return vector store statistics."""
        try:
            collection = self._get_collection()
            total = collection.count()
            dim = 0
            if total > 0:
                sample = collection.get(limit=1, include=["embeddings"])
                emb = sample.get("embeddings")
                if emb is not None and len(emb) > 0 and emb[0] is not None:
                    dim = len(emb[0])
            return {
                "collection_name": self.collection_name,
                "total_chunks": total,
                "embedding_dimension": dim,
                "persist_directory": self.persist_directory,
                "hnsw_config": {
                    "ef_construction": self.hnsw_ef_construction,
                    "M": self.hnsw_M,
                    "ef_search": self.hnsw_ef_search,
                },
            }
        except Exception:
            return {
                "collection_name": self.collection_name,
                "total_chunks": 0,
                "embedding_dimension": 0,
                "persist_directory": self.persist_directory,
                "hnsw_config": {},
            }