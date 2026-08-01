"""ChromaDB vector store wrapper for Hermes-RAG."""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger("hermes_rag")


class VectorStore:
    """ChromaDB-based vector store with optimized HNSW parameters."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "hermes_rag",
        hnsw_ef_construction: int = 100,
        hnsw_M: int = 8,
        hnsw_ef_search: int = 50,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        embedding_device: str = "cpu",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_M = hnsw_M
        self.hnsw_ef_search = hnsw_ef_search
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
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
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            client = self._get_client()
            try:
                self._collection = client.get_collection(
                    name=self.collection_name,
                )
            except Exception:
                self._collection = client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "hnsw:construction_ef": self.hnsw_ef_construction,
                        "hnsw:M": self.hnsw_M,
                        "hnsw:search_ef": self.hnsw_ef_search,
                    },
                )
        return self._collection

    def _get_embedder(self):
        """Lazy-load the embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                self.embedding_model,
                device=self.embedding_device,
            )
        return self._embedder

    def get_embedder(self):
        """Public accessor for the embedding model.

        Returns:
            The loaded SentenceTransformer model instance.
        """
        return self._get_embedder()

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Add chunks to the vector store.

        Args:
            chunks: List of chunk dicts with 'chunk_id', 'text', 'metadata'.
        """
        if not chunks:
            return

        embedder = self._get_embedder()
        collection = self._get_collection()

        ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]

        # Convert metadata values to ChromaDB-compatible types
        for meta in metadatas:
            for k, v in list(meta.items()):
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v  # Preserve native types
                elif v is None:
                    meta[k] = ""
                elif isinstance(v, (list, dict)):
                    meta[k] = str(v)
                else:
                    meta[k] = str(v)

        embeddings = embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def search(
        self,
        query: str,
        top_k: int = 100,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks.

        Args:
            query: Query text.
            top_k: Number of results to return.
            filter_metadata: Optional metadata filter for ChromaDB.

        Returns:
            List of result dicts with 'chunk_id', 'text', 'metadata', 'score'.
        """
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
                    "score": 1.0 - results["distances"][0][i],  # Convert distance to similarity
                })

        return output

    def count(self) -> int:
        """Return the number of chunks in the store."""
        try:
            collection = self._get_collection()
            return collection.count()
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
            # Catch NotFoundError from chromadb when collection doesn't exist
            self._collection = None

    def reset(self) -> None:
        """Reset the store (clear all data)."""
        self.clear()