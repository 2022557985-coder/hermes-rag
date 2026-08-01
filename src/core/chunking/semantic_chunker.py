"""Semantic chunking based on sentence embedding similarity."""

import numpy as np


class SemanticChunker:
    """Split text at semantic boundaries using sentence embeddings."""

    def __init__(
        self,
        threshold: float = 0.65,
        min_chunk_size: int = 50,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
    ):
        """Initialize the semantic chunker.

        Args:
            threshold: Cosine similarity threshold for split points.
            min_chunk_size: Minimum tokens for a chunk.
            model_name: SentenceTransformer model name.
            device: Device to load the model on ('cpu' or 'cuda').
        """
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self.model_name = model_name
        self.device = device
        self._embedder = None

    def _get_embedder(self):
        """Lazy-load the sentence embedder."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                self.model_name,
                device=self.device,
                local_files_only=True,
            )
        return self._embedder

    def split(self, text: str) -> list[str]:
        """Split text into semantically coherent chunks.

        Args:
            text: Input text.

        Returns:
            List of text chunks.
        """
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [text]

        # Get embeddings
        embedder = self._get_embedder()
        embeddings = embedder.encode(sentences, normalize_embeddings=True)

        # Find split points based on cosine similarity valleys
        split_points = self._find_split_points(embeddings)

        # Build chunks
        chunks = []
        start = 0
        for split_idx in split_points:
            chunk = " ".join(sentences[start:split_idx])
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            start = split_idx

        # Last chunk
        if start < len(sentences):
            chunk = " ".join(sentences[start:])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re
        pattern = r"(?<=[。！？.!?\n])\s*"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _find_split_points(self, embeddings: np.ndarray) -> list[int]:
        """Find semantic split points based on similarity valleys.

        Args:
            embeddings: Array of sentence embeddings, shape (n_sentences, dim).

        Returns:
            List of split point indices.
        """
        if len(embeddings) <= 1:
            return []

        # Compute pairwise cosine similarities of adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            similarities.append(sim)

        # Find local minima (valleys)
        split_points = []
        for i in range(1, len(similarities) - 1):
            if similarities[i] < similarities[i - 1] and similarities[i] <= similarities[i + 1]:
                if similarities[i] < self.threshold:
                    split_points.append(i + 1)

        return split_points