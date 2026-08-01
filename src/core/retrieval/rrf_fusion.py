"""Reciprocal Rank Fusion with dynamic weights."""

import re
from typing import List, Dict, Any


class RRFFusion:
    """RRF fusion with dynamic weight adjustment based on query features."""

    def __init__(
        self,
        k: int = 60,
        default_dense_weight: float = 0.5,
        default_sparse_weight: float = 0.5,
        product_code_dense_weight: float = 0.3,
        product_code_sparse_weight: float = 0.7,
        colloquial_dense_weight: float = 0.8,
        colloquial_sparse_weight: float = 0.2,
    ):
        self.k = k
        self.default_dense_weight = default_dense_weight
        self.default_sparse_weight = default_sparse_weight
        self.product_code_dense_weight = product_code_dense_weight
        self.product_code_sparse_weight = product_code_sparse_weight
        self.colloquial_dense_weight = colloquial_dense_weight
        self.colloquial_sparse_weight = colloquial_sparse_weight

    def fuse(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        query: str = "",
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """Fuse dense and sparse results using RRF with dynamic weights.

        Args:
            dense_results: Results from dense retrieval.
            sparse_results: Results from sparse retrieval.
            query: Original query for dynamic weight detection.
            top_k: Number of fused results to return.

        Returns:
            Fused and sorted result list.
        """
        # Handle edge case: both empty
        if not dense_results and not sparse_results:
            return []

        dense_weight, sparse_weight = self._get_dynamic_weights(query)

        scores = {}

        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            chunk_id = result["chunk_id"]
            rrf_score = dense_weight / (self.k + rank)
            if chunk_id in scores:
                scores[chunk_id]["score"] += rrf_score
                if "dense" not in scores[chunk_id]["sources"]:
                    scores[chunk_id]["sources"].append("dense")
            else:
                scores[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "metadata": result.get("metadata", {}),
                    "score": rrf_score,
                    "sources": ["dense"],
                }

        # Process sparse results
        for rank, result in enumerate(sparse_results, 1):
            chunk_id = result["chunk_id"]
            rrf_score = sparse_weight / (self.k + rank)
            if chunk_id in scores:
                scores[chunk_id]["score"] += rrf_score
                if "sparse" not in scores[chunk_id]["sources"]:
                    scores[chunk_id]["sources"].append("sparse")
            else:
                scores[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "metadata": result.get("metadata", {}),
                    "score": rrf_score,
                    "sources": ["sparse"],
                }

        # Sort by score descending
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        return sorted_results[:top_k]

    def _get_dynamic_weights(self, query: str) -> tuple:
        """Determine dynamic weights based on query features.

        - Product codes (contains digits/letters/symbols): boost BM25
        - Colloquial (contains conversational keywords): boost dense
        - Default: equal weights
        """
        # Detect product code patterns (e.g., "ABC-1234", "Model-X200", not "K-means")
        # Product codes typically have multiple digits and letters mixed with separators
        has_product_code = bool(
            re.search(r"[A-Za-z]{2,}[\d\-_]{2,}|[\d]{2,}[\-][A-Za-z]{2,}", query)
        )

        # Detect colloquial patterns (Chinese + English)
        colloquial_keywords_cn = [
            "怎么", "为啥", "为什么", "哪个", "哪一", "啥", "咋",
            "什么", "哪些", "是什么", "什么样", "怎么样", "如何",
            "帮我", "请问", "能否", "可以", "能不能",
        ]
        colloquial_keywords_en = [
            "how do i", "what is", "why is", "which one", "tell me",
            "explain", "show me", "can you", "help me", "what's the",
        ]
        is_colloquial = (
            any(kw in query for kw in colloquial_keywords_cn)
            or any(kw in query.lower() for kw in colloquial_keywords_en)
        )

        if has_product_code:
            return (self.product_code_dense_weight, self.product_code_sparse_weight)
        elif is_colloquial:
            return (self.colloquial_dense_weight, self.colloquial_sparse_weight)
        else:
            return (self.default_dense_weight, self.default_sparse_weight)