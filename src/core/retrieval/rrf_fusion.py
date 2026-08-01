"""Reciprocal Rank Fusion with dynamic weights."""

import re
from typing import Any


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
        min_score_threshold: float = 0.0,
    ):
        self.k = k
        self.default_dense_weight = default_dense_weight
        self.default_sparse_weight = default_sparse_weight
        self.product_code_dense_weight = product_code_dense_weight
        self.product_code_sparse_weight = product_code_sparse_weight
        self.colloquial_dense_weight = colloquial_dense_weight
        self.colloquial_sparse_weight = colloquial_sparse_weight
        self.min_score_threshold = min_score_threshold

    def _detect_query_features(self, query: str) -> dict[str, Any]:
        """Detect and return a dict of query features.

        Returns:
            Dict with keys: has_chinese, has_english, has_numbers,
            query_length, keyword_count, has_product_code, is_colloquial,
            ratio_chinese, ratio_english, ratio_numbers, is_short_query,
            has_special_chars, dominant_language.
        """
        features: dict[str, Any] = {}

        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', query))
        english_chars = len(re.findall(r'[a-zA-Z]', query))
        numbers = len(re.findall(r'\d', query))
        total_chars = len(query)

        features["has_chinese"] = chinese_chars > 0
        features["has_english"] = english_chars > 0
        features["has_numbers"] = numbers > 0
        features["query_length"] = total_chars
        features["has_special_chars"] = bool(re.search(r'[^\w\s\u4e00-\u9fff]', query))

        if total_chars > 0:
            features["ratio_chinese"] = chinese_chars / total_chars
            features["ratio_english"] = english_chars / total_chars
            features["ratio_numbers"] = numbers / total_chars
        else:
            features["ratio_chinese"] = 0.0
            features["ratio_english"] = 0.0
            features["ratio_numbers"] = 0.0

        # Detect product code patterns (e.g., "ABC-1234", "Model-X200")
        features["has_product_code"] = bool(
            re.search(r"[A-Za-z]{2,}[\d\-_]{2,}|[\d]{2,}[\-][A-Za-z]{2,}", query)
        )

        # Detect colloquial patterns
        colloquial_keywords_cn = [
            "怎么", "为啥", "为什么", "哪个", "哪一", "啥", "咋",
            "什么", "哪些", "是什么", "什么样", "怎么样", "如何",
            "帮我", "请问", "能否", "可以", "能不能",
        ]
        colloquial_keywords_en = [
            "how do i", "what is", "why is", "which one", "tell me",
            "explain", "show me", "can you", "help me", "what's the",
        ]
        features["is_colloquial"] = (
            any(kw in query for kw in colloquial_keywords_cn)
            or any(kw in query.lower() for kw in colloquial_keywords_en)
        )

        # Count keywords (non-stopword words)
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', query)
        features["keyword_count"] = len(words)

        # Short query detection
        features["is_short_query"] = total_chars <= 10

        # Dominant language
        if features["ratio_chinese"] > 0.5:
            features["dominant_language"] = "chinese"
        elif features["ratio_english"] > 0.5:
            features["dominant_language"] = "english"
        else:
            features["dominant_language"] = "mixed"

        return features

    def _compute_adaptive_weights(self, query: str) -> tuple[float, float]:
        """Compute adaptive weights using nuanced query features.

        Adjusts weights based on:
        - Product code vs colloquial patterns
        - Query length (shorter queries benefit from sparse/BM25)
        - Language dominance (Chinese queries with mixed English terms benefit from dense)
        - Keyword count (multi-keyword queries benefit from sparse)
        """
        features = self._detect_query_features(query)

        dense_weight = self.default_dense_weight
        sparse_weight = self.default_sparse_weight

        # Product code detection: heavily boost sparse
        if features["has_product_code"]:
            dense_weight = self.product_code_dense_weight
            sparse_weight = self.product_code_sparse_weight

        # Colloquial detection: boost dense
        elif features["is_colloquial"]:
            dense_weight = self.colloquial_dense_weight
            sparse_weight = self.colloquial_sparse_weight

        # Length-based adjustments
        if features["is_short_query"]:
            # Short queries: slightly boost sparse (exact match matters more)
            shift = 0.05
            dense_weight = max(0.1, dense_weight - shift)
            sparse_weight = min(0.9, sparse_weight + shift)

        if features["query_length"] > 100:
            # Very long queries: slightly boost dense (semantic understanding)
            shift = 0.05
            dense_weight = min(0.9, dense_weight + shift)
            sparse_weight = max(0.1, sparse_weight - shift)

        # Language-based adjustments
        if features["dominant_language"] == "chinese" and features["has_english"]:
            # Mixed Chinese-English: boost dense for semantic matching
            shift = 0.03
            dense_weight = min(0.9, dense_weight + shift)
            sparse_weight = max(0.1, sparse_weight - shift)

        # Keyword count adjustments
        if features["keyword_count"] >= 5:
            # Multi-keyword: boost sparse for precision
            shift = 0.03
            dense_weight = max(0.1, dense_weight - shift)
            sparse_weight = min(0.9, sparse_weight + shift)

        # Numbers-heavy queries: boost sparse
        if features["ratio_numbers"] > 0.3:
            shift = 0.05
            dense_weight = max(0.1, dense_weight - shift)
            sparse_weight = min(0.9, sparse_weight + shift)

        # Normalize to ensure sum = 1.0
        total = dense_weight + sparse_weight
        if total > 0:
            dense_weight = dense_weight / total
            sparse_weight = sparse_weight / total

        return (dense_weight, sparse_weight)

    def _get_dynamic_weights(self, query: str) -> tuple[float, float]:
        """Determine dynamic weights based on query features.

        Uses adaptive weights when query is non-empty, falls back to defaults.
        """
        if not query or not query.strip():
            return (self.default_dense_weight, self.default_sparse_weight)
        return self._compute_adaptive_weights(query)

    @staticmethod
    def normalize_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize RRF scores to [0, 1] range.

        Args:
            results: List of result dicts with 'score' key.

        Returns:
            Results with normalized scores in [0, 1] range.
        """
        if not results:
            return results

        scores = [r["score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            for r in results:
                r["score"] = 0.5
            return results

        score_range = max_score - min_score
        for r in results:
            r["score"] = (r["score"] - min_score) / score_range

        return results

    def fuse(
        self,
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, Any]],
        query: str = "",
        top_k: int = 50,
    ) -> list[dict[str, Any]]:
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

        scores: dict[str, dict[str, Any]] = {}

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

        score_list = list(scores.values())

        # Boundary handling: when all scores are zero or identical, return in original order
        unique_scores = set(r["score"] for r in score_list)
        if len(unique_scores) <= 1:
            sorted_results = score_list
        else:
            sorted_results = sorted(
                score_list,
                key=lambda x: x["score"],
                reverse=True,
            )

        # Filter by minimum score threshold
        if self.min_score_threshold > 0:
            sorted_results = [
                r for r in sorted_results
                if r["score"] >= self.min_score_threshold
            ]

        return sorted_results[:top_k]

    def fuse_with_boost(
        self,
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, Any]],
        query: str = "",
        top_k: int = 50,
        boost_factor: float = 1.0,
        boost_path: str = "dense",
    ) -> list[dict[str, Any]]:
        """Fuse results with a boost factor applied to one retrieval path.

        Args:
            dense_results: Results from dense retrieval.
            sparse_results: Results from sparse retrieval.
            query: Original query for dynamic weight detection.
            top_k: Number of fused results to return.
            boost_factor: Multiplier applied to the boosted path's scores.
            boost_path: Which path to boost ('dense' or 'sparse').

        Returns:
            Fused and sorted result list.
        """
        # Handle edge case: both empty
        if not dense_results and not sparse_results:
            return []

        dense_weight, sparse_weight = self._get_dynamic_weights(query)

        # Apply boost factor
        if boost_path == "dense":
            dense_weight *= boost_factor
        elif boost_path == "sparse":
            sparse_weight *= boost_factor

        # Renormalize weights
        total = dense_weight + sparse_weight
        if total > 0:
            dense_weight /= total
            sparse_weight /= total

        scores: dict[str, dict[str, Any]] = {}

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

        score_list = list(scores.values())

        # Boundary handling
        unique_scores = set(r["score"] for r in score_list)
        if len(unique_scores) <= 1:
            sorted_results = score_list
        else:
            sorted_results = sorted(
                score_list,
                key=lambda x: x["score"],
                reverse=True,
            )

        if self.min_score_threshold > 0:
            sorted_results = [
                r for r in sorted_results
                if r["score"] >= self.min_score_threshold
            ]

        return sorted_results[:top_k]