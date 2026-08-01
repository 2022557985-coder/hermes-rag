"""Lightweight heuristic reranker for Chinese retrieval.

Uses keyword overlap, positional signals, and source diversity to rerank
candidates without loading a large model. Suitable for 16GB machines.
"""

import logging
import re
from typing import Any

logger = logging.getLogger("hermes_rag")


class HeuristicReranker:
    """Lightweight heuristic reranker using keyword overlap + positional signals.

    Designed for Chinese retrieval scenarios where cross-encoder models
    (e.g., bge-reranker-v2-m3) are too large for the available hardware.

    Scoring components:
    1. Original RRF score: dominant, so re-ranking is conservative
    2. Keyword overlap: Jaccard similarity between query terms and document text
    3. Heading match: Bonus when query terms match document headings
    4. Positional bias: Earlier chunks get a slight boost
    5. Source diversity: Bonus for results from different sources
    """

    def __init__(
        self,
        keyword_overlap_weight: float = 0.20,
        position_weight: float = 0.05,
        heading_match_weight: float = 0.20,
        original_score_weight: float = 0.55,
        max_candidates: int = 50,
    ):
        self.keyword_overlap_weight = keyword_overlap_weight
        self.position_weight = position_weight
        self.heading_match_weight = heading_match_weight
        self.original_score_weight = original_score_weight
        self.max_candidates = max_candidates

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract meaningful keywords from text (Chinese + English).

        Filters out stop words, single characters, and pure numbers.
        """
        tokens = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{2,}|\d{2,}', text.lower())
        # Stop words (Chinese + English)
        stop_words = {
            "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一",
            "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
            "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
            "什么", "怎么", "如何", "为什么", "哪", "吗", "吧", "呢", "啊", "哦",
            "与", "或", "但", "而", "且", "从", "把", "被", "让", "给", "对",
            "以", "及", "向", "所", "其", "为", "之", "将", "已", "可", "能",
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "shall",
            "should", "can", "could", "may", "might", "must", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "and", "but", "or",
            "not", "no", "nor", "so", "if", "than", "too", "very", "this", "that",
            "these", "those", "here", "there", "then", "now", "also", "about",
        }
        return {t for t in tokens if t not in stop_words}

    def _keyword_overlap_score(self, query_keywords: set[str], doc_text: str) -> float:
        """Compute Jaccard similarity between query keywords and document keywords.

        Returns:
            Score in [0, 1].
        """
        if not query_keywords:
            return 0.0

        doc_keywords = self._extract_keywords(doc_text)
        if not doc_keywords:
            return 0.0

        intersection = query_keywords & doc_keywords
        union = query_keywords | doc_keywords

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _position_score(self, rank: int, total: int) -> float:
        """Compute position-based score: earlier chunks get a slight boost.

        Args:
            rank: 0-based rank of the chunk.
            total: Total number of candidates.

        Returns:
            Score in [0, 1].
        """
        if total <= 1:
            return 1.0
        # Linear decay: rank 0 -> 1.0, rank (total-1) -> 0.0
        return 1.0 - (rank / max(total - 1, 1))

    def _heading_match_score(self, query: str, heading: str) -> float:
        """Compute heading match score: bonus when query terms appear in heading.

        Args:
            query: The user query.
            heading: Document heading path.

        Returns:
            Score in [0, 1].
        """
        if not heading:
            return 0.0

        query_keywords = self._extract_keywords(query)
        if not query_keywords:
            return 0.0

        heading_lower = heading.lower()
        matched = sum(1 for kw in query_keywords if kw.lower() in heading_lower)
        return min(1.0, matched / max(len(query_keywords), 1))

    def _source_diversity_boost(
        self,
        candidates: list[dict[str, Any]],
        scores: list[float],
    ) -> list[float]:
        """Apply a small boost for results from diverse sources.

        First occurrence of each source gets a slight bonus.
        Later occurrences from the same source get slightly penalized.

        Args:
            candidates: List of candidate dicts.
            scores: Current scores list to modify.

        Returns:
            Modified scores list.
        """
        seen_sources: set[str] = set()
        boosted = list(scores)

        for i, candidate in enumerate(candidates):
            source = candidate.get("metadata", {}).get("source", "")
            if source and source not in seen_sources:
                boosted[i] = min(1.0, boosted[i] + 0.03)
                seen_sources.add(source)
            elif source:
                boosted[i] = max(0.0, boosted[i] - 0.01)

        return boosted

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rerank candidates using heuristic scoring.

        Args:
            query: Query string.
            candidates: List of candidate dicts with 'text', 'score', 'metadata'.

        Returns:
            Reranked candidates sorted by heuristic score.
        """
        if not query or not query.strip():
            return candidates

        if not candidates:
            return []

        candidates = candidates[: self.max_candidates]
        total = len(candidates)

        query_keywords = self._extract_keywords(query)

        raw_scores = [float(candidate.get("score", 0.0)) for candidate in candidates]
        min_raw, max_raw = min(raw_scores), max(raw_scores)
        raw_range = max_raw - min_raw

        scores: list[float] = []
        for i, candidate in enumerate(candidates):
            text = candidate.get("text", "")
            original_score = candidate.get("score", 0.0)
            heading = candidate.get("metadata", {}).get("heading_path", "")

            # Component scores
            kw_score = self._keyword_overlap_score(query_keywords, text)
            pos_score = self._position_score(i, total)
            heading_score = self._heading_match_score(query, heading)

            # Normalize the original RRF score to [0, 1] so the weighted
            # combination is scale-consistent: the fusion ranking stays dominant
            # while heading/keyword signals only break close ties.
            if raw_range > 0:
                normalized_original = (original_score - min_raw) / raw_range
            else:
                normalized_original = 1.0

            # Weighted combination
            final_score = (
                self.keyword_overlap_weight * kw_score
                + self.position_weight * pos_score
                + self.heading_match_weight * heading_score
                + self.original_score_weight * normalized_original
            )

            scores.append(final_score)

        # Apply source diversity boost
        scores = self._source_diversity_boost(candidates, scores)

        # Attach scores
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = scores[i]
            candidate["score"] = scores[i]

        # Sort by heuristic score descending
        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        return candidates