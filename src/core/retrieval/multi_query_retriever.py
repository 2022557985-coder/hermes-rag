"""Multi-query retrieval: generate query variants and merge results."""

import logging
import re
from typing import Any

logger = logging.getLogger("hermes_rag")


class MultiQueryRetriever:
    """Generate query variants and merge multi-path retrieval results.

    Improves recall by:
    1. Generating 2-3 query variants from the original query
    2. Running retrieval for each variant
    3. Merging all results using RRF-like scoring
    """

    # Chinese question patterns for rephrasing
    QUESTION_PATTERNS = [
        (r"^(.+)是什么[？?]?$", r"\1 定义 含义"),
        (r"^(.+)是谁[？?]?$", r"\1 人物 背景 介绍"),
        (r"^(.+)怎么(做|办|处理|解决)[？?]?$", r"\1 方法 步骤 教程"),
        (r"^如何(.+)[？?]?$", r"\1 方法 步骤"),
        (r"^(.+)和(.+)的区别[？?]?$", r"\1 \2 对比 不同"),
        (r"^(.+)有哪些[？?]?$", r"\1 列表 分类"),
        (r"^(.+)的特点[？?]?$", r"\1 特征 特性"),
        (r"^(.+)怎么样[？?]?$", r"\1 评价 特点"),
    ]

    # Conceptual question patterns - extract key terms
    CONCEPTUAL_PATTERNS = [
        r"什么是(.+)",
        r"(.+)的定义",
        r"(.+)的概念",
        r"介绍一下(.+)",
        r"请说明(.+)",
    ]

    def __init__(self, max_variants: int = 3, enabled: bool = True):
        self.max_variants = max_variants
        self.enabled = enabled

    def generate_variants(self, query: str) -> list[str]:
        """Generate query variants for better recall.

        Args:
            query: Original query string.

        Returns:
            List of query variant strings (includes original).
        """
        if not query or not query.strip():
            return [query]

        query = query.strip()
        variants: list[str] = [query]
        seen: set[str] = {query}

        # Strategy 1: Pattern-based rephrasing for Chinese questions
        for pattern, replacement in self.QUESTION_PATTERNS:
            m = re.match(pattern, query)
            if m:
                variant = m.expand(replacement).strip()
                if variant and variant != query and variant not in seen:
                    variants.append(variant)
                    seen.add(variant)
                break

        # Strategy 2: Extract key terms (remove question words)
        key_terms = self._extract_key_terms(query)
        if key_terms and key_terms not in seen:
            variants.append(key_terms)
            seen.add(key_terms)

        # Strategy 3: For mixed Chinese-English, add English-only variant
        english_terms = self._extract_english_terms(query)
        if english_terms and english_terms not in seen:
            variants.append(english_terms)
            seen.add(english_terms)

        # Limit to max_variants
        return variants[: self.max_variants]

    def _extract_key_terms(self, query: str) -> str:
        """Extract key terms by removing question words and stop words."""
        # Remove common question words
        question_words = [
            "是什么", "是谁", "什么是", "怎么", "如何", "怎样",
            "为什么", "哪些", "哪个", "请问", "能否", "可以",
            "能不能", "是否", "帮我", "请", "介绍一下",
            "？", "?", "吗", "呢", "吧",
        ]
        result = query
        for word in question_words:
            result = result.replace(word, " ")
        # Clean up and join
        terms = [t.strip() for t in result.split() if t.strip()]
        return " ".join(terms)

    def _extract_english_terms(self, query: str) -> str:
        """Extract English/technical terms from a mixed query."""
        english_words = re.findall(r'[a-zA-Z]{2,}', query)
        if len(english_words) >= 1:
            return " ".join(english_words)
        return ""

    def retrieve(
        self,
        query: str,
        dense_retriever: Any,
        sparse_retriever: Any,
        dense_top_k: int = 100,
        sparse_top_k: int = 100,
        rrf_fusion: Any = None,
        query_expander: Any = None,
    ) -> list[dict[str, Any]]:
        """Run multi-query retrieval: generate variants, retrieve, merge.

        Args:
            query: Original query.
            dense_retriever: Dense retriever instance.
            sparse_retriever: Sparse retriever instance.
            dense_top_k: Candidates per variant from dense.
            sparse_top_k: Candidates per variant from sparse.
            rrf_fusion: RRF fusion instance for merging.
            query_expander: Query expander for synonym expansion.

        Returns:
            Merged and deduplicated result list.
        """
        if not self.enabled:
            return []

        variants = self.generate_variants(query)
        if len(variants) <= 1:
            return []

        logger.debug(f"Multi-query variants: {variants}")

        all_dense: list[dict[str, Any]] = []
        all_sparse: list[dict[str, Any]] = []

        for variant in variants:
            # Apply query expansion to each variant
            expanded = variant
            if query_expander:
                try:
                    expansion = query_expander.expand(variant)
                    expanded = expansion.get("expanded", variant)
                except Exception:
                    pass

            # Dense retrieval
            if dense_retriever:
                try:
                    dense_results = dense_retriever.retrieve(expanded, dense_top_k // len(variants))
                    all_dense.extend(dense_results or [])
                except Exception as e:
                    logger.debug(f"Dense retrieval failed for variant '{variant[:50]}': {e}")

            # Sparse retrieval
            if sparse_retriever:
                try:
                    sparse_results = sparse_retriever.retrieve(variant, sparse_top_k // len(variants))
                    all_sparse.extend(sparse_results or [])
                except Exception as e:
                    logger.debug(f"Sparse retrieval failed for variant '{variant[:50]}': {e}")

        # Merge all results using RRF if available
        if rrf_fusion and (all_dense or all_sparse):
            try:
                merged = rrf_fusion.fuse(
                    all_dense, all_sparse, query=query,
                    top_k=max(dense_top_k, sparse_top_k),
                )
                return merged
            except Exception as e:
                logger.warning(f"Multi-query RRF fusion failed: {e}")

        # Fallback: deduplicate and return
        return self._deduplicate(all_dense + all_sparse)

    @staticmethod
    def _deduplicate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Deduplicate by chunk_id, keeping highest score."""
        seen: dict[str, dict[str, Any]] = {}
        for r in results:
            cid = r.get("chunk_id", "")
            if cid not in seen or r.get("score", 0) > seen[cid].get("score", 0):
                seen[cid] = r
        return sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)