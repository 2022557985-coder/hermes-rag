"""Multi-path retrieval pipeline orchestrator."""

import gc
import logging
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from src.utils.timer import Stopwatch

logger = logging.getLogger("hermes_rag")


class QueryClassifier:
    """Classify query type for adaptive retrieval strategy.

    Handles three query types:
    - factual: Exact-match queries (product codes, error codes, numeric IDs)
    - conceptual: Semantic/conceptual queries (definitions, comparisons, explanations)
    - procedural: How-to queries (steps, procedures, instructions)
    """

    # Factual patterns: product codes, error codes, version numbers, IDs
    FACTUAL_PATTERNS = [
        r"[A-Z]{2,}[\d\-_]{2,}[A-Z\d]*",          # Product codes: ABC-1234, XY_200
        r"[A-Z]{2,}\d{2,}",                         # Error codes: ERR404, HTTP500
        r"[A-Z]{2,}(?:_[A-Z]{2,})+",                # Underscore error codes: ERR_CONNECTION_REFUSED
        r"\d+\.\d+\.\d+",                            # Version numbers: 3.14.2
        r"#[A-Za-z0-9]+",                            # Hashtag IDs: #1234
        r"ID[\s:]+\d+",                              # Explicit IDs: ID 12345
    ]

    # Procedural keywords (Chinese + English)
    PROCEDURAL_KEYWORDS_CN = [
        "如何", "怎么", "怎样", "步骤", "流程",
        "教程", "指南", "怎么做", "如何做",
    ]
    PROCEDURAL_KEYWORDS_EN = [
        "how to", "how do i", "steps", "procedure",
        "tutorial", "guide", "walkthrough", "install",
    ]

    # Conceptual-only patterns: "如何" used in explanatory (not action) contexts
    CONCEPTUAL_HOW_PATTERNS = [
        r"如何(计算|工作|运作|运行|处理|影响|区别|不同|选择|判断|衡量|评估|检测)",
    ]

    @classmethod
    def classify(cls, query: str) -> str:
        """Classify query into: 'factual', 'procedural', or 'conceptual'.

        Args:
            query: Raw query string.

        Returns:
            One of 'factual', 'procedural', 'conceptual'.
        """
        has_factual = False
        has_procedural = False

        # Check factual patterns (but don't return immediately)
        for pattern in cls.FACTUAL_PATTERNS:
            if re.search(pattern, query):
                has_factual = True
                break

        # Check procedural keywords
        query_lower = query.lower()
        for kw in cls.PROCEDURAL_KEYWORDS_CN:
            if kw in query:
                has_procedural = True
                break
        if not has_procedural:
            for kw in cls.PROCEDURAL_KEYWORDS_EN:
                if kw in query_lower:
                    has_procedural = True
                    break

        # If both factual and procedural, prefer procedural when the query
        # has clear procedural intent (e.g., "ERR500 how to fix")
        if has_factual and has_procedural:
            return "procedural"

        if has_factual:
            return "factual"

        if has_procedural:
            # Check if "如何" is used in a conceptual (explanatory) context
            if "如何" in query:
                for pattern in cls.CONCEPTUAL_HOW_PATTERNS:
                    if re.search(pattern, query):
                        return "conceptual"
            return "procedural"

        return "conceptual"


class RetrievalPipeline:
    """Orchestrates multi-path retrieval: query expansion -> parallel recall -> RRF fusion -> reranking.

    Features:
    - Query type classification for adaptive retrieval
    - Result deduplication by chunk_id
    - Similarity threshold filtering for low-quality results
    - Production metrics collection (cache hit rate, latency, recall paths)
    - Embedding computation reuse (avoids double-encoding)
    """

    def __init__(
        self,
        index_manager,
        query_expander=None,
        dense_retriever=None,
        sparse_retriever=None,
        rule_retriever=None,
        rrf_fusion=None,
        cross_encoder=None,
        cache=None,
        config=None,
        metrics=None,
    ):
        self.index_manager = index_manager
        self.query_expander = query_expander
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rule_retriever = rule_retriever
        self.rrf_fusion = rrf_fusion
        self.cross_encoder = cross_encoder
        self.cache = cache
        self.config = config or {}
        self.metrics = metrics  # Production MetricsCollector

    def _get_query_embedding(self, query: str) -> Optional[Any]:
        """Compute query embedding once, with caching for reuse.

        This avoids the double-encoding bug: previously the embedding was computed
        once for cache lookup and again for cache set.
        """
        try:
            if self.index_manager and self.index_manager.vector_store:
                embedder = self.index_manager.vector_store.get_embedder()
                return embedder.encode([query], normalize_embeddings=True)[0]
        except Exception:
            return None
        return None

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate results by chunk_id, keeping the highest-scored entry.

        Args:
            results: List of result dicts, potentially with duplicates.

        Returns:
            Deduplicated list sorted by score descending.
        """
        seen = {}
        for r in results:
            chunk_id = r.get("chunk_id", "")
            if chunk_id not in seen or r.get("score", 0) > seen[chunk_id].get("score", 0):
                seen[chunk_id] = r
        return sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)

    def _filter_by_threshold(
        self,
        results: List[Dict[str, Any]],
        min_score: float = 0.001,
    ) -> List[Dict[str, Any]]:
        """Filter out results below a minimum score threshold.

        Args:
            results: List of result dicts.
            min_score: Minimum score to keep (default: 0.001).

        Returns:
            Filtered results list.
        """
        return [r for r in results if r.get("score", 0) >= min_score]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_reranker: bool = True,
    ) -> Dict[str, Any]:
        """Execute the full retrieval pipeline.

        Args:
            query: User query.
            top_k: Number of final results.
            use_reranker: Whether to use cross-encoder reranking.

        Returns:
            dict with 'results', 'query_info', 'timing'.
        """
        sw = Stopwatch()
        recall_paths = []
        reranker_used = False
        reranker_timed_out = False
        query_embedding = None

        # Step 0: Check cache
        if self.cache:
            query_embedding = self._get_query_embedding(query)
            cached = self.cache.get(query, query_embedding=query_embedding)
            if cached is not None:
                total_latency = sw.lap("total")
                if self.metrics:
                    self.metrics.record_query(
                        cached=True,
                        recall_paths=["cached"],
                        total_latency=total_latency,
                        component_timings={"cache_lookup": total_latency},
                    )
                return {
                    "results": cached[:top_k],
                    "query_info": {
                        "original": query,
                        "cached": True,
                        "query_type": QueryClassifier.classify(query),
                    },
                    "timing": {"total": sw.lap("total")},
                }

        # Step 1: Query expansion
        expanded = query
        hyde_text = None
        if self.query_expander:
            expansion = self.query_expander.expand(query)
            expanded = expansion["expanded"]
            hyde_text = expansion.get("hyde_text")
        sw.lap("query_expansion")

        # Step 2: Check rule-based filters
        metadata_filter = None
        if self.rule_retriever:
            metadata_filter = self.rule_retriever.build_filter(query)

        # Step 3: Parallel multi-path retrieval
        dense_results = []
        sparse_results = []

        dense_top_k = self.config.get("dense_top_k", 100)
        sparse_top_k = self.config.get("sparse_top_k", 100)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}

            if self.dense_retriever:
                futures["dense"] = executor.submit(
                    self.dense_retriever.retrieve,
                    expanded,
                    dense_top_k,
                    metadata_filter,
                )

            if self.sparse_retriever:
                futures["sparse"] = executor.submit(
                    self.sparse_retriever.retrieve,
                    query,
                    sparse_top_k,
                )

            for name, future in futures.items():
                try:
                    result = future.result(timeout=5)
                    if name == "dense":
                        dense_results = result
                        recall_paths.append("dense")
                    elif name == "sparse":
                        sparse_results = result
                        recall_paths.append("sparse")
                except Exception as e:
                    logger.warning(f"Error in {name} retrieval: {e}")

        sw.lap("multi_path_retrieval")

        # Step 4: RRF fusion
        fusion_top_k = self.config.get("fusion_top_k", 50)
        if self.rrf_fusion:
            fused = self.rrf_fusion.fuse(
                dense_results,
                sparse_results,
                query=query,
                top_k=fusion_top_k,
            )
        else:
            fused = dense_results[:fusion_top_k] if dense_results else sparse_results[:fusion_top_k]

        # Deduplicate after fusion
        fused = self._deduplicate_results(fused)

        sw.lap("rrf_fusion")

        # Step 5: Reranking (optional)
        if use_reranker and self.cross_encoder and self.config.get("reranking", {}).get("enabled", True):
            reranker_used = True
            try:
                rerank_timeout = self.config.get("reranking", {}).get("timeout_seconds", 1.5)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self.cross_encoder.rerank, query, fused
                    )
                    reranked = future.result(timeout=rerank_timeout)
                    fused = reranked
            except TimeoutError:
                logger.warning("Reranking timed out, using RRF results")
                reranker_timed_out = True
            except Exception as e:
                logger.warning(f"Reranking error: {e}, using RRF results")

        sw.lap("reranking")

        # Step 6: Deduplicate, filter by threshold, and slice to final top_k
        fused = self._deduplicate_results(fused)
        final_results = self._filter_by_threshold(fused)[:top_k]

        # Cache results
        if self.cache:
            try:
                # Reuse query_embedding from cache lookup (avoids double-encoding)
                if query_embedding is None:
                    query_embedding = self._get_query_embedding(query)
                self.cache.set(query, final_results, query_embedding=query_embedding)
            except (ValueError, TypeError, AttributeError):
                pass

        total_latency = sw.lap("total")

        # Record production metrics
        if self.metrics:
            durations = sw.durations()
            component_timings = {
                "query_expansion": durations.get("query_expansion", 0),
                "dense_retrieval": durations.get("multi_path_retrieval", 0),
                "rrf_fusion": durations.get("rrf_fusion", 0),
                "reranking": durations.get("reranking", 0),
                "total": total_latency,
            }
            self.metrics.record_query(
                cached=False,
                recall_paths=recall_paths,
                total_latency=total_latency,
                component_timings=component_timings,
                reranker_used=reranker_used,
                reranker_timed_out=reranker_timed_out,
            )

        # Cleanup
        gc.collect()

        return {
            "results": final_results,
            "query_info": {
                "original": query,
                "expanded": expanded,
                "hyde_used": hyde_text is not None,
                "query_type": QueryClassifier.classify(query),
            },
            "timing": {
                "total": total_latency,
                "details": sw.summary(),
            },
        }