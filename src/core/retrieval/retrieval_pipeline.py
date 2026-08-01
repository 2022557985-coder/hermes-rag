"""Multi-path retrieval pipeline orchestrator."""

import gc
import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Any

from src.utils.timer import Stopwatch

logger = logging.getLogger("hermes_rag")


class QueryClassifier:
    """Classify query type for adaptive retrieval strategy.

    Handles three query types:
    - factual: Exact-match queries (product codes, error codes, numeric IDs, dates, phone numbers)
    - conceptual: Semantic/conceptual queries (definitions, comparisons, explanations)
    - procedural: How-to queries (steps, procedures, instructions)
    """

    # Factual patterns: product codes, error codes, version numbers, IDs, dates, phone numbers
    FACTUAL_PATTERNS = [
        r"[A-Z]{2,}[\d\-_]{2,}[A-Z\d]*",          # Product codes: ABC-1234, XY_200
        r"[A-Z]{2,}\d{2,}",                         # Error codes: ERR404, HTTP500
        r"[A-Z]{2,}(?:_[A-Z]{2,})+",                # Underscore error codes: ERR_CONNECTION_REFUSED
        r"\d+\.\d+\.\d+",                            # Version numbers: 3.14.2
        r"#[A-Za-z0-9]+",                            # Hashtag IDs: #1234
        r"ID[\s:]+\d+",                              # Explicit IDs: ID 12345
        r"[\u4e00-\u9fa5]+[\-\s]?\d+号",             # Chinese product codes: 型号 A-1234号
        r"\d{4}[\-\/]\d{1,2}[\-\/]\d{1,2}",          # Dates: 2024-01-15, 2024/01/15
        r"1[3-9]\d{9}",                               # Chinese phone numbers: 13800138000
        r"\d{3}[\-\s]?\d{4}[\-\s]?\d{4}",            # Phone numbers with separators: 010-1234-5678
        r"\d{17}[\dXx]",                              # Chinese ID card numbers: 18 digits
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

    # Direct conceptual keywords
    CONCEPTUAL_KEYWORDS_CN = [
        "定义", "含义", "概念", "原理", "特点", "特征", "分类",
        "区别", "不同", "优点", "缺点", "优势", "劣势",
        "是什么", "什么是", "是谁", "谁是谁", "介绍",
        "概述", "总结", "背景", "历史", "发展",
        "对比", "比较", "关系", "作用", "功能", "用途",
    ]

    @classmethod
    def classify(cls, query: str) -> str:
        """Classify query into: 'factual', 'procedural', or 'conceptual'.

        Args:
            query: Raw query string.

        Returns:
            One of 'factual', 'procedural', 'conceptual'.
        """
        # Edge case: empty or whitespace-only query
        if not query or not query.strip():
            logger.debug("Empty or whitespace-only query, defaulting to conceptual")
            return "conceptual"

        query = query.strip()
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
            # Check if query contains explicit conceptual keywords
            for kw in cls.CONCEPTUAL_KEYWORDS_CN:
                if kw in query:
                    return "conceptual"
            return "procedural"

        # Check conceptual keywords even without procedural markers
        for kw in cls.CONCEPTUAL_KEYWORDS_CN:
            if kw in query:
                return "conceptual"

        return "conceptual"

    @classmethod
    def classify_with_confidence(cls, query: str) -> tuple[str, float]:
        """Classify query and return type with a confidence score.

        Confidence is based on how many signals matched the chosen type
        vs. signals that could have matched other types. Returns a score
        in [0.0, 1.0] where higher = more confident.

        Args:
            query: Raw query string.

        Returns:
            Tuple of (query_type, confidence_score).
        """
        if not query or not query.strip():
            return ("conceptual", 1.0)

        query = query.strip()
        query_lower = query.lower()

        # Count factual signal matches
        factual_count = sum(
            1 for pattern in cls.FACTUAL_PATTERNS if re.search(pattern, query)
        )
        # Count procedural signal matches
        procedural_count = sum(
            1 for kw in cls.PROCEDURAL_KEYWORDS_CN if kw in query
        ) + sum(1 for kw in cls.PROCEDURAL_KEYWORDS_EN if kw in query_lower)
        # Count conceptual signal matches
        conceptual_count = sum(
            1 for kw in cls.CONCEPTUAL_KEYWORDS_CN if kw in query
        ) + sum(1 for pattern in cls.CONCEPTUAL_HOW_PATTERNS if re.search(pattern, query))

        total_signals = factual_count + procedural_count + conceptual_count

        query_type = cls.classify(query)

        if total_signals == 0:
            # No signals matched, low-confidence default
            return (query_type, 0.3)

        if query_type == "factual":
            type_signals = factual_count
        elif query_type == "procedural":
            type_signals = procedural_count
        else:
            type_signals = conceptual_count

        confidence = type_signals / total_signals
        return (query_type, round(confidence, 2))


class RetrievalPipeline:
    """Orchestrates multi-path retrieval: query expansion -> parallel recall -> RRF fusion -> reranking.

    Features:
    - Query type classification for adaptive retrieval
    - Result deduplication by chunk_id
    - Similarity threshold filtering for low-quality results
    - Production metrics collection (cache hit rate, latency, recall paths)
    - Embedding computation reuse (avoids double-encoding)
    - Query validation and normalization
    - Batch retrieval for evaluation
    """

    # Maximum query length to prevent abuse / excessive embedding cost
    MAX_QUERY_LENGTH = 2000

    def __init__(
        self,
        index_manager: Any = None,
        query_expander: Any = None,
        dense_retriever: Any = None,
        sparse_retriever: Any = None,
        rule_retriever: Any = None,
        rrf_fusion: Any = None,
        cross_encoder: Any = None,
        cache: Any = None,
        config: dict[str, Any] | None = None,
        metrics: Any = None,
        multi_query_retriever: Any = None,
    ) -> None:
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
        self.multi_query_retriever = multi_query_retriever

    @staticmethod
    def _validate_query(query: str) -> tuple[bool, str | None]:
        """Validate query before processing.

        Args:
            query: Raw query string.

        Returns:
            Tuple of (is_valid, error_message). Error message is None if valid.
        """
        if not query or not isinstance(query, str):
            return False, "Query must be a non-empty string"

        stripped = query.strip()
        if not stripped:
            return False, "Query is empty or whitespace-only"

        if len(stripped) > RetrievalPipeline.MAX_QUERY_LENGTH:
            return False, f"Query exceeds maximum length of {RetrievalPipeline.MAX_QUERY_LENGTH} characters"

        # Check for excessive special characters (potential injection / garbage)
        special_ratio = sum(1 for c in stripped if not c.isalnum() and not c.isspace()) / max(len(stripped), 1)
        if special_ratio > 0.8:
            return False, "Query contains too many special characters"

        return True, None

    @staticmethod
    def _normalize_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ensure all result scores are normalized to [0, 1] range.

        If scores are outside [0, 1], min-max normalizes them into that range.
        Results with negative scores are clamped to 0.

        Args:
            results: List of result dicts with 'score' key.

        Returns:
            Results with normalized scores in [0, 1].
        """
        if not results:
            return results

        scores = [r.get("score", 0) for r in results]
        min_s = min(scores)
        max_s = max(scores)

        if 0 <= min_s and max_s <= 1:
            # Already in range, just clamp any stray negatives
            for r in results:
                if r.get("score", 0) < 0:
                    r["score"] = 0.0
            return results

        # Need normalization
        score_range = max_s - min_s
        if score_range == 0:
            for r in results:
                r["score"] = 0.5
            return results

        for r in results:
            r["score"] = max(0.0, min(1.0, (r.get("score", 0) - min_s) / score_range))
        return results

    def _safe_get_embedding(self, query: str) -> Any | None:
        """Compute embedding with edge-case handling.

        Handles edge cases:
        - Empty query (returns None)
        - Very long query (truncates to safe length)
        - Embedding failures (logs and returns None)

        Args:
            query: The query string to encode.

        Returns:
            Embedding vector or None on failure.
        """
        if not query or not query.strip():
            logger.debug("_safe_get_embedding: empty query, skipping")
            return None

        query = query.strip()
        if len(query) > self.MAX_QUERY_LENGTH:
            logger.debug(
                f"_safe_get_embedding: truncating query from {len(query)} to {self.MAX_QUERY_LENGTH} chars"
            )
            query = query[:self.MAX_QUERY_LENGTH]

        try:
            if self.index_manager and self.index_manager.vector_store:
                embedder = self.index_manager.vector_store.get_embedder()
                return embedder.encode([query], normalize_embeddings=True)[0]
        except Exception as e:
            logger.warning(f"Failed to compute query embedding: {e}")
        return None

    def _get_query_embedding(self, query: str) -> Any | None:
        """Compute query embedding once, with caching for reuse.

        Delegates to _safe_get_embedding for edge-case handling.
        """
        return self._safe_get_embedding(query)

    def _deduplicate_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        results: list[dict[str, Any]],
        min_score: float = 0.001,
    ) -> list[dict[str, Any]]:
        """Filter out results below a minimum score threshold.

        Args:
            results: List of result dicts.
            min_score: Minimum score to keep (default: 0.001).

        Returns:
            Filtered results list.
        """
        return [r for r in results if r.get("score", 0) >= min_score]

    def _expand_context_window(
        self,
        results: list[dict[str, Any]],
        top_k: int,
        window: int = 1,
    ) -> list[dict[str, Any]]:
        """Expand results with neighboring chunks for better context.

        For each result, fetches up to `window` neighboring chunks (before and after)
        and appends them to the result list. Neighbors get a slightly reduced score
        to keep them ordered after the primary results.

        Args:
            results: Primary retrieval results (already scored and sorted).
            top_k: Max results to return.
            window: Number of neighbors to fetch on each side.

        Returns:
            Expanded result list with at most top_k items.
        """
        if not results or window <= 0 or not self.index_manager:
            return results

        seen_ids = {r.get("chunk_id", "") for r in results}
        neighbors = []

        for r in results:
            chunk_id = r.get("chunk_id", "")
            neighbor_chunks = self.index_manager.get_neighbor_chunks(
                chunk_id, window=window
            )
            for nc in neighbor_chunks:
                nc_id = nc.get("chunk_id", "")
                if nc_id not in seen_ids:
                    seen_ids.add(nc_id)
                    # Neighbors get a score slightly below the primary result
                    nc["score"] = r.get("score", 0) * 0.85
                    nc["_is_neighbor"] = True
                    neighbors.append(nc)

        # Combine: primary results first, then neighbors. The contract is that
        # the caller never receives more than top_k results.
        expanded = list(results) + neighbors
        return expanded[:top_k]

    def _run_retrieval_path(
        self,
        retriever: Any,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
        path_name: str = "unknown",
    ) -> list[dict[str, Any]]:
        """Run a single retrieval path with retry logic.

        Args:
            retriever: The retriever instance.
            query: Query string.
            top_k: Number of results.
            metadata_filter: Optional metadata filter.
            path_name: Human-readable name for logging.

        Returns:
            List of result dicts (empty list on failure).
        """
        max_retries = self.config.get("retrieval_retries", 1)
        self.config.get("retrieval_timeout", 5.0)

        for attempt in range(max_retries + 1):
            try:
                if metadata_filter is not None:
                    result = retriever.retrieve(query, top_k, metadata_filter)
                else:
                    result = retriever.retrieve(query, top_k)
                if attempt > 0:
                    logger.info(f"{path_name} retrieval succeeded on retry {attempt}")
                return result or []
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"{path_name} retrieval attempt {attempt + 1} failed: {e}. Retrying..."
                    )
                else:
                    logger.warning(
                        f"{path_name} retrieval failed after {max_retries + 1} attempts: {e}"
                    )
        return []

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_reranker: bool = True,
    ) -> dict[str, Any]:
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

        # Step 0: Validate query
        is_valid, error_msg = self._validate_query(query)
        if not is_valid:
            logger.warning(f"Query validation failed: {error_msg}")
            return {
                "results": [],
                "recall_paths": [],
                "from_cache": False,
                "query_info": {
                    "original": query,
                    "error": error_msg,
                    "query_type": "conceptual",
                },
                "timing": {"total": sw.lap("total")},
            }

        logger.debug(f"Retrieving for query: {query[:100]}... (type: {QueryClassifier.classify(query)})")

        # Step 1: Check cache (only compute embedding if cache is enabled)
        if self.cache:
            query_embedding = self._safe_get_embedding(query)
            cached = self.cache.get(query, query_embedding=query_embedding)
            if cached is not None:
                logger.debug("Cache hit, returning cached results")
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
                    "recall_paths": ["cached"],
                    "from_cache": True,
                    "query_info": {
                        "original": query,
                        "cached": True,
                        "query_type": QueryClassifier.classify(query),
                    },
                    "timing": {"total": sw.lap("total")},
                }

        # Step 2: Query expansion
        expanded = query
        hyde_text = None
        if self.query_expander:
            logger.debug("Running query expansion")
            expansion = self.query_expander.expand(query)
            expanded = expansion["expanded"]
            hyde_text = expansion.get("hyde_text")
        sw.lap("query_expansion")

        # Step 3: Check rule-based filters
        metadata_filter = None
        if self.rule_retriever:
            metadata_filter = self.rule_retriever.build_filter(query)

        # Step 4: Multi-query retrieval — generate variants for better recall
        dense_top_k = self.config.get("dense_top_k", 100)
        sparse_top_k = self.config.get("sparse_top_k", 100)

        multi_query_results: list[dict[str, Any]] = []
        if self.multi_query_retriever and self.multi_query_retriever.enabled:
            logger.debug("Running multi-query retrieval")
            multi_query_results = self.multi_query_retriever.retrieve(
                query=query,
                dense_retriever=self.dense_retriever,
                sparse_retriever=self.sparse_retriever,
                dense_top_k=dense_top_k,
                sparse_top_k=sparse_top_k,
                rrf_fusion=self.rrf_fusion,
                query_expander=self.query_expander,
            )
            if multi_query_results:
                recall_paths.append("multi_query")
        sw.lap("multi_query_retrieval")

        # Step 5: Parallel multi-path retrieval with retry and configurable timeouts
        dense_results: list[dict[str, Any]] = []
        sparse_results: list[dict[str, Any]] = []

        retrieval_timeout = self.config.get("retrieval_timeout", 5.0)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}

            if self.dense_retriever:
                futures["dense"] = executor.submit(
                    self._run_retrieval_path,
                    self.dense_retriever,
                    query,
                    dense_top_k,
                    metadata_filter,
                    "dense",
                )

            if self.sparse_retriever:
                futures["sparse"] = executor.submit(
                    self._run_retrieval_path,
                    self.sparse_retriever,
                    expanded,
                    sparse_top_k,
                    None,
                    "sparse",
                )

            for name, future in futures.items():
                try:
                    result = future.result(timeout=retrieval_timeout)
                    if name == "dense":
                        dense_results = result
                        if result:
                            recall_paths.append("dense")
                            logger.debug(f"Dense retrieval returned {len(result)} results")
                    elif name == "sparse":
                        sparse_results = result
                        if result:
                            recall_paths.append("sparse")
                            logger.debug(f"Sparse retrieval returned {len(result)} results")
                except TimeoutError:
                    logger.warning(f"{name} retrieval timed out after {retrieval_timeout}s")
                except Exception as e:
                    logger.warning(f"Error in {name} retrieval: {e}")

        sw.lap("multi_path_retrieval")

        # Step 5.5: Merge multi-query results into candidate pool (append, not prepend)
        if multi_query_results:
            # Append multi-query results to sparse pool (keyword-oriented)
            sparse_results = sparse_results + multi_query_results

        # Step 7: RRF fusion
        fusion_top_k = self.config.get("fusion_top_k", 50)
        if self.rrf_fusion:
            logger.debug("Running RRF fusion")
            fused = self.rrf_fusion.fuse(
                dense_results,
                sparse_results,
                query=query,
                top_k=fusion_top_k,
            )
        else:
            fused = dense_results[:fusion_top_k] if dense_results else sparse_results[:fusion_top_k]

        sw.lap("rrf_fusion")

        # Step 8: Reranking (optional)
        reranking_config = self.config.get("reranking", {})
        reranking_enabled = reranking_config.get("enabled", True) if isinstance(reranking_config, dict) else True
        if use_reranker and self.cross_encoder and reranking_enabled:
            reranker_used = True
            logger.debug("Running cross-encoder reranking")
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

        # Step 9: Deduplicate, normalize scores, filter by threshold, and slice to final top_k
        fused = self._deduplicate_results(fused)
        fused = self._normalize_scores(fused)
        final_results = self._filter_by_threshold(fused)[:top_k]

        # Step 9.5: Context window expansion — fetch neighboring chunks
        context_window = self.config.get("context_window", 1)
        if context_window > 0 and self.index_manager:
            final_results = self._expand_context_window(
                final_results, top_k, window=context_window
            )

        logger.debug(f"Final results: {len(final_results)} items after dedup/normalize/filter/context")

        # Cache results (only if cache is enabled)
        if self.cache:
            try:
                # Reuse query_embedding from cache lookup (avoids double-encoding)
                if query_embedding is None:
                    query_embedding = self._safe_get_embedding(query)
                self.cache.set(query, final_results, query_embedding=query_embedding)
                logger.debug("Results cached")
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Failed to cache results: {e}")
            except Exception as e:
                logger.warning(f"Unexpected cache error: {e}")

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
            "recall_paths": recall_paths,
            "from_cache": False,
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

    def retrieve_batch(
        self,
        queries: list[str],
        top_k: int = 5,
        use_reranker: bool = True,
        max_workers: int = 4,
    ) -> list[dict[str, Any]]:
        """Retrieve results for multiple queries in parallel.

        Useful for evaluation and bulk processing. Each query is processed
        independently through the full pipeline.

        Args:
            queries: List of query strings.
            top_k: Number of final results per query.
            use_reranker: Whether to use cross-encoder reranking.
            max_workers: Maximum number of parallel workers.

        Returns:
            List of result dicts, one per query, in the same order as input.
        """
        if not queries:
            logger.debug("retrieve_batch: empty query list")
            return []

        logger.info(f"Batch retrieval for {len(queries)} queries with {max_workers} workers")

        results: list[dict[str, Any] | None] = [None] * len(queries)

        def _retrieve_one(idx: int, q: str) -> tuple[int, dict[str, Any]]:
            return idx, self.retrieve(q, top_k=top_k, use_reranker=use_reranker)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_retrieve_one, i, q): i
                for i, q in enumerate(queries)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    i, result = future.result()
                    results[i] = result
                except Exception as e:
                    logger.warning(f"Batch retrieval failed for query {idx}: {e}")
                    results[idx] = {
                        "results": [],
                        "recall_paths": [],
                        "from_cache": False,
                        "query_info": {
                            "original": queries[idx],
                            "error": str(e),
                            "query_type": "conceptual",
                        },
                        "timing": {"total": 0},
                    }

        return [r for r in results if r is not None]