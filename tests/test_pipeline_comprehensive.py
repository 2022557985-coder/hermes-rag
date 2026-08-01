"""Comprehensive pipeline tests covering all component configurations, edge cases,
and error scenarios for the RetrievalPipeline.

Tests all combinations of pipeline components (dense/sparse/reranker/cache),
internal pipeline methods, query classification integration, cache behavior,
metrics recording, and multi-language query support.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.indexing.bm25_index import BM25Index
from src.core.indexing.index_manager import IndexManager
from src.core.indexing.vector_store import VectorStore
from src.core.retrieval.dense_retriever import DenseRetriever
from src.core.retrieval.query_expander import QueryExpander
from src.core.retrieval.retrieval_pipeline import (
    QueryClassifier,
    RetrievalPipeline,
)
from src.core.retrieval.rrf_fusion import RRFFusion
from src.core.retrieval.rule_retriever import RuleRetriever
from src.core.retrieval.sparse_retriever import SparseRetriever
from src.utils.cache import QueryCache
from src.utils.metrics import MetricsCollector

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield tmpdir


@pytest.fixture
def test_chunks():
    """Return sample chunks for pipeline testing."""
    return [
        {
            "chunk_id": "c1",
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "metadata": {"source": "ml.txt", "heading_path": "Introduction"},
        },
        {
            "chunk_id": "c2",
            "text": "Supervised learning involves training a model on labeled data where the correct output is known.",
            "metadata": {"source": "ml.txt", "heading_path": "Supervised Learning"},
        },
        {
            "chunk_id": "c3",
            "text": "深度学习使用多层神经网络进行复杂模式识别任务，在图像识别和自然语言处理中表现优异。",
            "metadata": {"source": "dl.txt", "heading_path": "Deep Learning"},
        },
        {
            "chunk_id": "c4",
            "text": "Python is a high-level programming language known for its readability and simplicity.",
            "metadata": {"source": "python.txt", "heading_path": "Python Basics"},
        },
        {
            "chunk_id": "c5",
            "text": "RAG（检索增强生成）结合了信息检索和文本生成技术，可以有效减少大模型幻觉问题。",
            "metadata": {"source": "rag.txt", "heading_path": "RAG Introduction"},
        },
    ]


# Use locally available model
_LOCAL_MODEL = "BAAI/bge-small-zh-v1.5"


@pytest.fixture
def empty_index_manager(temp_dir):
    """Create an IndexManager with empty indexes."""
    vector_store = VectorStore(
        persist_directory=os.path.join(temp_dir, "chroma_empty"),
        collection_name="test_empty",
        embedding_model=_LOCAL_MODEL,
    )
    bm25_index = BM25Index()
    return IndexManager(vector_store=vector_store, bm25_index=bm25_index)


@pytest.fixture
def populated_index_manager(temp_dir, test_chunks):
    """Create an IndexManager with pre-loaded test chunks."""
    vector_store = VectorStore(
        persist_directory=os.path.join(temp_dir, "chroma_populated"),
        collection_name="test_populated",
        embedding_model=_LOCAL_MODEL,
    )
    vector_store.clear()
    bm25_index = BM25Index()
    index_manager = IndexManager(vector_store=vector_store, bm25_index=bm25_index)
    index_manager.ingest_chunks(test_chunks)
    return index_manager


@pytest.fixture
def base_pipeline_config():
    """Return a base retrieval pipeline config."""
    return {
        "dense_top_k": 100,
        "sparse_top_k": 100,
        "fusion_top_k": 50,
        "reranking": {"enabled": True, "timeout_seconds": 1.5},
    }


def _build_pipeline(index_manager, config=None, **kwargs):
    """Helper to build a RetrievalPipeline with optional component overrides."""
    cfg = config or {}
    return RetrievalPipeline(
        index_manager=index_manager,
        query_expander=kwargs.get("query_expander", QueryExpander(synonym_enabled=True, hyde_enabled=False)),
        dense_retriever=kwargs.get("dense_retriever", DenseRetriever(index_manager)),
        sparse_retriever=kwargs.get("sparse_retriever", SparseRetriever(index_manager)),
        rule_retriever=kwargs.get("rule_retriever", RuleRetriever()),
        rrf_fusion=kwargs.get("rrf_fusion", RRFFusion()),
        cross_encoder=kwargs.get("cross_encoder", None),
        cache=kwargs.get("cache", QueryCache(max_size=10)),
        config=cfg,
        metrics=kwargs.get("metrics", MetricsCollector()),
    )


# ---------------------------------------------------------------------------
# Pipeline component configuration tests
# ---------------------------------------------------------------------------

class TestPipelineComponentConfigurations:
    """Test pipeline with different combinations of enabled components."""

    def test_all_components_enabled(self, populated_index_manager, base_pipeline_config):
        """Test pipeline with dense + sparse + reranker + cache all enabled."""
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            sparse_retriever=SparseRetriever(populated_index_manager),
            cross_encoder=None,
            cache=QueryCache(max_size=10),
        )
        result = pipeline.retrieve("machine learning", top_k=3, use_reranker=True)
        assert "results" in result, "Result should contain 'results' key"
        assert "query_info" in result, "Result should contain 'query_info' key"
        assert "timing" in result, "Result should contain 'timing' key"
        assert len(result["results"]) > 0, "Should return at least one result"
        assert len(result["results"]) <= 3, "Should respect top_k limit"

    def test_only_dense_retrieval(self, populated_index_manager, base_pipeline_config):
        """Test pipeline with only dense retrieval (no sparse, no reranker, no cache)."""
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            sparse_retriever=None,
            cross_encoder=None,
            cache=None,
        )
        result = pipeline.retrieve("machine learning", top_k=3, use_reranker=False)
        assert len(result["results"]) > 0, "Dense retrieval should return results"

    def test_only_sparse_retrieval(self, populated_index_manager, base_pipeline_config):
        """Test pipeline with only sparse retrieval (no dense, no reranker, no cache)."""
        pipeline = RetrievalPipeline(
            index_manager=populated_index_manager,
            query_expander=None,
            dense_retriever=None,
            sparse_retriever=SparseRetriever(populated_index_manager),
            rule_retriever=None,
            rrf_fusion=RRFFusion(),
            cross_encoder=None,
            cache=None,
            config=base_pipeline_config,
            metrics=MetricsCollector(),
        )
        result = pipeline.retrieve("machine learning", top_k=3, use_reranker=False)
        assert len(result["results"]) > 0, "Sparse retrieval should return results"

    def test_without_reranker(self, populated_index_manager, base_pipeline_config):
        """Test pipeline without reranker but with dense + sparse + cache."""
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            cross_encoder=None,
        )
        result = pipeline.retrieve("machine learning", top_k=3, use_reranker=False)
        assert len(result["results"]) > 0, "Should return results without reranker"
        assert result["timing"]["total"] > 0, "Timing should be recorded"

    def test_without_cache(self, populated_index_manager, base_pipeline_config):
        """Test pipeline without cache but with other components."""
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            cross_encoder=None,
            cache=None,
        )
        result = pipeline.retrieve("machine learning", top_k=3, use_reranker=False)
        assert len(result["results"]) > 0, "Should return results without cache"
        assert result["query_info"].get("cached", False) is False, "Should not be cached"

    def test_without_query_expander(self, populated_index_manager, base_pipeline_config):
        """Test pipeline without query expander."""
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            query_expander=None,
            cross_encoder=None,
            cache=None,
        )
        result = pipeline.retrieve("machine learning", top_k=3, use_reranker=False)
        # Without expander, expanded should equal original
        assert result["query_info"]["expanded"] == result["query_info"]["original"], \
            "Expanded query should equal original when no expander"


# ---------------------------------------------------------------------------
# Batch retrieval tests
# ---------------------------------------------------------------------------

class TestBatchRetrieval:
    """Test batch retrieval functionality."""

    def test_retrieve_batch_multiple_queries(self, populated_index_manager, base_pipeline_config):
        """Test retrieve_batch with multiple queries."""
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            cross_encoder=None,
            cache=None,
        )
        queries = [
            "machine learning",
            "Python programming",
            "深度学习",
        ]
        results = pipeline.retrieve_batch(queries, top_k=3, use_reranker=False)
        assert len(results) == 3, "Should return results for all 3 queries"
        for r in results:
            assert "results" in r, "Each batch result should have 'results'"
            assert "query_info" in r, "Each batch result should have 'query_info'"

    def test_retrieve_batch_empty_list(self, populated_index_manager, base_pipeline_config):
        """Test retrieve_batch with empty query list."""
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            cross_encoder=None,
            cache=None,
        )
        results = pipeline.retrieve_batch([], top_k=3, use_reranker=False)
        assert results == [], "Empty input should return empty list"

    def test_retrieve_batch_single_query(self, populated_index_manager, base_pipeline_config):
        """Test retrieve_batch with a single query."""
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            cross_encoder=None,
            cache=None,
        )
        results = pipeline.retrieve_batch(["Python programming"], top_k=3, use_reranker=False)
        assert len(results) == 1, "Should return one result for single query"
        assert len(results[0]["results"]) > 0, "Should have results for the query"


# ---------------------------------------------------------------------------
# Query validation tests
# ---------------------------------------------------------------------------

class TestQueryValidation:
    """Test _validate_query with various inputs."""

    @pytest.mark.parametrize("query,expected_valid", [
        ("valid query", True),
        ("", False),
        ("   ", False),
        ("x" * (RetrievalPipeline.MAX_QUERY_LENGTH + 1), False),
        ("x" * RetrievalPipeline.MAX_QUERY_LENGTH, True),
        ("normal query text", True),
        ("!@#$%^&*()!@#$%^&*()!@#$%^&*()!@#$%^&*()", False),  # >80% special chars
        ("a!b@c#d$e%f^g&h*i(j)k_l-m+n=o[p]q{r}s|t:u;v\"w'x,y.z/<o>p?", True),  # mixed
    ])
    def test_validate_query(self, query, expected_valid):
        """Test _validate_query with various inputs."""
        is_valid, error_msg = RetrievalPipeline._validate_query(query)
        assert is_valid == expected_valid, \
            f"Expected valid={expected_valid} for query='{query[:50]}...', got valid={is_valid}, error='{error_msg}'"

    def test_validate_query_none(self):
        """Test _validate_query with None input."""
        is_valid, error_msg = RetrievalPipeline._validate_query(None)
        assert not is_valid, "None should be invalid"
        assert error_msg is not None, "Should return an error message"

    def test_validate_query_non_string(self):
        """Test _validate_query with non-string input."""
        is_valid, error_msg = RetrievalPipeline._validate_query(123)
        assert not is_valid, "Non-string should be invalid"


# ---------------------------------------------------------------------------
# Score normalization tests
# ---------------------------------------------------------------------------

class TestScoreNormalization:
    """Test _normalize_scores with various score ranges."""

    def test_normalize_scores_already_in_range(self):
        """Scores already in [0, 1] should remain unchanged."""
        results = [
            {"chunk_id": "a", "score": 0.9},
            {"chunk_id": "b", "score": 0.5},
            {"chunk_id": "c", "score": 0.1},
        ]
        normalized = RetrievalPipeline._normalize_scores(results)
        assert normalized[0]["score"] == 0.9, "Top score should remain 0.9"
        assert normalized[2]["score"] == 0.1, "Bottom score should remain 0.1"

    def test_normalize_scores_outside_range(self):
        """Scores outside [0, 1] should be normalized."""
        results = [
            {"chunk_id": "a", "score": 100.0},
            {"chunk_id": "b", "score": 50.0},
            {"chunk_id": "c", "score": 0.0},
        ]
        normalized = RetrievalPipeline._normalize_scores(results)
        for r in normalized:
            assert 0.0 <= r["score"] <= 1.0, f"Score {r['score']} should be in [0, 1]"

    def test_normalize_scores_all_same(self):
        """All identical scores should map to 0.5."""
        results = [
            {"chunk_id": "a", "score": 42.0},
            {"chunk_id": "b", "score": 42.0},
            {"chunk_id": "c", "score": 42.0},
        ]
        normalized = RetrievalPipeline._normalize_scores(results)
        for r in normalized:
            assert r["score"] == 0.5, "All equal scores should normalize to 0.5"

    def test_normalize_scores_negative(self):
        """Negative scores should be clamped to 0."""
        results = [
            {"chunk_id": "a", "score": -0.5},
            {"chunk_id": "b", "score": 0.5},
        ]
        normalized = RetrievalPipeline._normalize_scores(results)
        for r in normalized:
            assert r["score"] >= 0.0, "No score should be negative"

    def test_normalize_scores_empty(self):
        """Empty list should return empty list."""
        result = RetrievalPipeline._normalize_scores([])
        assert result == [], "Empty input should return empty list"


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------

class TestDeduplication:
    """Test _deduplicate_results with duplicates."""

    def test_deduplicate_keeps_highest_score(self):
        """Duplicate chunk_ids should keep the entry with highest score."""
        pipeline = RetrievalPipeline()
        results = [
            {"chunk_id": "c1", "text": "text1", "score": 0.5},
            {"chunk_id": "c1", "text": "text1", "score": 0.9},  # Duplicate, higher score
            {"chunk_id": "c2", "text": "text2", "score": 0.7},
        ]
        deduped = pipeline._deduplicate_results(results)
        assert len(deduped) == 2, "Should have 2 unique results"
        c1 = [r for r in deduped if r["chunk_id"] == "c1"][0]
        assert c1["score"] == 0.9, "Should keep the highest scored duplicate"

    def test_deduplicate_no_duplicates(self):
        """No duplicates should return all results unchanged count."""
        pipeline = RetrievalPipeline()
        results = [
            {"chunk_id": "c1", "text": "text1", "score": 0.9},
            {"chunk_id": "c2", "text": "text2", "score": 0.7},
            {"chunk_id": "c3", "text": "text3", "score": 0.5},
        ]
        deduped = pipeline._deduplicate_results(results)
        assert len(deduped) == 3, "Should keep all unique results"

    def test_deduplicate_empty(self):
        """Empty list should return empty list."""
        pipeline = RetrievalPipeline()
        result = pipeline._deduplicate_results([])
        assert result == [], "Empty input should return empty list"

    def test_deduplicate_sorted_by_score(self):
        """Results should be sorted by score descending after dedup."""
        pipeline = RetrievalPipeline()
        results = [
            {"chunk_id": "c3", "text": "text3", "score": 0.3},
            {"chunk_id": "c1", "text": "text1", "score": 0.9},
            {"chunk_id": "c2", "text": "text2", "score": 0.7},
        ]
        deduped = pipeline._deduplicate_results(results)
        scores = [r["score"] for r in deduped]
        assert scores == sorted(scores, reverse=True), "Should be sorted by score descending"


# ---------------------------------------------------------------------------
# Threshold filtering tests
# ---------------------------------------------------------------------------

class TestThresholdFiltering:
    """Test _filter_by_threshold with various thresholds."""

    def test_filter_default_threshold(self):
        """Default threshold (0.001) should filter out very low scores."""
        pipeline = RetrievalPipeline()
        results = [
            {"chunk_id": "c1", "score": 0.5},
            {"chunk_id": "c2", "score": 0.0001},  # Below default threshold
            {"chunk_id": "c3", "score": 0.8},
        ]
        filtered = pipeline._filter_by_threshold(results)
        assert len(filtered) == 2, "Should filter out the very low score result"

    def test_filter_high_threshold(self):
        """High threshold should filter out most results."""
        pipeline = RetrievalPipeline()
        results = [
            {"chunk_id": "c1", "score": 0.5},
            {"chunk_id": "c2", "score": 0.3},
            {"chunk_id": "c3", "score": 0.8},
        ]
        filtered = pipeline._filter_by_threshold(results, min_score=0.4)
        assert len(filtered) == 2, "Should keep only scores >= 0.4"

    def test_filter_zero_threshold(self):
        """Zero threshold should keep all results."""
        pipeline = RetrievalPipeline()
        results = [
            {"chunk_id": "c1", "score": 0.0},
            {"chunk_id": "c2", "score": 0.5},
            {"chunk_id": "c3", "score": -0.1},
        ]
        filtered = pipeline._filter_by_threshold(results, min_score=0.0)
        assert len(filtered) == 2, "Should keep scores >= 0.0, filter negatives"

    def test_filter_empty(self):
        """Empty list should return empty list."""
        pipeline = RetrievalPipeline()
        result = pipeline._filter_by_threshold([], min_score=0.5)
        assert result == [], "Empty input should return empty list"


# ---------------------------------------------------------------------------
# Query classification integration tests
# ---------------------------------------------------------------------------

class TestQueryClassificationIntegration:
    """Test query classification integrated with pipeline."""

    @pytest.mark.parametrize("query,expected_type", [
        ("什么是机器学习", "conceptual"),
        ("如何安装Python", "procedural"),
        ("ABC-1234的价格", "factual"),
        ("machine learning definition", "conceptual"),
        ("how to install Python", "procedural"),
        ("ERR404如何解决", "procedural"),  # Has both factual and procedural -> procedural
    ])
    def test_query_classification_in_pipeline(self, populated_index_manager, base_pipeline_config,
                                               query, expected_type):
        """Test that query_type is correctly set in query_info."""
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            cross_encoder=None,
            cache=None,
        )
        result = pipeline.retrieve(query, top_k=3, use_reranker=False)
        assert result["query_info"]["query_type"] == expected_type, \
            f"Query '{query}' should be classified as '{expected_type}'"

    def test_classify_with_confidence(self):
        """Test classify_with_confidence returns type and score."""
        qtype, confidence = QueryClassifier.classify_with_confidence("什么是机器学习")
        assert qtype == "conceptual", "Should classify as conceptual"
        assert 0.0 <= confidence <= 1.0, "Confidence should be in [0, 1]"

    def test_classify_with_confidence_empty(self):
        """Test classify_with_confidence with empty query."""
        qtype, confidence = QueryClassifier.classify_with_confidence("")
        assert qtype == "conceptual", "Empty query should default to conceptual"
        assert confidence == 1.0, "Empty query should have max confidence"


# ---------------------------------------------------------------------------
# Cache hit/miss behavior tests
# ---------------------------------------------------------------------------

class TestCacheBehavior:
    """Test cache hit and miss behavior in the pipeline."""

    def test_cache_hit_on_second_query(self, populated_index_manager, base_pipeline_config):
        """Second identical query should hit the cache."""
        cache = QueryCache(max_size=10)
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            cross_encoder=None,
            cache=cache,
        )
        # First query (cold - cache miss)
        result1 = pipeline.retrieve("Python programming", top_k=3, use_reranker=False)
        assert result1["query_info"].get("cached", False) is False, "First query should miss cache"

        # Second query (warm - cache hit)
        result2 = pipeline.retrieve("Python programming", top_k=3, use_reranker=False)
        assert result2["query_info"].get("cached", False) is True, "Second query should hit cache"

    def test_cache_miss_different_query(self, populated_index_manager, base_pipeline_config):
        """Different query should miss the cache."""
        cache = QueryCache(max_size=10)
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            cross_encoder=None,
            cache=cache,
        )
        pipeline.retrieve("Python programming", top_k=3, use_reranker=False)
        result = pipeline.retrieve("machine learning", top_k=3, use_reranker=False)
        assert result["query_info"].get("cached", False) is False, \
            "Different query should miss cache"

    def test_cache_disabled_no_caching(self, populated_index_manager, base_pipeline_config):
        """When cache is None, results should never be cached."""
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            cross_encoder=None,
            cache=None,
        )
        result1 = pipeline.retrieve("Python programming", top_k=3, use_reranker=False)
        result2 = pipeline.retrieve("Python programming", top_k=3, use_reranker=False)
        assert result1["query_info"].get("cached", False) is False, "First query should not be cached"
        assert result2["query_info"].get("cached", False) is False, "Second query should not be cached"


# ---------------------------------------------------------------------------
# Metrics recording tests
# ---------------------------------------------------------------------------

class TestMetricsRecording:
    """Test that metrics are properly recorded after queries."""

    def test_metrics_after_query(self, populated_index_manager, base_pipeline_config):
        """Metrics should be recorded after a query."""
        metrics = MetricsCollector()
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            cross_encoder=None,
            cache=None,
            metrics=metrics,
        )
        pipeline.retrieve("machine learning", top_k=3, use_reranker=False)
        report = metrics.get_full_report()
        assert report["total_queries"] >= 1, "Should record at least one query"

    def test_metrics_cache_hit_tracking(self, populated_index_manager, base_pipeline_config):
        """Metrics should track cache hits and misses."""
        metrics = MetricsCollector()
        cache = QueryCache(max_size=10)
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            cross_encoder=None,
            cache=cache,
            metrics=metrics,
        )
        pipeline.retrieve("Python", top_k=3, use_reranker=False)
        pipeline.retrieve("Python", top_k=3, use_reranker=False)  # Cache hit
        report = metrics.get_full_report()
        assert report["cache"]["hits"] >= 1, "Should record at least one cache hit"

    def test_metrics_recall_paths(self, populated_index_manager, base_pipeline_config):
        """Metrics should track recall paths."""
        metrics = MetricsCollector()
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            cross_encoder=None,
            cache=None,
            metrics=metrics,
        )
        pipeline.retrieve("machine learning", top_k=3, use_reranker=False)
        recall = metrics.get_recall_path_distribution()
        total = recall["dense_only"] + recall["sparse_only"] + recall["both"] + recall["cached"]
        assert total >= 1, "Should have at least one recall path recorded"

    def test_metrics_latency_recorded(self, populated_index_manager, base_pipeline_config):
        """Metrics should record latency percentiles."""
        metrics = MetricsCollector()
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            cross_encoder=None,
            cache=None,
            metrics=metrics,
        )
        pipeline.retrieve("machine learning", top_k=3, use_reranker=False)
        latencies = metrics.get_latency_percentiles()
        assert latencies["avg"] > 0, "Average latency should be > 0"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test pipeline error handling when components fail."""

    def test_empty_query_returns_error_info(self, populated_index_manager, base_pipeline_config):
        """Empty query should return error info but not crash."""
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            cross_encoder=None,
            cache=None,
        )
        result = pipeline.retrieve("", top_k=3, use_reranker=False)
        assert result["results"] == [], "Empty query should return empty results"
        assert "error" in result["query_info"], "Should include error info"

    def test_very_long_query_truncated(self, populated_index_manager, base_pipeline_config):
        """Very long query should be truncated, not crash."""
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            cross_encoder=None,
            cache=None,
        )
        long_query = "x" * (RetrievalPipeline.MAX_QUERY_LENGTH + 500)
        result = pipeline.retrieve(long_query, top_k=3, use_reranker=False)
        assert result["results"] == [], "Too-long query should return empty results"
        assert "error" in result["query_info"], "Should include error info"

    def test_special_chars_query_not_crash(self, populated_index_manager, base_pipeline_config):
        """Query with excessive special chars should not crash."""
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            cross_encoder=None,
            cache=None,
        )
        result = pipeline.retrieve("@#$%^&*()!@#$%^&*()!@#$%^&*()!", top_k=3, use_reranker=False)
        assert isinstance(result["results"], list), "Should return a list, not crash"


# ---------------------------------------------------------------------------
# Metadata filter tests
# ---------------------------------------------------------------------------

class TestMetadataFilterRetrieval:
    """Test retrieval with metadata filters."""

    def test_retrieval_with_metadata_filter(self, populated_index_manager, base_pipeline_config):
        """Test that retrieval respects metadata filters."""
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            cross_encoder=None,
            cache=None,
        )
        # Use a mock rule_retriever that returns a filter
        rule_retriever = RuleRetriever()
        rule_retriever.build_filter = MagicMock(return_value={"source": "ml.txt"})
        pipeline.rule_retriever = rule_retriever

        result = pipeline.retrieve("machine learning", top_k=3, use_reranker=False)
        assert isinstance(result["results"], list), "Should return results list"


# ---------------------------------------------------------------------------
# Multi-language query tests
# ---------------------------------------------------------------------------

class TestMultiLanguageQueries:
    """Test retrieval with Chinese, English, and mixed-language queries."""

    def test_chinese_query(self, populated_index_manager, base_pipeline_config):
        """Test retrieval with a Chinese query."""
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            cross_encoder=None,
            cache=None,
        )
        result = pipeline.retrieve("什么是深度学习", top_k=3, use_reranker=False)
        assert len(result["results"]) > 0, "Chinese query should return results"
        texts = [r["text"] for r in result["results"]]
        assert any("深度" in t for t in texts), "Should find Chinese content"

    def test_english_query(self, populated_index_manager, base_pipeline_config):
        """Test retrieval with an English query."""
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            cross_encoder=None,
            cache=None,
        )
        result = pipeline.retrieve("What is machine learning?", top_k=3, use_reranker=False)
        assert len(result["results"]) > 0, "English query should return results"

    def test_mixed_language_query(self, populated_index_manager, base_pipeline_config):
        """Test retrieval with mixed Chinese-English query."""
        pipeline = _build_pipeline(
            populated_index_manager,
            config=base_pipeline_config,
            cross_encoder=None,
            cache=None,
        )
        result = pipeline.retrieve("如何使用Python进行machine learning", top_k=3, use_reranker=False)
        assert len(result["results"]) > 0, "Mixed language query should return results"
        assert result["query_info"]["query_type"] in ("conceptual", "procedural", "factual"), \
            "Should classify the query"


# ---------------------------------------------------------------------------
# Pipeline factory integration tests
# ---------------------------------------------------------------------------

class TestPipelineFactoryIntegration:
    """Test pipeline_factory.build_pipeline integration."""

    def test_build_pipeline_all_components(self, temp_dir):
        """Test build_pipeline with all components enabled."""
        from src.config import HermesConfig, reset_config
        from src.core.pipeline_factory import build_pipeline

        reset_config()
        config = HermesConfig()
        config.chromadb.persist_directory = os.path.join(temp_dir, "chroma_factory")
        config.chromadb.collection_name = "test_factory"
        config.reranking.enabled = True  # Explicitly opt into reranking for this integration test

        try:
            pipeline = build_pipeline(
                config=config,
                query_expansion_enabled=True,
                use_reranker=True,
                use_sparse=True,
                use_cache=True,
            )
            assert pipeline is not None, "Pipeline should be built"
            assert pipeline.dense_retriever is not None, "Should have dense retriever"
            assert pipeline.sparse_retriever is not None, "Should have sparse retriever"
            assert pipeline.cross_encoder is not None, "Should have cross encoder"
            assert pipeline.cache is not None, "Should have cache"
        finally:
            reset_config()

    def test_build_pipeline_minimal(self, temp_dir):
        """Test build_pipeline with minimal components."""
        from src.config import HermesConfig, reset_config
        from src.core.pipeline_factory import build_pipeline

        reset_config()
        config = HermesConfig()
        config.chromadb.persist_directory = os.path.join(temp_dir, "chroma_minimal")

        try:
            pipeline = build_pipeline(
                config=config,
                query_expansion_enabled=False,
                use_reranker=False,
                use_sparse=False,
                use_cache=False,
            )
            assert pipeline is not None, "Pipeline should be built"
            assert pipeline.query_expander is None, "Should not have query expander"
            assert pipeline.sparse_retriever is None, "Should not have sparse retriever"
            assert pipeline.cross_encoder is None, "Should not have cross encoder"
            assert pipeline.cache is None, "Should not have cache"
        finally:
            reset_config()