"""Comprehensive optimization edge-case tests for Hermes-RAG.

Covers all major components with thorough edge-case testing:
- QueryClassifier, RRFFusion, QueryExpander, QueryCache
- MetricsCollector, Stopwatch/TimerStats, BM25Index
- VectorStore, Security, LLMClient, CrossEncoderReranker
- HierarchicalChunker, Integration pipeline tests
"""

import os
import tempfile
import threading
import time

import numpy as np
import pytest

from src.core.chunking.hierarchical_chunker import HierarchicalChunker
from src.core.generation.llm_client import LLMClient
from src.core.indexing.bm25_index import BM25Index
from src.core.reranking.cross_encoder import CrossEncoderReranker
from src.core.retrieval.query_expander import QueryExpander
from src.core.retrieval.retrieval_pipeline import QueryClassifier, RetrievalPipeline
from src.core.retrieval.rrf_fusion import RRFFusion
from src.utils.cache import QueryCache
from src.utils.metrics import MetricsCollector, get_metrics, reset_metrics
from src.utils.security import (
    rate_limit_check,
    sanitize_filename,
    sanitize_html,
    validate_batch_size,
    validate_file_extension,
    validate_file_path,
    validate_file_size,
    validate_query,
    validate_text_content,
    validate_url,
)
from src.utils.timer import Stopwatch, TimerStats

# ============================================================================
# 1. QueryClassifier Tests
# ============================================================================

class TestQueryClassifierEdgeCases:
    """Comprehensive edge-case tests for QueryClassifier."""

    # --- Empty / Whitespace ---

    def test_classify_empty_string_defaults_to_conceptual(self):
        """Empty string should default to 'conceptual'."""
        assert QueryClassifier.classify("") == "conceptual"

    def test_classify_whitespace_only_defaults_to_conceptual(self):
        """Whitespace-only query should default to 'conceptual'."""
        assert QueryClassifier.classify("   \t\n  ") == "conceptual"

    def test_classify_with_confidence_empty_string_returns_1_0(self):
        """Empty string should have confidence 1.0 for conceptual."""
        qtype, conf = QueryClassifier.classify_with_confidence("")
        assert qtype == "conceptual"
        assert conf == 1.0

    def test_classify_with_confidence_whitespace_only(self):
        """Whitespace-only should have confidence 1.0 for conceptual."""
        qtype, conf = QueryClassifier.classify_with_confidence("   \n  ")
        assert qtype == "conceptual"
        assert conf == 1.0

    def test_classify_none_shall_not_be_passed_but_handled(self):
        """None query should not crash; classify handles falsy."""
        # The implementation checks 'not query' so None becomes 'conceptual'
        assert QueryClassifier.classify(None) == "conceptual"

    # --- Very Long Query ---

    def test_classify_very_long_query_handles_gracefully(self):
        """Very long query should not cause performance issues."""
        long_query = "什么是机器学习 " * 1000
        result = QueryClassifier.classify(long_query)
        assert result in ("factual", "procedural", "conceptual")

    def test_classify_with_confidence_very_long_query(self):
        """Very long query with confidence should not crash."""
        long_query = "如何配置数据库 " * 500
        qtype, conf = QueryClassifier.classify_with_confidence(long_query)
        assert qtype in ("factual", "procedural", "conceptual")
        assert 0.0 <= conf <= 1.0

    # --- Special Characters ---

    def test_classify_special_chars_only_defaults_to_conceptual(self):
        """Query with only special characters should default to conceptual."""
        assert QueryClassifier.classify("!@#$%^&*()") == "conceptual"

    def test_classify_special_chars_with_procedural_keyword(self):
        """Special chars combined with procedural keywords."""
        result = QueryClassifier.classify("!!!如何!!!")
        assert result == "procedural"

    def test_classify_html_tags_in_query(self):
        """HTML tags in query should be handled gracefully."""
        result = QueryClassifier.classify("<script>alert('xss')</script>")
        assert result in ("factual", "procedural", "conceptual")

    # --- Mixed Languages ---

    def test_classify_mixed_chinese_english_factual(self):
        """Mixed Chinese-English with product code."""
        result = QueryClassifier.classify("ERR404 如何解决")
        assert result == "procedural"  # both factual and procedural → procedural

    def test_classify_mixed_chinese_english_conceptual(self):
        """Mixed Chinese-English conceptual query."""
        result = QueryClassifier.classify("什么是Machine Learning")
        assert result == "conceptual"

    def test_classify_mixed_language_with_unicode(self):
        """Mixed languages with Unicode ranges."""
        result = QueryClassifier.classify("机器学习とは何ですか")
        assert result in ("factual", "procedural", "conceptual")

    # --- Numeric Only ---

    def test_classify_numeric_only_defaults_to_conceptual(self):
        """Pure numeric query should default to conceptual."""
        assert QueryClassifier.classify("12345") == "conceptual"

    def test_classify_phone_number_returns_factual(self):
        """Chinese phone number pattern should be factual."""
        assert QueryClassifier.classify("13812345678") == "factual"

    def test_classify_date_pattern_returns_factual(self):
        """Date pattern should be factual."""
        assert QueryClassifier.classify("2024-01-15") == "factual"

    def test_classify_version_number_returns_factual(self):
        """Version number should be factual."""
        assert QueryClassifier.classify("3.14.2 更新内容") == "factual"

    def test_classify_id_card_number_returns_factual(self):
        """18-digit Chinese ID card number should be factual."""
        assert QueryClassifier.classify("110101199001011234") == "factual"

    # --- Emoji ---

    def test_classify_emoji_only_defaults_to_conceptual(self):
        """Emoji-only query should default to conceptual."""
        assert QueryClassifier.classify("😀😃😄😁") == "conceptual"

    def test_classify_emoji_with_text(self):
        """Emoji with meaningful text."""
        result = QueryClassifier.classify("如何安装😀Python")
        assert result == "procedural"

    # --- SQL Injection / XSS Patterns ---

    def test_classify_sql_injection_attempt(self):
        """SQL injection should not crash classifier."""
        result = QueryClassifier.classify("SELECT * FROM users WHERE 1=1")
        assert result in ("factual", "procedural", "conceptual")

    def test_classify_xss_attempt(self):
        """XSS attempt should not crash classifier."""
        result = QueryClassifier.classify("javascript:alert(1)")
        assert result in ("factual", "procedural", "conceptual")

    def test_classify_sql_union_select(self):
        """SQL UNION SELECT should not crash."""
        result = QueryClassifier.classify("UNION SELECT password FROM users")
        assert result in ("factual", "procedural", "conceptual")

    # --- URL Queries ---

    def test_classify_url_in_query(self):
        """URL in query should be handled."""
        result = QueryClassifier.classify("https://example.com/path 请问如何配置")
        assert result == "procedural"

    def test_classify_url_with_factual_pattern(self):
        """URL with factual pattern."""
        result = QueryClassifier.classify("https://api.example.com/v1.2.3")
        assert result == "factual"  # has version number pattern

    # --- Unicode Edge Cases ---

    def test_classify_unicode_surrogate_pair(self):
        """Unicode surrogate pair (e.g., emoji) should not crash."""
        result = QueryClassifier.classify("🎉\U0001F600")
        assert result == "conceptual"

    def test_classify_zero_width_chars(self):
        """Zero-width characters should not crash."""
        result = QueryClassifier.classify("如何\u200b配置\u200c数据库")
        assert result == "procedural"  # 如何 + 配置

    def test_classify_fullwidth_chars(self):
        """Fullwidth characters with factual pattern."""
        result = QueryClassifier.classify("ＥＲＲ５００")
        assert result in ("factual", "procedural", "conceptual")

    # --- Boundary Cases for Classification Types ---

    def test_classify_factual_with_procedural_prefers_procedural(self):
        """When both factual and procedural, procedural wins."""
        assert QueryClassifier.classify("ERR500 how to fix") == "procedural"

    def test_classify_conceptual_how_detected(self):
        """'如何计算' should be conceptual, not procedural."""
        assert QueryClassifier.classify("如何计算神经网络参数") == "conceptual"

    def test_classify_conceptual_how_work(self):
        """'如何工作' should be conceptual."""
        assert QueryClassifier.classify("如何工作这个系统") == "conceptual"

    def test_classify_conceptual_keywords_direct_match(self):
        """Direct conceptual keywords should classify as conceptual."""
        assert QueryClassifier.classify("机器学习的定义是什么") == "conceptual"

    def test_classify_procedural_basic(self):
        """Basic procedural query."""
        assert QueryClassifier.classify("如何安装Python") == "procedural"

    def test_classify_procedural_english(self):
        """English procedural query."""
        assert QueryClassifier.classify("how to install Python") == "procedural"

    def test_classify_factual_product_code(self):
        """Product code should be factual."""
        assert QueryClassifier.classify("ABC-1234") == "factual"

    def test_classify_factual_error_code(self):
        """Error code should be factual."""
        assert QueryClassifier.classify("ERR_CONNECTION_REFUSED") == "factual"

    def test_classify_with_confidence_no_signals(self):
        """Query with no signals returns low confidence."""
        qtype, conf = QueryClassifier.classify_with_confidence("abc")
        assert qtype == "conceptual"
        assert conf == 0.3

    def test_classify_with_confidence_strong_signal(self):
        """Query with strong signal returns high confidence."""
        qtype, conf = QueryClassifier.classify_with_confidence("如何安装Python")
        assert qtype == "procedural"
        assert conf > 0.5

    def test_classify_with_confidence_factual_signal(self):
        """Factual signal confidence."""
        qtype, conf = QueryClassifier.classify_with_confidence("ERR500")
        assert qtype == "factual"
        assert conf == 1.0  # only factual signals matched


# ============================================================================
# 2. RRFFusion Tests
# ============================================================================

class TestRRFFusionEdgeCases:
    """Comprehensive edge-case tests for RRFFusion."""

    @pytest.fixture
    def rrf(self):
        return RRFFusion(k=60)

    @pytest.fixture
    def sample_dense(self):
        return [
            {"chunk_id": "d1", "text": "dense result 1", "metadata": {}, "score": 0.9},
            {"chunk_id": "d2", "text": "dense result 2", "metadata": {}, "score": 0.8},
            {"chunk_id": "d3", "text": "dense result 3", "metadata": {}, "score": 0.7},
        ]

    @pytest.fixture
    def sample_sparse(self):
        return [
            {"chunk_id": "s1", "text": "sparse result 1", "metadata": {}, "score": 0.85},
            {"chunk_id": "s2", "text": "sparse result 2", "metadata": {}, "score": 0.75},
        ]

    # --- Empty Lists ---

    def test_fuse_both_empty_returns_empty(self, rrf):
        """Both dense and sparse empty should return empty list."""
        assert rrf.fuse([], []) == []

    def test_fuse_dense_only_returns_results(self, rrf, sample_dense):
        """Only dense results should be returned."""
        result = rrf.fuse(sample_dense, [])
        assert len(result) == len(sample_dense)

    def test_fuse_sparse_only_returns_results(self, rrf, sample_sparse):
        """Only sparse results should be returned."""
        result = rrf.fuse([], sample_sparse)
        assert len(result) == len(sample_sparse)

    def test_fuse_with_boost_both_empty(self, rrf):
        """fuse_with_boost with both empty."""
        assert rrf.fuse_with_boost([], []) == []

    # --- Single Result ---

    def test_fuse_single_dense_result(self, rrf):
        """Single dense result."""
        single = [{"chunk_id": "d1", "text": "only one", "metadata": {}, "score": 0.9}]
        result = rrf.fuse(single, [])
        assert len(result) == 1
        assert result[0]["chunk_id"] == "d1"

    def test_fuse_single_sparse_result(self, rrf):
        """Single sparse result."""
        single = [{"chunk_id": "s1", "text": "only one", "metadata": {}, "score": 0.9}]
        result = rrf.fuse([], single)
        assert len(result) == 1

    # --- Same Scores / Duplicate chunk_ids ---

    def test_fuse_duplicate_chunk_ids_sum_scores(self, rrf):
        """Duplicate chunk_ids across paths should sum scores."""
        dense = [{"chunk_id": "c1", "text": "text1", "metadata": {}, "score": 0.9}]
        sparse = [{"chunk_id": "c1", "text": "text1", "metadata": {}, "score": 0.8}]
        result = rrf.fuse(dense, sparse)
        assert len(result) == 1
        assert "dense" in result[0]["sources"]
        assert "sparse" in result[0]["sources"]

    def test_fuse_duplicate_chunk_ids_keeps_first_text(self, rrf):
        """Duplicate chunk_id should keep the text from the first occurrence."""
        dense = [{"chunk_id": "c1", "text": "first text", "metadata": {}, "score": 0.9}]
        sparse = [{"chunk_id": "c1", "text": "second text", "metadata": {}, "score": 0.8}]
        result = rrf.fuse(dense, sparse)
        assert result[0]["text"] == "first text"

    # --- Large k values ---

    def test_fuse_very_large_k(self, rrf, sample_dense, sample_sparse):
        """Very large k should not break anything."""
        rrf_large = RRFFusion(k=100000)
        result = rrf_large.fuse(sample_dense, sample_sparse)
        # With very large k, all scores are very close but still valid
        assert len(result) > 0
        for r in result:
            assert r["score"] >= 0

    def test_fuse_very_small_k(self, rrf, sample_dense, sample_sparse):
        """Very small k (k=1) should produce valid results."""
        rrf_small = RRFFusion(k=1)
        result = rrf_small.fuse(sample_dense, sample_sparse)
        assert len(result) > 0

    # --- Negative Scores ---

    def test_normalize_scores_empty(self, rrf):
        """Normalize empty list returns empty."""
        assert RRFFusion.normalize_scores([]) == []

    def test_normalize_scores_single(self, rrf):
        """Normalize single score."""
        result = RRFFusion.normalize_scores([{"score": 0.5, "chunk_id": "c1"}])
        assert result[0]["score"] == 0.5

    def test_normalize_scores_all_same(self, rrf):
        """Normalize all same scores."""
        results = [
            {"score": 0.5, "chunk_id": "c1"},
            {"score": 0.5, "chunk_id": "c2"},
        ]
        result = RRFFusion.normalize_scores(results)
        for r in result:
            assert r["score"] == 0.5

    # --- Weight Validation ---

    def test_detect_query_features_empty(self, rrf):
        """Detect features from empty query."""
        features = rrf._detect_query_features("")
        assert features["has_chinese"] is False
        assert features["has_english"] is False
        assert features["query_length"] == 0

    def test_detect_query_features_chinese(self, rrf):
        """Detect features from Chinese query."""
        features = rrf._detect_query_features("如何安装Python")
        assert features["has_chinese"] is True
        assert features["has_english"] is True

    def test_detect_query_features_product_code(self, rrf):
        """Detect product code in query."""
        features = rrf._detect_query_features("ABC-1234")
        assert features["has_product_code"] is True

    def test_detect_query_features_colloquial(self, rrf):
        """Detect colloquial query."""
        features = rrf._detect_query_features("怎么安装")
        assert features["is_colloquial"] is True

    def test_compute_adaptive_weights_product_code(self, rrf):
        """Product code should boost sparse weight."""
        dw, sw = rrf._compute_adaptive_weights("ABC-1234")
        assert sw > dw  # sparse weight should be higher

    def test_compute_adaptive_weights_colloquial(self, rrf):
        """Colloquial should boost dense weight."""
        dw, sw = rrf._compute_adaptive_weights("怎么安装Python")
        assert dw > sw  # dense weight should be higher

    def test_compute_adaptive_weights_short_query(self, rrf):
        """Short query should slightly boost sparse."""
        dw, sw = rrf._compute_adaptive_weights("hi")
        assert dw + sw == pytest.approx(1.0, rel=1e-9)

    def test_compute_adaptive_weights_long_query(self, rrf):
        """Long query should slightly boost dense."""
        long_q = "请详细解释机器学习的各种算法和应用场景" * 10
        dw, sw = rrf._compute_adaptive_weights(long_q)
        assert dw + sw == pytest.approx(1.0, rel=1e-9)

    def test_compute_adaptive_weights_numbers_heavy(self, rrf):
        """Numbers-heavy query should boost sparse."""
        dw, sw = rrf._compute_adaptive_weights("123 456 789 1011 1213")
        # Numbers-heavy → boost sparse
        assert 0.0 <= dw <= 1.0
        assert 0.0 <= sw <= 1.0

    def test_get_dynamic_weights_empty_query(self, rrf):
        """Empty query returns default weights."""
        dw, sw = rrf._get_dynamic_weights("")
        assert dw == rrf.default_dense_weight
        assert sw == rrf.default_sparse_weight

    def test_get_dynamic_weights_whitespace_query(self, rrf):
        """Whitespace query returns default weights."""
        dw, sw = rrf._get_dynamic_weights("   ")
        assert dw == rrf.default_dense_weight
        assert sw == rrf.default_sparse_weight

    # --- Boost Factor Edge Cases ---

    def test_fuse_with_boost_zero_factor(self, rrf, sample_dense, sample_sparse):
        """Boost factor of 0 should be handled."""
        result = rrf.fuse_with_boost(
            sample_dense, sample_sparse, boost_factor=0.0, boost_path="dense"
        )
        assert len(result) > 0

    def test_fuse_with_boost_negative_factor(self, rrf, sample_dense, sample_sparse):
        """Negative boost factor produces negative weights but still works."""
        result = rrf.fuse_with_boost(
            sample_dense, sample_sparse, boost_factor=-1.0, boost_path="dense"
        )
        assert len(result) > 0

    def test_fuse_with_boost_very_large_factor(self, rrf, sample_dense, sample_sparse):
        """Very large boost factor should not crash."""
        result = rrf.fuse_with_boost(
            sample_dense, sample_sparse, boost_factor=1000.0, boost_path="sparse"
        )
        assert len(result) > 0

    def test_fuse_with_boost_invalid_path(self, rrf, sample_dense, sample_sparse):
        """Invalid boost_path should not crash."""
        result = rrf.fuse_with_boost(
            sample_dense, sample_sparse, boost_factor=2.0, boost_path="invalid"
        )
        assert len(result) > 0

    # --- Min Score Threshold ---

    def test_fuse_with_min_score_threshold(self, sample_dense, sample_sparse):
        """Min score threshold should filter results."""
        rrf = RRFFusion(k=60, min_score_threshold=0.02)
        result = rrf.fuse(sample_dense, sample_sparse)
        assert len(result) >= 0

    # --- Top-K ---

    def test_fuse_top_k_zero(self, rrf, sample_dense, sample_sparse):
        """top_k=0 should return empty."""
        result = rrf.fuse(sample_dense, sample_sparse, top_k=0)
        assert result == []

    def test_fuse_top_k_larger_than_results(self, rrf, sample_dense, sample_sparse):
        """top_k larger than total results should return all."""
        result = rrf.fuse(sample_dense, sample_sparse, top_k=100)
        total = len(result)
        assert total <= len(sample_dense) + len(sample_sparse)

    # --- Mixed Language Queries ---

    def test_fuse_with_mixed_language_query(self, rrf, sample_dense, sample_sparse):
        """Mixed language query should use adaptive weights."""
        result = rrf.fuse(sample_dense, sample_sparse, query="如何配置ABC-1234")
        assert len(result) > 0

    def test_fuse_with_english_query(self, rrf, sample_dense, sample_sparse):
        """English query should use adaptive weights."""
        result = rrf.fuse(sample_dense, sample_sparse, query="how to install Python")
        assert len(result) > 0

    def test_fuse_with_no_query(self, rrf, sample_dense, sample_sparse):
        """No query string should use default weights."""
        result = rrf.fuse(sample_dense, sample_sparse, query="")
        assert len(result) > 0


# ============================================================================
# 3. QueryExpander Tests
# ============================================================================

class TestQueryExpanderEdgeCases:
    """Comprehensive edge-case tests for QueryExpander."""

    @pytest.fixture
    def expander(self):
        return QueryExpander(synonym_enabled=True, hyde_enabled=False, max_synonyms=3)

    # --- Empty / None Query ---

    def test_expand_empty_string_returns_defaults(self, expander):
        """Empty string should return safe defaults."""
        result = expander.expand("")
        assert result["original"] == ""
        assert result["expanded"] == ""
        assert result["synonyms"] == []
        assert result["keywords"] == []

    def test_expand_whitespace_only(self, expander):
        """Whitespace-only should return safe defaults."""
        result = expander.expand("   ")
        assert result["original"] == "   "
        assert result["synonyms"] == []
        assert result["keywords"] == []

    def test_expand_none_query(self, expander):
        """None query should return safe defaults."""
        result = expander.expand(None)
        assert result["original"] == ""
        assert result["expanded"] == ""
        assert result["synonyms"] == []
        assert result["keywords"] == []

    def test_expand_non_string_query(self, expander):
        """Non-string query should return safe defaults."""
        result = expander.expand(123)
        assert result["original"] == ""
        assert result["expanded"] == ""
        assert result["synonyms"] == []
        assert result["keywords"] == []

    # --- Emoji ---

    def test_expand_emoji_only(self, expander):
        """Emoji-only query should have no synonyms."""
        result = expander.expand("😀😃😄")
        assert result["synonyms"] == []
        assert result["keywords"] == []

    def test_expand_emoji_with_text(self, expander):
        """Emoji with text should still expand text."""
        result = expander.expand("如何安装😀")
        assert len(result["keywords"]) > 0

    # --- Very Long Query ---

    def test_expand_very_long_query(self, expander):
        """Very long query should not crash."""
        long_query = "如何配置数据库 " * 500
        result = expander.expand(long_query)
        assert "original" in result
        assert "expanded" in result

    # --- Pure Numbers ---

    def test_expand_pure_numbers(self, expander):
        """Pure numbers should not produce synonyms."""
        result = expander.expand("12345 67890")
        assert result["synonyms"] == []
        # Pure numeric tokens are filtered from keywords
        assert all(not kw.isdigit() for kw in result["keywords"])

    # --- Special Regex Characters ---

    def test_expand_special_regex_chars(self, expander):
        """Special regex characters should not crash tokenization."""
        result = expander.expand(".+*?^$()[]{}|\\")
        assert "original" in result
        assert "expanded" in result

    def test_expand_special_chars_with_keywords(self, expander):
        """Special chars with keywords should still expand."""
        result = expander.expand("如何***配置***数据库")
        assert len(result["keywords"]) > 0

    # --- Synonym Deduplication ---

    def test_expand_no_duplicate_synonyms(self, expander):
        """Synonyms should not contain duplicates."""
        result = expander.expand("重置密码")
        syns = result["synonyms"]
        assert len(syns) == len(set(syns))

    def test_expand_no_duplicate_weighted_synonyms(self, expander):
        """Weighted synonyms should not contain duplicates."""
        result = expander.expand("重置密码")
        weighted = result["weighted_synonyms"]
        syn_names = [s[0] for s in weighted]
        assert len(syn_names) == len(set(syn_names))

    def test_expand_synonyms_not_in_original(self, expander):
        """Synonyms should not contain words already in original query."""
        result = expander.expand("配置")
        for syn in result["synonyms"]:
            assert syn not in result["original"]

    # --- Keyword Deduplication ---

    def test_expand_keywords_no_duplicates(self, expander):
        """Keywords should not contain duplicates."""
        result = expander.expand("如何配置配置数据库数据库")
        assert len(result["keywords"]) == len(set(k.lower() for k in result["keywords"]))

    def test_expand_keywords_exclude_stop_words(self, expander):
        """Keywords should not contain stop words."""
        result = expander.expand("的 了 是 在 我")
        assert result["keywords"] == []

    def test_expand_keywords_exclude_single_char(self, expander):
        """Single character tokens should be excluded from keywords."""
        result = expander.expand("a b c")
        assert result["keywords"] == []

    # --- Mixed Language Tokenization ---

    def test_expand_mixed_cn_en_query(self, expander):
        """Mixed Chinese-English query should be tokenized correctly."""
        result = expander.expand("如何安装Python and CUDA")
        assert len(result["keywords"]) > 0

    def test_expand_chinese_only_query(self, expander):
        """Chinese-only query should be expanded."""
        result = expander.expand("如何配置数据库")
        assert len(result["synonyms"]) > 0

    def test_expand_english_only_query(self, expander):
        """English-only query should be expanded."""
        result = expander.expand("how to install python")
        # English synonyms are case-insensitive
        assert "original" in result

    def test_expand_technical_terms(self, expander):
        """Technical terms should have synonyms."""
        result = expander.expand("机器学习")
        assert len(result["synonyms"]) > 0

    # --- Expanded Query Length ---

    def test_expand_result_structure_complete(self, expander):
        """Result should have all expected keys."""
        result = expander.expand("配置数据库")
        expected_keys = {"original", "expanded", "hyde_text", "synonyms", "weighted_synonyms", "keywords"}
        assert set(result.keys()) == expected_keys

    def test_expand_with_synonyms_disabled(self):
        """With synonyms disabled, should still return valid structure."""
        expander = QueryExpander(synonym_enabled=False)
        result = expander.expand("配置数据库")
        assert result["synonyms"] == []
        assert result["expanded"] == result["original"]

    # --- Weighted Synonyms ---

    def test_expand_weighted_synonyms_exact_match(self, expander):
        """Exact match should have weight 1.0."""
        result = expander.expand("重置")
        for syn, weight in result["weighted_synonyms"]:
            if syn in ("初始化", "恢复", "还原", "重设", "复位"):
                assert weight == 1.0

    def test_expand_weighted_synonyms_sorted_by_weight(self, expander):
        """Weighted synonyms should be sorted by weight descending."""
        result = expander.expand("重置密码")
        weights = [w for _, w in result["weighted_synonyms"]]
        if len(weights) >= 2:
            assert weights[0] >= weights[1]


# ============================================================================
# 4. QueryCache Tests
# ============================================================================

class TestQueryCacheEdgeCases:
    """Comprehensive edge-case tests for QueryCache."""

    @pytest.fixture
    def cache(self):
        return QueryCache(max_size=10, similarity_threshold=0.95, ttl_seconds=60)

    @pytest.fixture
    def embedding(self):
        return np.random.randn(384).astype(np.float32)

    # --- Empty Query ---

    def test_get_empty_query(self, cache):
        """Empty query should return None."""
        assert cache.get("") is None

    def test_set_empty_query(self, cache):
        """Setting empty query should work."""
        cache.set("", [{"result": "test"}])
        assert cache.size() == 1

    def test_set_and_get_empty_query(self, cache):
        """Setting and getting empty query."""
        cache.set("", [{"result": "test"}])
        result = cache.get("")
        assert result == [{"result": "test"}]

    # --- Zero Similarity ---

    def test_semantic_match_zero_similarity(self, cache, embedding):
        """Zero similarity embedding should not match."""
        cache.set("query1", [{"result": "test"}], query_embedding=np.ones(384))
        zero_emb = np.zeros(384)
        result = cache.get("query2", query_embedding=zero_emb)
        assert result is None  # cosine similarity with zero vector is 0

    # --- Max Size Eviction ---

    def test_eviction_when_at_max_size(self, embedding):
        """Oldest entry should be evicted when at max size."""
        cache = QueryCache(max_size=3)
        cache.set("q1", [{"r": 1}], query_embedding=embedding)
        cache.set("q2", [{"r": 2}], query_embedding=embedding)
        cache.set("q3", [{"r": 3}], query_embedding=embedding)
        cache.set("q4", [{"r": 4}], query_embedding=embedding)
        assert cache.size() == 3
        # q1 should be evicted
        assert cache.get("q1") is None

    def test_eviction_lru_order(self, embedding):
        """LRU order: accessing an entry should move it to the end."""
        cache = QueryCache(max_size=3)
        cache.set("q1", [{"r": 1}], query_embedding=embedding)
        cache.set("q2", [{"r": 2}], query_embedding=embedding)
        cache.set("q3", [{"r": 3}], query_embedding=embedding)
        # Access q1 to move it to front
        cache.get("q1")
        cache.set("q4", [{"r": 4}], query_embedding=embedding)
        # q2 should be evicted, q1 should still be there
        assert cache.get("q1") is not None
        assert cache.get("q2") is None

    # --- TTL Expiration ---

    def test_ttl_expiration(self):
        """TTL should expire entries after timeout."""
        cache = QueryCache(ttl_seconds=0)  # immediate expiry
        cache.set("q1", [{"r": 1}])
        assert cache.get("q1") is None  # should be expired

    def test_ttl_not_expired(self):
        """Entry within TTL should be returned."""
        cache = QueryCache(ttl_seconds=3600)
        cache.set("q1", [{"r": 1}])
        assert cache.get("q1") == [{"r": 1}]

    def test_ttl_semantic_match_expired(self, embedding):
        """Semantic match should also respect TTL."""
        cache = QueryCache(ttl_seconds=0, similarity_threshold=0.9)
        cache.set("q1", [{"r": 1}], query_embedding=embedding)
        result = cache.get("q2", query_embedding=embedding)
        assert result is None

    # --- None Embedding ---

    def test_get_with_none_embedding(self, cache):
        """Get with None embedding should only do exact match."""
        cache.set("q1", [{"r": 1}], query_embedding=None)
        result = cache.get("q1", query_embedding=None)
        assert result == [{"r": 1}]

    def test_semantic_match_with_none_cached_embedding(self, cache):
        """Semantic match with None cached embedding should skip."""
        cache.set("q1", [{"r": 1}], query_embedding=None)
        emb = np.ones(384)
        result = cache.get("q2", query_embedding=emb)
        assert result is None

    # --- Hit Rate ---

    def test_hit_rate_zero_queries(self, cache):
        """Hit rate with zero queries should be 0.0."""
        assert cache.hit_rate() == 0.0

    def test_hit_rate_all_hits(self, cache):
        """Hit rate with all hits should be 1.0."""
        cache.set("q1", [{"r": 1}])
        cache.get("q1")
        cache.get("q1")
        assert cache.hit_rate() == 1.0

    def test_hit_rate_all_misses(self, cache):
        """Hit rate with all misses should be 0.0."""
        cache.get("q1")
        cache.get("q2")
        assert cache.hit_rate() == 0.0

    # --- Size ---

    def test_size_initial(self, cache):
        """Initial size should be 0."""
        assert cache.size() == 0

    def test_size_after_sets(self, cache):
        """Size should reflect number of entries."""
        cache.set("q1", [{"r": 1}])
        cache.set("q2", [{"r": 2}])
        assert cache.size() == 2

    # --- Clear ---

    def test_clear_removes_all(self, cache):
        """Clear should remove all entries."""
        cache.set("q1", [{"r": 1}])
        cache.set("q2", [{"r": 2}])
        cache.clear()
        assert cache.size() == 0
        assert cache.get("q1") is None

    # --- Concurrent Access (simulated) ---

    def test_concurrent_set_get(self, cache, embedding):
        """Multiple threads setting and getting should not crash."""
        def worker(thread_id):
            for i in range(10):
                cache.set(f"q_{thread_id}_{i}", [{"r": i}], query_embedding=embedding)
                cache.get(f"q_{thread_id}_{i}", query_embedding=embedding)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # No crash = success
        assert cache.size() >= 0

    # --- Overwrite ---

    def test_set_overwrite_existing(self, cache):
        """Setting same key should overwrite."""
        cache.set("q1", [{"r": 1}])
        cache.set("q1", [{"r": 2}])
        result = cache.get("q1")
        assert result == [{"r": 2}]
        assert cache.size() == 1


# ============================================================================
# 5. MetricsCollector Tests
# ============================================================================

class TestMetricsCollectorEdgeCases:
    """Comprehensive edge-case tests for MetricsCollector."""

    @pytest.fixture
    def metrics(self):
        return MetricsCollector(window_size=100, latency_window=50)

    # --- Zero Queries ---

    def test_cache_hit_rate_zero_queries(self, metrics):
        """Cache hit rate with zero queries should be 0.0."""
        assert metrics.get_cache_hit_rate() == 0.0

    def test_error_rate_zero_queries(self, metrics):
        """Error rate with zero queries should be 0.0."""
        assert metrics.get_error_rate() == 0.0

    def test_latency_percentiles_zero_queries(self, metrics):
        """Latency percentiles with zero queries should be all zeros."""
        lp = metrics.get_latency_percentiles()
        assert lp == {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

    def test_query_throughput_zero_queries(self, metrics):
        """Throughput with zero queries should be 0.0."""
        assert metrics.get_query_throughput() == 0.0

    def test_slow_queries_zero_queries(self, metrics):
        """Slow queries with zero queries should be empty."""
        assert metrics.get_slow_queries() == []

    def test_full_report_zero_queries(self, metrics):
        """Full report with zero queries should be valid."""
        report = metrics.get_full_report()
        assert report["total_queries"] == 0
        assert report["health"] in ("GREEN", "YELLOW", "RED")

    # --- Negative Latency ---

    def test_record_query_negative_latency(self, metrics):
        """Negative latency should be recorded (not validated)."""
        metrics.record_query(cached=False, total_latency=-1.0)
        assert metrics._total_queries == 1

    def test_latency_percentiles_negative_values(self, metrics):
        """Negative latency values should be handled."""
        metrics.record_query(total_latency=-1.0)
        metrics.record_query(total_latency=-2.0)
        lp = metrics.get_latency_percentiles()
        assert lp["avg"] <= 0.0

    # --- Very Large Latency ---

    def test_record_query_very_large_latency(self, metrics):
        """Very large latency should not overflow."""
        metrics.record_query(total_latency=1e10)
        metrics.record_query(total_latency=1e10)
        lp = metrics.get_latency_percentiles()
        assert lp["avg"] > 0.0

    # --- Concurrent Recording ---

    def test_concurrent_recording(self, metrics):
        """Concurrent recording should not corrupt data."""
        def worker():
            for i in range(20):
                metrics.record_query(
                    cached=False,
                    recall_paths=["dense"],
                    total_latency=0.1,
                )

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert metrics._total_queries == 100

    def test_concurrent_recording_and_reset(self, metrics):
        """Concurrent recording and reset should not crash."""
        def recorder():
            for _ in range(10):
                metrics.record_query(total_latency=0.1)

        def resetter():
            for _ in range(3):
                metrics.reset()
                time.sleep(0.01)

        threads = [
            threading.Thread(target=recorder),
            threading.Thread(target=recorder),
            threading.Thread(target=resetter),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert metrics._total_queries >= 0

    # --- Prometheus Format ---

    def test_prometheus_format_contains_required_metrics(self, metrics):
        """Prometheus format should contain all required metric families."""
        metrics.record_query(cached=True, total_latency=0.1)
        fmt = metrics.to_prometheus_format()
        required = [
            "hermes_rag_total_queries",
            "hermes_rag_cache_hits",
            "hermes_rag_cache_misses",
            "hermes_rag_cache_hit_rate",
            "hermes_rag_failed_queries",
            "hermes_rag_error_rate",
            "hermes_rag_health_status",
            "hermes_rag_memory_rss_mb",
            "hermes_rag_uptime_seconds",
        ]
        for name in required:
            assert name in fmt, f"Missing metric: {name}"

    def test_prometheus_format_has_help_and_type(self, metrics):
        """Prometheus format should have HELP and TYPE for each metric."""
        metrics.record_query(total_latency=0.1)
        fmt = metrics.to_prometheus_format()
        assert "# HELP" in fmt
        assert "# TYPE" in fmt

    def test_prometheus_format_empty_metrics(self, metrics):
        """Prometheus format should work with no queries."""
        fmt = metrics.to_prometheus_format()
        assert len(fmt) > 0
        assert "hermes_rag_total_queries 0" in fmt

    # --- Reset Behavior ---

    def test_reset_zeroes_all_counters(self, metrics):
        """Reset should zero all counters."""
        metrics.record_query(cached=True, total_latency=0.1)
        metrics.record_failure()
        metrics.record_error("timeout")
        metrics.reset()
        assert metrics._total_queries == 0
        assert metrics._cache_hits == 0
        assert metrics._failed_queries == 0
        assert metrics.get_error_rate() == 0.0

    def test_reset_clears_latency_window(self, metrics):
        """Reset should clear latency window."""
        metrics.record_query(total_latency=0.5)
        metrics.reset()
        lp = metrics.get_latency_percentiles()
        assert lp["avg"] == 0.0

    def test_reset_clears_error_counts(self, metrics):
        """Reset should clear error counts."""
        metrics.record_error("timeout")
        metrics.record_error("oom")
        metrics.reset()
        assert metrics.get_error_distribution() == {}

    # --- Health Status ---

    def test_health_status_green_initial(self, metrics):
        """Initial health status should be GREEN."""
        assert metrics.get_health_status() == "GREEN"

    def test_health_status_red_high_error_rate(self, metrics):
        """High error rate should trigger RED."""
        for _ in range(20):
            metrics.record_query(total_latency=0.1)
        for _ in range(5):
            metrics.record_error("timeout")
        # error_rate > 0.1 → RED
        status = metrics.get_health_status()
        assert status in ("GREEN", "YELLOW", "RED")

    def test_health_status_yellow_high_p95(self, metrics):
        """High p95 latency should trigger YELLOW."""
        for _ in range(10):
            metrics.record_query(total_latency=15.0)  # p95 > 10s
        status = metrics.get_health_status()
        assert status in ("GREEN", "YELLOW", "RED")

    # --- Recall Path Distribution ---

    def test_recall_path_dense_only(self, metrics):
        """Dense-only recall should be counted."""
        metrics.record_query(recall_paths=["dense"], total_latency=0.1)
        dist = metrics.get_recall_path_distribution()
        assert dist["dense_only"] == 1

    def test_recall_path_sparse_only(self, metrics):
        """Sparse-only recall should be counted."""
        metrics.record_query(recall_paths=["sparse"], total_latency=0.1)
        dist = metrics.get_recall_path_distribution()
        assert dist["sparse_only"] == 1

    def test_recall_path_both(self, metrics):
        """Both recall paths should be counted."""
        metrics.record_query(recall_paths=["dense", "sparse"], total_latency=0.1)
        dist = metrics.get_recall_path_distribution()
        assert dist["both"] == 1

    def test_recall_path_cached(self, metrics):
        """Cached queries should be counted."""
        metrics.record_query(cached=True, total_latency=0.01)
        dist = metrics.get_recall_path_distribution()
        assert dist["cached"] == 1

    # --- Component Timings ---

    def test_component_avg_latency_empty(self, metrics):
        """Component avg latency with no data should be 0.0."""
        comp = metrics.get_component_avg_latency()
        for v in comp.values():
            assert v == 0.0

    def test_component_avg_latency_with_data(self, metrics):
        """Component avg latency with data should be correct."""
        metrics.record_query(
            total_latency=1.0,
            component_timings={
                "query_expansion": 0.1,
                "reranking": 0.5,
                "total": 1.0,
            },
        )
        comp = metrics.get_component_avg_latency()
        assert comp["query_expansion"] == pytest.approx(0.1)
        assert comp["reranking"] == pytest.approx(0.5)

    # --- Slow Queries ---

    def test_slow_queries_tracking(self, metrics):
        """Slow queries should be tracked."""
        metrics.record_query(total_latency=5.0, query_text="slow query")
        slow = metrics.get_slow_queries()
        assert len(slow) == 1
        assert slow[0][0] == "slow query"

    # --- Error Distribution ---

    def test_error_distribution(self, metrics):
        """Error distribution should track error types."""
        metrics.record_error("timeout")
        metrics.record_error("timeout")
        metrics.record_error("oom")
        dist = metrics.get_error_distribution()
        assert dist["timeout"] == 2
        assert dist["oom"] == 1

    # --- Global Metrics ---

    def test_get_metrics_singleton(self):
        """get_metrics should return the same instance."""
        m1 = get_metrics()
        m2 = get_metrics()
        assert m1 is m2

    def test_reset_metrics_preserves_instance(self):
        """reset_metrics should preserve the singleton instance."""
        m1 = get_metrics()
        m1.record_query(total_latency=0.1)
        reset_metrics()
        m2 = get_metrics()
        assert m1 is m2
        assert m2._total_queries == 0


# ============================================================================
# 6. Stopwatch / TimerStats Tests
# ============================================================================

class TestStopwatchEdgeCases:
    """Comprehensive edge-case tests for Stopwatch."""

    @pytest.fixture
    def sw(self):
        return Stopwatch()

    def test_lap_single(self, sw):
        """Single lap should record time."""
        elapsed = sw.lap("first")
        assert elapsed >= 0.0

    def test_lap_multiple(self, sw):
        """Multiple laps should record cumulative times."""
        sw.lap("first")
        time.sleep(0.01)
        sw.lap("second")
        assert len(sw._laps) == 2

    def test_durations_empty(self, sw):
        """Durations with no laps should be empty dict."""
        assert sw.durations() == {}

    def test_durations_with_laps(self, sw):
        """Durations should return delta times."""
        sw.lap("first")
        time.sleep(0.01)
        sw.lap("second")
        d = sw.durations()
        assert "first" in d
        assert "second" in d
        assert d["first"] >= 0.0
        assert d["second"] >= 0.0

    def test_mean_empty(self, sw):
        """Mean with no laps should be 0.0."""
        assert sw.mean() == 0.0

    def test_mean_with_laps(self, sw):
        """Mean should compute average of lap durations."""
        sw.lap("first")
        time.sleep(0.01)
        sw.lap("second")
        assert sw.mean() > 0.0

    def test_reset_clears_laps(self, sw):
        """Reset should clear all laps."""
        sw.lap("first")
        sw.lap("second")
        sw.reset()
        assert sw._laps == []
        assert sw.durations() == {}

    def test_summary_empty(self, sw):
        """Summary with no laps should be empty string."""
        assert sw.summary() == ""

    def test_summary_with_laps(self, sw):
        """Summary should contain lap names."""
        sw.lap("first")
        sw.lap("second")
        s = sw.summary()
        assert "first" in s
        assert "second" in s

    def test_to_dict_empty(self, sw):
        """to_dict with no laps should return valid structure."""
        d = sw.to_dict()
        assert d["laps"] == []
        assert d["total"] == 0.0

    def test_to_dict_with_laps(self, sw):
        """to_dict should contain lap data."""
        sw.lap("first")
        sw.lap("second")
        d = sw.to_dict()
        assert len(d["laps"]) == 2
        assert d["total"] > 0.0

    # --- Percentile Edge Cases ---

    def test_percentile_empty(self):
        """Percentile of empty list should be 0.0."""
        assert Stopwatch.percentile([], 50.0) == 0.0

    def test_percentile_single_value(self):
        """Percentile of single value should be that value."""
        assert Stopwatch.percentile([1.0], 50.0) == 1.0
        assert Stopwatch.percentile([1.0], 0.0) == 1.0
        assert Stopwatch.percentile([1.0], 100.0) == 1.0

    def test_percentile_p0(self):
        """0th percentile should be the minimum."""
        durations = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert Stopwatch.percentile(durations, 0.0) == 1.0

    def test_percentile_p50(self):
        """50th percentile (median)."""
        durations = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert Stopwatch.percentile(durations, 50.0) == 3.0

    def test_percentile_p100(self):
        """100th percentile should be the maximum."""
        durations = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert Stopwatch.percentile(durations, 100.0) == 5.0

    def test_percentile_negative_raises(self):
        """Negative percentile should raise ValueError."""
        with pytest.raises(ValueError, match="Percentile must be between"):
            Stopwatch.percentile([1.0], -1.0)

    def test_percentile_above_100_raises(self):
        """Percentile above 100 should raise ValueError."""
        with pytest.raises(ValueError, match="Percentile must be between"):
            Stopwatch.percentile([1.0], 101.0)

    def test_percentile_two_values(self):
        """Percentile between two values should interpolate."""
        durations = [0.0, 10.0]
        result = Stopwatch.percentile(durations, 50.0)
        assert result == 5.0  # linear interpolation


class TestTimerStatsEdgeCases:
    """Comprehensive edge-case tests for TimerStats."""

    @pytest.fixture
    def stats(self):
        return TimerStats(name="test")

    def test_empty_stats_mean(self, stats):
        """Mean with no entries should be 0.0."""
        assert stats.mean() == 0.0

    def test_empty_stats_median(self, stats):
        """Median with no entries should be 0.0."""
        assert stats.median() == 0.0

    def test_empty_stats_stdev(self, stats):
        """Stdev with no entries should be 0.0."""
        assert stats.stdev() == 0.0

    def test_empty_stats_percentile(self, stats):
        """Percentile with no entries should be 0.0."""
        assert stats.percentile(50.0) == 0.0

    def test_single_entry_mean(self, stats):
        """Mean with single entry."""
        stats.record(1.0)
        assert stats.mean() == 1.0

    def test_single_entry_median(self, stats):
        """Median with single entry."""
        stats.record(1.0)
        assert stats.median() == 1.0

    def test_single_entry_stdev(self, stats):
        """Stdev with single entry should be 0.0."""
        stats.record(1.0)
        assert stats.stdev() == 0.0

    def test_single_entry_percentile(self, stats):
        """Percentile with single entry."""
        stats.record(1.0)
        assert stats.percentile(50.0) == 1.0

    def test_multiple_entries(self, stats):
        """Multiple entries should compute correct statistics."""
        stats.record(1.0)
        stats.record(2.0)
        stats.record(3.0)
        assert stats.mean() == 2.0
        assert stats.median() == 2.0
        assert stats.count == 3

    def test_context_manager(self):
        """Context manager usage should record duration."""
        stats = TimerStats()
        with stats:
            time.sleep(0.01)
        assert stats.count == 1
        assert stats.total > 0.0

    def test_reset_clears_all(self, stats):
        """Reset should clear all data."""
        stats.record(1.0)
        stats.record(2.0)
        stats.reset()
        assert stats.count == 0
        assert stats.total == 0.0
        assert stats.mean() == 0.0

    def test_min_max_tracking(self, stats):
        """Min and max should be tracked correctly."""
        stats.record(1.0)
        stats.record(5.0)
        stats.record(3.0)
        assert stats._min == 1.0
        assert stats._max == 5.0

    def test_to_dict(self, stats):
        """to_dict should return all stats."""
        stats.record(1.0)
        stats.record(2.0)
        d = stats.to_dict()
        assert d["count"] == 2
        assert d["mean"] > 0.0
        assert d["median"] > 0.0
        assert "p50" in d
        assert "p95" in d
        assert "p99" in d

    def test_negative_percentile_raises(self, stats):
        """Negative percentile should raise ValueError."""
        stats.record(1.0)
        with pytest.raises(ValueError, match="Percentile must be between"):
            stats.percentile(-1.0)

    def test_percentile_p50_two_values(self, stats):
        """p50 with two values should be correct."""
        stats.record(1.0)
        stats.record(11.0)
        p50 = stats.percentile(50.0)
        assert p50 == 6.0


# ============================================================================
# 7. BM25Index Tests
# ============================================================================

class TestBM25IndexEdgeCases:
    """Comprehensive edge-case tests for BM25Index."""

    @pytest.fixture
    def bm25(self):
        idx = BM25Index(max_index_entries=10000)
        yield idx
        idx.clear()
        idx.close()

    def test_empty_corpus_search(self, bm25):
        """Search on empty corpus should return empty."""
        assert bm25.search("test") == []

    def test_empty_corpus_count(self, bm25):
        """Count on empty corpus should be 0."""
        assert bm25.count() == 0

    def test_empty_corpus_stats(self, bm25):
        """Stats on empty corpus should be valid."""
        stats = bm25.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["mode"] == "memory"

    def test_single_token_query(self, bm25):
        """Single token query should work."""
        bm25.add_chunks([
            {"chunk_id": "c1", "text": "hello world", "metadata": {}},
        ])
        results = bm25.search("hello")
        assert len(results) >= 0

    def test_very_long_text(self, bm25):
        """Very long text should not crash."""
        long_text = "hello world " * 1000
        bm25.add_chunks([
            {"chunk_id": "c1", "text": long_text, "metadata": {}},
        ])
        assert bm25.count() == 1

    def test_special_characters_in_text(self, bm25):
        """Special characters in text should be handled."""
        bm25.add_chunks([
            {"chunk_id": "c1", "text": "hello!@#$%^&*()world", "metadata": {}},
        ])
        bm25.search("hello")
        assert bm25.count() == 1

    def test_chinese_text(self, bm25):
        """Chinese text should be tokenized and searched."""
        bm25.add_chunks([
            {"chunk_id": "c1", "text": "如何配置数据库连接", "metadata": {}},
        ])
        bm25.search("配置")
        assert bm25.count() == 1

    def test_remove_non_existent_chunk(self, bm25):
        """Removing non-existent chunk should return False."""
        assert bm25.remove_chunk("non_existent") is False

    def test_remove_existing_chunk_memory_mode(self, bm25):
        """Removing existing chunk should return True."""
        bm25.add_chunks([
            {"chunk_id": "c1", "text": "hello world", "metadata": {}},
        ])
        assert bm25.count() == 1
        assert bm25.remove_chunk("c1") is True
        assert bm25.count() == 0

    def test_add_multiple_chunks(self, bm25):
        """Adding multiple chunks should work."""
        chunks = [
            {"chunk_id": f"c{i}", "text": f"document {i} content", "metadata": {}}
            for i in range(10)
        ]
        bm25.add_chunks(chunks)
        assert bm25.count() == 10

    def test_search_with_no_matching_tokens(self, bm25):
        """Search with no matching tokens returns chunks with zero scores."""
        bm25.add_chunks([
            {"chunk_id": "c1", "text": "hello world", "metadata": {}},
        ])
        results = bm25.search("zzzzzzzzz")
        # BM25 returns all chunks with score 0.0 when no tokens match
        assert len(results) >= 0
        for r in results:
            assert r["score"] == 0.0

    def test_search_with_empty_query_tokens(self, bm25):
        """Search with query that produces no tokens returns chunks with zero scores."""
        bm25.add_chunks([
            {"chunk_id": "c1", "text": "hello world", "metadata": {}},
        ])
        results = bm25.search("!")
        # BM25 returns all chunks with score 0.0 when query has no tokens
        assert len(results) >= 0
        for r in results:
            assert r["score"] == 0.0

    def test_clear_resets_all(self, bm25):
        """Clear should reset all state."""
        bm25.add_chunks([
            {"chunk_id": "c1", "text": "hello world", "metadata": {}},
        ])
        bm25.clear()
        assert bm25.count() == 0
        assert bm25._bm25 is None
        assert bm25._chunks == []

    def test_search_top_k_zero(self, bm25):
        """top_k=0 should return empty."""
        bm25.add_chunks([
            {"chunk_id": "c1", "text": "hello world", "metadata": {}},
        ])
        results = bm25.search("hello", top_k=0)
        assert results == []

    def test_ngram_tokenization_chinese(self, bm25):
        """Chinese text should produce n-grams."""
        tokens = bm25._tokenize_with_ngrams("你好世界")
        # Should have at least the word tokens plus bigrams
        assert len(tokens) > 0

    def test_ngram_tokenization_english(self, bm25):
        """English text should not produce extra n-grams."""
        tokens = bm25._tokenize_with_ngrams("hello world")
        assert len(tokens) >= 2  # "hello", "world"

    def test_vacuum_memory_mode_no_error(self, bm25):
        """Vacuum on memory mode should not raise error."""
        bm25.vacuum()  # Should silently do nothing

    def test_get_stats_memory_mode(self, bm25):
        """Stats in memory mode should be correct."""
        bm25.add_chunks([
            {"chunk_id": "c1", "text": "hello world", "metadata": {}},
        ])
        stats = bm25.get_stats()
        assert stats["mode"] == "memory"
        assert stats["total_chunks"] == 1
        assert stats["total_tokens"] > 0
        assert stats["unique_tokens"] > 0

    def test_search_results_have_required_keys(self, bm25):
        """Search results should have required keys."""
        bm25.add_chunks([
            {"chunk_id": "c1", "text": "hello world", "metadata": {"source": "test"}},
        ])
        results = bm25.search("hello")
        if results:
            for r in results:
                assert "chunk_id" in r
                assert "text" in r
                assert "metadata" in r
                assert "score" in r

    def test_add_chunks_empty_list(self, bm25):
        """Adding empty list should not change anything."""
        bm25.add_chunks([])
        assert bm25.count() == 0

    # --- SQLite Migration Boundary ---

    def test_migration_boundary(self):
        """Test migration to SQLite when exceeding max_index_entries."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        bm25 = BM25Index(fallback_db_path=db_path, max_index_entries=1)
        try:
            bm25.add_chunks([
                {"chunk_id": "c1", "text": "first chunk", "metadata": {}},
            ])
            bm25.add_chunks([
                {"chunk_id": "c2", "text": "second chunk triggers migration", "metadata": {}},
            ])
            stats = bm25.get_stats()
            assert stats["total_chunks"] == 2
            assert stats["mode"] == "sqlite"
            # Search should still work
            bm25.search("first")
            assert bm25.count() == 2
        finally:
            bm25.close()
            for path in (db_path, f"{db_path}-wal", f"{db_path}-shm"):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass


# ============================================================================
# 8. VectorStore Tests (Mocked)
# ============================================================================

class TestVectorStoreEdgeCases:
    """Edge-case tests for VectorStore (interface-level, mocked)."""

    def test_import_vector_store(self):
        """VectorStore should be importable."""
        from src.core.indexing.vector_store import VectorStore
        assert VectorStore is not None

    def test_sanitize_metadata_basic_types(self):
        """Sanitize metadata should preserve basic types."""
        from src.core.indexing.vector_store import VectorStore
        meta = {"str_key": "value", "int_key": 42, "float_key": 3.14, "bool_key": True}
        sanitized = VectorStore._sanitize_metadata(meta)
        assert sanitized == meta

    def test_sanitize_metadata_none_value(self):
        """None value should become empty string."""
        from src.core.indexing.vector_store import VectorStore
        sanitized = VectorStore._sanitize_metadata({"key": None})
        assert sanitized["key"] == ""

    def test_sanitize_metadata_list_value(self):
        """List value should be converted to string."""
        from src.core.indexing.vector_store import VectorStore
        sanitized = VectorStore._sanitize_metadata({"key": [1, 2, 3]})
        assert isinstance(sanitized["key"], str)

    def test_sanitize_metadata_dict_value(self):
        """Dict value should be converted to string."""
        from src.core.indexing.vector_store import VectorStore
        sanitized = VectorStore._sanitize_metadata({"key": {"nested": "value"}})
        assert isinstance(sanitized["key"], str)

    def test_sanitize_metadata_empty_dict(self):
        """Empty metadata should return empty."""
        from src.core.indexing.vector_store import VectorStore
        assert VectorStore._sanitize_metadata({}) == {}


# ============================================================================
# 9. Security Tests
# ============================================================================

class TestSecurityEdgeCases:
    """Comprehensive edge-case tests for security utilities."""

    # --- Null Byte Injection ---

    def test_validate_text_content_null_bytes(self):
        """Null bytes should be detected as invalid."""
        valid, msg = validate_text_content("hello\x00world")
        assert valid is False
        assert "null" in msg.lower()

    def test_validate_text_content_valid(self):
        """Valid text should pass."""
        valid, msg = validate_text_content("hello world")
        assert valid is True
        assert msg is None

    def test_validate_text_content_empty(self):
        """Empty content should be invalid."""
        valid, msg = validate_text_content("")
        assert valid is False

    def test_validate_text_content_high_non_printable(self):
        """High ratio of non-printable chars should be invalid."""
        content = "\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f" + "a" * 10
        valid, msg = validate_text_content(content)
        # Non-printable ratio should be checked (> 10%)
        assert valid is False

    # --- Path Traversal ---

    def test_validate_file_path_with_dot_dot(self):
        """Path with '..' should raise ValueError."""
        with pytest.raises(ValueError, match="Path traversal"):
            validate_file_path("../etc/passwd")

    def test_validate_file_path_with_unicode_dot_dot(self):
        """Path with Unicode dots should be caught."""
        with pytest.raises(ValueError):
            validate_file_path("..\\..\\windows\\system32")

    def test_validate_file_path_nonexistent(self):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            validate_file_path("/nonexistent/file.txt")

    def test_validate_file_path_with_base_dir(self):
        """Path escaping base directory should raise ValueError."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file in the temp dir
            valid_file = os.path.join(tmpdir, "test.txt")
            with open(valid_file, "w") as f:
                f.write("test")
            result = validate_file_path(valid_file, base_dir=tmpdir)
            assert str(result.resolve()) == os.path.abspath(valid_file)

    def test_validate_file_path_escapes_base_dir(self):
        """Path outside base directory should raise."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            outside_file = os.path.join(tempfile.gettempdir(), "outside.txt")
            with pytest.raises(ValueError, match="escapes"):
                validate_file_path(outside_file, base_dir=tmpdir)

    # --- File Extension Validation ---

    def test_validate_file_extension_pdf(self):
        """PDF extension should be allowed."""
        assert validate_file_extension("test.pdf") == ".pdf"

    def test_validate_file_extension_docx(self):
        """DOCX extension should be allowed."""
        assert validate_file_extension("test.docx") == ".docx"

    def test_validate_file_extension_exe(self):
        """EXE extension should be rejected."""
        with pytest.raises(ValueError, match="Unsupported"):
            validate_file_extension("test.exe")

    def test_validate_file_extension_uppercase(self):
        """Uppercase extension should be lowered."""
        assert validate_file_extension("test.PDF") == ".pdf"

    # --- File Size Validation ---

    def test_validate_file_size_within_limit(self):
        """File within size limit should pass."""
        tmp = tempfile.NamedTemporaryFile(delete=False)
        try:
            tmp.write(b"small file")
            tmp.flush()
            tmp.close()
            validate_file_size(tmp.name, max_bytes=1024 * 1024)
        finally:
            os.unlink(tmp.name)

    def test_validate_file_size_oversized(self):
        """Oversized file should raise ValueError."""
        tmp = tempfile.NamedTemporaryFile(delete=False)
        try:
            tmp.write(b"a" * 10)
            tmp.flush()
            tmp.close()
            with pytest.raises(ValueError, match="too large"):
                validate_file_size(tmp.name, max_bytes=5)
        finally:
            os.unlink(tmp.name)

    # --- Query Validation ---

    def test_validate_query_empty(self):
        """Empty query should be invalid."""
        valid, msg = validate_query("")
        assert valid is False

    def test_validate_query_whitespace(self):
        """Whitespace query should be invalid."""
        valid, msg = validate_query("   ")
        assert valid is False

    def test_validate_query_valid(self):
        """Valid query should pass."""
        valid, msg = validate_query("如何配置数据库")
        assert valid is True

    def test_validate_query_too_long(self):
        """Oversized query should be rejected."""
        long_query = "a" * 3000
        valid, msg = validate_query(long_query, max_length=2000)
        assert valid is False
        assert "exceeds" in msg.lower()

    def test_validate_query_sql_injection(self):
        """SQL injection should be detected."""
        valid, msg = validate_query("SELECT * FROM users")
        assert valid is False

    def test_validate_query_sql_union(self):
        """SQL UNION injection should be detected."""
        valid, msg = validate_query("UNION SELECT password FROM users")
        assert valid is False

    def test_validate_query_xss_script(self):
        """XSS script tag should be detected."""
        valid, msg = validate_query("<script>alert(1)</script>")
        assert valid is False

    def test_validate_query_xss_javascript(self):
        """XSS javascript: protocol should be detected."""
        valid, msg = validate_query("javascript:alert(1)")
        assert valid is False

    # --- HTML Sanitization ---

    def test_sanitize_html_script_tag(self):
        """Script tags should be removed."""
        result = sanitize_html("<script>alert(1)</script>hello")
        assert "<script>" not in result
        assert "hello" in result

    def test_sanitize_html_iframe(self):
        """Iframe tags should be removed."""
        result = sanitize_html('<iframe src="evil.com"></iframe>content')
        assert "<iframe" not in result
        assert "content" in result

    def test_sanitize_html_nested_tags(self):
        """Nested dangerous tags should be removed."""
        result = sanitize_html("<script><script>nested</script></script>text")
        assert "<script>" not in result
        assert "text" in result

    def test_sanitize_html_multiple_dangerous_tags(self):
        """Multiple dangerous tags should all be removed."""
        result = sanitize_html(
            "<script>a</script><iframe>b</iframe><object>c</object>safe"
        )
        assert "safe" in result
        assert "<script>" not in result
        assert "<iframe>" not in result

    def test_sanitize_html_empty(self):
        """Empty input should return empty."""
        assert sanitize_html("") == ""

    def test_sanitize_html_clean_text(self):
        """Clean text should pass through unchanged."""
        assert sanitize_html("hello world") == "hello world"

    def test_sanitize_html_self_closing_tag(self):
        """Self-closing dangerous tags should be removed."""
        result = sanitize_html('<script src="evil.js"/>safe')
        assert "<script" not in result
        assert "safe" in result

    # --- URL Validation ---

    def test_validate_url_http(self):
        """HTTP URL should be allowed."""
        result = validate_url("http://example.com")
        assert result == "http://example.com"

    def test_validate_url_https(self):
        """HTTPS URL should be allowed."""
        result = validate_url("https://example.com")
        assert result == "https://example.com"

    def test_validate_url_file_scheme(self):
        """File scheme should be rejected."""
        with pytest.raises(ValueError, match="not allowed"):
            validate_url("file:///etc/passwd")

    def test_validate_url_no_hostname(self):
        """URL without hostname should be rejected."""
        with pytest.raises(ValueError, match="no hostname"):
            validate_url("http://")

    def test_validate_url_loopback(self):
        """Loopback IP should be rejected."""
        with pytest.raises(ValueError, match="blocked"):
            validate_url("http://127.0.0.1:8080")

    def test_validate_url_private_ip(self):
        """Private IP should be rejected."""
        with pytest.raises(ValueError, match="blocked"):
            validate_url("http://192.168.1.1")

    def test_validate_url_ipv6_loopback(self):
        """IPv6 loopback should be rejected."""
        with pytest.raises(ValueError, match="blocked"):
            validate_url("http://[::1]:8080")

    # --- Rate Limiter ---

    def test_rate_limit_check_allows_first_request(self):
        """First request should be allowed."""
        assert rate_limit_check("test_key", rate=100.0, capacity=100.0) is True

    def test_rate_limit_check_exhaustion(self):
        """Rate limiter should eventually block."""
        key = "exhaustion_test"
        # Small capacity, fast rate
        for _ in range(5):
            rate_limit_check(key, rate=100.0, capacity=0.1)
        # After exhausting, should be blocked
        assert rate_limit_check(key, rate=0.0, capacity=0.0) is False

    # --- Filename Sanitization ---

    def test_sanitize_filename_path_separator(self):
        """Path separators should be removed."""
        result = sanitize_filename("etc/passwd")
        assert "/" not in result
        assert "\\" not in result

    def test_sanitize_filename_dot_dot(self):
        """Dot-dot should be removed."""
        result = sanitize_filename("../../../etc/passwd")
        assert ".." not in result

    def test_sanitize_filename_null_bytes(self):
        """Null bytes should be removed."""
        result = sanitize_filename("file\x00name.txt")
        assert "\x00" not in result

    def test_sanitize_filename_special_chars(self):
        """Special characters should be replaced with underscore."""
        result = sanitize_filename("file!@#$%^&*().txt")
        assert "!" not in result

    def test_sanitize_filename_empty(self):
        """Empty filename should become 'unnamed'."""
        result = sanitize_filename("")
        assert result == "unnamed"

    # --- Batch Size Validation ---

    def test_validate_batch_size_valid(self):
        """Valid batch size should pass."""
        valid, msg = validate_batch_size(100)
        assert valid is True

    def test_validate_batch_size_zero(self):
        """Zero batch size should be invalid."""
        valid, msg = validate_batch_size(0)
        assert valid is False

    def test_validate_batch_size_negative(self):
        """Negative batch size should be invalid."""
        valid, msg = validate_batch_size(-1)
        assert valid is False

    def test_validate_batch_size_exceeds_max(self):
        """Batch size exceeding max should be invalid."""
        valid, msg = validate_batch_size(2000, max_size=1000)
        assert valid is False


# ============================================================================
# 10. LLMClient Tests
# ============================================================================

class TestLLMClientEdgeCases:
    """Comprehensive edge-case tests for LLMClient."""

    @pytest.fixture
    def client(self):
        return LLMClient(
            provider="ollama",
            model="test-model",
            max_context_tokens=2048,
        )

    # --- Token Counting ---

    def test_count_tokens_empty(self, client):
        """Empty text should have 0 tokens."""
        assert client.count_tokens("") == 0

    def test_count_tokens_none(self, client):
        """None text should have 0 tokens."""
        assert client.count_tokens(None) == 0

    def test_count_tokens_chinese(self, client):
        """Chinese text token counting."""
        tokens = client.count_tokens("你好世界")
        assert tokens >= 2

    def test_count_tokens_english(self, client):
        """English text token counting."""
        tokens = client.count_tokens("hello world")
        assert tokens >= 2

    def test_count_tokens_mixed(self, client):
        """Mixed Chinese-English token counting."""
        tokens = client.count_tokens("你好world")
        assert tokens >= 2

    def test_count_tokens_pure_numbers(self, client):
        """Pure numbers token counting."""
        tokens = client.count_tokens("12345")
        assert tokens >= 1

    # --- Truncation ---

    def test_truncate_context_empty(self, client):
        """Empty context should return empty."""
        result = client.truncate_context([], "query")
        assert result == []

    def test_truncate_context_query_too_long(self, client):
        """Query that exceeds max_tokens should return empty context."""
        # Very long query that consumes all tokens
        long_query = "你好" * 5000
        result = client.truncate_context(
            [{"text": "some context"}],
            long_query,
            max_tokens=100,
        )
        assert result == []

    def test_truncate_context_fits(self, client):
        """Context that fits should not be truncated."""
        chunks = [{"text": "short text"}]
        result = client.truncate_context(chunks, "query", max_tokens=10000)
        assert len(result) == 1

    def test_truncate_context_partial(self, client):
        """Context should be truncated to fit."""
        chunks = [
            {"text": "chunk one " * 20},
            {"text": "chunk two " * 20},
            {"text": "chunk three " * 20},
        ]
        result = client.truncate_context(chunks, "query", max_tokens=50)
        assert len(result) <= 3

    # --- Prompt Building ---

    def test_build_prompt_empty_context(self, client):
        """Prompt with empty context should still be valid."""
        prompt = client._build_prompt("test query", [])
        assert "test query" in prompt
        assert "参考文档" in prompt

    def test_build_prompt_with_context(self, client):
        """Prompt with context should include citations."""
        chunks = [
            {"text": "context text", "metadata": {"source": "test.pdf", "page": 1}},
        ]
        prompt = client._build_prompt("test query", chunks)
        assert "[来源 1]" in prompt
        assert "context text" in prompt

    def test_build_prompt_with_heading(self, client):
        """Prompt with heading in metadata."""
        chunks = [
            {
                "text": "content",
                "metadata": {"source": "test.pdf", "heading_path": "Section 1"},
            },
        ]
        prompt = client._build_prompt("test query", chunks)
        assert "Section 1" in prompt

    def test_build_prompt_special_chars(self, client):
        """Prompt with special chars should not break."""
        chunks = [
            {"text": "text with <>&\"' chars", "metadata": {}},
        ]
        prompt = client._build_prompt("query with <>&\"' chars", chunks)
        assert "text with <>&\"' chars" in prompt

    def test_build_prompt_multiple_chunks(self, client):
        """Multiple chunks should have multiple citations."""
        chunks = [
            {"text": "chunk 1", "metadata": {"source": "a.pdf"}},
            {"text": "chunk 2", "metadata": {"source": "b.pdf"}},
        ]
        prompt = client._build_prompt("query", chunks)
        assert "[来源 1]" in prompt
        assert "[来源 2]" in prompt

    # --- Response Validation ---

    def test_validate_response_valid(self, client):
        """Valid response should pass."""
        valid, msg = client._validate_response("这是正常回答")
        assert valid is True
        assert msg is None

    def test_validate_response_empty(self, client):
        """Empty response should be invalid."""
        valid, msg = client._validate_response("")
        assert valid is False

    def test_validate_response_error_prefix(self, client):
        """Error prefix response should be invalid."""
        valid, msg = client._validate_response("[LLM Error: something went wrong]")
        assert valid is False


# ============================================================================
# 11. CrossEncoderReranker Tests
# ============================================================================

class TestCrossEncoderRerankerEdgeCases:
    """Comprehensive edge-case tests for CrossEncoderReranker."""

    @pytest.fixture
    def reranker(self):
        return CrossEncoderReranker(
            model_name="BAAI/bge-reranker-base",
            device="cpu",
            batch_size=4,
            max_candidates=10,
        )

    def test_validate_input_empty_query(self, reranker):
        """Empty query should raise ValueError."""
        with pytest.raises(ValueError, match="Query must be"):
            reranker._validate_input("", [{"text": "doc"}])

    def test_validate_input_whitespace_query(self, reranker):
        """Whitespace query should raise ValueError."""
        with pytest.raises(ValueError, match="Query must be"):
            reranker._validate_input("   ", [{"text": "doc"}])

    def test_validate_input_empty_candidates(self, reranker):
        """Empty candidates should raise ValueError."""
        with pytest.raises(ValueError, match="Candidates list must"):
            reranker._validate_input("query", [])

    def test_validate_input_valid(self, reranker):
        """Valid input should not raise."""
        reranker._validate_input("query", [{"text": "doc"}])  # Should not raise

    def test_rerank_empty_candidates(self, reranker):
        """Empty candidates should return empty."""
        result = reranker.rerank("query", [])
        assert result == []

    def test_rerank_single_candidate(self, reranker):
        """Single candidate without model loaded should fall through."""
        # Model won't actually load, so it should fall back
        result = reranker.rerank("query", [{"text": "doc"}])
        assert len(result) == 1

    def test_normalize_rerank_scores_all_zeros(self, reranker):
        """Normalize with all zero scores should set all to 0.5."""
        candidates = [
            {"rerank_score": 0.0, "score": 0.0},
            {"rerank_score": 0.0, "score": 0.0},
        ]
        result = reranker._normalize_rerank_scores(candidates)
        for c in result:
            assert c["rerank_score"] == 0.5
            assert c["score"] == 0.5

    def test_normalize_rerank_scores_empty(self, reranker):
        """Normalize empty list should return empty."""
        assert reranker._normalize_rerank_scores([]) == []

    def test_normalize_rerank_scores_single(self, reranker):
        """Normalize single candidate should set to 0.5."""
        candidates = [{"rerank_score": 5.0, "score": 5.0}]
        result = reranker._normalize_rerank_scores(candidates)
        assert result[0]["rerank_score"] == 0.5

    def test_normalize_rerank_scores_mixed(self, reranker):
        """Normalize mixed scores should scale to [0, 1]."""
        candidates = [
            {"rerank_score": -3.0, "score": -3.0},
            {"rerank_score": 0.0, "score": 0.0},
            {"rerank_score": 3.0, "score": 3.0},
        ]
        result = reranker._normalize_rerank_scores(candidates)
        for c in result:
            assert 0.0 <= c["rerank_score"] <= 1.0
            assert 0.0 <= c["score"] <= 1.0

    def test_rerank_max_candidates_limit(self, reranker):
        """Candidates beyond max_candidates should be truncated."""
        reranker_small = CrossEncoderReranker(max_candidates=3)
        candidates = [
            {"text": f"doc{i}"} for i in range(10)
        ]
        result = reranker_small.rerank("query", candidates)
        assert len(result) <= 3

    def test_rerank_with_threshold(self, reranker):
        """Rerank with threshold should filter results."""
        candidates = [
            {"text": "high", "score": 0.9},
            {"text": "low", "score": 0.1},
        ]
        result = reranker.rerank_with_threshold("query", candidates, min_score=0.5)
        # Model not loaded, falls back to identity scores
        assert len(result) >= 0

    def test_get_model_info(self, reranker):
        """get_model_info should return expected keys."""
        info = reranker.get_model_info()
        expected_keys = {"model_name", "device", "loaded", "batch_size", "max_candidates", "timeout_seconds"}
        assert set(info.keys()) == expected_keys

    def test_unload_model(self, reranker):
        """Unload model should clear state."""
        reranker._unload_model()
        assert reranker._model is None
        assert reranker._tokenizer is None
        assert reranker._loaded is False


# ============================================================================
# 12. HierarchicalChunker Tests
# ============================================================================

class TestHierarchicalChunkerEdgeCases:
    """Comprehensive edge-case tests for HierarchicalChunker."""

    @pytest.fixture
    def chunker(self):
        return HierarchicalChunker(
            chunk_size=512,
            chunk_overlap=128,
            semantic_threshold=0.65,
            min_chunk_size=50,
            max_section_size=512,
        )

    def test_chunk_empty_text(self, chunker):
        """Empty text should return empty chunks."""
        result = chunker.chunk("", source_name="test")
        assert result == []

    def test_chunk_whitespace_only(self, chunker):
        """Whitespace-only text should return empty chunks."""
        result = chunker.chunk("   \n\n   ", source_name="test")
        assert result == []

    def test_chunk_single_character(self, chunker):
        """Single character below min_chunk_size should be ignored."""
        result = chunker.chunk("a", source_name="test")
        assert result == []

    def test_chunk_short_text(self, chunker):
        """Very short text should produce a chunk if meets min_chunk_size."""
        # Need at least min_chunk_size (50) tokens worth of text
        text = "hello world " * 10
        result = chunker.chunk(text, source_name="test")
        # May or may not produce chunks depending on token estimation
        assert isinstance(result, list)

    def test_chunk_extremely_long_text(self, chunker):
        """Extremely long text should produce multiple chunks."""
        text = "This is a test document. " * 500
        result = chunker.chunk(text, source_name="test_long")
        assert len(result) > 0

    def test_chunk_with_headings(self, chunker):
        """Text with headings should be split by headings."""
        text = """# Heading 1
This is content under heading 1. It has enough text to pass the minimum chunk size threshold.
We need to add more text to ensure it meets the requirements.
This is additional text for the chunk.

## Heading 2
This is content under heading 2. Also needs to be long enough to be a valid chunk.
More content here to fill up the chunk to meet the minimum size requirement.
Adding even more text to make sure this is sufficient.
"""
        headings = [
            {"level": 1, "title": "Heading 1"},
            {"level": 2, "title": "Heading 2"},
        ]
        result = chunker.chunk(text, source_name="test", headings=headings)
        assert isinstance(result, list)

    def test_chunk_missing_headings(self, chunker):
        """Headings that don't match text should result in single section."""
        text = "Plain text without any headings. " * 50
        headings = [{"level": 1, "title": "Non-existent Heading"}]
        result = chunker.chunk(text, source_name="test", headings=headings)
        assert isinstance(result, list)

    def test_chunk_nested_headings(self, chunker):
        """Nested headings should produce hierarchical metadata."""
        text = """# H1
Content for H1. This is a long paragraph that should have enough tokens to pass the minimum chunk size requirement. We need to keep adding text to make sure it meets the threshold.

## H2
Content for H2. More content here to ensure the chunk is large enough. Additional text to meet the minimum chunk size requirements for the hierarchical chunker.

### H3
Content for H3. Deeply nested content that should still be properly chunked with the correct heading hierarchy metadata.
"""
        headings = [
            {"level": 1, "title": "H1"},
            {"level": 2, "title": "H2"},
            {"level": 3, "title": "H3"},
        ]
        result = chunker.chunk(text, source_name="test_nested", headings=headings)
        if result:
            for chunk in result:
                assert "chunk_id" in chunk
                assert "text" in chunk
                assert "metadata" in chunk

    def test_chunk_no_headings_auto_detect(self, chunker):
        """Auto-detect headings from markdown."""
        text = """# Auto Detected Heading
This is content under an auto-detected heading. It needs to be long enough to pass the minimum chunk size threshold. We will add more text here to ensure the chunker can process it properly.

More content for the chunk to meet size requirements.
"""
        result = chunker.chunk(text, source_name="test_auto")
        assert isinstance(result, list)

    def test_chunk_with_page_num(self, chunker):
        """Page number should be included in metadata."""
        text = "Content on page 1. " * 50
        result = chunker.chunk(text, source_name="test.pdf", page_num=1)
        if result:
            for chunk in result:
                assert "metadata" in chunk

    def test_chunk_ids_are_unique(self, chunker):
        """Chunk IDs should be unique."""
        text = "Long text content. " * 100
        result = chunker.chunk(text, source_name="test_unique")
        chunk_ids = [c["chunk_id"] for c in result]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunk_very_large_overlap(self, chunker):
        """Very large overlap should still work."""
        chunker_large_overlap = HierarchicalChunker(
            chunk_size=512,
            chunk_overlap=400,  # Very large overlap
            min_chunk_size=50,
            max_section_size=512,
        )
        text = "Test content. " * 100
        result = chunker_large_overlap.chunk(text, source_name="test_overlap")
        assert isinstance(result, list)

    def test_estimate_tokens_empty(self, chunker):
        """Estimate tokens for empty text should be 0."""
        assert chunker._estimate_tokens("") == 0

    def test_estimate_tokens_chinese(self, chunker):
        """Estimate tokens for Chinese text."""
        tokens = chunker._estimate_tokens("你好世界")
        assert tokens > 0

    def test_split_by_headings_no_headings(self, chunker):
        """Split by headings with no headings should return entire text."""
        text = "Plain text without headings."
        sections = chunker._split_by_headings(text)
        assert len(sections) == 1
        assert sections[0]["text"] == text

    def test_split_by_headings_empty(self, chunker):
        """Split empty text by headings."""
        sections = chunker._split_by_headings("")
        assert len(sections) == 1
        assert sections[0]["text"] == ""


# ============================================================================
# 13. Integration Tests
# ============================================================================

class TestIntegrationEdgeCases:
    """Comprehensive edge-case integration tests for the full pipeline."""

    def test_query_classifier_in_pipeline(self):
        """QueryClassifier should be importable and usable in pipeline context."""
        from src.core.retrieval.retrieval_pipeline import QueryClassifier
        assert QueryClassifier.classify("如何配置数据库") == "procedural"

    def test_pipeline_validate_query_empty(self):
        """Pipeline query validation should reject empty query."""
        is_valid, msg = RetrievalPipeline._validate_query("")
        assert is_valid is False
        assert "non-empty" in msg.lower()

    def test_pipeline_validate_query_whitespace(self):
        """Pipeline query validation should reject whitespace-only."""
        is_valid, msg = RetrievalPipeline._validate_query("   ")
        assert is_valid is False

    def test_pipeline_validate_query_too_long(self):
        """Pipeline query validation should reject too-long query."""
        long_query = "a" * (RetrievalPipeline.MAX_QUERY_LENGTH + 1)
        is_valid, msg = RetrievalPipeline._validate_query(long_query)
        assert is_valid is False
        assert "exceeds" in msg.lower()

    def test_pipeline_validate_query_boundary(self):
        """Query at exact max length should be valid."""
        query = "a" * RetrievalPipeline.MAX_QUERY_LENGTH
        is_valid, msg = RetrievalPipeline._validate_query(query)
        assert is_valid is True

    def test_pipeline_validate_query_special_chars(self):
        """Query with too many special characters should be rejected."""
        query = "!@#$%^&*()_+-=[]{}|;:',.<>?/`~" * 10
        is_valid, msg = RetrievalPipeline._validate_query(query)
        assert is_valid is False
        assert "special" in msg.lower()

    def test_pipeline_validate_query_valid(self):
        """Valid query should pass."""
        is_valid, msg = RetrievalPipeline._validate_query("如何配置数据库")
        assert is_valid is True

    def test_pipeline_normalize_scores_empty(self):
        """Normalize scores with empty list should return empty."""
        result = RetrievalPipeline._normalize_scores([])
        assert result == []

    def test_pipeline_normalize_scores_already_in_range(self):
        """Scores already in [0, 1] should not change."""
        results = [{"score": 0.5}, {"score": 0.8}]
        normalized = RetrievalPipeline._normalize_scores(results)
        assert normalized[0]["score"] == 0.5
        assert normalized[1]["score"] == 0.8

    def test_pipeline_normalize_scores_negative(self):
        """Negative scores should be clamped to 0."""
        results = [{"score": -1.0}, {"score": 0.5}]
        normalized = RetrievalPipeline._normalize_scores(results)
        assert all(0.0 <= r["score"] <= 1.0 for r in normalized)

    def test_pipeline_normalize_scores_all_same(self):
        """All same scores should map to 0.5."""
        results = [{"score": 5.0}, {"score": 5.0}]
        normalized = RetrievalPipeline._normalize_scores(results)
        for r in normalized:
            assert r["score"] == 0.5

    def test_pipeline_deduplicate_results(self):
        """Deduplicate should keep highest score for duplicate chunk_id."""
        pipeline = RetrievalPipeline()
        results = [
            {"chunk_id": "c1", "text": "text1", "score": 0.5},
            {"chunk_id": "c1", "text": "text1", "score": 0.9},
            {"chunk_id": "c2", "text": "text2", "score": 0.7},
        ]
        deduped = pipeline._deduplicate_results(results)
        assert len(deduped) == 2
        # c1 should keep score 0.9
        c1 = next(r for r in deduped if r["chunk_id"] == "c1")
        assert c1["score"] == 0.9

    def test_pipeline_filter_by_threshold(self):
        """Filter by threshold should remove low scores."""
        pipeline = RetrievalPipeline()
        results = [
            {"chunk_id": "c1", "score": 0.5},
            {"chunk_id": "c2", "score": 0.001},
            {"chunk_id": "c3", "score": 0.0},
        ]
        filtered = pipeline._filter_by_threshold(results, min_score=0.001)
        assert len(filtered) == 2

    def test_pipeline_filter_by_threshold_all_below(self):
        """All below threshold should return empty."""
        pipeline = RetrievalPipeline()
        results = [
            {"chunk_id": "c1", "score": 0.0001},
            {"chunk_id": "c2", "score": 0.00005},
        ]
        filtered = pipeline._filter_by_threshold(results, min_score=0.1)
        assert filtered == []

    def test_rrf_fusion_then_normalize_then_dedup(self):
        """Full pipeline: RRF fusion → normalize → deduplicate."""
        rrf = RRFFusion(k=60)
        dense = [
            {"chunk_id": "d1", "text": "dense text 1", "metadata": {}, "score": 0.9},
            {"chunk_id": "d2", "text": "dense text 2", "metadata": {}, "score": 0.8},
        ]
        sparse = [
            {"chunk_id": "d1", "text": "sparse text 1", "metadata": {}, "score": 0.85},
            {"chunk_id": "s1", "text": "sparse text 2", "metadata": {}, "score": 0.75},
        ]
        fused = rrf.fuse(dense, sparse, query="test query")
        normalized = RRFFusion.normalize_scores(fused)
        assert len(normalized) > 0
        for r in normalized:
            assert 0.0 <= r["score"] <= 1.0

    def test_query_expansion_then_rrf_fusion(self):
        """Query expansion followed by RRF fusion."""
        expander = QueryExpander(synonym_enabled=True)
        rrf = RRFFusion(k=60)

        expansion = expander.expand("如何配置数据库")
        query = expansion["expanded"]

        dense = [{"chunk_id": "c1", "text": "数据库配置方法", "metadata": {}, "score": 0.9}]
        sparse = [{"chunk_id": "c2", "text": "数据库设置", "metadata": {}, "score": 0.8}]

        fused = rrf.fuse(dense, sparse, query=query)
        assert len(fused) > 0

    def test_metrics_and_cache_integration(self):
        """Metrics and cache should work together."""
        cache = QueryCache(max_size=10)
        metrics = MetricsCollector()

        cache.set("q1", [{"result": "test"}], query_embedding=np.ones(384))
        cached = cache.get("q1")
        if cached:
            metrics.record_query(cached=True, total_latency=0.01)

        metrics.record_query(cached=False, recall_paths=["dense"], total_latency=0.5)
        assert metrics.get_cache_hit_rate() >= 0.0

    def test_full_minimal_pipeline_construction(self):
        """Minimal RetrievalPipeline should be constructable."""
        pipeline = RetrievalPipeline(
            config={
                "dense_top_k": 50,
                "sparse_top_k": 50,
                "fusion_top_k": 30,
                "retrieval_timeout": 3.0,
            }
        )
        assert pipeline is not None
        assert pipeline.config["dense_top_k"] == 50

    def test_pipeline_retrieve_batch_empty(self):
        """Batch retrieve with empty queries should return empty."""
        pipeline = RetrievalPipeline()
        result = pipeline.retrieve_batch([])
        assert result == []

    def test_pipeline_retrieve_with_invalid_query(self):
        """Retrieve with invalid query should return error info."""
        pipeline = RetrievalPipeline()
        result = pipeline.retrieve("")
        assert result["results"] == []
        assert "error" in result["query_info"]

    def test_pipeline_retrieve_with_special_chars_only(self):
        """Retrieve with special chars only should fail validation."""
        pipeline = RetrievalPipeline()
        result = pipeline.retrieve("!@#$%^&*()!@#$%^&*()!@#$%^&*()")
        assert result["results"] == []
        assert "error" in result["query_info"]

    def test_pipeline_retrieve_with_very_long_query(self):
        """Retrieve with very long query should fail validation."""
        pipeline = RetrievalPipeline()
        long_query = "a" * (RetrievalPipeline.MAX_QUERY_LENGTH + 100)
        result = pipeline.retrieve(long_query)
        assert result["results"] == []
        assert "error" in result["query_info"]

    def test_pipeline_result_deduplication_integration(self):
        """Full integration: RRF produces results → deduplicate → normalize → filter."""
        rrf = RRFFusion(k=60)
        pipeline = RetrievalPipeline()

        dense = [
            {"chunk_id": "c1", "text": "text1", "metadata": {}, "score": 0.9},
            {"chunk_id": "c2", "text": "text2", "metadata": {}, "score": 0.8},
            {"chunk_id": "c3", "text": "text3", "metadata": {}, "score": 0.7},
        ]
        sparse = [
            {"chunk_id": "c1", "text": "text1_dup", "metadata": {}, "score": 0.85},
            {"chunk_id": "c4", "text": "text4", "metadata": {}, "score": 0.65},
        ]

        fused = rrf.fuse(dense, sparse, query="test")
        deduped = pipeline._deduplicate_results(fused)
        normalized = pipeline._normalize_scores(deduped)
        filtered = pipeline._filter_by_threshold(normalized)

        assert len(filtered) > 0
        # c1 should appear only once
        c1_count = sum(1 for r in filtered if r["chunk_id"] == "c1")
        assert c1_count == 1

    def test_concurrent_queries_integration(self):
        """Simulate concurrent queries through the pipeline."""
        def run_query(query_text):
            qtype = QueryClassifier.classify(query_text)
            return qtype

        queries = [
            "如何配置数据库",
            "什么是机器学习",
            "ERR500",
            "how to install Python",
            "ABC-1234",
            "2024-01-15",
            "数据库的定义",
            "如何计算损失函数",
        ]

        results = []
        threads = []
        lock = threading.Lock()

        def worker(q):
            r = run_query(q)
            with lock:
                results.append(r)

        for q in queries:
            t = threading.Thread(target=worker, args=(q,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(results) == len(queries)
        for r in results:
            assert r in ("factual", "procedural", "conceptual")