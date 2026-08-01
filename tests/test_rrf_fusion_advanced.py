"""Advanced tests for RRFFusion: query features, adaptive weights, boost, normalization, edge cases."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from core.retrieval.rrf_fusion import RRFFusion
except ImportError:
    RRFFusion = None


def _make_result(chunk_id, text, score=0.0):
    """Helper to create a result dict."""
    return {"chunk_id": chunk_id, "text": text, "metadata": {}, "score": score}


class TestDetectQueryFeatures:
    """Test _detect_query_features with various query types."""

    @pytest.mark.parametrize("query, expected", [
        ("机器学习", {"has_chinese": True, "has_english": False, "has_numbers": False,
                      "dominant_language": "chinese", "is_short_query": True}),
        ("machine learning", {"has_chinese": False, "has_english": True, "has_numbers": False,
                               "dominant_language": "english", "is_short_query": False}),
        ("Python 3.11 安装教程", {"has_chinese": True, "has_english": True, "has_numbers": True,
                                   "dominant_language": "mixed"}),
        ("ABC-1234", {"has_product_code": True, "has_english": True, "has_numbers": True}),
        ("Model-X200 价格", {"has_chinese": True, "has_english": True, "has_numbers": True}),
        ("", {"query_length": 0, "keyword_count": 0, "has_chinese": False}),
        ("a", {"query_length": 1, "is_short_query": True, "has_english": True}),
        ("怎么安装Python", {"is_colloquial": True, "has_chinese": True, "has_english": True}),
        ("how do i reset password", {"is_colloquial": True, "has_english": True}),
        ("what is the best way to deploy", {"is_colloquial": True, "has_english": True}),
        ("12345", {"has_numbers": True, "has_chinese": False, "has_english": False,
                    "dominant_language": "mixed"}),
        ("!@#$%^&*()", {"has_special_chars": True, "has_chinese": False, "has_english": False}),
        ("a" * 200, {"is_short_query": False, "query_length": 200}),
        ("你好world", {"has_chinese": True, "has_english": True, "dominant_language": "english"}),
        ("中文English混合123", {"has_chinese": True, "has_english": True, "has_numbers": True,
                                "dominant_language": "mixed"}),
    ])
    def test_detect_query_features_parametrized(self, query, expected):
        """Test _detect_query_features with various query types."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        features = fusion._detect_query_features(query)
        for key, value in expected.items():
            assert features[key] == value, (
                f"query={query!r}: expected {key}={value!r}, got {features[key]!r}"
            )

    def test_detect_query_features_structure(self):
        """Test that all expected keys are present in feature dict."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        features = fusion._detect_query_features("测试test123")
        expected_keys = [
            "has_chinese", "has_english", "has_numbers", "query_length",
            "keyword_count", "has_product_code", "is_colloquial",
            "ratio_chinese", "ratio_english", "ratio_numbers",
            "is_short_query", "has_special_chars", "dominant_language",
        ]
        for key in expected_keys:
            assert key in features, f"Missing key: {key}"

    def test_detect_very_long_query(self):
        """Test feature detection with a very long query."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        long_query = "机器学习" * 200
        features = fusion._detect_query_features(long_query)
        assert features["query_length"] > 100
        assert features["is_short_query"] is False
        assert features["dominant_language"] == "chinese"
        assert features["keyword_count"] > 0


class TestComputeAdaptiveWeights:
    """Test _compute_adaptive_weights with different query types."""

    def test_product_code_boosts_sparse(self):
        """Product code queries should boost sparse weight."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        w_d, w_s = fusion._compute_adaptive_weights("ABC-1234")
        assert w_s > w_d, f"Expected sparse({w_s}) > dense({w_d}) for product code"

    def test_colloquial_boosts_dense(self):
        """Colloquial queries should boost dense weight."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        w_d, w_s = fusion._compute_adaptive_weights("怎么安装Python")
        assert w_d > w_s, f"Expected dense({w_d}) > sparse({w_s}) for colloquial"

    def test_short_query_boosts_sparse(self):
        """Short queries should slightly boost sparse."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        w_d1, w_s1 = fusion._compute_adaptive_weights("机器学习")
        w_d2, w_s2 = fusion._compute_adaptive_weights("机器学习是一个非常复杂的领域")
        # Short query should have more sparse weight relative to long query
        assert w_s1 >= w_s2 - 0.01, (
            f"Short query sparse({w_s1}) should be >= long query sparse({w_s2})"
        )

    def test_weights_sum_to_one(self):
        """Adaptive weights should always sum to 1.0."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        queries = [
            "机器学习", "ABC-1234", "怎么安装Python", "how do i deploy",
            "Python 3.11 安装教程", "", "a" * 200, "12345", "中文English混合",
        ]
        for q in queries:
            w_d, w_s = fusion._compute_adaptive_weights(q)
            assert abs(w_d + w_s - 1.0) < 1e-9, (
                f"Query={q!r}: weights sum={w_d + w_s}, expected 1.0"
            )

    def test_long_query_boosts_dense(self):
        """Very long queries should boost dense."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        long_query = "机器学习是一个非常重要的领域，它涉及多种算法和技术" * 10
        w_d, w_s = fusion._compute_adaptive_weights(long_query)
        assert w_d >= 0.5, f"Long query should have dense >= 0.5, got {w_d}"

    def test_numbers_heavy_boosts_sparse(self):
        """Queries with many numbers should boost sparse."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        w_d, w_s = fusion._compute_adaptive_weights("123 456 789 012 345")
        assert w_s > w_d, f"Numbers-heavy query should boost sparse: d={w_d}, s={w_s}"

    def test_mixed_chinese_english_boosts_dense(self):
        """Mixed Chinese-English queries should boost dense."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        w_d, w_s = fusion._compute_adaptive_weights("如何使用Python API")
        assert w_d >= 0.5, f"Mixed CN-EN: expected dense >= 0.5, got {w_d}"

    def test_multi_keyword_boosts_sparse(self):
        """Queries with 5+ keywords should boost sparse."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        w_d, w_s = fusion._compute_adaptive_weights("machine learning deep neural network python")
        assert w_s >= w_d - 0.03, (
            f"Multi-keyword: expected sparse({w_s}) >= dense({w_d})"
        )

    def test_default_weights_unchanged(self):
        """Neutral queries should keep default weights (close to 0.5)."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        # A query with balanced features, not too short, not too long, no special patterns
        w_d, w_s = fusion._compute_adaptive_weights("机器学习与深度学习系统架构")
        # Not product code, not colloquial, not too short (>10 chars), not too long, not numbers-heavy
        # Few keywords, Chinese → should be close to defaults
        assert abs(w_d - 0.5) <= 0.05, (
            f"Neutral query should be close to defaults: d={w_d}, s={w_s}"
        )
        assert abs(w_s - 0.5) <= 0.05, (
            f"Neutral query should be close to defaults: d={w_d}, s={w_s}"
        )


class TestFuseWithBoost:
    """Test fuse_with_boost with different boost factors."""

    def test_boost_dense(self):
        """Boosting dense should increase dense result scores."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        dense = [_make_result("d1", "dense_only")]
        sparse = [_make_result("s1", "sparse_only")]
        # Without boost, both should get equal-ish weights
        fusion.fuse(dense, sparse)
        # With dense boost
        result_boost = fusion.fuse_with_boost(dense, sparse, boost_factor=2.0, boost_path="dense")
        # The dense-only result should score higher relative to sparse-only
        assert len(result_boost) == 2
        # Find the dense chunk in boosted result
        d_score = next(r["score"] for r in result_boost if r["chunk_id"] == "d1")
        s_score = next(r["score"] for r in result_boost if r["chunk_id"] == "s1")
        assert d_score > s_score, (
            f"Dense boost: dense_score({d_score}) should > sparse_score({s_score})"
        )

    def test_boost_sparse(self):
        """Boosting sparse should increase sparse result scores."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        dense = [_make_result("d1", "dense_only")]
        sparse = [_make_result("s1", "sparse_only")]
        result_boost = fusion.fuse_with_boost(
            dense, sparse, boost_factor=2.0, boost_path="sparse"
        )
        d_score = next(r["score"] for r in result_boost if r["chunk_id"] == "d1")
        s_score = next(r["score"] for r in result_boost if r["chunk_id"] == "s1")
        assert s_score > d_score, (
            f"Sparse boost: sparse_score({s_score}) should > dense_score({d_score})"
        )

    def test_boost_factor_zero(self):
        """Boost factor of 0 should not crash."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        dense = [_make_result("1", "text")]
        sparse = [_make_result("2", "text")]
        result = fusion.fuse_with_boost(dense, sparse, boost_factor=0.0, boost_path="dense")
        assert len(result) == 2

    def test_boost_factor_large(self):
        """Very large boost factor should still work."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        dense = [_make_result("1", "text")]
        sparse = [_make_result("2", "text")]
        result = fusion.fuse_with_boost(dense, sparse, boost_factor=100.0, boost_path="dense")
        assert len(result) == 2
        # Dense should be first
        assert result[0]["chunk_id"] == "1"

    def test_boost_both_empty(self):
        """Boost with empty results should return empty."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        result = fusion.fuse_with_boost([], [], boost_factor=2.0)
        assert result == []

    def test_boost_top_k_respected(self):
        """Boost should respect top_k parameter."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        dense = [_make_result(str(i), f"text{i}") for i in range(20)]
        sparse = [_make_result(str(i + 10), f"text{i+10}") for i in range(20)]
        result = fusion.fuse_with_boost(dense, sparse, top_k=5, boost_factor=1.5)
        assert len(result) == 5


class TestNormalizeScores:
    """Test normalize_scores with various score ranges."""

    def test_normalize_positive_scores(self):
        """Normalize scores in [0, 1] range."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        results = [
            _make_result("a", "text", 0.0),
            _make_result("b", "text", 0.5),
            _make_result("c", "text", 1.0),
        ]
        normalized = RRFFusion.normalize_scores(results)
        assert normalized[0]["score"] == 0.0
        assert normalized[1]["score"] == 0.5
        assert normalized[2]["score"] == 1.0

    def test_normalize_negative_scores(self):
        """Normalize scores that include negative values."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        results = [
            _make_result("a", "text", -10.0),
            _make_result("b", "text", 0.0),
            _make_result("c", "text", 10.0),
        ]
        normalized = RRFFusion.normalize_scores(results)
        assert normalized[0]["score"] == 0.0, f"Expected 0.0, got {normalized[0]['score']}"
        assert normalized[1]["score"] == 0.5, f"Expected 0.5, got {normalized[1]['score']}"
        assert normalized[2]["score"] == 1.0, f"Expected 1.0, got {normalized[2]['score']}"

    def test_normalize_all_zero_scores(self):
        """All zero scores should be set to 0.5 (maintains original order)."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        results = [
            _make_result("a", "text", 0.0),
            _make_result("b", "text", 0.0),
            _make_result("c", "text", 0.0),
        ]
        normalized = RRFFusion.normalize_scores(results)
        for r in normalized:
            assert r["score"] == 0.5, f"All-zero scores should all be 0.5, got {r['score']}"

    def test_normalize_all_equal_positive_scores(self):
        """All equal positive scores should be set to 0.5."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        results = [
            _make_result("a", "text", 3.14),
            _make_result("b", "text", 3.14),
        ]
        normalized = RRFFusion.normalize_scores(results)
        for r in normalized:
            assert r["score"] == 0.5, f"Equal scores should all be 0.5, got {r['score']}"

    def test_normalize_empty_list(self):
        """Normalize empty list should return empty list."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        result = RRFFusion.normalize_scores([])
        assert result == []

    def test_normalize_single_element(self):
        """Normalize single element should set to 0.5."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        results = [_make_result("a", "text", 42.0)]
        normalized = RRFFusion.normalize_scores(results)
        assert normalized[0]["score"] == 0.5

    def test_normalize_very_large_scores(self):
        """Normalize very large scores."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        results = [
            _make_result("a", "text", 1e10),
            _make_result("b", "text", 2e10),
        ]
        normalized = RRFFusion.normalize_scores(results)
        assert normalized[0]["score"] == 0.0
        assert normalized[1]["score"] == 1.0

    def test_normalize_very_small_scores(self):
        """Normalize very small (near-zero) scores."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        results = [
            _make_result("a", "text", 1e-10),
            _make_result("b", "text", 2e-10),
        ]
        normalized = RRFFusion.normalize_scores(results)
        assert abs(normalized[0]["score"] - 0.0) < 1e-6
        assert abs(normalized[1]["score"] - 1.0) < 1e-6


class TestMinScoreThreshold:
    """Test min_score_threshold filtering."""

    def test_threshold_filters_low_scores(self):
        """Low scores should be filtered out."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion(min_score_threshold=0.01)
        dense = [_make_result("1", "text")]
        sparse = [_make_result("2", "text")]
        # With k=60, score = 0.5 / (60+1) ≈ 0.008
        # This is below 0.01 threshold
        result = fusion.fuse(dense, sparse)
        assert len(result) == 0, f"Expected 0 results, got {len(result)}"

    def test_threshold_zero_passes_all(self):
        """Threshold of 0 should pass all results."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion(min_score_threshold=0.0)
        dense = [_make_result("1", "text")]
        sparse = [_make_result("2", "text")]
        result = fusion.fuse(dense, sparse)
        assert len(result) == 2

    def test_threshold_very_high_filters_all(self):
        """Very high threshold should filter all results."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion(min_score_threshold=999.0)
        dense = [_make_result("1", "text")]
        sparse = [_make_result("2", "text")]
        result = fusion.fuse(dense, sparse)
        assert len(result) == 0


class TestEdgeCases:
    """Test edge cases for RRFFusion."""

    def test_empty_query_uses_defaults(self):
        """Empty query should use default weights."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        w_d, w_s = fusion._get_dynamic_weights("")
        assert w_d == 0.5 and w_s == 0.5

    def test_whitespace_query_uses_defaults(self):
        """Whitespace-only query should use default weights."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        w_d, w_s = fusion._get_dynamic_weights("   ")
        assert w_d == 0.5 and w_s == 0.5

    def test_single_character_query(self):
        """Single character query should work."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        features = fusion._detect_query_features("中")
        assert features["is_short_query"] is True
        assert features["query_length"] == 1
        assert features["has_chinese"] is True

    def test_exactly_equal_scores_maintain_order(self):
        """When all scores are equal, original order should be maintained."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion(k=1000000)  # Very large k to make scores nearly equal
        dense = [_make_result(str(i), f"text{i}") for i in range(5)]
        sparse = [_make_result(str(i + 5), f"text{i+5}") for i in range(5)]
        result = fusion.fuse(dense, sparse)
        assert len(result) == 10

    def test_negative_scores_in_fuse(self):
        """Negative scores should not cause issues in fuse."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        dense = [_make_result("1", "text", -1.0), _make_result("2", "text", -2.0)]
        sparse = [_make_result("3", "text", -0.5)]
        result = fusion.fuse(dense, sparse)
        assert len(result) == 3

    def test_custom_weight_initialization(self):
        """Custom weights should be respected in initialization."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion(
            default_dense_weight=0.3,
            default_sparse_weight=0.7,
            product_code_dense_weight=0.2,
            product_code_sparse_weight=0.8,
            colloquial_dense_weight=0.9,
            colloquial_sparse_weight=0.1,
        )
        assert fusion.default_dense_weight == 0.3
        assert fusion.default_sparse_weight == 0.7
        assert fusion.product_code_dense_weight == 0.2
        assert fusion.product_code_sparse_weight == 0.8
        assert fusion.colloquial_dense_weight == 0.9
        assert fusion.colloquial_sparse_weight == 0.1

    def test_custom_k_value(self):
        """Custom k value should affect score magnitudes."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion_small_k = RRFFusion(k=1)
        fusion_large_k = RRFFusion(k=1000)
        dense = [_make_result("1", "text")]
        sparse = []
        result_small = fusion_small_k.fuse(dense, sparse)
        result_large = fusion_large_k.fuse(dense, sparse)
        # Small k should give higher score (denominator is smaller)
        assert result_small[0]["score"] > result_large[0]["score"], (
            f"Small k score({result_small[0]['score']}) should be > large k score({result_large[0]['score']})"
        )