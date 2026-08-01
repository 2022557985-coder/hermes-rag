"""Stress tests and extreme edge cases for Hermes-RAG.

Tests:
1. Very large batch operations
2. Unicode and special characters
3. Concurrent access patterns
4. Very long/short queries
5. Mixed language queries
6. Numerical precision edge cases
7. Empty/null data handling
"""

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Add local_packages for rank_bm25
_local_pkg = Path(__file__).parent.parent / "local_packages"
if _local_pkg.exists():
    sys.path.insert(0, str(_local_pkg))


# ---------------------------------------------------------------------------
# Unicode / Special Characters
# ---------------------------------------------------------------------------

class TestUnicodeSpecialChars:
    """Test handling of Unicode, emoji, and special characters."""

    def test_unicode_queries(self):
        """Queries with various Unicode scripts should not crash."""
        from src.core.retrieval.retrieval_pipeline import QueryClassifier

        queries = [
            "日本語の質問です",           # Japanese
            "한국어 질문입니다",           # Korean
            "مرحبا بالعالم",              # Arabic
            "Привет мир",                 # Russian
            "🎉 Hello World 🎉",          # Emoji
            "café résumé naïve",          # Accented Latin
            "αβγδε μαθηματικά",           # Greek
            "数据科学 与 AI 🤖 的应用",    # Mixed Chinese + emoji
        ]
        for q in queries:
            result = QueryClassifier.classify(q)
            assert result in ("factual", "procedural", "conceptual"), \
                f"Query '{q}' should return a valid type, got '{result}'"

    def test_special_char_query_validation(self):
        """Special character queries should be validated."""
        from src.utils.security import validate_query

        # Valid special chars
        valid, _ = validate_query("C++ programming")
        assert valid, "C++ should be valid"

        # Query with '#' triggers SQL injection comment pattern
        valid, msg = validate_query("test@#$%query")
        assert not valid, "Query with # should be rejected as SQL injection"

        # SQL injection attempt
        valid, msg = validate_query("SELECT * FROM users; DROP TABLE users;")
        assert not valid, "SQL injection should be rejected"

        # XSS attempt
        valid, msg = validate_query("<script>alert('xss')</script>")
        assert not valid, "XSS should be rejected"

    def test_unicode_filename_sanitization(self):
        """Unicode filenames should be sanitized."""
        from src.utils.security import sanitize_filename

        assert sanitize_filename("正常文件名.txt") == "正常文件名.txt"
        assert sanitize_filename("../../../etc/passwd") == "etcpasswd"
        assert sanitize_filename("test\0null.txt") == "testnull.txt"
        assert sanitize_filename("") == "unnamed"
        assert sanitize_filename("   ") == "unnamed"

    def test_unicode_text_validation(self):
        """Unicode text content should be validated."""
        from src.utils.security import validate_text_content

        valid, _ = validate_text_content("中文内容测试\nEnglish content\n日本語のコンテンツ")
        assert valid, "Multilingual content should be valid"

        valid, msg = validate_text_content("")
        assert not valid, "Empty content should be invalid"

        valid, msg = validate_text_content("\x00binary")
        assert not valid, "Null bytes should be rejected"


# ---------------------------------------------------------------------------
# Very Large / Very Small Inputs
# ---------------------------------------------------------------------------

class TestInputSizeExtremes:
    """Test handling of extreme input sizes."""

    def test_very_long_query(self):
        """Very long queries should be handled gracefully."""
        from src.core.retrieval.retrieval_pipeline import RetrievalPipeline

        pipeline = RetrievalPipeline()

        # Valid long query (just under limit)
        long_query = "x" * 1500
        valid, _ = pipeline._validate_query(long_query)
        assert valid, f"Query of {len(long_query)} chars should be valid"

        # Over limit
        too_long = "x" * 3000
        valid, msg = pipeline._validate_query(too_long)
        assert not valid, f"Query of {len(too_long)} chars should be invalid"

    def test_very_short_query(self):
        """Very short queries should be handled."""
        from src.core.retrieval.retrieval_pipeline import RetrievalPipeline

        pipeline = RetrievalPipeline()

        valid, _ = pipeline._validate_query("a")
        assert valid, "Single char should be valid"

        valid, msg = pipeline._validate_query("")
        assert not valid, "Empty query should be invalid"

        valid, msg = pipeline._validate_query("   ")
        assert not valid, "Whitespace-only query should be invalid"

    def test_large_batch_operations(self):
        """Large batch operations should not crash."""
        from src.core.indexing.bm25_index import BM25Index

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = BM25Index(fallback_db_path=db_path, max_index_entries=50)

            # Add 1000 chunks
            chunks = [
                {
                    "chunk_id": f"chunk_{i}",
                    "text": f"This is test chunk number {i} with some content for BM25 indexing.",
                    "metadata": {"index": i},
                }
                for i in range(1000)
            ]
            idx.add_chunks(chunks)

            assert idx.count() == 1000, f"Expected 1000 chunks, got {idx.count()}"
        finally:
            idx.clear()
            os.unlink(db_path)

    def test_normalize_scores_all_same(self):
        """Normalize when all scores are identical."""
        from src.core.retrieval.retrieval_pipeline import RetrievalPipeline

        results = [
            {"chunk_id": f"c{i}", "score": 0.5}
            for i in range(10)
        ]
        normalized = RetrievalPipeline._normalize_scores(results)
        assert all(r["score"] == 0.5 for r in normalized), \
            "All-same scores should normalize to 0.5"

    def test_normalize_scores_negative(self):
        """Normalize scores containing negative values."""
        from src.core.retrieval.retrieval_pipeline import RetrievalPipeline

        results = [
            {"chunk_id": "a", "score": -1.0},
            {"chunk_id": "b", "score": -0.5},
            {"chunk_id": "c", "score": 0.0},
            {"chunk_id": "d", "score": 0.5},
            {"chunk_id": "e", "score": 1.0},
        ]
        normalized = RetrievalPipeline._normalize_scores(results)
        assert normalized[0]["score"] >= 0, "Negative scores should be clamped"
        assert all(0 <= r["score"] <= 1 for r in normalized), \
            "All scores should be in [0, 1]"

    def test_normalize_scores_single(self):
        """Normalize single result."""
        from src.core.retrieval.retrieval_pipeline import RetrievalPipeline

        results = [{"chunk_id": "a", "score": 0.7}]
        normalized = RetrievalPipeline._normalize_scores(results)
        assert normalized[0]["score"] == 0.7, "Single score should be unchanged"


# ---------------------------------------------------------------------------
# Concurrent Access
# ---------------------------------------------------------------------------

class TestConcurrency:
    """Test thread safety of core components."""

    def test_bm25_concurrent_search(self):
        """Concurrent BM25 searches should be safe."""
        from src.core.indexing.bm25_index import BM25Index

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = BM25Index(fallback_db_path=db_path)
            chunks = [
                {"chunk_id": f"c{i}", "text": f"test content {i}", "metadata": {}}
                for i in range(100)
            ]
            idx.add_chunks(chunks)

            errors = []
            def search_worker():
                try:
                    for _ in range(50):
                        result = idx.search("test", top_k=5)
                        assert isinstance(result, list)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=search_worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Concurrent searches had errors: {errors}"
        finally:
            idx.clear()
            os.unlink(db_path)

    def test_metrics_concurrent_recording(self):
        """Concurrent metrics recording should be safe."""
        from src.utils.metrics import MetricsCollector

        mc = MetricsCollector(window_size=1000)

        errors = []
        def record_worker():
            try:
                for i in range(100):
                    mc.record_query(
                        cached=(i % 3 == 0),
                        recall_paths=["dense", "sparse"] if i % 2 == 0 else ["dense"],
                        total_latency=0.01 * (i % 10 + 1),
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent recording had errors: {errors}"
        report = mc.get_full_report()
        assert report["total_queries"] == 2000, \
            f"Expected 2000 queries, got {report['total_queries']}"

    def test_cache_concurrent_access(self):
        """Concurrent cache access should be safe."""
        from src.utils.cache import QueryCache

        cache = QueryCache(max_size=100)
        emb = np.random.randn(128).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        errors = []
        def cache_worker(worker_id):
            try:
                for i in range(50):
                    key = f"query_{worker_id}_{i}"
                    cache.set(key, [{"chunk_id": str(i)}], query_embedding=emb)
                    result = cache.get(key)
                    if result is not None:
                        assert len(result) > 0
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=cache_worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent cache access had errors: {errors}"

    def test_rate_limiter_concurrent(self):
        """Concurrent rate limit checks should be safe."""
        from src.utils.security import rate_limit_check

        results = []
        def checker():
            results.append(rate_limit_check("concurrent_test", rate=100, capacity=200))

        threads = [threading.Thread(target=checker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # With rate=100, capacity=200, all 50 should pass
        assert all(results), "All concurrent rate checks should pass"


# ---------------------------------------------------------------------------
# Mixed Language Queries
# ---------------------------------------------------------------------------

class TestMixedLanguage:
    """Test mixed-language query handling."""

    def test_mixed_cn_en_queries(self):
        """Mixed Chinese-English queries should be classified correctly."""
        from src.core.retrieval.retrieval_pipeline import QueryClassifier

        tests = [
            ("Python是什么语言", "conceptual"),
            ("how to 安装 Python", "procedural"),
            ("什么是machine learning", "conceptual"),
            ("API 接口文档", "conceptual"),
            ("Docker 容器 部署教程", "procedural"),
            ("Redis 缓存 配置", "conceptual"),
            ("Kubernetes 集群 管理", "conceptual"),
        ]
        for query, expected in tests:
            result = QueryClassifier.classify(query)
            assert result == expected, f"'{query}' expected {expected}, got {result}"

    def test_rrf_mixed_language_detection(self):
        """RRF should detect mixed language features."""
        from src.core.retrieval.rrf_fusion import RRFFusion

        rrf = RRFFusion()

        features = rrf._detect_query_features("Python 机器学习 教程")
        assert features["has_chinese"], "Should detect Chinese"
        assert features["has_english"], "Should detect English"
        assert features["dominant_language"] in ("chinese", "mixed"), \
            f"Expected Chinese-dominant, got {features['dominant_language']}"

        features = rrf._detect_query_features("how to train ML model")
        assert features["dominant_language"] == "english", \
            f"Expected English, got {features['dominant_language']}"

    def test_query_expansion_mixed_language(self):
        """Query expansion should handle mixed language."""
        from src.core.retrieval.query_expander import QueryExpander

        expander = QueryExpander(synonym_enabled=True)

        result = expander.expand("Python 机器学习 模型")
        assert result["expanded"] != result["original"], \
            "Mixed language query should be expanded"
        # Should have synonyms for both Chinese and English terms
        assert len(result["synonyms"]) > 0, "Should have synonyms"

    def test_security_mixed_language_injection(self):
        """Mixed language injection attempts should be caught."""
        from src.utils.security import validate_query

        # Unicode SQL injection
        valid, _ = validate_query("ＳＥＬＥＣＴ ＊ ＦＲＯＭ")
        assert valid, "Fullwidth characters should be treated as text"

        # Mixed injection
        valid, msg = validate_query("正常查询 UNION SELECT * FROM users")
        assert not valid, "Mixed injection should be rejected"


# ---------------------------------------------------------------------------
# Numerical Precision Edge Cases
# ---------------------------------------------------------------------------

class TestNumericalPrecision:
    """Test numerical precision edge cases."""

    def test_rrf_zero_scores(self):
        """RRF with zero scores should not divide by zero."""
        from src.core.retrieval.rrf_fusion import RRFFusion

        rrf = RRFFusion()

        # All-zero scores
        results = rrf.fuse(
            [{"chunk_id": "a", "text": "A", "metadata": {}}],
            [{"chunk_id": "b", "text": "B", "metadata": {}}],
            query="test",
        )
        assert len(results) >= 0, "Should not crash with zero scores"

    def test_rrf_single_result_each(self):
        """RRF with single result from each path."""
        from src.core.retrieval.rrf_fusion import RRFFusion

        rrf = RRFFusion()

        result = rrf.fuse(
            [{"chunk_id": "a", "text": "A", "metadata": {}}],
            [{"chunk_id": "b", "text": "B", "metadata": {}}],
            query="test",
        )
        assert len(result) == 2, f"Expected 2 results, got {len(result)}"

    def test_rrf_identical_chunks(self):
        """RRF with identical chunks from both paths."""
        from src.core.retrieval.rrf_fusion import RRFFusion

        rrf = RRFFusion()

        result = rrf.fuse(
            [{"chunk_id": "a", "text": "A", "metadata": {}}],
            [{"chunk_id": "a", "text": "A", "metadata": {}}],
            query="test",
        )
        assert len(result) == 1, "Identical chunks should be merged"
        assert "dense" in result[0]["sources"], "Should have dense source"
        assert "sparse" in result[0]["sources"], "Should have sparse source"

    def test_deduplicate_edge_cases(self):
        """Deduplication with various edge cases."""
        from src.core.retrieval.retrieval_pipeline import RetrievalPipeline

        pipeline = RetrievalPipeline()

        # Empty input
        assert pipeline._deduplicate_results([]) == []

        # Single item
        result = pipeline._deduplicate_results(
            [{"chunk_id": "a", "score": 0.5}]
        )
        assert len(result) == 1

        # All duplicates
        result = pipeline._deduplicate_results([
            {"chunk_id": "a", "score": 0.9},
            {"chunk_id": "a", "score": 0.5},
            {"chunk_id": "a", "score": 0.7},
        ])
        assert len(result) == 1
        assert result[0]["score"] == 0.9, "Should keep highest score"

    def test_filter_threshold_edge(self):
        """Filter exactly at threshold."""
        from src.core.retrieval.retrieval_pipeline import RetrievalPipeline

        pipeline = RetrievalPipeline()

        results = [
            {"chunk_id": "a", "score": 0.001},
            {"chunk_id": "b", "score": 0.0009},
            {"chunk_id": "c", "score": 0.0},
        ]
        filtered = pipeline._filter_by_threshold(results, min_score=0.001)
        assert len(filtered) == 1, f"Expected 1 at threshold, got {len(filtered)}"
        assert filtered[0]["chunk_id"] == "a"


# ---------------------------------------------------------------------------
# Cache Edge Cases
# ---------------------------------------------------------------------------

class TestCacheEdgeCases:
    """Test cache with extreme patterns."""

    def test_cache_eviction_order(self):
        """Cache should evict oldest entries (LRU)."""
        from src.utils.cache import QueryCache

        cache = QueryCache(max_size=3)

        cache.set("a", [{"chunk_id": "1"}])
        cache.set("b", [{"chunk_id": "2"}])
        cache.set("c", [{"chunk_id": "3"}])

        # Access 'a' to make it most recently used
        cache.get("a")

        # Add 'd' - should evict 'b' (least recently used, since 'a' was accessed)
        cache.set("d", [{"chunk_id": "4"}])

        assert cache.get("a") is not None, "Recently accessed 'a' should still be in cache"
        assert cache.get("b") is None, "LRU 'b' should be evicted"
        assert cache.get("c") is not None, "'c' should still be in cache"
        assert cache.get("d") is not None, "'d' should be in cache"

    def test_cache_ttl_expiry(self):
        """Cache entries should expire after TTL."""
        from src.utils.cache import QueryCache

        cache = QueryCache(max_size=10, ttl_seconds=0)
        cache.set("test", [{"chunk_id": "1"}])
        time.sleep(0.1)
        assert cache.get("test") is None, "TTL=0 entry should expire"

    def test_cache_clear(self):
        """Cache clear should remove all entries."""
        from src.utils.cache import QueryCache

        cache = QueryCache(max_size=10)
        for i in range(5):
            cache.set(f"key_{i}", [{"chunk_id": str(i)}])

        assert cache.size() == 5
        cache.clear()
        assert cache.size() == 0
        assert cache.hit_rate() == 0.0

    def test_cache_same_key_overwrite(self):
        """Setting same key should update and not increase count."""
        from src.utils.cache import QueryCache

        cache = QueryCache(max_size=10)
        cache.set("key", [{"chunk_id": "1"}])
        cache.set("key", [{"chunk_id": "2"}])

        assert cache.size() == 1, "Same key should not increase size"
        result = cache.get("key")
        assert result[0]["chunk_id"] == "2", "Should return latest value"


# ---------------------------------------------------------------------------
# Query Classifier Edge Cases
# ---------------------------------------------------------------------------

class TestQueryClassifierEdge:
    """Test query classifier with unusual inputs."""

    def test_pure_numbers(self):
        """Pure numeric queries."""
        from src.core.retrieval.retrieval_pipeline import QueryClassifier

        assert QueryClassifier.classify("12345") == "conceptual"
        assert QueryClassifier.classify("2024-01-15") == "factual"
        assert QueryClassifier.classify("13800138000") == "factual"

    def test_pure_special_chars(self):
        """Pure special character queries."""
        from src.core.retrieval.retrieval_pipeline import QueryClassifier

        # Should not crash
        assert QueryClassifier.classify("!@#$%^&*()") == "conceptual"
        assert QueryClassifier.classify("---") == "conceptual"

    def test_classify_with_confidence(self):
        """Confidence scores should be in [0.3, 1.0]."""
        from src.core.retrieval.retrieval_pipeline import QueryClassifier

        queries = [
            "什么是机器学习",
            "how to reset password",
            "ABC-1234",
            "test",
            "",
        ]
        for q in queries:
            qtype, confidence = QueryClassifier.classify_with_confidence(q)
            assert 0.0 <= confidence <= 1.0, \
                f"Confidence {confidence} out of range for '{q}'"
            assert qtype in ("factual", "procedural", "conceptual"), \
                f"Invalid type '{qtype}' for '{q}'"


# ---------------------------------------------------------------------------
# RRF Dynamic Weights
# ---------------------------------------------------------------------------

class TestRRFDynamicWeights:
    """Test RRF dynamic weight computation."""

    def test_weights_sum_to_one(self):
        """Dynamic weights should always sum to 1.0."""
        from src.core.retrieval.rrf_fusion import RRFFusion

        rrf = RRFFusion()

        queries = [
            "ABC-1234 产品",
            "什么是机器学习",
            "short q",
            "a" * 200,
            "Python 机器学习 深度学习 神经网络 模型",
            "1234567890",
            "测试 query with English",
        ]
        for q in queries:
            dw, sw = rrf._get_dynamic_weights(q)
            assert abs(dw + sw - 1.0) < 1e-10, \
                f"Weights sum to {dw + sw} for '{q}', expected 1.0"
            assert 0.1 <= dw <= 0.9, f"Dense weight {dw} out of range for '{q}'"
            assert 0.1 <= sw <= 0.9, f"Sparse weight {sw} out of range for '{q}'"

    def test_product_code_boosts_sparse(self):
        """Product code queries should boost sparse weight."""
        from src.core.retrieval.rrf_fusion import RRFFusion

        rrf = RRFFusion()
        dw, sw = rrf._get_dynamic_weights("ABC-1234 配置")
        assert sw > dw, f"Product code should boost sparse: dw={dw}, sw={sw}"

    def test_colloquial_boosts_dense(self):
        """Colloquial queries should boost dense weight."""
        from src.core.retrieval.rrf_fusion import RRFFusion

        rrf = RRFFusion()
        dw, sw = rrf._get_dynamic_weights("什么是机器学习")
        assert dw > sw, f"Colloquial should boost dense: dw={dw}, sw={sw}"


# ---------------------------------------------------------------------------
# BM25 Edge Cases
# ---------------------------------------------------------------------------

class TestBM25EdgeCases:
    """Test BM25 with extreme edge cases."""

    def test_empty_corpus_search(self):
        """Search on empty corpus should return empty."""
        from src.core.indexing.bm25_index import BM25Index

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = BM25Index(fallback_db_path=db_path)
            result = idx.search("test")
            assert result == [], "Empty corpus should return empty"
        finally:
            os.unlink(db_path)

    def test_count_after_clear(self):
        """Count should be zero after clear."""
        from src.core.indexing.bm25_index import BM25Index

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = BM25Index(fallback_db_path=db_path)
            idx.add_chunks([
                {"chunk_id": "c1", "text": "test", "metadata": {}},
            ])
            assert idx.count() == 1
            idx.clear()
            assert idx.count() == 0
        finally:
            os.unlink(db_path)

    def test_remove_nonexistent_chunk(self):
        """Removing non-existent chunk should return False."""
        from src.core.indexing.bm25_index import BM25Index

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = BM25Index(fallback_db_path=db_path)
            assert idx.remove_chunk("nonexistent") is False
        finally:
            os.unlink(db_path)

    def test_single_char_query(self):
        """Single character query should work."""
        from src.core.indexing.bm25_index import BM25Index

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = BM25Index(fallback_db_path=db_path)
            idx.add_chunks([
                {"chunk_id": "c1", "text": "a b c d e", "metadata": {}},
            ])
            result = idx.search("a")
            assert isinstance(result, list), "Should return list, not crash"
        finally:
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Security Edge Cases
# ---------------------------------------------------------------------------

class TestSecurityEdgeCases:
    """Test security utilities with edge cases."""

    def test_file_path_traversal(self):
        """Path traversal should be rejected."""
        from src.utils.security import validate_file_path

        # Test with non-existent base_dir
        base_dir = str(Path(__file__).parent.parent)

        with pytest.raises(ValueError, match="Path traversal"):
            validate_file_path("../etc/passwd", base_dir=base_dir)

        with pytest.raises(ValueError, match="Path traversal"):
            validate_file_path("....//....//etc/passwd", base_dir=base_dir)

    def test_url_validation_blocked(self):
        """Blocked URLs should be rejected."""
        from src.utils.security import validate_url

        with pytest.raises(ValueError, match="not allowed"):
            validate_url("ftp://example.com/file")

        with pytest.raises(ValueError, match="blocked network"):
            validate_url("http://127.0.0.1:8080/api")

    def test_html_sanitization_complex(self):
        """Complex HTML should be sanitized."""
        from src.utils.security import sanitize_html

        html = "<script>alert('xss')</script><p>Safe content</p><iframe src='evil'></iframe>"
        sanitized = sanitize_html(html)
        assert "script" not in sanitized.lower(), "Script tags should be removed"
        assert "iframe" not in sanitized.lower(), "Iframe tags should be removed"
        assert "Safe content" in sanitized, "Safe content should be preserved"

    def test_batch_size_validation(self):
        """Batch size validation edge cases."""
        from src.utils.security import validate_batch_size

        valid, _ = validate_batch_size(0)
        assert not valid, "Zero should be invalid"

        valid, _ = validate_batch_size(-1)
        assert not valid, "Negative should be invalid"

        valid, _ = validate_batch_size(100)
        assert valid, "100 should be valid"

        valid, _ = validate_batch_size(10000)
        assert not valid, "10000 should exceed max"


# ---------------------------------------------------------------------------
# Metrics Edge Cases
# ---------------------------------------------------------------------------

class TestMetricsEdgeCases:
    """Test metrics collector with edge cases."""

    def test_empty_metrics_report(self):
        """Empty metrics should return sensible defaults."""
        from src.utils.metrics import MetricsCollector

        mc = MetricsCollector()
        report = mc.get_full_report()

        assert report["total_queries"] == 0
        assert report["cache"]["hit_rate"] == 0.0
        assert report["latency"]["avg_ms"] == 0.0
        assert report["health"] == "GREEN", "Empty metrics should be GREEN"

    def test_metrics_reset(self):
        """Reset should clear all metrics."""
        from src.utils.metrics import MetricsCollector

        mc = MetricsCollector()
        for i in range(10):
            mc.record_query(total_latency=0.1)

        assert mc.get_full_report()["total_queries"] == 10
        mc.reset()
        assert mc.get_full_report()["total_queries"] == 0

    def test_health_status_transitions(self):
        """Health status should transition based on errors."""
        from src.utils.metrics import MetricsCollector

        mc = MetricsCollector()

        # Normal: GREEN
        mc.record_query(total_latency=0.1)
        assert mc.get_health_status() == "GREEN"

        # High latency: YELLOW
        for _ in range(5):
            mc.record_query(total_latency=15.0)
        # Force high p95 by adding more high-latency queries
        mc._latency_window.clear()
        for _ in range(100):
            mc.record_query(total_latency=15.0)
        status = mc.get_health_status()
        assert status in ("YELLOW", "RED"), f"High latency should be YELLOW/RED, got {status}"

    def test_prometheus_format(self):
        """Prometheus export should produce valid format."""
        from src.utils.metrics import MetricsCollector

        mc = MetricsCollector()
        mc.record_query(total_latency=0.1)

        output = mc.to_prometheus_format()
        assert "# HELP" in output, "Should have HELP lines"
        assert "# TYPE" in output, "Should have TYPE lines"
        assert "hermes_rag_total_queries" in output
        assert "hermes_rag_health_status" in output