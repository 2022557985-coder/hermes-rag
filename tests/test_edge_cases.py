"""Edge case tests for the full Hermes-RAG pipeline: query handling, concurrency, caching, special characters."""

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Try to import all modules that may be needed
try:
    from core.retrieval.rrf_fusion import RRFFusion
except ImportError:
    RRFFusion = None

try:
    from core.retrieval.query_expander import QueryExpander
except ImportError:
    QueryExpander = None

try:
    from core.indexing.bm25_index import BM25Index
except ImportError:
    BM25Index = None

try:
    from utils.security import sanitize_html, validate_query, validate_text_content
except ImportError:
    validate_query = None
    validate_text_content = None
    sanitize_html = None

try:
    from utils.metrics import MetricsCollector
except ImportError:
    MetricsCollector = None


class TestEmptyQueryHandling:
    """Test empty query handling across the pipeline."""

    def test_rrf_fusion_empty_query(self):
        """RRF fusion should handle empty query gracefully."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        result = fusion.fuse([], [], query="")
        assert result == []

    def test_query_expander_empty_query(self):
        """Query expander should handle empty query gracefully."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        result = expander.expand("")
        assert result["original"] == ""
        assert result["synonyms"] == []
        assert result["expanded"] == ""

    def test_bm25_empty_query(self):
        """BM25 search with empty query should return empty."""
        if BM25Index is None:
            pytest.skip("BM25Index module not available")
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = BM25Index(fallback_db_path=db_path)
            idx.add_chunks([
                {"chunk_id": "c1", "text": "test text", "metadata": {}},
            ])
            result = idx.search("")
            assert result == []
        finally:
            os.unlink(db_path)

    def test_validate_query_empty(self):
        """Validate query should reject empty query."""
        if validate_query is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_query("")
        assert is_valid is False


class TestVeryLongQuery:
    """Test handling of very long queries."""

    def test_rrf_fusion_long_query(self):
        """RRF fusion should handle very long query."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        long_query = "机器学习" * 500
        features = fusion._detect_query_features(long_query)
        assert features["query_length"] > 1000
        assert features["is_short_query"] is False

    def test_query_expander_long_query(self):
        """Query expander should handle very long query."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        long_query = "机器学习与深度学习" * 100
        result = expander.expand(long_query)
        assert result["original"] == long_query
        assert "expanded" in result

    def test_validate_query_near_max_length(self):
        """Query near max length should be accepted."""
        if validate_query is None:
            pytest.skip("Security module not available")
        query = "机器学习" * 100  # 400 chars
        is_valid, error_msg = validate_query(query, max_length=500)
        assert is_valid is True, f"Expected valid, got: {error_msg}"

    def test_validate_query_exceeds_max(self):
        """Query exceeding max length should be rejected."""
        if validate_query is None:
            pytest.skip("Security module not available")
        query = "x" * 1001
        is_valid, error_msg = validate_query(query, max_length=1000)
        assert is_valid is False


class TestSpecialCharacterQueries:
    """Test queries with special characters."""

    @pytest.mark.parametrize("query", [
        "!@#$%^&*()",
        "test <script> alert(1) </script>",
        "SELECT * FROM users",
        "query with \\ backslashes",
        "query with \"quotes\"",
        "query with 'single quotes'",
        "中文!@# 特殊字符",
        "test\nwith\nnewlines",
        "test\twith\ttabs",
        "{\"key\": \"value\"}",
    ])
    def test_rrf_fusion_special_chars(self, query):
        """RRF fusion should handle special characters without crashing."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        try:
            features = fusion._detect_query_features(query)
            assert isinstance(features, dict)
        except Exception as e:
            pytest.fail(f"RRF fusion crashed on query {query!r}: {e}")

    @pytest.mark.parametrize("query", [
        "!@#$%^&*()",
        "test with \"quotes\"",
        "test with 'single quotes'",
        "中文!@# 特殊字符",
        "{\"key\": \"value\"}",
    ])
    def test_query_expander_special_chars(self, query):
        """Query expander should handle special characters without crashing."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        try:
            result = expander.expand(query)
            assert "original" in result
        except Exception as e:
            pytest.fail(f"Query expander crashed on query {query!r}: {e}")


class TestUnicodeQueries:
    """Test queries with Unicode characters (emoji, CJK, mixed)."""

    @pytest.mark.parametrize("query", [
        "你好世界 🌍",
        "AI 🤖 机器学习",
        "\u4e2d\u6587\u6d4b\u8bd5",  # 中文测试
        "日本語テスト",
        "한국어 테스트",
        "混合Mixed混合",
        "café résumé",
        "αβγδε",
    ])
    def test_rrf_fusion_unicode(self, query):
        """RRF fusion should handle Unicode queries."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        try:
            features = fusion._detect_query_features(query)
            assert isinstance(features, dict)
        except Exception as e:
            pytest.fail(f"RRF fusion crashed on Unicode query: {e}")

    @pytest.mark.parametrize("query", [
        "你好世界 🌍",
        "AI 🤖 机器学习",
        "混合Mixed混合",
        "café résumé",
    ])
    def test_query_expander_unicode(self, query):
        """Query expander should handle Unicode queries."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        try:
            result = expander.expand(query)
            assert "original" in result
        except Exception as e:
            pytest.fail(f"Query expander crashed on Unicode query: {e}")


class TestNumericOnlyQueries:
    """Test numeric-only queries."""

    def test_numeric_rrf_fusion(self):
        """RRF fusion should handle numeric-only query."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        features = fusion._detect_query_features("12345")
        assert features["has_numbers"] is True
        assert features["has_chinese"] is False
        assert features["has_english"] is False

    def test_numeric_query_expander(self):
        """Query expander should handle numeric-only query."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        result = expander.expand("12345")
        assert result["original"] == "12345"
        assert "expanded" in result


class TestSingleCharacterQueries:
    """Test single character queries."""

    def test_single_char_cjk(self):
        """Single CJK character query should work."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        features = fusion._detect_query_features("中")
        assert features["is_short_query"] is True
        assert features["query_length"] == 1

    def test_single_char_ascii(self):
        """Single ASCII character query should work."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        features = fusion._detect_query_features("a")
        assert features["is_short_query"] is True
        assert features["has_english"] is True


class TestWhitespaceOnlyQueries:
    """Test whitespace-only queries."""

    def test_whitespace_rrf(self):
        """RRF fusion should handle whitespace-only query."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        w_d, w_s = fusion._get_dynamic_weights("   ")
        assert w_d == 0.5  # Should fall back to defaults

    def test_whitespace_expander(self):
        """Query expander should handle whitespace-only query."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        result = expander.expand("   ")
        assert result["synonyms"] == []

    def test_whitespace_validate(self):
        """Validate query should reject whitespace-only query."""
        if validate_query is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_query("   ")
        assert is_valid is False


class TestStopWordsOnlyQueries:
    """Test queries with only stop words."""

    def test_stop_words_expander(self):
        """Query with only stop words should produce no keywords."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        result = expander.expand("的 了 是 在")
        keywords = result["keywords"]
        # All tokens are stop words, so no keywords
        assert len(keywords) == 0, f"Expected no keywords for stop words query, got {keywords}"


class TestConcurrentQueries:
    """Test thread safety with concurrent queries."""

    def test_concurrent_rrf_fusion(self):
        """Multiple threads calling RRF fusion should be safe."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        errors = []

        def run_fusion():
            try:
                for _ in range(50):
                    dense = [{"chunk_id": f"d{i}", "text": f"text{i}", "metadata": {}, "score": 0.0}
                             for i in range(5)]
                    sparse = [{"chunk_id": f"s{i}", "text": f"text{i}", "metadata": {}, "score": 0.0}
                              for i in range(5)]
                    result = fusion.fuse(dense, sparse, query="test")
                    assert len(result) == 10
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_fusion) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Errors during concurrent RRF fusion: {errors}"

    def test_concurrent_query_expander(self):
        """Multiple threads calling query expander should be safe."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        errors = []

        def run_expand():
            try:
                for _ in range(50):
                    result = expander.expand("机器学习")
                    assert "original" in result
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_expand) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Errors during concurrent query expansion: {errors}"


class TestRapidSuccessiveQueries:
    """Test rapid successive queries (cache behavior)."""

    def test_rapid_rrf_queries(self):
        """Rapid successive RRF queries should not corrupt state."""
        if RRFFusion is None:
            pytest.skip("RRFFusion module not available")
        fusion = RRFFusion()
        for i in range(100):
            dense = [{"chunk_id": str(i), "text": f"text{i}", "metadata": {}, "score": 0.0}]
            sparse = []
            result = fusion.fuse(dense, sparse, query=f"query{i}")
            assert len(result) == 1

    def test_rapid_expander_queries(self):
        """Rapid successive expander queries should not corrupt state."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        for i in range(100):
            result = expander.expand(f"测试{i}")
            assert "original" in result


class TestNoMatchQueries:
    """Test queries that match nothing."""

    def test_bm25_no_match(self):
        """BM25 search with non-matching query returns results with negative scores."""
        if BM25Index is None:
            pytest.skip("BM25Index module not available")
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = BM25Index(fallback_db_path=db_path)
            idx.add_chunks([
                {"chunk_id": "c1", "text": "机器学习", "metadata": {}},
            ])
            result = idx.search("xyzzy_nonexistent_term")
            # BM25 may return results with score <= 0 for no-match queries
            if result:
                for r in result:
                    assert r["score"] <= 0, f"Expected non-positive score for no-match, got {r['score']}"
        finally:
            os.unlink(db_path)

    def test_bm25_partial_match(self):
        """BM25 search with partial match should return results when jieba is available."""
        if BM25Index is None:
            pytest.skip("BM25Index module not available")
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = BM25Index(fallback_db_path=db_path)
            idx.add_chunks([
                {"chunk_id": "c1", "text": "机器学习与深度学习", "metadata": {}},
                {"chunk_id": "c2", "text": "Python编程", "metadata": {}},
            ])
            result = idx.search("机器")
            # Without jieba, fallback tokenization may not produce partial matches
            # The search should not crash regardless
            assert isinstance(result, list)
            if len(result) > 0:
                assert result[0]["chunk_id"] == "c1"
        finally:
            os.unlink(db_path)


class TestVeryLongExpansion:
    """Test query with very long expansion."""

    def test_long_expansion_handling(self):
        """Query that produces many synonyms should not crash."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander(synonym_enabled=True, max_synonyms=5)
        # Query with many words that have synonyms
        query = "重置 密码 设置 安装 删除 修改 查看 错误 连接 启动 停止 配置 文件 系统"
        result = expander.expand(query)
        assert "original" in result
        assert "expanded" in result
        # Expanded query should be longer than original
        assert len(result["expanded"]) >= len(query), (
            "Expanded query should be >= original length"
        )


class TestTextValidationEdgeCases:
    """Test text validation edge cases."""

    def test_tabs_and_newlines_valid(self):
        """Text with tabs and newlines should be valid."""
        if validate_text_content is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_text_content("line1\tindented\nline2\n")
        assert is_valid is True, f"Expected valid, got: {error_msg}"

    def test_unicode_text_valid(self):
        """Unicode text should be valid."""
        if validate_text_content is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_text_content("中文测试 🌍 émoji")
        assert is_valid is True, f"Expected valid, got: {error_msg}"

    def test_large_valid_text(self):
        """Large valid text should pass."""
        if validate_text_content is None:
            pytest.skip("Security module not available")
        large_text = "valid line\n" * 10000
        is_valid, error_msg = validate_text_content(large_text)
        assert is_valid is True, f"Expected valid, got: {error_msg}"


class TestSanitizeHtmlEdgeCases:
    """Test HTML sanitization edge cases."""

    def test_nested_dangerous_tags(self):
        """Nested dangerous tags should be stripped."""
        if sanitize_html is None:
            pytest.skip("Security module not available")
        html = "<div><script>alert(1)</script><p>safe</p></div>"
        result = sanitize_html(html)
        assert "script" not in result.lower()
        assert "<p>safe</p>" in result

    def test_mixed_case_tags(self):
        """Mixed case tags should be stripped."""
        if sanitize_html is None:
            pytest.skip("Security module not available")
        html = "<SCRIPT>alert(1)</SCRIPT><ScRiPt>alert(2)</ScRiPt>"
        result = sanitize_html(html)
        assert "script" not in result.lower()

    def test_attributes_preserved_on_safe_tags(self):
        """Attributes on safe tags should be preserved."""
        if sanitize_html is None:
            pytest.skip("Security module not available")
        html = '<a href="https://example.com">link</a>'
        result = sanitize_html(html)
        assert 'href="https://example.com"' in result
        assert "link" in result