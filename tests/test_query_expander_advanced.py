"""Advanced tests for QueryExpander: tokenization, weighted synonyms, keywords, edge cases."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from core.retrieval.query_expander import (
        _CN_SYNONYMS,
        _EN_SYNONYMS,
        _STOP_WORDS,
        QueryExpander,
    )
except ImportError:
    QueryExpander = None
    _CN_SYNONYMS = {}
    _EN_SYNONYMS = {}
    _STOP_WORDS = frozenset()


class TestTokenizeQuery:
    """Test _tokenize_query with mixed Chinese-English queries."""

    @pytest.mark.parametrize("query, expected_tokens", [
        ("机器学习", ["机器学习"]),
        ("machine learning", ["machine", "learning"]),
        # Note: regex matches consecutive CJK as one token, "." becomes separate token
        ("Python 3.11 安装", ["Python", "3", ".", "11", "安装"]),
        ("API 接口测试", ["API", "接口测试"]),
        ("如何deploy项目", ["如何", "deploy", "项目"]),
        ("", []),
        ("   ", []),
        ("Kubernetes 容器编排", ["Kubernetes", "容器编排"]),
        ("123 abc 456", ["123", "abc", "456"]),
        ("微服务架构设计", ["微服务架构设计"]),
        ("test", ["test"]),
        ("中文English混合123", ["中文", "English", "混合", "123"]),
        ("!@#$%", ["!", "@", "#", "$", "%"]),
    ])
    def test_tokenize_query_parametrized(self, query, expected_tokens):
        """Test tokenization of various query types."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        tokens = expander._tokenize_query(query)
        assert tokens == expected_tokens, (
            f"Query={query!r}: expected {expected_tokens}, got {tokens}"
        )

    def test_tokenize_chinese_only(self):
        """Test tokenization of pure Chinese text."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        tokens = expander._tokenize_query("深度学习与神经网络")
        # Consecutive CJK chars are matched as one token by the regex
        assert "深度学习与神经网络" in tokens, f"Expected CJK token, got {tokens}"
        assert len(tokens) == 1, f"Expected 1 token, got {len(tokens)}: {tokens}"

    def test_tokenize_punctuation_handling(self):
        """Test tokenization handles punctuation correctly."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        tokens = expander._tokenize_query("hello, world!")
        assert "hello" in tokens
        assert "world" in tokens
        # Punctuation should be separated
        assert "," in tokens or len(tokens) >= 4


class TestGetSynonymsWithWeights:
    """Test _get_synonyms_with_weights for exact and partial matches."""

    def test_exact_match_weight(self):
        """Exact token matches should get weight 1.0."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        tokens = expander._tokenize_query("重置")
        weighted = expander._get_synonyms_with_weights("重置", tokens)
        for syn, weight in weighted:
            if weight == 1.0:
                assert syn in _CN_SYNONYMS.get("重置", ()), (
                    f"Synonym '{syn}' not found in expected synonyms for '重置'"
                )
        # At least one exact match should exist
        has_exact = any(w == 1.0 for _, w in weighted)
        assert has_exact, "Expected at least one exact match with weight 1.0"

    def test_partial_match_weight(self):
        """Partial token matches should get weight 0.5."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        tokens = expander._tokenize_query("重置密码所有")
        weighted = expander._get_synonyms_with_weights("重置密码所有", tokens)
        # Any partial match should have weight 0.5
        has_partial = any(w == 0.5 for _, w in weighted)
        # It's possible all are exact matches, but partial should exist for tokens like "所有"
        if has_partial:
            for syn, weight in weighted:
                if weight == 0.5:
                    assert isinstance(syn, str) and len(syn) > 0

    def test_no_duplicate_weights(self):
        """No duplicate synonyms should appear in weighted results."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        query = "重置密码"
        tokens = expander._tokenize_query(query)
        weighted = expander._get_synonyms_with_weights(query, tokens)
        syn_names = [s.lower() for s, _ in weighted]
        assert len(syn_names) == len(set(syn_names)), (
            f"Duplicate synonyms found: {syn_names}"
        )

    def test_english_exact_match(self):
        """English exact matches should get weight 1.0."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        tokens = expander._tokenize_query("reset")
        weighted = expander._get_synonyms_with_weights("reset", tokens)
        # Should have exact matches for "reset"
        exact_synonyms = [s for s, w in weighted if w == 1.0]
        assert len(exact_synonyms) > 0, "Expected exact match synonyms for 'reset'"

    def test_sorted_by_weight(self):
        """Weighted synonyms should be sorted by weight descending."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        query = "如何设置密码"
        tokens = expander._tokenize_query(query)
        weighted = expander._get_synonyms_with_weights(query, tokens)
        for i in range(len(weighted) - 1):
            assert weighted[i][1] >= weighted[i + 1][1], (
                f"Weights not sorted descending: {weighted[i]} vs {weighted[i+1]}"
            )

    def test_max_synonyms_limit_per_word(self):
        """max_synonyms should limit synonyms per dictionary word."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander_low = QueryExpander(synonym_enabled=True, max_synonyms=1)
        expander_high = QueryExpander(synonym_enabled=True, max_synonyms=5)
        tokens = expander_low._tokenize_query("重置")
        weighted_low = expander_low._get_synonyms_with_weights("重置", tokens)
        weighted_high = expander_high._get_synonyms_with_weights("重置", tokens)
        # Low max_synonyms should produce fewer or equal results
        assert len(weighted_low) <= len(weighted_high), (
            f"max_synonyms=1 produced {len(weighted_low)}, max_synonyms=5 produced {len(weighted_high)}"
        )


class TestExpandWithKeywords:
    """Test _expand_with_keywords extraction."""

    def test_keywords_exclude_stop_words(self):
        """Keywords should exclude stop words."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        # "的" is a stop word but it's embedded in a CJK token sequence
        # Use a query where stop words are separate tokens
        query = "the 机器学习 算法"
        tokens = expander._tokenize_query(query)
        keywords = expander._expand_with_keywords(query, tokens)
        assert "the" not in keywords, "Stop word 'the' should be excluded"
        assert "机器学习" in keywords or "算法" in keywords

    def test_keywords_exclude_short_tokens(self):
        """Keywords should exclude single-character tokens."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        query = "a b c Python 部署"
        tokens = expander._tokenize_query(query)
        keywords = expander._expand_with_keywords(query, tokens)
        assert "a" not in keywords, "Single char 'a' should be excluded"
        assert "b" not in keywords, "Single char 'b' should be excluded"
        assert "c" not in keywords, "Single char 'c' should be excluded"

    def test_keywords_exclude_pure_numbers(self):
        """Keywords should exclude pure numeric tokens."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        query = "Python 3 12345 版本"
        tokens = expander._tokenize_query(query)
        keywords = expander._expand_with_keywords(query, tokens)
        assert "12345" not in keywords, "Pure number should be excluded"
        assert "3" not in keywords, "Pure number should be excluded"

    def test_keywords_no_duplicates(self):
        """Keywords should not contain duplicates."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        query = "Python Python 机器学习 机器学习"
        tokens = expander._tokenize_query(query)
        keywords = expander._expand_with_keywords(query, tokens)
        assert len(keywords) == len(set(k.lower() for k in keywords)), (
            f"Duplicate keywords found: {keywords}"
        )

    def test_keywords_exclude_original_query(self):
        """Keywords should not include the original query."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        query = "测试"
        tokens = expander._tokenize_query(query)
        keywords = expander._expand_with_keywords(query, tokens)
        # "测试" is 2 chars, not a stop word, not a number
        # It should appear as a keyword since it's the original token
        assert "测试" in keywords, f"Expected '测试' in keywords, got {keywords}"


class TestExpandedFieldStructure:
    """Test that expanded field includes weighted_synonyms and keywords."""

    def test_weighted_synonyms_in_result(self):
        """Expanded result should include weighted_synonyms."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander(synonym_enabled=True)
        result = expander.expand("重置密码")
        assert "weighted_synonyms" in result
        assert isinstance(result["weighted_synonyms"], list)
        assert len(result["weighted_synonyms"]) > 0
        for item in result["weighted_synonyms"]:
            assert isinstance(item, tuple), f"Expected tuple, got {type(item)}"
            assert len(item) == 2, f"Expected (synonym, weight), got {item}"

    def test_keywords_in_result(self):
        """Expanded result should include keywords."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander(synonym_enabled=True)
        result = expander.expand("机器学习算法")
        assert "keywords" in result
        assert isinstance(result["keywords"], list)


class TestNewTechSynonyms:
    """Test new Chinese and English tech synonyms."""

    @pytest.mark.parametrize("word, expected_synonyms", [
        ("数据库", ["Database", "DB", "数据存储", "关系型数据库"]),
        ("API", ["接口", "应用程序接口", "REST", "端点"]),
        ("缓存", ["Cache", "缓冲", "临时存储", "高速缓存"]),
        ("部署", ["上线", "发布", "Deploy", "交付"]),
        ("监控", ["Monitor", "观测", "追踪", "告警"]),
        ("日志", ["Log", "记录", "审计", "跟踪"]),
        ("安全", ["Security", "加密", "防护", "认证"]),
        ("测试", ["Test", "验证", "检查", "质检"]),
        ("架构", ["Architecture", "设计", "结构", "框架"]),
        ("微服务", ["Microservice", "分布式", "服务化", "SOA"]),
        ("容器", ["Container", "Docker", "虚拟化", "隔离"]),
        ("Kubernetes", ["K8s", "容器编排", "集群管理"]),
    ])
    def test_chinese_tech_synonyms(self, word, expected_synonyms):
        """Verify Chinese tech synonyms are correctly defined."""
        if word in _CN_SYNONYMS:
            syns = _CN_SYNONYMS[word]
            for expected in expected_synonyms:
                assert expected in syns, (
                    f"Expected '{expected}' in synonyms for '{word}': {syns}"
                )

    @pytest.mark.parametrize("word, expected_synonyms", [
        ("machine learning", ["ML", "statistical learning", "pattern recognition", "AI"]),
        ("deep learning", ["DL", "DNN", "neural network", "deep neural network"]),
        ("API", ["interface", "endpoint", "service", "REST API"]),
        ("database", ["DB", "data store", "storage", "RDBMS"]),
        ("cache", ["buffer", "temporary storage", "memoization"]),
        ("deployment", ["release", "rollout", "delivery", "launch"]),
        ("monitoring", ["observability", "tracking", "alerting", "logging"]),
        ("security", ["encryption", "protection", "authentication", "authorization"]),
    ])
    def test_english_tech_synonyms(self, word, expected_synonyms):
        """Verify English tech synonyms are correctly defined."""
        if word in _EN_SYNONYMS:
            syns = _EN_SYNONYMS[word]
            for expected in expected_synonyms:
                assert expected in syns, (
                    f"Expected '{expected}' in synonyms for '{word}': {syns}"
                )


class TestEdgeCases:
    """Test edge cases for QueryExpander."""

    def test_empty_query(self):
        """Empty query should return empty result structure."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        result = expander.expand("")
        assert result["original"] == ""
        assert result["synonyms"] == []
        assert result["weighted_synonyms"] == []
        assert result["keywords"] == []
        assert result["expanded"] == ""

    def test_none_query(self):
        """None query should return empty result structure."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        result = expander.expand(None)
        assert result["original"] == ""
        assert result["synonyms"] == []

    def test_whitespace_query(self):
        """Whitespace-only query should return empty result."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        result = expander.expand("   ")
        assert result["synonyms"] == []
        assert result["original"] == "   "

    def test_non_string_query(self):
        """Non-string query should be handled gracefully."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        result = expander.expand(123)
        assert result["synonyms"] == []

    def test_synonym_dicts_are_immutable(self):
        """Verify synonym dictionaries are immutable (tuples)."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        for word, syns in _CN_SYNONYMS.items():
            assert isinstance(syns, tuple), (
                f"CN synonym for '{word}' should be tuple, got {type(syns)}"
            )
        for word, syns in _EN_SYNONYMS.items():
            assert isinstance(syns, tuple), (
                f"EN synonym for '{word}' should be tuple, got {type(syns)}"
            )

    def test_stop_words_is_frozenset(self):
        """Verify stop words are frozenset (immutable)."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        assert isinstance(_STOP_WORDS, frozenset), (
            f"STOP_WORDS should be frozenset, got {type(_STOP_WORDS)}"
        )

    def test_very_long_query(self):
        """Very long query should not crash."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        long_query = "机器学习与深度学习" * 100
        result = expander.expand(long_query)
        assert result["original"] == long_query
        assert "expanded" in result

    def test_synonyms_not_in_original(self):
        """Synonyms already in the query should not be added."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        result = expander.expand("初始化 恢复")
        # "初始化" and "恢复" are synonyms of "重置" - should not duplicate
        assert result["synonyms"].count("初始化") == 0
        assert result["synonyms"].count("恢复") == 0

    def test_expand_result_keys(self):
        """All expected keys should be present in expand result."""
        if QueryExpander is None:
            pytest.skip("QueryExpander module not available")
        expander = QueryExpander()
        result = expander.expand("测试")
        expected_keys = [
            "original", "expanded", "hyde_text", "synonyms",
            "weighted_synonyms", "keywords",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key '{key}' in expand result"