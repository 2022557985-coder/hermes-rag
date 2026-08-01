"""Comprehensive tests for QueryExpander."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from core.retrieval.query_expander import QueryExpander


class TestQueryExpander:
    """Test query expansion functionality."""

    def test_expand_with_synonyms(self):
        expander = QueryExpander(synonym_enabled=True, hyde_enabled=False)
        result = expander.expand("重置密码")
        assert result["original"] == "重置密码"
        assert len(result["synonyms"]) > 0
        assert result["expanded"] != "重置密码"

    def test_expand_no_synonyms(self):
        expander = QueryExpander(synonym_enabled=True)
        result = expander.expand("xyzzy_not_a_real_word")
        assert result["synonyms"] == []
        assert result["expanded"] == "xyzzy_not_a_real_word"

    def test_expand_synonym_disabled(self):
        expander = QueryExpander(synonym_enabled=False)
        result = expander.expand("重置密码")
        assert result["synonyms"] == []
        assert result["expanded"] == "重置密码"

    def test_expand_hyde_disabled(self):
        expander = QueryExpander(hyde_enabled=False)
        result = expander.expand("test")
        assert result["hyde_text"] is None

    def test_max_synonyms_limit(self):
        # max_synonyms applies per-word, and "重置密码" has 2 words each with synonyms
        expander = QueryExpander(synonym_enabled=True, max_synonyms=1)
        result = expander.expand("重置密码")
        # "重置" -> at most 1, "密码" -> at most 1, total <= 2
        assert len(result["synonyms"]) <= 2

    def test_chinese_synonyms(self):
        expander = QueryExpander(synonym_enabled=True)
        result = expander.expand("机器学习")
        syns = result["synonyms"]
        assert any("ML" in s or "Machine Learning" in s for s in syns)

    def test_english_synonyms(self):
        expander = QueryExpander(synonym_enabled=True)
        result = expander.expand("reset password")
        syns = result["synonyms"]
        assert len(syns) > 0

    def test_no_duplicate_synonyms(self):
        """Synonyms already in query should not be added."""
        expander = QueryExpander(synonym_enabled=True)
        result = expander.expand("重置 初始化")
        # "初始化" is a synonym of "重置", should not duplicate
        assert result["synonyms"].count("初始化") == 0

    def test_expand_empty_query(self):
        expander = QueryExpander(synonym_enabled=True)
        result = expander.expand("")
        assert result["original"] == ""
        assert result["synonyms"] == []

    def test_expand_result_structure(self):
        expander = QueryExpander(synonym_enabled=True)
        result = expander.expand("test")
        assert "original" in result
        assert "expanded" in result
        assert "hyde_text" in result
        assert "synonyms" in result

    def test_frozen_synonym_dict(self):
        """Verify synonym dictionaries are immutable tuples."""
        from core.retrieval.query_expander import _CN_SYNONYMS, _EN_SYNONYMS
        assert isinstance(_CN_SYNONYMS["重置"], tuple)
        assert isinstance(_EN_SYNONYMS["reset"], tuple)

    def test_ml_technical_synonyms(self):
        expander = QueryExpander(synonym_enabled=True)
        result = expander.expand("神经网络")
        syns = result["synonyms"]
        assert any("Neural Network" in s or "ANN" in s for s in syns)

    def test_iters_运维_synonyms(self):
        expander = QueryExpander(synonym_enabled=True)
        result = expander.expand("安装")
        syns = result["synonyms"]
        assert any("部署" in s or "配置" in s for s in syns)