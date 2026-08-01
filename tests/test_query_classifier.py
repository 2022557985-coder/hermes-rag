"""Comprehensive tests for QueryClassifier."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from core.retrieval.retrieval_pipeline import QueryClassifier


class TestQueryClassifier:
    """Test query classification accuracy and edge cases."""

    # ---- Factual Queries ----
    @pytest.mark.parametrize("query", [
        "ABC-1234的价格是多少",
        "v3.14.2版本有什么新功能",
        "#1234号工单状态",
        "ID 12345的文档",
        "XY_200型号",
        "HTTP500错误",
        "产品代码 PX-9001",
        "版本号2.0.1",
    ])
    def test_classify_factual_by_code(self, query):
        assert QueryClassifier.classify(query) == "factual"

    # Factual + procedural combo -> procedural wins (by design)
    @pytest.mark.parametrize("query", [
        "ERR404怎么解决",
        "ERR_CONNECTION_REFUSED怎么处理",
    ])
    def test_classify_factual_procedural_combo(self, query):
        assert QueryClassifier.classify(query) == "procedural"

    # ---- Conceptual Queries ----
    @pytest.mark.parametrize("query", [
        "什么是机器学习",
        "深度学习的定义",
        "神经网络的概念",
        "RAG的原理是什么",
        "Python的特点",
        "分类和回归的区别",
        "监督学习的优点",
        "向量检索的优势",
        "介绍一下陈祖敬",
        "总结一下公司政策",
        "Python的发展历史",
        "什么是交叉验证",
        "李碧宣是谁",
        "谁是谁",
        "如何计算余弦相似度",
        "如何评估模型性能",
        "如何选择超参数",
        "如何判断过拟合",
        "如何衡量检索质量",
        "如何检测异常",
    ])
    def test_classify_conceptual(self, query):
        assert QueryClassifier.classify(query) == "conceptual"

    # ---- Procedural Queries ----
    @pytest.mark.parametrize("query", [
        "如何安装Python",
        "怎么配置环境变量",
        "怎样重置密码",
        "安装教程",
        "使用指南",
        "how to install docker",
        "steps to deploy",
        "tutorial for beginners",
        "guide to setup",
        "如何部署服务",
        "怎么做数据清洗",
        "如何重置系统",
    ])
    def test_classify_procedural(self, query):
        assert QueryClassifier.classify(query) == "procedural"

    # ---- Edge Cases ----
    @pytest.mark.parametrize("query,expected", [
        ("", "conceptual"),          # Empty query
        ("   ", "conceptual"),       # Whitespace
        ("?", "conceptual"),         # Only punctuation
        ("123", "conceptual"),       # Not a product code pattern
        ("a", "conceptual"),         # Single letter
        ("你好", "conceptual"),      # Simple greeting
        ("test", "conceptual"),      # Unknown single word
    ])
    def test_classify_edge_cases(self, query, expected):
        assert QueryClassifier.classify(query) == expected

    # ---- Mixed Priorities ----
    @pytest.mark.parametrize("query,expected", [
        ("ERR500如何修复", "procedural"),   # Factual + Procedural -> Procedural
        ("ABC-123如何安装", "procedural"),  # Product code + how-to
        ("如何理解ABC-123的工作原理", "procedural"),  # Procedural wins
    ])
    def test_classify_mixed_priority(self, query, expected):
        assert QueryClassifier.classify(query) == expected

    # ---- Conceptual How Patterns ----
    @pytest.mark.parametrize("query", [
        "如何计算TF-IDF值",
        "如何工作",
        "如何运作",
        "如何运行",
        "如何区别过拟合和欠拟合",
        "如何评估模型效果",
        "如何检测数据泄露",
    ])
    def test_conceptual_how_patterns(self, query):
        assert QueryClassifier.classify(query) == "conceptual"

    # ---- Immutability ----
    def test_classify_is_deterministic(self):
        """Multiple calls should return same result."""
        query = "什么是深度学习"
        results = [QueryClassifier.classify(query) for _ in range(10)]
        assert all(r == results[0] for r in results)
        assert results[0] == "conceptual"