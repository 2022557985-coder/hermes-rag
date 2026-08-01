"""Tests for retrieval module."""

import pytest
from src.core.retrieval.query_expander import QueryExpander
from src.core.retrieval.rrf_fusion import RRFFusion
from src.core.retrieval.rule_retriever import RuleRetriever


class TestQueryExpander:
    """Tests for QueryExpander."""

    def test_synonym_expansion(self):
        expander = QueryExpander(synonym_enabled=True, hyde_enabled=False)
        result = expander.expand("如何重置密码？")
        assert result["original"] == "如何重置密码？"
        assert len(result["synonyms"]) > 0
        assert "重置" in result["original"]

    def test_no_synonym_for_unknown_words(self):
        expander = QueryExpander(synonym_enabled=True, hyde_enabled=False)
        result = expander.expand("xyzabc123")
        assert result["synonyms"] == []

    def test_hyde_disabled_by_default(self):
        expander = QueryExpander()
        result = expander.expand("test query")
        assert result["hyde_text"] is None

    def test_expand_empty_query(self):
        expander = QueryExpander()
        result = expander.expand("")
        assert result["original"] == ""


class TestRRFFusion:
    """Tests for RRFFusion."""

    def test_fuse_basic(self):
        fusion = RRFFusion(k=60)
        dense_results = [
            {"chunk_id": "a", "text": "text a", "score": 0.9},
            {"chunk_id": "b", "text": "text b", "score": 0.8},
        ]
        sparse_results = [
            {"chunk_id": "b", "text": "text b", "score": 5.0},
            {"chunk_id": "c", "text": "text c", "score": 4.0},
        ]
        fused = fusion.fuse(dense_results, sparse_results, top_k=5)
        assert len(fused) == 3  # a, b, c
        # b should have highest score (appears in both)
        assert fused[0]["chunk_id"] == "b"

    def test_fuse_empty_inputs(self):
        fusion = RRFFusion()
        fused = fusion.fuse([], [], top_k=5)
        assert fused == []

    def test_dynamic_weights_product_code(self):
        fusion = RRFFusion()
        dense_w, sparse_w = fusion._get_dynamic_weights("product ABC-123 is broken")
        assert sparse_w > dense_w  # BM25 should be boosted

    def test_dynamic_weights_colloquial(self):
        fusion = RRFFusion()
        dense_w, sparse_w = fusion._get_dynamic_weights("怎么重置密码？")
        assert dense_w > sparse_w  # Dense should be boosted

    def test_dynamic_weights_default(self):
        fusion = RRFFusion()
        dense_w, sparse_w = fusion._get_dynamic_weights("what is machine learning")
        assert dense_w == 0.5
        assert sparse_w == 0.5


class TestRuleRetriever:
    """Tests for RuleRetriever."""

    def test_detect_chapter_hint(self):
        retriever = RuleRetriever()
        hint = retriever.detect_chapter_hint("第三章的内容是什么？")
        assert hint == "第三章"

    def test_detect_english_chapter(self):
        retriever = RuleRetriever()
        hint = retriever.detect_chapter_hint("what is Chapter 5 about")
        assert hint == "Chapter 5"

    def test_no_chapter_hint(self):
        retriever = RuleRetriever()
        hint = retriever.detect_chapter_hint("hello world")
        assert hint is None

    def test_build_filter(self):
        retriever = RuleRetriever()
        f = retriever.build_filter("第三章的内容")
        assert f is not None
        assert "heading_path" in f

    def test_build_filter_no_hint(self):
        retriever = RuleRetriever()
        f = retriever.build_filter("hello")
        assert f is None