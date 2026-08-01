"""Tests for reranking module."""

import pytest
from src.core.reranking.cross_encoder import CrossEncoderReranker


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""

    def test_initialization(self):
        reranker = CrossEncoderReranker(
            model_name="BAAI/bge-reranker-base",
            device="cpu",
            batch_size=8,
            max_candidates=50,
        )
        assert reranker.model_name == "BAAI/bge-reranker-base"
        assert reranker.device == "cpu"
        assert reranker.max_candidates == 50

    def test_rerank_empty_candidates(self):
        reranker = CrossEncoderReranker()
        result = reranker.rerank("test query", [])
        assert result == []

    def test_rerank_single_candidate(self):
        reranker = CrossEncoderReranker()
        candidates = [
            {"chunk_id": "1", "text": "This is a test document about machine learning."},
        ]
        result = reranker.rerank("machine learning", candidates)
        assert len(result) == 1
        assert "rerank_score" in result[0]

    def test_model_unload(self):
        reranker = CrossEncoderReranker()
        reranker._load_model()
        assert reranker._model is not None
        reranker._unload_model()
        assert reranker._model is None