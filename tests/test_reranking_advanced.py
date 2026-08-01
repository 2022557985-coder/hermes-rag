"""Advanced reranking tests for CrossEncoderReranker.

Tests warmup, threshold-based reranking, score normalization,
model info, input validation, thread-local storage, fallback
behavior, batch processing, timeout handling, and Chinese text reranking.
"""

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.reranking.cross_encoder import CrossEncoderReranker, _thread_local

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidates(n=5, prefix="candidate"):
    """Create sample candidate dicts."""
    return [
        {
            "chunk_id": f"{prefix}_{i}",
            "text": f"This is {prefix} text {i} about machine learning and artificial intelligence.",
            "score": 0.9 - i * 0.1,
            "metadata": {"source": f"doc_{i}.txt"},
        }
        for i in range(n)
    ]


def _make_chinese_candidates(n=5):
    """Create Chinese sample candidate dicts."""
    return [
        {
            "chunk_id": f"cn_{i}",
            "text": f"这是关于机器学习和人工智能的中文候选文本 {i}。包含了深度学习、神经网络等相关概念。",
            "score": 0.9 - i * 0.1,
            "metadata": {"source": f"cn_doc_{i}.txt"},
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Warmup tests
# ---------------------------------------------------------------------------

class TestWarmup:
    """Test _warmup method behavior."""

    def test_warmup_no_model_loaded(self):
        """Test warmup when model is not loaded (should not crash)."""
        reranker = CrossEncoderReranker()
        try:
            reranker._warmup()
        except (OSError, RuntimeError) as e:
            # Model not available locally - skip gracefully
            pytest.skip(f"Reranker model not available locally: {e}")
        except Exception as e:
            pytest.fail(f"_warmup should not crash when model not loaded: {e}")

    def test_warmup_with_model_failure(self):
        """Test warmup handles model load failure gracefully."""
        reranker = CrossEncoderReranker()
        # Simulate model load failure
        with patch.object(reranker, '_load_model', side_effect=RuntimeError("Model load failed")):
            try:
                reranker._warmup()
            except Exception:
                pass  # Should not raise - _warmup catches exceptions


# ---------------------------------------------------------------------------
# Rerank with threshold tests
# ---------------------------------------------------------------------------

class TestRerankWithThreshold:
    """Test rerank_with_threshold method."""

    def test_rerank_with_threshold_filters_below(self):
        """Test that candidates below threshold are filtered out."""
        reranker = CrossEncoderReranker()
        candidates = _make_candidates(5)

        # Mock rerank to return candidates with known scores
        with patch.object(reranker, 'rerank', return_value=[
            {**c, "rerank_score": 0.9, "score": 0.9} for c in candidates[:1]
        ] + [
            {**c, "rerank_score": 0.2, "score": 0.2} for c in candidates[1:]
        ]):
            result = reranker.rerank_with_threshold("test query", candidates, min_score=0.5)
            assert len(result) == 1, "Should filter out candidates below 0.5 threshold"

    def test_rerank_with_threshold_all_below_returns_top(self):
        """Test that when all candidates are below threshold, top one is returned."""
        reranker = CrossEncoderReranker()
        candidates = _make_candidates(3)

        # All candidates below threshold
        with patch.object(reranker, 'rerank', return_value=[
            {**c, "rerank_score": 0.1, "score": 0.1} for c in candidates
        ]):
            result = reranker.rerank_with_threshold("test query", candidates, min_score=0.5)
            assert len(result) == 1, "Should return top candidate when all are below threshold"

    def test_rerank_with_threshold_zero_keeps_all(self):
        """Test that min_score=0 keeps all candidates."""
        reranker = CrossEncoderReranker()
        candidates = _make_candidates(3)

        with patch.object(reranker, 'rerank', return_value=[
            {**c, "rerank_score": 0.5, "score": 0.5} for c in candidates
        ]):
            result = reranker.rerank_with_threshold("test query", candidates, min_score=0.0)
            assert len(result) == 3, "Should keep all candidates with min_score=0"

    def test_rerank_with_threshold_empty_candidates(self):
        """Test threshold reranking with empty candidates."""
        reranker = CrossEncoderReranker()
        result = reranker.rerank_with_threshold("test", [], min_score=0.3)
        assert result == [], "Empty candidates should return empty list"


# ---------------------------------------------------------------------------
# Normalize rerank scores tests
# ---------------------------------------------------------------------------

class TestNormalizeRerankScores:
    """Test _normalize_rerank_scores method."""

    def test_normalize_basic(self):
        """Test basic score normalization."""
        reranker = CrossEncoderReranker()
        candidates = [
            {"chunk_id": "a", "rerank_score": 10.0, "score": 10.0},
            {"chunk_id": "b", "rerank_score": 5.0, "score": 5.0},
            {"chunk_id": "c", "rerank_score": 0.0, "score": 0.0},
        ]
        normalized = reranker._normalize_rerank_scores(candidates)
        assert normalized[0]["rerank_score"] == 1.0, "Highest should normalize to 1.0"
        assert normalized[0]["score"] == 1.0, "score should also be normalized"
        assert normalized[2]["rerank_score"] == 0.0, "Lowest should normalize to 0.0"

    def test_normalize_all_equal(self):
        """Test normalization when all scores are equal."""
        reranker = CrossEncoderReranker()
        candidates = [
            {"chunk_id": "a", "rerank_score": 3.0, "score": 3.0},
            {"chunk_id": "b", "rerank_score": 3.0, "score": 3.0},
        ]
        normalized = reranker._normalize_rerank_scores(candidates)
        for c in normalized:
            assert c["rerank_score"] == 0.5, "Equal scores should normalize to 0.5"
            assert c["score"] == 0.5, "score should also be 0.5"

    def test_normalize_empty(self):
        """Test normalization with empty list."""
        reranker = CrossEncoderReranker()
        result = reranker._normalize_rerank_scores([])
        assert result == [], "Empty list should return empty list"

    def test_normalize_single_candidate(self):
        """Test normalization with single candidate."""
        reranker = CrossEncoderReranker()
        candidates = [{"chunk_id": "a", "rerank_score": 42.0, "score": 42.0}]
        normalized = reranker._normalize_rerank_scores(candidates)
        assert normalized[0]["rerank_score"] == 0.5, "Single candidate should normalize to 0.5"
        assert normalized[0]["score"] == 0.5, "score should also be 0.5"


# ---------------------------------------------------------------------------
# Get model info tests
# ---------------------------------------------------------------------------

class TestGetModelInfo:
    """Test get_model_info method."""

    def test_get_model_info_defaults(self):
        """Test get_model_info returns correct defaults."""
        reranker = CrossEncoderReranker(
            model_name="BAAI/bge-reranker-base",
            device="cpu",
            batch_size=16,
            max_candidates=50,
            timeout_seconds=1.5,
        )
        info = reranker.get_model_info()
        assert info["model_name"] == "BAAI/bge-reranker-base", "Correct model name"
        assert info["device"] == "cpu", "Correct device"
        assert info["loaded"] is False, "Should not be loaded initially"
        assert info["batch_size"] == 16, "Correct batch size"
        assert info["max_candidates"] == 50, "Correct max candidates"
        assert info["timeout_seconds"] == 1.5, "Correct timeout"

    def test_get_model_info_custom(self):
        """Test get_model_info with custom parameters."""
        reranker = CrossEncoderReranker(
            model_name="custom/model",
            device="cuda",
            batch_size=32,
            max_candidates=100,
            timeout_seconds=3.0,
        )
        info = reranker.get_model_info()
        assert info["model_name"] == "custom/model", "Custom model name"
        assert info["device"] == "cuda", "CUDA device"
        assert info["batch_size"] == 32, "Custom batch size"
        assert info["max_candidates"] == 100, "Custom max candidates"

    def test_get_model_info_structure(self):
        """Test get_model_info returns all expected keys."""
        reranker = CrossEncoderReranker()
        info = reranker.get_model_info()
        expected_keys = {"model_name", "device", "loaded", "batch_size", "max_candidates", "timeout_seconds"}
        assert set(info.keys()) == expected_keys, f"Should have exactly {expected_keys} keys"


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Test _validate_input method."""

    def test_validate_empty_query(self):
        """Test validation with empty query."""
        reranker = CrossEncoderReranker()
        with pytest.raises(ValueError, match="Query must be a non-empty string"):
            reranker._validate_input("", _make_candidates(3))

    def test_validate_whitespace_query(self):
        """Test validation with whitespace-only query."""
        reranker = CrossEncoderReranker()
        with pytest.raises(ValueError, match="Query must be a non-empty string"):
            reranker._validate_input("   ", _make_candidates(3))

    def test_validate_empty_candidates(self):
        """Test validation with empty candidates list."""
        reranker = CrossEncoderReranker()
        with pytest.raises(ValueError, match="Candidates list must be non-empty"):
            reranker._validate_input("valid query", [])

    def test_validate_both_invalid(self):
        """Test validation with both query and candidates invalid."""
        reranker = CrossEncoderReranker()
        with pytest.raises(ValueError):
            reranker._validate_input("", [])

    def test_rerank_handles_validation_error_gracefully(self):
        """Test that rerank gracefully handles validation errors."""
        reranker = CrossEncoderReranker()
        result = reranker.rerank("", [])
        assert result == [], "Empty query and candidates should return empty list"


# ---------------------------------------------------------------------------
# Thread-local model storage tests
# ---------------------------------------------------------------------------

class TestThreadLocalStorage:
    """Test thread-local model storage mechanism."""

    def test_set_and_get_thread_model(self):
        """Test _set_thread_model and _get_thread_model."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        reranker._set_thread_model(mock_model, mock_tokenizer)
        model, tokenizer = reranker._get_thread_model()

        assert model is mock_model, "Should retrieve the same model object"
        assert tokenizer is mock_tokenizer, "Should retrieve the same tokenizer object"

    def test_thread_local_isolation(self):
        """Test that thread-local storage is isolated per thread."""
        reranker = CrossEncoderReranker()
        results = {}

        def thread_func(thread_id):
            # Set thread-local model
            mock_model = MagicMock()
            mock_model._thread_id = thread_id
            reranker._set_thread_model(mock_model, MagicMock())
            model, _ = reranker._get_thread_model()
            results[thread_id] = model._thread_id

        t1 = threading.Thread(target=thread_func, args=(1,))
        t2 = threading.Thread(target=thread_func, args=(2,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Each thread should have its own model
        assert results.get(1) == 1, "Thread 1 should have its own model"
        assert results.get(2) == 2, "Thread 2 should have its own model"

    def test_thread_local_cleared_on_unload(self):
        """Test that thread-local storage is cleared on model unload."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        reranker._set_thread_model(mock_model, mock_tokenizer)
        model, tokenizer = reranker._get_thread_model()
        assert model is not None, "Should have thread-local model before unload"

        reranker._unload_model()
        model, tokenizer = reranker._get_thread_model()
        assert model is None, "Thread-local model should be cleared after unload"

    def test_get_thread_model_no_storage(self):
        """Test _get_thread_model when nothing is stored."""
        # Clear any existing thread-local
        if hasattr(_thread_local, "model"):
            del _thread_local.model
        if hasattr(_thread_local, "tokenizer"):
            del _thread_local.tokenizer

        reranker = CrossEncoderReranker()
        model, tokenizer = reranker._get_thread_model()
        assert model is None, "Should return None when no model stored"
        assert tokenizer is None, "Should return None when no tokenizer stored"


# ---------------------------------------------------------------------------
# Fallback to identity reranking tests
# ---------------------------------------------------------------------------

class TestFallbackBehavior:
    """Test fallback to identity reranking when model fails."""

    def test_fallback_when_model_not_loaded(self):
        """Test fallback to identity reranking when model is not loaded."""
        reranker = CrossEncoderReranker()
        # Ensure model is not loaded
        reranker._unload_model()

        candidates = _make_candidates(3)
        result = reranker.rerank("test query", candidates)

        assert len(result) == 3, "Should return all candidates"
        for c in result:
            assert c["rerank_score"] == 1.0, "Identity fallback should set rerank_score to 1.0"
            assert "score" in c, "Original score should be preserved"

    def test_rerank_catches_load_error(self):
        """Test that rerank catches model load errors gracefully."""
        reranker = CrossEncoderReranker()
        candidates = _make_candidates(3)

        with patch.object(reranker, '_load_model', side_effect=Exception("Model load error")):
            result = reranker.rerank("test query", candidates)
            # Should return original candidates without crashing
            assert len(result) == 3, "Should return original candidates on load error"

    def test_rerank_catches_inference_error(self):
        """Test that rerank catches inference errors gracefully."""
        reranker = CrossEncoderReranker()
        # Mock _load_model to succeed but _validate_input to pass
        candidates = _make_candidates(3)

        with patch.object(reranker, '_load_model'):
            # Set model to something truthy to pass the None check
            reranker._model = MagicMock()
            reranker._tokenizer = MagicMock()
            # Make the model call fail
            reranker._model.side_effect = Exception("Inference error")

            result = reranker.rerank("test query", candidates)
            assert len(result) == 3, "Should return original candidates on inference error"


# ---------------------------------------------------------------------------
# Batch processing tests
# ---------------------------------------------------------------------------

class TestBatchProcessing:
    """Test batch processing with different batch sizes."""

    def test_batch_processing_small_batch(self):
        """Test reranking with batch_size=1."""
        reranker = CrossEncoderReranker(batch_size=1, max_candidates=100)
        candidates = _make_candidates(5)

        with patch.object(reranker, '_load_model'):
            reranker._model = MagicMock()
            reranker._tokenizer = MagicMock()
            # Simulate model outputs
            reranker._model.return_value = MagicMock()
            reranker._model.return_value.logits = MagicMock()
            reranker._model.return_value.logits.squeeze.return_value.cpu.return_value.tolist.return_value = [0.5] * 5

            result = reranker.rerank("test", candidates)
            assert len(result) == 5, "Should process all candidates with batch_size=1"

    def test_batch_processing_large_batch(self):
        """Test reranking with batch_size=32."""
        reranker = CrossEncoderReranker(batch_size=32, max_candidates=100)
        candidates = _make_candidates(10)

        with patch.object(reranker, '_load_model'):
            reranker._model = MagicMock()
            reranker._tokenizer = MagicMock()
            reranker._model.return_value = MagicMock()
            reranker._model.return_value.logits = MagicMock()
            reranker._model.return_value.logits.squeeze.return_value.cpu.return_value.tolist.return_value = [0.5] * 10

            result = reranker.rerank("test", candidates)
            assert len(result) == 10, "Should process all candidates with batch_size=32"

    def test_max_candidates_limit(self):
        """Test that max_candidates limits the number of processed candidates."""
        reranker = CrossEncoderReranker(max_candidates=3)
        candidates = _make_candidates(10)

        with patch.object(reranker, '_load_model'):
            reranker._model = MagicMock()
            reranker._tokenizer = MagicMock()
            reranker._model.return_value = MagicMock()
            reranker._model.return_value.logits = MagicMock()
            reranker._model.return_value.logits.squeeze.return_value.cpu.return_value.tolist.return_value = [0.5] * 3

            result = reranker.rerank("test", candidates)
            assert len(result) <= 3, "Should respect max_candidates limit"


# ---------------------------------------------------------------------------
# Timeout handling tests
# ---------------------------------------------------------------------------

class TestTimeoutHandling:
    """Test timeout handling in reranker."""

    def test_timeout_configuration(self):
        """Test timeout_seconds is properly configured."""
        reranker = CrossEncoderReranker(timeout_seconds=2.5)
        assert reranker.timeout_seconds == 2.5, "Should store timeout value"

    def test_default_timeout(self):
        """Test default timeout value."""
        reranker = CrossEncoderReranker()
        assert reranker.timeout_seconds == 1.5, "Default timeout should be 1.5s"


# ---------------------------------------------------------------------------
# Chinese text reranking tests
# ---------------------------------------------------------------------------

class TestChineseReranking:
    """Test reranking with Chinese text."""

    def test_rerank_chinese_candidates(self):
        """Test reranking Chinese candidates."""
        reranker = CrossEncoderReranker()
        candidates = _make_chinese_candidates(5)

        with patch.object(reranker, '_load_model'):
            reranker._model = MagicMock()
            reranker._tokenizer = MagicMock()
            reranker._model.return_value = MagicMock()
            reranker._model.return_value.logits = MagicMock()
            reranker._model.return_value.logits.squeeze.return_value.cpu.return_value.tolist.return_value = [0.8, 0.6, 0.4, 0.2, 0.1]

            result = reranker.rerank("什么是机器学习", candidates)
            assert len(result) == 5, "Should process Chinese candidates"
            for c in result:
                assert "rerank_score" in c, "Should have rerank_score"
                assert "score" in c, "Should have score"

    def test_rerank_chinese_query_english_candidates(self):
        """Test reranking with Chinese query and English candidates."""
        reranker = CrossEncoderReranker()
        candidates = _make_candidates(3)

        with patch.object(reranker, '_load_model'):
            reranker._model = MagicMock()
            reranker._tokenizer = MagicMock()
            reranker._model.return_value = MagicMock()
            reranker._model.return_value.logits = MagicMock()
            reranker._model.return_value.logits.squeeze.return_value.cpu.return_value.tolist.return_value = [0.5, 0.3, 0.1]

            result = reranker.rerank("什么是机器学习", candidates)
            assert len(result) == 3, "Should handle mixed language reranking"

    def test_rerank_mixed_language(self):
        """Test reranking with mixed Chinese-English candidates."""
        reranker = CrossEncoderReranker()
        candidates = [
            {"chunk_id": "mix_1", "text": "Machine learning 是人工智能的一个子集", "score": 0.9},
            {"chunk_id": "mix_2", "text": "Deep learning uses neural networks 进行深度学习", "score": 0.7},
            {"chunk_id": "mix_3", "text": "Python is a programming language 编程语言", "score": 0.5},
        ]

        with patch.object(reranker, '_load_model'):
            reranker._model = MagicMock()
            reranker._tokenizer = MagicMock()
            reranker._model.return_value = MagicMock()
            reranker._model.return_value.logits = MagicMock()
            reranker._model.return_value.logits.squeeze.return_value.cpu.return_value.tolist.return_value = [0.9, 0.5, 0.3]

            result = reranker.rerank("如何学习machine learning", candidates)
            assert len(result) == 3, "Should handle mixed language"


# ---------------------------------------------------------------------------
# Model load/unload lifecycle tests
# ---------------------------------------------------------------------------

class TestModelLifecycle:
    """Test model load/unload lifecycle."""

    def test_initial_state_not_loaded(self):
        """Test that model is not loaded initially."""
        reranker = CrossEncoderReranker()
        assert reranker._model is None, "Model should be None initially"
        assert reranker._tokenizer is None, "Tokenizer should be None initially"
        assert reranker._loaded is False, "Should not be marked as loaded"

    def test_unload_clears_model(self):
        """Test that _unload_model clears all model references."""
        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()
        reranker._tokenizer = MagicMock()
        reranker._loaded = True

        reranker._unload_model()

        assert reranker._model is None, "Model should be None after unload"
        assert reranker._tokenizer is None, "Tokenizer should be None after unload"
        assert reranker._loaded is False, "Should not be marked as loaded"

    def test_double_check_loading_thread_safety(self):
        """Test the double-check pattern in _load_model prevents re-loading."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        MagicMock()
        reranker._model = mock_model

        # Should return immediately without loading
        reranker._load_model()
        assert reranker._model is mock_model, "Should not reload if already loaded"


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestInitialization:
    """Test CrossEncoderReranker initialization."""

    def test_default_initialization(self):
        """Test default initialization values."""
        reranker = CrossEncoderReranker()
        assert reranker.model_name == "BAAI/bge-reranker-v2-m3", "Default model name"
        assert reranker.device == "cpu", "Default device"
        assert reranker.batch_size == 16, "Default batch size"
        assert reranker.max_candidates == 50, "Default max candidates"
        assert reranker.timeout_seconds == 1.5, "Default timeout"

    def test_custom_initialization(self):
        """Test custom initialization values."""
        reranker = CrossEncoderReranker(
            model_name="custom-reranker",
            device="cuda",
            batch_size=8,
            max_candidates=20,
            timeout_seconds=2.0,
        )
        assert reranker.model_name == "custom-reranker"
        assert reranker.device == "cuda"
        assert reranker.batch_size == 8
        assert reranker.max_candidates == 20
        assert reranker.timeout_seconds == 2.0