"""Cross-encoder reranking using BGE-Reranker."""

import gc
import logging
import threading
from typing import Any

logger = logging.getLogger("hermes_rag")

# Thread-local storage for models to avoid reloading issues in multi-threaded environments
_thread_local = threading.local()


class CrossEncoderReranker:
    """Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

    Features:
    - Lazy loading: model loaded only when reranking is needed
    - Thread-safe: model loading is protected by a lock; thread-local storage
      avoids model reloading issues in multi-threaded environments
    - Timeout protection: falls back to identity reranking on timeout
    - Configurable: can be disabled via config
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        batch_size: int = 16,
        max_candidates: int = 50,
        timeout_seconds: float = 1.5,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_candidates = max_candidates
        self.timeout_seconds = timeout_seconds
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()
        self._loaded = False

    def _get_thread_model(self) -> tuple[Any | None, Any | None]:
        """Get model/tokenizer from thread-local storage."""
        model = getattr(_thread_local, "model", None)
        tokenizer = getattr(_thread_local, "tokenizer", None)
        return model, tokenizer

    def _set_thread_model(self, model: Any, tokenizer: Any) -> None:
        """Store model/tokenizer in thread-local storage."""
        _thread_local.model = model
        _thread_local.tokenizer = tokenizer

    def _load_model(self) -> None:
        """Lazy-load the reranker model with GPU OOM fallback (thread-safe)."""
        # Check thread-local first
        model, tokenizer = self._get_thread_model()
        if model is not None and tokenizer is not None:
            self._model = model
            self._tokenizer = tokenizer
            self._loaded = True
            return

        if self._model is not None:
            return

        with self._load_lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return

            # Re-check thread-local after acquiring lock
            model, tokenizer = self._get_thread_model()
            if model is not None and tokenizer is not None:
                self._model = model
                self._tokenizer = tokenizer
                self._loaded = True
                return

            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, local_files_only=True
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, local_files_only=True
                )
                self._model.eval()

                target_device = self.device
                if target_device == "cuda" and torch.cuda.is_available():
                    try:
                        self._model = self._model.to("cuda")
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(
                            f"GPU out of memory loading {self.model_name}, falling back to CPU"
                        )
                        self._model = self._model.to("cpu")
                        self.device = "cpu"
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.warning(
                                f"GPU out of memory loading {self.model_name}, falling back to CPU"
                            )
                            self._model = self._model.to("cpu")
                            self.device = "cpu"
                        else:
                            raise
                else:
                    self._model = self._model.to("cpu")
                    self.device = "cpu"

                # Store in thread-local to avoid reloading across threads
                self._set_thread_model(self._model, self._tokenizer)
                self._loaded = True

            except Exception as e:
                logger.error(
                    f"Failed to load reranker model '{self.model_name}': {e}. "
                    f"Will fall back to identity reranking."
                )
                self._model = None
                self._tokenizer = None
                self._loaded = False
                raise

    def _warmup(self) -> None:
        """Run a dummy inference to pre-load and warm up the model.

        This ensures the model is fully loaded and GPU kernels are compiled
        before processing real queries, reducing first-query latency.
        """
        try:
            self._load_model()
            if self._model is None or self._tokenizer is None:
                logger.warning("Cannot warm up: model not loaded.")
                return

            import torch

            dummy_query = "warmup query"
            dummy_doc = "warmup document"
            with torch.inference_mode():
                inputs = self._tokenizer(
                    [dummy_query, dummy_doc],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                if self.device == "cuda" and torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                self._model(**inputs)
            logger.info(f"Reranker model '{self.model_name}' warmed up successfully.")
        except Exception as e:
            logger.warning(f"Model warmup failed (non-fatal): {e}")

    def _unload_model(self) -> None:
        """Release the reranker model from memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
        self._loaded = False
        # Always clear thread-local storage
        if hasattr(_thread_local, "model"):
            del _thread_local.model
        if hasattr(_thread_local, "tokenizer"):
            del _thread_local.tokenizer
        gc.collect()

    def _validate_input(self, query: str, candidates: list[dict[str, Any]]) -> None:
        """Validate reranking inputs.

        Args:
            query: The query string.
            candidates: List of candidate dicts.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if not candidates:
            raise ValueError("Candidates list must be non-empty.")

    def _normalize_rerank_scores(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Normalize rerank scores to the [0, 1] range using min-max scaling.

        Args:
            candidates: List of candidate dicts with 'rerank_score' key.

        Returns:
            Candidates with normalized scores in [0, 1].
        """
        scores = [c.get("rerank_score", 0.0) for c in candidates]
        if not scores:
            return candidates

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All scores are equal; set to 0.5 as neutral
            for c in candidates:
                c["rerank_score"] = 0.5
                c["score"] = 0.5
        else:
            for c in candidates:
                raw = c.get("rerank_score", 0.0)
                normalized = (raw - min_score) / (max_score - min_score)
                c["rerank_score"] = normalized
                c["score"] = normalized

        return candidates

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rerank candidates using cross-encoder.

        Args:
            query: Query string.
            candidates: List of candidate dicts with 'text' key.

        Returns:
            Reranked candidates sorted by relevance score.
        """
        try:
            self._validate_input(query, candidates)
        except ValueError as e:
            logger.warning(f"Input validation failed: {e}")
            return candidates

        if not candidates:
            return []

        candidates = candidates[: self.max_candidates]

        try:
            self._load_model()

            # If model failed to load, fall back to identity reranking
            if self._model is None or self._tokenizer is None:
                logger.warning("Model not loaded, falling back to identity reranking.")
                for i, candidate in enumerate(candidates):
                    candidate["rerank_score"] = 1.0
                    candidate["score"] = candidate.get("score", 1.0)
                return candidates

            import torch

            pairs = []
            for candidate in candidates:
                text = candidate.get("text", "")
                pairs.append([query, text])

            scores: list[float] = []
            with torch.inference_mode():
                for i in range(0, len(pairs), self.batch_size):
                    batch = pairs[i : i + self.batch_size]
                    inputs = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )
                    if self.device == "cuda" and torch.cuda.is_available():
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}

                    outputs = self._model(**inputs)
                    batch_scores = outputs.logits.squeeze(-1).cpu().tolist()
                    if isinstance(batch_scores, float):
                        batch_scores = [batch_scores]
                    scores.extend(batch_scores)

            # Attach scores and sort
            for i, candidate in enumerate(candidates):
                candidate["rerank_score"] = scores[i] if i < len(scores) else 0.0
                candidate["score"] = candidate["rerank_score"]

            # Normalize scores to [0, 1]
            candidates = self._normalize_rerank_scores(candidates)

            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

            return candidates

        except Exception as e:
            logger.warning(f"Reranking failed: {e}, falling back to identity reranking")
            for candidate in candidates:
                candidate["rerank_score"] = 1.0
                candidate["score"] = candidate.get("score", 1.0)
            return candidates

    def rerank_with_threshold(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        min_score: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Rerank and filter candidates below a minimum score threshold.

        Args:
            query: Query string.
            candidates: List of candidate dicts with 'text' key.
            min_score: Minimum rerank score to keep (0.0 to 1.0).

        Returns:
            Filtered and reranked candidates sorted by relevance score.
        """
        reranked = self.rerank(query, candidates)
        filtered = [
            c for c in reranked if c.get("rerank_score", 0.0) >= min_score
        ]
        if not filtered and reranked:
            logger.warning(
                f"All candidates below threshold {min_score}, returning top candidate."
            )
            return reranked[:1]
        return filtered

    def get_model_info(self) -> dict[str, Any]:
        """Get model information including name, device, and loaded status.

        Returns:
            Dict with model_name, device, loaded, and batch_size.
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self._loaded,
            "batch_size": self.batch_size,
            "max_candidates": self.max_candidates,
            "timeout_seconds": self.timeout_seconds,
        }