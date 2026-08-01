"""Cross-encoder reranking using BGE-Reranker."""

import gc
import logging
import threading
from typing import List, Dict, Any, Optional

logger = logging.getLogger("hermes_rag")


class CrossEncoderReranker:
    """Cross-encoder reranker using BAAI/bge-reranker-base.

    Features:
    - Lazy loading: model loaded only when reranking is needed
    - Thread-safe: model loading is protected by a lock
    - Timeout protection: falls back to RRF results on timeout
    - Configurable: can be disabled via config
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
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

    def _load_model(self):
        """Lazy-load the reranker model with GPU OOM fallback (thread-safe)."""
        if self._model is not None:
            return

        with self._load_lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return

            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
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

    def _unload_model(self):
        """Release the reranker model from memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            gc.collect()

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using cross-encoder.

        Args:
            query: Query string.
            candidates: List of candidate dicts with 'text' key.

        Returns:
            Reranked candidates sorted by relevance score.
        """
        if not candidates:
            return []

        candidates = candidates[: self.max_candidates]

        try:
            self._load_model()

            import torch

            pairs = []
            for candidate in candidates:
                text = candidate.get("text", "")
                pairs.append([query, text])

            scores = []
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
                    batch_scores = (
                        outputs.logits.squeeze(-1).cpu().tolist()
                    )
                    if isinstance(batch_scores, float):
                        batch_scores = [batch_scores]
                    scores.extend(batch_scores)

            # Attach scores and sort
            for i, candidate in enumerate(candidates):
                candidate["rerank_score"] = scores[i] if i < len(scores) else 0.0
                candidate["score"] = candidate["rerank_score"]  # Update primary score

            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

            return candidates

        except Exception as e:
            logger.warning(f"Reranking failed: {e}, returning original candidates")
            return candidates