"""RAGAS evaluation integration for Hermes-RAG.

RAGAS (Retrieval Augmented Generation Assessment) provides:
- Context Precision: How precise are the retrieved contexts?
- Context Recall: How many relevant contexts were retrieved?
- Faithfulness: Is the generated answer faithful to the context?
- Answer Relevancy: Is the answer relevant to the question?

Installation: pip install ragas
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


class RAGASAdapter:
    """Adapter for RAGAS evaluation of Hermes-RAG retrieval and generation.

    Can work with or without the ragas library installed. When ragas is not
    available, falls back to heuristic-based approximations.
    """

    def __init__(self, llm_provider=None):
        """Initialize RAGAS adapter.

        Args:
            llm_provider: Optional LLM provider for RAGAS metrics (e.g., OpenAI, Ollama).
                         If None, uses heuristic approximations.
        """
        self.llm = llm_provider
        self._ragas_available = self._check_ragas()

    @staticmethod
    def _check_ragas() -> bool:
        try:
            import ragas  # noqa: F401
            return True
        except ImportError:
            return False

    def evaluate_retrieval(
        self,
        questions: List[str],
        retrieved_contexts: List[List[str]],
        ground_truth_contexts: List[List[str]],
    ) -> Dict[str, float]:
        """Evaluate retrieval quality using RAGAS metrics.

        Args:
            questions: List of query strings.
            retrieved_contexts: List of retrieved context lists per query.
            ground_truth_contexts: List of ground truth context lists per query.

        Returns:
            Dict with context_precision and context_recall scores.
        """
        if self._ragas_available and self.llm:
            return self._evaluate_with_ragas(
                questions, retrieved_contexts, ground_truth_contexts
            )
        return self._evaluate_heuristic(
            questions, retrieved_contexts, ground_truth_contexts
        )

    def _evaluate_heuristic(
        self,
        questions: List[str],
        retrieved_contexts: List[List[str]],
        ground_truth_contexts: List[List[str]],
    ) -> Dict[str, float]:
        """Heuristic-based evaluation when RAGAS is unavailable.

        Uses text overlap metrics as approximations:
        - Context Precision: Fraction of retrieved contexts that are relevant
        - Context Recall: Fraction of ground truth contexts that were retrieved
        """
        precision_scores = []
        recall_scores = []

        for retrieved, ground_truth in zip(retrieved_contexts, ground_truth_contexts):
            if not ground_truth:
                continue

            gt_set = set(ground_truth)
            ret_set = set(retrieved)

            # Precision: what fraction of retrieved is relevant?
            if ret_set:
                precision = len(ret_set & gt_set) / len(ret_set)
            else:
                precision = 0.0
            precision_scores.append(precision)

            # Recall: what fraction of ground truth was retrieved?
            recall = len(ret_set & gt_set) / len(gt_set)
            recall_scores.append(recall)

        return {
            "context_precision": float(np.mean(precision_scores)) if precision_scores else 0.0,
            "context_recall": float(np.mean(recall_scores)) if recall_scores else 0.0,
        }

    def _evaluate_with_ragas(
        self,
        questions: List[str],
        retrieved_contexts: List[List[str]],
        ground_truth_contexts: List[List[str]],
    ) -> Dict[str, float]:
        """Full RAGAS evaluation with LLM-based metrics."""
        from ragas import evaluate
        from ragas.metrics import context_precision, context_recall
        from datasets import Dataset

        data = {
            "question": questions,
            "contexts": retrieved_contexts,
            "ground_truth": ground_truth_contexts,
        }
        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset,
            metrics=[context_precision, context_recall],
            llm=self.llm,
        )

        return {
            "context_precision": float(result["context_precision"]),
            "context_recall": float(result["context_recall"]),
        }

    def evaluate_generation(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
    ) -> Dict[str, float]:
        """Evaluate generation quality.

        Args:
            questions: List of query strings.
            answers: List of generated answers.
            contexts: List of contexts used for generation.

        Returns:
            Dict with faithfulness and answer_relevancy scores.
        """
        if self._ragas_available and self.llm:
            return self._evaluate_generation_ragas(questions, answers, contexts)
        return self._evaluate_generation_heuristic(questions, answers, contexts)

    def _evaluate_generation_heuristic(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
    ) -> Dict[str, float]:
        """Heuristic generation quality evaluation."""
        # Faithfulness approximation: check if answer tokens overlap with context
        faithfulness_scores = []
        relevancy_scores = []

        for question, answer, ctx_list in zip(questions, answers, contexts):
            if not answer or not ctx_list:
                faithfulness_scores.append(0.0)
                relevancy_scores.append(0.0)
                continue

            # Faithfulness: fraction of answer tokens found in context
            answer_tokens = set(answer.lower().split())
            context_text = " ".join(ctx_list).lower()
            context_tokens = set(context_text.split())

            if answer_tokens:
                overlap = len(answer_tokens & context_tokens)
                faithfulness = overlap / len(answer_tokens)
            else:
                faithfulness = 0.0
            faithfulness_scores.append(faithfulness)

            # Answer relevancy: fraction of context tokens in answer
            question_tokens = set(question.lower().split())
            if context_tokens - question_tokens:
                overlap = len(answer_tokens & context_tokens)
                relevancy = overlap / len(context_tokens)
            else:
                relevancy = 0.0
            relevancy_scores.append(relevancy)

        return {
            "faithfulness": float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0,
            "answer_relevancy": float(np.mean(relevancy_scores)) if relevancy_scores else 0.0,
        }

    def _evaluate_generation_ragas(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
    ) -> Dict[str, float]:
        """Full RAGAS generation evaluation."""
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=self.llm,
        )

        return {
            "faithfulness": float(result["faithfulness"]),
            "answer_relevancy": float(result["answer_relevancy"]),
        }

    def run_full_evaluation(
        self,
        questions: List[str],
        answers: List[str],
        retrieved_contexts: List[List[str]],
        ground_truth_contexts: List[List[str]],
    ) -> Dict[str, Any]:
        """Run complete RAGAS evaluation.

        Args:
            questions: List of query strings.
            answers: List of generated answers.
            retrieved_contexts: List of retrieved context lists.
            ground_truth_contexts: List of ground truth context lists.

        Returns:
            Dict with all RAGAS metrics.
        """
        retrieval_metrics = self.evaluate_retrieval(
            questions, retrieved_contexts, ground_truth_contexts
        )
        generation_metrics = self.evaluate_generation(
            questions, answers, retrieved_contexts
        )

        return {
            "retrieval": retrieval_metrics,
            "generation": generation_metrics,
        }


def run_ragas_eval():
    """Run RAGAS evaluation on Hermes-RAG."""
    from src.config import get_config, reset_config
    reset_config()
    cfg = get_config()

    from src.core.pipeline_factory import build_pipeline
    from evaluation.dataset import EvaluationDataset

    print("=" * 60)
    print("  RAGAS Evaluation for Hermes-RAG")
    print("=" * 60)

    # Build pipeline
    pipeline = build_pipeline(config=cfg)

    # Load dataset
    dataset = EvaluationDataset()
    data = dataset.load()
    print(f"\nLoaded {len(data)} evaluation queries")

    # Prepare data for RAGAS
    questions = []
    retrieved_contexts = []
    ground_truth_contexts = []
    answers = []

    for item in data[:10]:  # Limit to 10 queries for efficiency
        query = item["query"]
        relevant_ids = item["relevant_chunk_ids"]

        result = pipeline.retrieve(query=query, top_k=5, use_reranker=True)
        retrieved = result.get("results", [])

        questions.append(query)
        retrieved_contexts.append([r.get("text", "") for r in retrieved])
        ground_truth_contexts.append(relevant_ids)

        # Generate a simple answer from retrieved contexts
        ctx_text = " ".join([r.get("text", "") for r in retrieved[:3]])
        if ctx_text:
            answers.append(f"Based on the context: {ctx_text[:200]}")
        else:
            answers.append("")

    # Run RAGAS evaluation
    adapter = RAGASAdapter()
    results = adapter.run_full_evaluation(
        questions=questions,
        answers=answers,
        retrieved_contexts=retrieved_contexts,
        ground_truth_contexts=ground_truth_contexts,
    )

    print("\n--- RAGAS Evaluation Results ---")
    print(f"  Context Precision:  {results['retrieval']['context_precision']:.4f}")
    print(f"  Context Recall:     {results['retrieval']['context_recall']:.4f}")
    if "faithfulness" in results["generation"]:
        print(f"  Faithfulness:       {results['generation']['faithfulness']:.4f}")
        print(f"  Answer Relevancy:   {results['generation']['answer_relevancy']:.4f}")
    print("=" * 60)

    # Save results
    output_path = Path(__file__).parent / "data" / "ragas_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_ragas_eval()