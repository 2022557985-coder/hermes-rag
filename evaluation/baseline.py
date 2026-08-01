"""Baseline comparison: Hermes-RAG vs default LangChain-style retriever."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class BaselineRetriever:
    """Simple baseline retriever simulating LangChain's default VectorStoreRetriever.

    Uses single-path dense retrieval with fixed-size chunking, no RRF, no reranker.
    """

    def __init__(self, index_manager):
        self.index_manager = index_manager

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """Simple dense-only retrieval."""
        results = self.index_manager.search_dense(query, top_k=top_k)
        for r in results:
            r["source"] = "baseline_dense"
        return results


def compare_hermes_vs_baseline(pipeline_hermes, pipeline_baseline, dataset):
    """Compare Hermes-RAG against baseline retriever.

    Args:
        pipeline_hermes: Hermes-RAG pipeline.
        pipeline_baseline: Baseline retriever.
        dataset: EvaluationDataset.

    Returns:
        dict with comparison metrics.
    """
    from evaluation.eval import evaluate_pipeline, print_report

    print("=" * 70)
    print("  Hermes-RAG vs Baseline Comparison")
    print("=" * 70)

    # Evaluate Hermes-RAG
    hermes_metrics = evaluate_pipeline(pipeline_hermes, dataset, use_reranker=True)
    print_report(hermes_metrics, "Hermes-RAG (Full Pipeline)")

    # Evaluate Baseline
    baseline_metrics = evaluate_pipeline(pipeline_baseline, dataset, use_reranker=False)
    print_report(baseline_metrics, "Baseline (Dense-only, Fixed Chunking)")

    # Comparison
    print("\n--- Comparative Analysis ---")
    metrics_to_compare = ["hit_rate@1", "hit_rate@3", "hit_rate@5", "mrr", "ndcg@10"]
    for metric in metrics_to_compare:
        h_val = hermes_metrics[metric]
        b_val = baseline_metrics[metric]
        improvement = (h_val - b_val) / max(b_val, 0.001) * 100
        print(f"  {metric:15s}: Baseline={b_val:.4f}, Hermes={h_val:.4f}, Improvement={improvement:+.1f}%")

    return {
        "hermes": hermes_metrics,
        "baseline": baseline_metrics,
    }