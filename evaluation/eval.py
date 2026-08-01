"""Evaluation script for Hermes-RAG: Hit Rate, MRR, NDCG, Precision, Recall."""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def hit_rate_at_k(results: List[Dict[str, Any]], relevant_ids: List[str], k: int) -> float:
    """Calculate Hit Rate@K.

    Args:
        results: Retrieved results (sorted by relevance).
        relevant_ids: List of relevant chunk IDs.
        k: Top-K threshold.

    Returns:
        1.0 if any relevant document in top-K, else 0.0.
    """
    if not relevant_ids:
        return 0.0

    top_k_ids = [r["chunk_id"] for r in results[:k]]
    return 1.0 if any(rid in top_k_ids for rid in relevant_ids) else 0.0


def precision_at_k(results: List[Dict[str, Any]], relevant_ids: List[str], k: int) -> float:
    """Calculate Precision@K.

    Precision@K = (# of relevant documents in top-K) / K

    Args:
        results: Retrieved results (sorted by relevance).
        relevant_ids: List of relevant chunk IDs.
        k: Top-K threshold.

    Returns:
        Precision@K score (0.0 to 1.0).
    """
    if not relevant_ids or k <= 0:
        return 0.0

    top_k_ids = [r["chunk_id"] for r in results[:k]]
    relevant_in_top_k = sum(1 for rid in top_k_ids if rid in relevant_ids)
    return relevant_in_top_k / k


def recall_at_k(results: List[Dict[str, Any]], relevant_ids: List[str], k: int) -> float:
    """Calculate Recall@K.

    Recall@K = (# of relevant documents in top-K) / (total # of relevant documents)

    Args:
        results: Retrieved results (sorted by relevance).
        relevant_ids: List of relevant chunk IDs.
        k: Top-K threshold.

    Returns:
        Recall@K score (0.0 to 1.0).
    """
    if not relevant_ids or k <= 0:
        return 0.0

    top_k_ids = [r["chunk_id"] for r in results[:k]]
    relevant_in_top_k = sum(1 for rid in top_k_ids if rid in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def mrr(results: List[Dict[str, Any]], relevant_ids: List[str]) -> float:
    """Calculate Mean Reciprocal Rank.

    Args:
        results: Retrieved results (sorted by relevance).
        relevant_ids: List of relevant chunk IDs.

    Returns:
        MRR score.
    """
    if not relevant_ids:
        return 0.0

    for rank, result in enumerate(results, 1):
        if result["chunk_id"] in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: List[Dict[str, Any]], relevant_ids: List[str], k: int) -> float:
    """Calculate NDCG@K.

    Args:
        results: Retrieved results (sorted by relevance).
        relevant_ids: List of relevant chunk IDs.
        k: Top-K threshold.

    Returns:
        NDCG@K score.
    """
    if not relevant_ids:
        return 0.0

    # Binary relevance: 1 if relevant, 0 otherwise
    dcg = 0.0
    for i, result in enumerate(results[:k]):
        rel = 1.0 if result["chunk_id"] in relevant_ids else 0.0
        dcg += rel / np.log2(i + 2)  # i+2 because i starts at 0

    # Ideal DCG (all relevant documents at top)
    idcg = 0.0
    for i in range(min(len(relevant_ids), k)):
        idcg += 1.0 / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_pipeline(
    pipeline,
    dataset,
    use_reranker: bool = True,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Evaluate a retrieval pipeline on a dataset.

    Args:
        pipeline: RetrievalPipeline instance.
        dataset: EvaluationDataset instance.
        use_reranker: Whether to use reranker.
        top_k: Top-K for final results.

    Returns:
        dict with evaluation metrics.
    """
    import logging
    logger = logging.getLogger("hermes_rag")

    # Clear cache before evaluation to avoid cross-run contamination
    if pipeline.cache:
        pipeline.cache.clear()

    data = dataset.load()
    queries = dataset.get_queries()

    metrics = {
        "hit_rate@1": [],
        "hit_rate@3": [],
        "hit_rate@5": [],
        "precision@1": [],
        "precision@3": [],
        "precision@5": [],
        "recall@1": [],
        "recall@3": [],
        "recall@5": [],
        "mrr": [],
        "ndcg@10": [],
        "total_queries": len(queries),
        "total_time": 0.0,
        "avg_latency": 0.0,
        "failed_queries": 0,
        "failed_query_details": [],
    }

    for item in data:
        query = item["query"]
        relevant_ids = item["relevant_chunk_ids"]

        try:
            start = time.perf_counter()
            result = pipeline.retrieve(
                query=query,
                top_k=top_k,
                use_reranker=use_reranker,
            )
            elapsed = time.perf_counter() - start

            results = result.get("results", [])

            metrics["hit_rate@1"].append(hit_rate_at_k(results, relevant_ids, 1))
            metrics["hit_rate@3"].append(hit_rate_at_k(results, relevant_ids, 3))
            metrics["hit_rate@5"].append(hit_rate_at_k(results, relevant_ids, 5))
            metrics["precision@1"].append(precision_at_k(results, relevant_ids, 1))
            metrics["precision@3"].append(precision_at_k(results, relevant_ids, 3))
            metrics["precision@5"].append(precision_at_k(results, relevant_ids, 5))
            metrics["recall@1"].append(recall_at_k(results, relevant_ids, 1))
            metrics["recall@3"].append(recall_at_k(results, relevant_ids, 3))
            metrics["recall@5"].append(recall_at_k(results, relevant_ids, 5))
            metrics["mrr"].append(mrr(results, relevant_ids))
            metrics["ndcg@10"].append(ndcg_at_k(results, relevant_ids, 10))

            metrics["total_time"] += elapsed

        except Exception as e:
            logger.warning(f"Query failed: '{query[:80]}...' - {type(e).__name__}: {e}")
            metrics["failed_queries"] += 1
            metrics["failed_query_details"].append({
                "query": query,
                "error": str(e),
                "error_type": type(e).__name__,
            })

    # Aggregate
    total_successful = max(metrics["total_queries"] - metrics["failed_queries"], 0)
    aggregated = {
        "hit_rate@1": float(np.mean(metrics["hit_rate@1"])) if metrics["hit_rate@1"] else 0.0,
        "hit_rate@3": float(np.mean(metrics["hit_rate@3"])) if metrics["hit_rate@3"] else 0.0,
        "hit_rate@5": float(np.mean(metrics["hit_rate@5"])) if metrics["hit_rate@5"] else 0.0,
        "precision@1": float(np.mean(metrics["precision@1"])) if metrics["precision@1"] else 0.0,
        "precision@3": float(np.mean(metrics["precision@3"])) if metrics["precision@3"] else 0.0,
        "precision@5": float(np.mean(metrics["precision@5"])) if metrics["precision@5"] else 0.0,
        "recall@1": float(np.mean(metrics["recall@1"])) if metrics["recall@1"] else 0.0,
        "recall@3": float(np.mean(metrics["recall@3"])) if metrics["recall@3"] else 0.0,
        "recall@5": float(np.mean(metrics["recall@5"])) if metrics["recall@5"] else 0.0,
        "mrr": float(np.mean(metrics["mrr"])) if metrics["mrr"] else 0.0,
        "ndcg@10": float(np.mean(metrics["ndcg@10"])) if metrics["ndcg@10"] else 0.0,
        "total_queries": metrics["total_queries"],
        "successful_queries": total_successful,
        "failed_queries": metrics["failed_queries"],
        "total_time": metrics["total_time"],
        "avg_latency": metrics["total_time"] / total_successful if total_successful > 0 else 0,
    }

    return aggregated


def print_report(metrics: Dict[str, Any], title: str = "Evaluation Results"):
    """Print a formatted evaluation report."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Total Queries:     {metrics['total_queries']}")
    if metrics.get("failed_queries", 0) > 0:
        print(f"  Successful:        {metrics.get('successful_queries', metrics['total_queries'])}")
        print(f"  Failed:            {metrics['failed_queries']}")
    print(f"  --- Recall-Oriented ---")
    print(f"  Hit Rate@1:        {metrics['hit_rate@1']:.4f} ({metrics['hit_rate@1']*100:.1f}%)")
    print(f"  Hit Rate@3:        {metrics['hit_rate@3']:.4f} ({metrics['hit_rate@3']*100:.1f}%)")
    print(f"  Hit Rate@5:        {metrics['hit_rate@5']:.4f} ({metrics['hit_rate@5']*100:.1f}%)")
    print(f"  Recall@1:          {metrics.get('recall@1', 0):.4f}")
    print(f"  Recall@5:          {metrics.get('recall@5', 0):.4f}")
    print(f"  --- Precision-Oriented ---")
    print(f"  Precision@1:       {metrics.get('precision@1', 0):.4f}")
    print(f"  Precision@5:       {metrics.get('precision@5', 0):.4f}")
    print(f"  --- Ranking ---")
    print(f"  MRR:               {metrics['mrr']:.4f}")
    print(f"  NDCG@10:           {metrics['ndcg@10']:.4f}")
    print(f"  --- Performance ---")
    print(f"  Total Time:        {metrics['total_time']:.3f}s")
    print(f"  Avg Latency:       {metrics['avg_latency']*1000:.1f}ms")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Hermes-RAG Evaluation")
    parser.add_argument("--baseline", action="store_true", help="Run baseline comparison")
    parser.add_argument("--no-reranker", action="store_true", help="Disable reranker")
    parser.add_argument("--output", type=str, default="", help="Output JSON file")
    args = parser.parse_args()

    from src.config import get_config
    from src.core.pipeline_factory import build_pipeline
    from evaluation.dataset import EvaluationDataset

    cfg = get_config()

    # Build pipeline
    use_reranker = not args.no_reranker
    pipeline = build_pipeline(
        config=cfg,
        use_reranker=use_reranker,
    )

    # Load dataset
    dataset = EvaluationDataset()
    print(f"Loaded {dataset.size()} evaluation queries")

    # Evaluate Hermes-RAG
    metrics = evaluate_pipeline(pipeline, dataset, use_reranker=use_reranker)
    print_report(metrics, f"Hermes-RAG (Reranker={'On' if use_reranker else 'Off'})")

    # Baseline comparison
    if args.baseline:
        print("\n--- Baseline Comparison ---")
        # Dense-only (no RRF, no reranker)
        pipeline_baseline = build_pipeline(
            config=cfg,
            query_expansion_enabled=False,
            use_reranker=False,
            use_sparse=False,
            use_cache=False,
        )
        baseline_metrics = evaluate_pipeline(pipeline_baseline, dataset, use_reranker=False)
        print_report(baseline_metrics, "Baseline (Dense-only, no RRF, no Reranker)")

        # Comparison summary
        print(f"\n--- Improvement Summary ---")
        improvement = (metrics["hit_rate@5"] - baseline_metrics["hit_rate@5"]) / max(baseline_metrics["hit_rate@5"], 0.001) * 100
        print(f"  Hit Rate@5 Improvement: {improvement:+.1f}%")

    # Save results
    if args.output:
        output = {
            "hermes_rag": metrics,
            "config": {
                "use_reranker": use_reranker,
            },
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()