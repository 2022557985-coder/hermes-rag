"""Standalone evaluation runner for Hermes-RAG."""
import sys
import json
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

# Step 1: Ingest sample documents first
print("=" * 60)
print("Step 1: Ingesting sample documents...")
print("=" * 60)

from src.config import get_config, reset_config
reset_config()
cfg = get_config()

from src.core.ingestion.parser_factory import ParserFactory
from src.core.chunking.hierarchical_chunker import HierarchicalChunker
from src.core.pipeline_factory import build_pipeline

# Build pipeline first to get index_manager
pipeline = build_pipeline(config=cfg)
index_manager = pipeline.index_manager

# Clear existing data
index_manager.clear()

chunker = HierarchicalChunker(
    chunk_size=cfg.chunking.chunk_size,
    chunk_overlap=cfg.chunking.chunk_overlap,
    semantic_threshold=cfg.chunking.semantic_threshold,
    min_chunk_size=cfg.chunking.min_chunk_size,
    max_section_size=cfg.chunking.max_section_size,
    embedding_model=cfg.embedding.model_name,
    embedding_device=cfg.embedding.device,
)

sample_docs_dir = Path(__file__).parent / "evaluation" / "data" / "sample_docs"
files = list(sample_docs_dir.glob("*.md"))
print(f"Found {len(files)} sample documents")

total_chunks = 0
for file_path in files:
    parser = ParserFactory.get_parser(str(file_path))
    parsed = parser.parse(str(file_path))
    source_name = file_path.name
    chunks = chunker.chunk(
        text=parsed["text"],
        source_name=source_name,
        headings=parsed.get("metadata", {}).get("headings"),
    )
    if chunks:
        counts = index_manager.ingest_chunks(chunks)
        total_chunks += len(chunks)
        print(f"  {source_name}: {len(chunks)} chunks, vector={counts['vector_count']}, bm25={counts['bm25_count']}")

print(f"Total ingested: {total_chunks} chunks")

# Step 2: Run evaluation
print("\n" + "=" * 60)
print("Step 2: Running evaluation...")
print("=" * 60)

from evaluation.eval import evaluate_pipeline, print_report, hit_rate_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k
from evaluation.dataset import EvaluationDataset

dataset = EvaluationDataset()
data = dataset.load()
print(f"Loaded {dataset.size()} evaluation queries")

# == Test 1: Hermes-RAG Full Pipeline (with reranker) ==
print("\n--- Test 1: Hermes-RAG Full Pipeline ---")
metrics_full = evaluate_pipeline(pipeline, dataset, use_reranker=True)
print_report(metrics_full, "Hermes-RAG (Full Pipeline with Reranker)")

# == Test 2: Hermes-RAG without Reranker ==
pipeline_no_rerank = build_pipeline(
    config=cfg,
    query_expansion_enabled=True,
    use_reranker=False,
    use_sparse=True,
    use_cache=False,
)
print("\n--- Test 2: Hermes-RAG without Reranker ---")
metrics_no_rerank = evaluate_pipeline(pipeline_no_rerank, dataset, use_reranker=False)
print_report(metrics_no_rerank, "Hermes-RAG (No Reranker)")

# == Test 3: Baseline (Dense-only, no RRF, no Reranker) ==
print("\n--- Test 3: Baseline (Dense-only) ---")
pipeline_baseline = build_pipeline(
    config=cfg,
    query_expansion_enabled=False,
    use_reranker=False,
    use_sparse=False,
    use_cache=False,
)
baseline_metrics = evaluate_pipeline(pipeline_baseline, dataset, use_reranker=False)
print_report(baseline_metrics, "Baseline (Dense-only, no Query Expansion)")

# == Test 4: Dense + Sparse without Reranker ==
print("\n--- Test 4: Dense + Sparse (RRF, no Reranker) ---")
pipeline_rrf = build_pipeline(
    config=cfg,
    query_expansion_enabled=True,
    use_reranker=False,
    use_sparse=True,
    use_cache=False,
)
rrf_metrics = evaluate_pipeline(pipeline_rrf, dataset, use_reranker=False)
print_report(rrf_metrics, "Dense + Sparse (RRF, no Reranker)")

# == Summary ==
print("\n" + "=" * 70)
print("  COMPARATIVE SUMMARY")
print("=" * 70)
print(f"{'Metric':<15} {'Baseline':>10} {'+RRF':>10} {'+Rerank':>10} {'Full':>10}")
print("-" * 55)
for metric in ["hit_rate@1", "hit_rate@3", "hit_rate@5", "precision@1", "precision@5", "recall@1", "recall@5", "mrr", "ndcg@10"]:
    print(f"{metric:<15} {baseline_metrics[metric]:>10.4f} {rrf_metrics[metric]:>10.4f} {metrics_no_rerank[metric]:>10.4f} {metrics_full[metric]:>10.4f}")

print("\n--- Improvement over Baseline ---")
for metric in ["hit_rate@1", "hit_rate@3", "hit_rate@5", "precision@1", "precision@5", "mrr", "ndcg@10"]:
    imp = (metrics_full[metric] - baseline_metrics[metric]) / max(baseline_metrics[metric], 0.001) * 100
    print(f"  {metric:<15}: {imp:+.1f}%")

# == Difficulty-based analysis ==
print("\n" + "=" * 70)
print("  DIFFICULTY-BASED BREAKDOWN (Full Pipeline)")
print("=" * 70)

for difficulty in ["easy", "medium", "hard"]:
    difficulty_items = [item for item in data if item.get("difficulty", "easy") == difficulty]
    if not difficulty_items:
        continue
    
    hit1_list, hit3_list, hit5_list, mrr_list, ndcg_list = [], [], [], [], []
    prec1_list, prec5_list, rec1_list, rec5_list = [], [], [], []
    for item in difficulty_items:
        result = pipeline.retrieve(query=item["query"], top_k=5, use_reranker=True)
        results = result.get("results", [])
        relevant_ids = item["relevant_chunk_ids"]
        hit1_list.append(hit_rate_at_k(results, relevant_ids, 1))
        hit3_list.append(hit_rate_at_k(results, relevant_ids, 3))
        hit5_list.append(hit_rate_at_k(results, relevant_ids, 5))
        prec1_list.append(precision_at_k(results, relevant_ids, 1))
        prec5_list.append(precision_at_k(results, relevant_ids, 5))
        rec1_list.append(recall_at_k(results, relevant_ids, 1))
        rec5_list.append(recall_at_k(results, relevant_ids, 5))
        mrr_list.append(mrr(results, relevant_ids))
        ndcg_list.append(ndcg_at_k(results, relevant_ids, 10))
    
    print(f"\n  [{difficulty.upper()}] {len(difficulty_items)} queries:")
    print(f"    Hit Rate@1:  {np.mean(hit1_list):.4f} ({np.mean(hit1_list)*100:.1f}%)")
    print(f"    Hit Rate@3:  {np.mean(hit3_list):.4f} ({np.mean(hit3_list)*100:.1f}%)")
    print(f"    Hit Rate@5:  {np.mean(hit5_list):.4f} ({np.mean(hit5_list)*100:.1f}%)")
    print(f"    Precision@1: {np.mean(prec1_list):.4f}")
    print(f"    Precision@5: {np.mean(prec5_list):.4f}")
    print(f"    Recall@1:    {np.mean(rec1_list):.4f}")
    print(f"    Recall@5:    {np.mean(rec5_list):.4f}")
    print(f"    MRR:         {np.mean(mrr_list):.4f}")
    print(f"    NDCG@10:     {np.mean(ndcg_list):.4f}")

# == Per-query analysis ==
print("\n" + "=" * 70)
print("  PER-QUERY ANALYSIS (Full Pipeline)")
print("=" * 70)

# Clear cache before per-query analysis
if pipeline.cache:
    pipeline.cache.clear()

failed_queries = []
for item in data:
    query = item["query"]
    relevant_ids = item["relevant_chunk_ids"]
    difficulty = item.get("difficulty", "easy")
    
    result = pipeline.retrieve(query=query, top_k=5, use_reranker=True)
    results = result.get("results", [])
    
    retrieved_ids = [r["chunk_id"] for r in results[:5]]
    hit = any(rid in retrieved_ids for rid in relevant_ids)
    
    # Find rank of first relevant result
    rank = None
    for i, r in enumerate(results, 1):
        if r["chunk_id"] in relevant_ids:
            rank = i
            break
    
    status = "PASS" if hit else "FAIL"
    if not hit:
        failed_queries.append((query, difficulty, relevant_ids, retrieved_ids))
    
    print(f"\n[{status}] [{difficulty}] {query}")
    print(f"  Relevant: {relevant_ids}")
    print(f"  Retrieved (top-5): {retrieved_ids}")
    print(f"  Rank of 1st relevant: {rank if rank else 'N/A'}")
    scores_list = [round(r.get("score", 0), 4) for r in results[:5]]
    print(f"  Scores: {scores_list}")

# == Failed queries summary ==
if failed_queries:
    print("\n" + "=" * 70)
    print(f"  FAILED QUERIES ({len(failed_queries)} failed)")
    print("=" * 70)
    for query, difficulty, relevant, retrieved in failed_queries:
        print(f"\n  [{difficulty}] {query}")
        print(f"    Expected: {relevant}")
        print(f"    Got:      {retrieved}")

print("\nEvaluation complete!")