"""Reproducible benchmark harness for Hermes-RAG.

Runs retrieval evaluation against a temporary index that is built in-process,
so vector and BM25 indexes are guaranteed to be populated. This removes the
previous cross-process persistence footgun where BM25 was silently empty.
"""

import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_DOCS = [
    "ml_arch.md",
    "ml_classification.md",
    "ml_clustering.md",
    "ml_cv.md",
    "ml_eval.md",
    "ml_intro.md",
    "ml_neural.md",
    "ml_regression.md",
    "ml_supervised.md",
    "py_basics.md",
    "py_features.md",
]


def create_benchmark_env(
    docs_dir: Path | None = None,
    extra_docs: bool = False,
) -> tuple:
    """Build a pipeline with a fresh temporary index and ingest benchmark docs.

    Returns:
        (pipeline, config, temp_dir_handle)
    """
    from src.config import get_config, reset_config

    reset_config()
    cfg = get_config()

    temp_handle = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    temp_path = Path(temp_handle.name)
    cfg.chromadb.persist_directory = str(temp_path / "chroma")
    cfg.chromadb.document_store_path = str(temp_path / "document_store.db")
    cfg.bm25.fallback_db_path = str(temp_path / "bm25_fallback.db")

    from src.core.pipeline_factory import build_pipeline

    pipeline = build_pipeline(config=cfg)
    index_manager = pipeline.index_manager

    docs_root = docs_dir or (PROJECT_ROOT / "evaluation" / "data" / "sample_docs")
    doc_names = list(BENCHMARK_DOCS)
    if extra_docs:
        doc_names += sorted(
            p.name
            for p in docs_root.iterdir()
            if p.is_file() and p.suffix.lower() in (".md", ".txt") and p.name not in doc_names
        )

    from src.core.chunking.hierarchical_chunker import HierarchicalChunker
    from src.core.ingestion.parser_factory import ParserFactory

    chunker = HierarchicalChunker(
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
        semantic_threshold=cfg.chunking.semantic_threshold,
        min_chunk_size=cfg.chunking.min_chunk_size,
        max_section_size=cfg.chunking.max_section_size,
        embedding_model=cfg.embedding.model_name,
        embedding_device=cfg.embedding.device,
    )

    total = 0
    for name in doc_names:
        fp = docs_root / name
        if not fp.exists():
            continue
        try:
            parsed = ParserFactory.get_parser(str(fp)).parse(str(fp))
            chunks = chunker.chunk(
                text=parsed["text"],
                source_name=name,
                headings=parsed.get("metadata", {}).get("headings"),
            )
            if chunks:
                index_manager.ingest_chunks(chunks)
                total += len(chunks)
        except Exception as e:  # noqa: BLE001
            print(f"  skipped {name}: {e}")

    print(f"Benchmark index ready: {total} chunks")
    return pipeline, cfg, temp_handle


def _variant_name(key: str) -> str:
    return {
        "baseline": "baseline_dense_only",
        "rrf": "rrf_no_rerank",
        "rerank": "heuristic_rerank",
        "cross": "cross_encoder",
    }.get(key, key)


def run_benchmark(
    output_path: str | None = None,
    variants: list[str] | None = None,
    extra_docs: bool = False,
) -> dict[str, Any]:
    """Run the canonical benchmark and optionally save JSON results."""
    if variants is None:
        variants = ["baseline", "rrf", "rerank"]

    from evaluation.dataset import EvaluationDataset
    from evaluation.eval import (
        evaluate_pipeline,
        hit_rate_at_k,
        mrr,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
    )
    from src.core.pipeline_factory import build_pipeline

    pipeline, cfg, temp_handle = create_benchmark_env(extra_docs=extra_docs)
    dataset = EvaluationDataset()
    data = dataset.load()
    print(f"Loaded {len(data)} evaluation queries")

    results: dict[str, Any] = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "embedding_model": cfg.embedding.model_name,
            "reranker_enabled_default": cfg.reranking.enabled,
            "dataset_version": "2.4.0",
            "metrics": ["hit_rate", "mrr", "ndcg", "doc_hit_rate"],
            "dataset_size": len(data),
            "chunks": pipeline.index_manager.document_store.count() if pipeline.index_manager.document_store else 0,
            "variants": [_variant_name(v) for v in variants],
        },
        "variants": {},
        "difficulty_breakdown": {},
        "failed_queries": [],
        "doc_failed_queries": [],
    }

    for key in variants:
        name = _variant_name(key)
        if key == "baseline":
            p = build_pipeline(
                config=cfg,
                query_expansion_enabled=False,
                use_reranker=False,
                use_sparse=False,
                use_cache=False,
            )
        elif key == "rrf":
            p = build_pipeline(config=cfg, use_reranker=False, use_cache=False)
        elif key == "rerank":
            p = build_pipeline(config=cfg, use_reranker=True, use_cache=False)
        elif key == "cross":
            p = build_pipeline(config=cfg, use_reranker=True, use_cache=False)
        else:
            continue

        print(f"\n--- Evaluating variant: {name} ---")
        metrics = evaluate_pipeline(p, dataset, use_reranker=(key in ("rerank", "cross")), top_k=5)
        results["variants"][name] = metrics

    # Difficulty breakdown for the strongest available variant
    breakdown_key = None
    for preferred in ("rrf_no_rerank", "heuristic_rerank"):
        if preferred in results["variants"]:
            breakdown_key = preferred
            break
    if breakdown_key is None:
        breakdown_key = next(iter(results["variants"]))

    p_final = build_pipeline(config=cfg, use_reranker=(breakdown_key == "heuristic_rerank"), use_cache=False)
    for difficulty in ("easy", "medium", "hard"):
        items = [item for item in data if item.get("difficulty", "easy") == difficulty]
        if not items:
            continue
        acc = {
            "hit_rate@1": [], "hit_rate@3": [], "hit_rate@5": [],
            "precision@1": [], "precision@5": [],
            "recall@1": [], "recall@5": [],
            "mrr": [], "ndcg@10": [],
            "doc_hit_rate@1": [], "doc_hit_rate@3": [], "doc_hit_rate@5": [],
            "doc_mrr": [],
        }
        for item in items:
            result = p_final.retrieve(query=item["query"], top_k=5, use_reranker=(breakdown_key == "heuristic_rerank"))
            retrieved = result.get("results", [])
            relevant = item["relevant_chunk_ids"]
            relevant_docs = item.get("relevant_doc_ids") or sorted({rid.rsplit("_", 1)[0] for rid in relevant})
            seen_docs: set[str] = set()
            doc_retrieved = []
            for r in retrieved:
                doc = r["chunk_id"].rsplit("_", 1)[0]
                if doc not in seen_docs:
                    seen_docs.add(doc)
                    doc_retrieved.append({"chunk_id": doc})
            acc["hit_rate@1"].append(hit_rate_at_k(retrieved, relevant, 1))
            acc["hit_rate@3"].append(hit_rate_at_k(retrieved, relevant, 3))
            acc["hit_rate@5"].append(hit_rate_at_k(retrieved, relevant, 5))
            acc["precision@1"].append(precision_at_k(retrieved, relevant, 1))
            acc["precision@5"].append(precision_at_k(retrieved, relevant, 5))
            acc["recall@1"].append(recall_at_k(retrieved, relevant, 1))
            acc["recall@5"].append(recall_at_k(retrieved, relevant, 5))
            acc["mrr"].append(mrr(retrieved, relevant))
            acc["ndcg@10"].append(ndcg_at_k(retrieved, relevant, 10))
            acc["doc_hit_rate@1"].append(hit_rate_at_k(doc_retrieved, relevant_docs, 1))
            acc["doc_hit_rate@3"].append(hit_rate_at_k(doc_retrieved, relevant_docs, 3))
            acc["doc_hit_rate@5"].append(hit_rate_at_k(doc_retrieved, relevant_docs, 5))
            acc["doc_mrr"].append(mrr(doc_retrieved, relevant_docs))

        results["difficulty_breakdown"][difficulty] = {
            k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in acc.items()
        }
        results["difficulty_breakdown"][difficulty]["queries"] = len(items)

    # Failed queries against the primary variant
    if breakdown_key in results["variants"]:
        p_primary = build_pipeline(config=cfg, use_reranker=(breakdown_key == "heuristic_rerank"), use_cache=False)
        for item in data:
            result = p_primary.retrieve(query=item["query"], top_k=5, use_reranker=(breakdown_key == "heuristic_rerank"))
            retrieved = [r["chunk_id"] for r in result.get("results", [])[:5]]
            if not any(rid in retrieved for rid in item["relevant_chunk_ids"]):
                results["failed_queries"].append(
                    {
                        "query": item["query"],
                        "difficulty": item.get("difficulty", "easy"),
                        "expected": item["relevant_chunk_ids"],
                        "retrieved_top5": retrieved,
                    }
                )
            relevant_docs = item.get("relevant_doc_ids") or sorted(
                {rid.rsplit("_", 1)[0] for rid in item["relevant_chunk_ids"]}
            )
            retrieved_docs = sorted({rid.rsplit("_", 1)[0] for rid in retrieved})
            if not any(doc in retrieved_docs for doc in relevant_docs):
                results["doc_failed_queries"].append(
                    {
                        "query": item["query"],
                        "difficulty": item.get("difficulty", "easy"),
                        "expected_docs": relevant_docs,
                        "retrieved_docs": retrieved_docs,
                    }
                )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResults saved to {out}")

    temp_handle.cleanup()
    return results


def print_benchmark_report(results: dict[str, Any]) -> None:
    """Pretty-print a benchmark report to the console."""
    print("\n" + "=" * 70)
    print("  HERMES-RAG BENCHMARK REPORT")
    print("=" * 70)
    meta = results.get("meta", {})
    print(
        f"  Date: {meta.get('date')} | Python: {meta.get('python')} | "
        f"Model: {meta.get('embedding_model')}"
    )
    print(f"  Queries: {meta.get('dataset_size')} | Chunks: {meta.get('chunks')}")

    rows = []
    for name, metrics in results.get("variants", {}).items():
        rows.append(
            (
                name,
                metrics.get("hit_rate@1", 0),
                metrics.get("hit_rate@3", 0),
                metrics.get("hit_rate@5", 0),
                metrics.get("doc_hit_rate@5", 0),
                metrics.get("mrr", 0),
                metrics.get("ndcg@10", 0),
                metrics.get("avg_latency", 0),
            )
        )

    print(f"\n  {'Variant':<22}{'H@1':>8}{'H@3':>8}{'H@5':>8}{'DocH5':>8}{'MRR':>8}{'NDCG':>8}{'Lat(ms)':>10}")
    print("  " + "-" * 78)
    for row in rows:
        name, h1, h3, h5, dh5, mrr_v, ndcg_v, lat = row
        print(
            f"  {name:<22}{h1*100:>7.1f}%{h3*100:>7.1f}%{h5*100:>7.1f}%"
            f"{dh5*100:>7.1f}%{mrr_v:>8.4f}{ndcg_v:>8.4f}{lat*1000:>9.1f}"
        )

    print("\n  Difficulty breakdown (primary variant):")
    for diff, metrics in results.get("difficulty_breakdown", {}).items():
        print(
            f"    [{diff.upper()}] H@1={metrics.get('hit_rate@1', 0)*100:.1f}% "
            f"H@3={metrics.get('hit_rate@3', 0)*100:.1f}% "
            f"H@5={metrics.get('hit_rate@5', 0)*100:.1f}% "
            f"DocH5={metrics.get('doc_hit_rate@5', 0)*100:.1f}% "
            f"MRR={metrics.get('mrr', 0):.4f} (n={metrics.get('queries', 0)})"
        )

    failed = results.get("failed_queries", [])
    if failed:
        print(f"\n  Failed chunk-level queries: {len(failed)}")
        for item in failed[:10]:
            print(f"    [{item['difficulty']}] {item['query']}")
    else:
        print("\n  Failed chunk-level queries: 0")
    doc_failed = results.get("doc_failed_queries", [])
    if doc_failed:
        print(f"  Failed document-level queries: {len(doc_failed)}")
        for item in doc_failed[:10]:
            print(f"    [{item['difficulty']}] {item['query']}")
    else:
        print("  Failed document-level queries: 0")
    print("=" * 70)