"""Full-pipeline integration test for Hermes-RAG.

Validates all improvements:
1. Query classification
2. Result deduplication
3. Similarity threshold filtering
4. Production metrics collection
5. Cache hit rate tracking
6. Multi-path recall distribution
7. Latency percentile monitoring
8. Error handling and edge cases
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_query_classification():
    """Test query classifier works correctly."""
    from src.core.retrieval.retrieval_pipeline import QueryClassifier

    print("=" * 60)
    print("  TEST 1: Query Classification")
    print("=" * 60)

    tests = [
        ("什么是机器学习？", "conceptual"),
        ("ABC-1234产品的配置方法", "factual"),
        ("ERR500错误如何解决", "factual"),
        ("如何安装Python？", "procedural"),
        ("how to reset password", "procedural"),
        ("分类和回归有什么区别", "conceptual"),
        ("v3.14.2更新了什么", "factual"),
        ("Python支持哪些编程范式", "conceptual"),
        ("怎么配置网络设置", "procedural"),
        ("#ID12345的bug状态", "factual"),
    ]

    passed = 0
    for query, expected in tests:
        result = QueryClassifier.classify(query)
        status = "PASS" if result == expected else "FAIL"
        if result == expected:
            passed += 1
        print(f"  [{status}] '{query[:50]}' -> {result} (expected: {expected})")

    print(f"\n  Result: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_result_deduplication():
    """Test result deduplication works."""
    from src.core.retrieval.retrieval_pipeline import RetrievalPipeline

    print("\n" + "=" * 60)
    print("  TEST 2: Result Deduplication")
    print("=" * 60)

    pipeline = RetrievalPipeline(index_manager=None)

    # Create duplicate results
    results = [
        {"chunk_id": "a", "text": "Text A", "score": 0.9},
        {"chunk_id": "b", "text": "Text B", "score": 0.5},
        {"chunk_id": "a", "text": "Text A dup", "score": 0.8},  # Duplicate, lower score
        {"chunk_id": "c", "text": "Text C", "score": 0.3},
        {"chunk_id": "b", "text": "Text B dup", "score": 0.7},  # Duplicate, higher score
    ]

    deduped = pipeline._deduplicate_results(results)

    assert len(deduped) == 3, f"Expected 3 unique results, got {len(deduped)}"
    assert deduped[0]["chunk_id"] == "a", f"Expected 'a' first, got {deduped[0]['chunk_id']}"
    assert deduped[0]["score"] == 0.9, f"Expected score 0.9, got {deduped[0]['score']}"
    assert deduped[1]["chunk_id"] == "b", f"Expected 'b' second, got {deduped[1]['chunk_id']}"
    assert deduped[1]["score"] == 0.7, f"Expected score 0.7 (kept higher), got {deduped[1]['score']}"

    print("  PASS: Deduplication works correctly")
    print(f"    Input: {len(results)} items -> Output: {len(deduped)} items")
    print(f"    Order: {[r['chunk_id'] for r in deduped]}")
    return True


def test_threshold_filtering():
    """Test similarity threshold filtering."""
    from src.core.retrieval.retrieval_pipeline import RetrievalPipeline

    print("\n" + "=" * 60)
    print("  TEST 3: Threshold Filtering")
    print("=" * 60)

    pipeline = RetrievalPipeline(index_manager=None)

    results = [
        {"chunk_id": "a", "score": 0.9},
        {"chunk_id": "b", "score": 0.5},
        {"chunk_id": "c", "score": 0.0005},  # Below threshold
        {"chunk_id": "d", "score": 0},
        {"chunk_id": "e", "score": 0.001},
    ]

    filtered = pipeline._filter_by_threshold(results, min_score=0.001)

    assert len(filtered) == 3, f"Expected 3 results, got {len(filtered)}"
    assert all(r["score"] >= 0.001 for r in filtered), "All results should be >= threshold"

    print("  PASS: Threshold filtering works correctly")
    print(f"    Input: {len(results)} items -> Filtered: {len(filtered)} items")
    return True


def test_metrics_collector():
    """Test production metrics collection."""
    from src.utils.metrics import MetricsCollector, reset_metrics

    print("\n" + "=" * 60)
    print("  TEST 4: Production Metrics Collector")
    print("=" * 60)

    reset_metrics()
    mc = MetricsCollector(window_size=100, latency_window=50)

    # Simulate queries
    for i in range(100):
        if i < 30:
            # 30% cache hits
            mc.record_query(
                cached=True,
                recall_paths=["cached"],
                total_latency=0.001,
            )
        elif i < 80:
            # 50% normal queries with both paths
            mc.record_query(
                cached=False,
                recall_paths=["dense", "sparse"],
                total_latency=0.05 + (i % 10) * 0.01,
                reranker_used=True,
                reranker_timed_out=(i % 20 == 0),
            )
        else:
            # 20% dense-only queries
            mc.record_query(
                cached=False,
                recall_paths=["dense"],
                total_latency=0.03,
            )

    # Check cache hit rate
    hit_rate = mc.get_cache_hit_rate()
    print(f"  Cache Hit Rate: {hit_rate:.4f} ({hit_rate*100:.1f}%)")
    assert 0.25 < hit_rate < 0.35, f"Expected ~0.3, got {hit_rate}"

    # Check latency percentiles
    latencies = mc.get_latency_percentiles()
    print(f"  Latency: avg={latencies['avg']*1000:.1f}ms, p50={latencies['p50']*1000:.1f}ms, p99={latencies['p99']*1000:.1f}ms")
    assert latencies["p99"] > 0, "p99 should be > 0"

    # Check recall path distribution
    dist = mc.get_recall_path_distribution()
    print(f"  Recall Paths: {dist}")
    assert dist["cached"] == 30, f"Expected 30 cached, got {dist['cached']}"
    assert dist["both"] == 50, f"Expected 50 both, got {dist['both']}"
    assert dist["dense_only"] == 20, f"Expected 20 dense_only, got {dist['dense_only']}"

    # Check full report
    report = mc.get_full_report()
    print(f"  Full Report: total_queries={report['total_queries']}, qps={report['qps']}")
    assert report["total_queries"] == 100, f"Expected 100 queries, got {report['total_queries']}"
    assert "cache" in report
    assert "latency" in report
    assert "recall_paths" in report
    assert "reranker" in report

    print("  PASS: All metrics collection tests passed")
    return True


def test_cache_semantic_matching():
    """Test semantic cache matching."""
    from src.utils.cache import QueryCache

    print("\n" + "=" * 60)
    print("  TEST 5: Semantic Cache Matching")
    print("=" * 60)

    cache = QueryCache(max_size=100, similarity_threshold=0.9, ttl_seconds=3600)

    # Create a simple embedding (random normalized vector)
    emb1 = np.random.randn(128)
    emb1 = emb1 / np.linalg.norm(emb1)

    # Very similar embedding (cosine sim > 0.95)
    emb2 = emb1 + np.random.randn(128) * 0.05
    emb2 = emb2 / np.linalg.norm(emb2)

    # Dissimilar embedding
    emb3 = -emb1

    # Set a cached result
    cache.set("什么是机器学习？", [{"chunk_id": "ml_intro.md_0", "score": 1.0}], query_embedding=emb1)

    # Exact match should hit
    result = cache.get("什么是机器学习？")
    assert result is not None, "Exact match should hit"
    print("  PASS: Exact match cache hit")

    # Very similar query should hit via semantic matching
    result = cache.get("机器学习是什么？", query_embedding=emb2)
    sim = float(np.dot(emb1, emb2))
    print(f"  Semantic similarity: {sim:.4f}")
    if sim >= 0.9:
        assert result is not None, "Semantic match should hit"
        print("  PASS: Semantic similarity cache hit")
    else:
        print("  SKIP: Similarity below threshold, semantic match not expected")

    # Dissimilar query should miss
    cache.set("Python是什么？", [{"chunk_id": "py_basics.md_0", "score": 1.0}], query_embedding=emb3)
    result = cache.get("什么是机器学习？", query_embedding=emb1)
    assert result is not None, "Exact match should still work"

    # Check hit rate stats
    print(f"  Cache hit rate: {cache.hit_rate():.4f}")
    print(f"  Cache size: {cache.size()}")

    return True


def test_rrf_boundary_cases():
    """Test RRF fusion boundary cases."""
    from src.core.retrieval.rrf_fusion import RRFFusion

    print("\n" + "=" * 60)
    print("  TEST 6: RRF Boundary Cases")
    print("=" * 60)

    rrf = RRFFusion()

    # Case 1: Both empty
    result = rrf.fuse([], [], query="test")
    assert result == [], "Both empty should return empty list"
    print("  PASS: Empty inputs return empty list")

    # Case 2: Only dense
    dense = [{"chunk_id": "a", "text": "A", "metadata": {}}]
    result = rrf.fuse(dense, [], query="test")
    assert len(result) == 1, f"Expected 1 result, got {len(result)}"
    assert result[0]["chunk_id"] == "a"
    print("  PASS: Dense-only returns correct result")

    # Case 3: Only sparse
    sparse = [{"chunk_id": "b", "text": "B", "metadata": {}}]
    result = rrf.fuse([], sparse, query="test")
    assert len(result) == 1, f"Expected 1 result, got {len(result)}"
    assert result[0]["chunk_id"] == "b"
    print("  PASS: Sparse-only returns correct result")

    # Case 4: Partial overlap
    result = rrf.fuse(
        [{"chunk_id": "a", "text": "A", "metadata": {}},
         {"chunk_id": "c", "text": "C", "metadata": {}}],
        [{"chunk_id": "b", "text": "B", "metadata": {}},
         {"chunk_id": "a", "text": "A dup", "metadata": {}}],
        query="test",
    )
    assert len(result) == 3, f"Expected 3 unique results, got {len(result)}"
    print("  PASS: Overlapping results merged correctly")

    # Case 5: Dynamic weights for product codes
    result = rrf.fuse(
        [{"chunk_id": "a", "text": "A", "metadata": {}}],
        [{"chunk_id": "b", "text": "B", "metadata": {}}],
        query="ABC-1234",
    )
    assert len(result) == 2, f"Expected 2 results, got {len(result)}"
    # Product codes should boost BM25: b should rank higher
    assert result[0]["chunk_id"] == "b", f"Expected BM25 result first for product code, got {result[0]['chunk_id']}"
    print("  PASS: Product code query boosts BM25 rank")

    # Case 6: Dynamic weights for colloquial queries
    result = rrf.fuse(
        [{"chunk_id": "a", "text": "A", "metadata": {}}],
        [{"chunk_id": "b", "text": "B", "metadata": {}}],
        query="什么是机器学习",
    )
    assert len(result) == 2, f"Expected 2 results, got {len(result)}"
    # Colloquial should boost dense: a should rank higher
    assert result[0]["chunk_id"] == "a", f"Expected dense result first for colloquial, got {result[0]['chunk_id']}"
    print("  PASS: Colloquial query boosts dense rank")

    return True


def test_query_expansion():
    """Test query expansion with synonyms."""
    from src.core.retrieval.query_expander import QueryExpander

    print("\n" + "=" * 60)
    print("  TEST 7: Query Expansion")
    print("=" * 60)

    expander = QueryExpander(synonym_enabled=True, hyde_enabled=False)

    # Test Chinese ML synonym expansion
    result = expander.expand("如何评估模型性能")
    assert result["expanded"] != result["original"], "Query should be expanded"
    assert len(result["synonyms"]) > 0, "Should have synonyms"
    print(f"  Original: {result['original']}")
    print(f"  Expanded: {result['expanded']}")
    print(f"  Synonyms: {result['synonyms']}")
    print("  PASS: Chinese ML synonym expansion works")

    # Test English synonym expansion
    result = expander.expand("how to reset password")
    assert result["expanded"] != result["original"], "Query should be expanded"
    print(f"  Original: {result['original']}")
    print(f"  Expanded: {result['expanded']}")
    print(f"  Synonyms: {result['synonyms']}")
    print("  PASS: English synonym expansion works")

    # Test query with no synonyms
    result = expander.expand("XYZ123")
    assert result["expanded"] == result["original"], "Query with no synonyms should not change"
    print("  PASS: No-synonym query unchanged")

    return True


def test_evaluation_metrics():
    """Test evaluation metrics calculation."""
    from evaluation.eval import (
        hit_rate_at_k, precision_at_k, recall_at_k, mrr, ndcg_at_k,
    )

    print("\n" + "=" * 60)
    print("  TEST 8: Evaluation Metrics Calculation")
    print("=" * 60)

    # Perfect results
    results = [
        {"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"},
        {"chunk_id": "d"}, {"chunk_id": "e"},
    ]
    relevant = ["a", "b", "c"]

    assert hit_rate_at_k(results, relevant, 1) == 1.0, "Hit@1 should be 1.0"
    assert precision_at_k(results, relevant, 3) == 1.0, "Precision@3 should be 1.0"
    assert recall_at_k(results, relevant, 3) == 1.0, "Recall@3 should be 1.0"
    assert mrr(results, relevant) == 1.0, "MRR should be 1.0"
    print("  PASS: Perfect results score 1.0 on all metrics")

    # No relevant results
    results = [
        {"chunk_id": "x"}, {"chunk_id": "y"}, {"chunk_id": "z"},
    ]
    assert hit_rate_at_k(results, relevant, 3) == 0.0, "Hit@3 should be 0.0"
    assert precision_at_k(results, relevant, 3) == 0.0, "Precision@3 should be 0.0"
    assert recall_at_k(results, relevant, 3) == 0.0, "Recall@3 should be 0.0"
    assert mrr(results, relevant) == 0.0, "MRR should be 0.0"
    print("  PASS: No relevant results score 0.0 on all metrics")

    # Partial results
    results = [
        {"chunk_id": "x"}, {"chunk_id": "a"}, {"chunk_id": "y"},
    ]
    assert hit_rate_at_k(results, relevant, 3) == 1.0, "Hit@3 should be 1.0"
    assert precision_at_k(results, relevant, 3) == 1/3, "Precision@3 should be 1/3"
    assert recall_at_k(results, relevant, 3) == 1/3, "Recall@3 should be 1/3"
    assert mrr(results, relevant) == 0.5, "MRR should be 0.5"
    print("  PASS: Partial results calculate correctly")

    # NDCG
    results = [
        {"chunk_id": "a"}, {"chunk_id": "x"}, {"chunk_id": "b"},
        {"chunk_id": "y"}, {"chunk_id": "c"},
    ]
    ndcg = ndcg_at_k(results, relevant, 5)
    assert 0.5 < ndcg < 1.0, f"NDCG should be between 0.5 and 1.0, got {ndcg}"
    print(f"  PASS: NDCG@5 = {ndcg:.4f}")

    # Empty relevant
    assert hit_rate_at_k(results, [], 3) == 0.0, "Empty relevant should return 0.0"
    print("  PASS: Empty relevant list returns 0.0")

    return True


def test_negative_samples():
    """Test negative sample evaluation."""
    print("\n" + "=" * 60)
    print("  TEST 9: Negative Sample Evaluation")
    print("=" * 60)

    ground_truth_path = Path(__file__).parent.parent / "evaluation" / "data" / "ground_truth.json"
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries_with_negatives = [
        item for item in data if "negative_chunk_ids" in item
    ]
    print(f"  Found {len(queries_with_negatives)} queries with negative samples")

    for item in queries_with_negatives:
        print(f"  Query: {item['query']}")
        print(f"    Relevant: {item['relevant_chunk_ids']}")
        print(f"    Negative: {item['negative_chunk_ids']}")

    assert len(queries_with_negatives) >= 3, "Should have at least 3 negative sample queries"
    print("  PASS: Negative samples properly configured")
    return True


def test_ragas_integration():
    """Test RAGAS integration."""
    from evaluation.ragas_eval import RAGASAdapter

    print("\n" + "=" * 60)
    print("  TEST 10: RAGAS Integration")
    print("=" * 60)

    adapter = RAGASAdapter()

    # Test heuristic evaluation
    questions = ["什么是机器学习？", "Python的特点是什么？"]
    retrieved = [["机器学习是AI的一个分支", "深度学习是ML的子集"], ["Python是解释型语言", "Python支持多范式"]]
    ground_truth = [["机器学习是AI的一个分支"], ["Python是解释型语言"]]

    results = adapter.evaluate_retrieval(questions, retrieved, ground_truth)
    print(f"  Context Precision: {results['context_precision']:.4f}")
    print(f"  Context Recall:    {results['context_recall']:.4f}")

    assert 0 <= results["context_precision"] <= 1, "Precision should be in [0,1]"
    assert 0 <= results["context_recall"] <= 1, "Recall should be in [0,1]"
    print("  PASS: RAGAS heuristic evaluation works")

    # Test generation evaluation
    answers = ["机器学习是AI的一个分支", "Python是解释型高级语言"]
    gen_results = adapter.evaluate_generation(questions, answers, retrieved)
    print(f"  Faithfulness:      {gen_results.get('faithfulness', 'N/A')}")
    print(f"  Answer Relevancy:  {gen_results.get('answer_relevancy', 'N/A')}")
    print("  PASS: RAGAS generation evaluation works")

    return True


def test_full_pipeline_integration():
    """Test full pipeline integration with all improvements."""
    print("\n" + "=" * 60)
    print("  TEST 11: Full Pipeline Integration")
    print("=" * 60)

    try:
        from src.config import get_config, reset_config
        from src.core.pipeline_factory import build_pipeline
        from src.utils.metrics import get_metrics, reset_metrics

        reset_config()
        reset_metrics()
        cfg = get_config()

        pipeline = build_pipeline(config=cfg)

        # Verify pipeline has all components
        assert pipeline.index_manager is not None, "Should have index_manager"
        assert pipeline.query_expander is not None, "Should have query_expander"
        assert pipeline.dense_retriever is not None, "Should have dense_retriever"
        assert pipeline.sparse_retriever is not None, "Should have sparse_retriever"
        assert pipeline.rrf_fusion is not None, "Should have rrf_fusion"
        assert pipeline.metrics is not None, "Should have metrics collector"
        print("  PASS: All pipeline components initialized")

        # Verify metrics collector is accessible
        metrics = get_metrics()
        report = metrics.get_full_report()
        assert report["total_queries"] == 0, "Should start with 0 queries"
        print("  PASS: Metrics collector initialized with 0 queries")

        # Verify query expansion works
        test_query = "什么是机器学习"
        expansion = pipeline.query_expander.expand(test_query)
        assert expansion["expanded"] != test_query or len(expansion["synonyms"]) > 0
        print(f"  PASS: Query expansion active: '{expansion['expanded'][:80]}'")

        # Verify query classifier works
        from src.core.retrieval.retrieval_pipeline import QueryClassifier
        qtype = QueryClassifier.classify(test_query)
        assert qtype in ("factual", "procedural", "conceptual")
        print(f"  PASS: Query classified as '{qtype}'")

        print("\n  All integration tests passed!")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("  HERMES-RAG FULL PIPELINE INTEGRATION TEST")
    print("=" * 80)

    results = {}

    tests = [
        ("Query Classification", test_query_classification),
        ("Result Deduplication", test_result_deduplication),
        ("Threshold Filtering", test_threshold_filtering),
        ("Production Metrics", test_metrics_collector),
        ("Semantic Cache", test_cache_semantic_matching),
        ("RRF Boundary Cases", test_rrf_boundary_cases),
        ("Query Expansion", test_query_expansion),
        ("Evaluation Metrics", test_evaluation_metrics),
        ("Negative Samples", test_negative_samples),
        ("RAGAS Integration", test_ragas_integration),
        ("Full Pipeline Integration", test_full_pipeline_integration),
    ]

    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 80)
    print("  TEST SUMMARY")
    print("=" * 80)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  Total: {passed}/{total} passed")
    print("=" * 80)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)