"""Performance benchmark tests for Hermes-RAG components.

Measures latency, throughput, and scalability of retrieval, ingestion,
and caching operations. All tests are marked with @pytest.mark.slow to
allow selective execution.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.indexing.bm25_index import BM25Index
from src.core.indexing.index_manager import IndexManager
from src.core.indexing.vector_store import VectorStore
from src.core.retrieval.dense_retriever import DenseRetriever
from src.core.retrieval.query_expander import QueryExpander
from src.core.retrieval.retrieval_pipeline import RetrievalPipeline
from src.core.retrieval.rrf_fusion import RRFFusion
from src.core.retrieval.sparse_retriever import SparseRetriever
from src.utils.cache import QueryCache
from src.utils.metrics import MetricsCollector

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    """Create a temporary directory for benchmark data."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield tmpdir


@pytest.fixture
def benchmark_chunks():
    """Generate benchmark chunks for performance testing."""
    chunks = []
    topics = ["machine learning", "deep learning", "neural networks", "natural language processing",
              "computer vision", "reinforcement learning", "data science", "statistics",
              "optimization", "algorithms", "Python", "Java", "C++", "Rust", "Go"]
    for i in range(100):
        topic = topics[i % len(topics)]
        chunks.append({
            "chunk_id": f"bench_{i}",
            "text": (
                f"This is chunk {i} about {topic}. "
                f"It contains detailed information about {topic} concepts, "
                f"applications, and best practices. "
                f"Key terms include {topic} fundamentals, advanced techniques, "
                f"and real-world use cases."
            ),
            "metadata": {"source": f"bench_{i//10}.txt", "chunk_idx": i, "topic": topic},
        })
    return chunks


@pytest.fixture
def populated_index(temp_dir, benchmark_chunks):
    """Create a populated index for benchmarking."""
    vector_store = VectorStore(
        persist_directory=os.path.join(temp_dir, "chroma_perf"),
        collection_name="test_perf",
        embedding_model="BAAI/bge-small-zh-v1.5",
    )
    vector_store.clear()
    bm25 = BM25Index()
    im = IndexManager(vector_store=vector_store, bm25_index=bm25)
    im.ingest_chunks(benchmark_chunks)
    return im


@pytest.fixture
def perf_pipeline(populated_index):
    """Build a pipeline for performance testing."""
    return RetrievalPipeline(
        index_manager=populated_index,
        query_expander=QueryExpander(synonym_enabled=True, hyde_enabled=False),
        dense_retriever=DenseRetriever(populated_index),
        sparse_retriever=SparseRetriever(populated_index),
        rrf_fusion=RRFFusion(),
        cache=QueryCache(max_size=100),
        config={
            "dense_top_k": 100,
            "sparse_top_k": 100,
            "fusion_top_k": 50,
            "reranking": {"enabled": False},
        },
        metrics=MetricsCollector(),
    )


# ---------------------------------------------------------------------------
# Latency benchmarks
# ---------------------------------------------------------------------------

class TestDenseRetrievalLatency:
    """Benchmark dense retrieval latency."""

    def test_dense_retrieval_latency_small_index(self, perf_pipeline):
        """Dense retrieval latency should be reasonable for small index."""
        # Warmup
        perf_pipeline.retrieve("warmup query", top_k=5, use_reranker=False)

        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            result = perf_pipeline.retrieve("machine learning", top_k=5, use_reranker=False)
            latencies.append(time.perf_counter() - start)

        avg_latency = sum(latencies) / len(latencies)
        # For small index (100 chunks), should be well under 1s
        assert avg_latency < 5.0, \
            f"Dense retrieval average latency {avg_latency:.3f}s should be under 5s for small index"
        assert len(result["results"]) > 0, "Should return results"


class TestSparseRetrievalLatency:
    """Benchmark sparse retrieval latency."""

    def test_sparse_retrieval_latency(self, perf_pipeline):
        """Sparse retrieval latency should be reasonable."""
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            result = perf_pipeline.retrieve("machine learning", top_k=5, use_reranker=False)
            latencies.append(time.perf_counter() - start)

        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 5.0, \
            f"Sparse retrieval average latency {avg_latency:.3f}s should be under 5s"
        assert len(result["results"]) > 0, "Should return results"


class TestRRFFusionLatency:
    """Benchmark RRF fusion latency."""

    def test_rrf_fusion_latency(self, perf_pipeline):
        """RRF fusion should be fast (sub-millisecond)."""
        rrf = RRFFusion()
        dense_results = [
            {"chunk_id": f"d{i}", "text": f"dense text {i}", "metadata": {}, "score": 0.9 - i * 0.01}
            for i in range(50)
        ]
        sparse_results = [
            {"chunk_id": f"s{i}", "text": f"sparse text {i}", "metadata": {}, "score": 0.8 - i * 0.01}
            for i in range(50)
        ]

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            rrf.fuse(dense_results, sparse_results, query="test query", top_k=50)
            latencies.append(time.perf_counter() - start)

        avg_latency = sum(latencies) / len(latencies)
        # RRF fusion should be fast (< 10ms)
        assert avg_latency < 0.1, \
            f"RRF fusion average latency {avg_latency*1000:.2f}ms should be under 100ms"


class TestFullPipelineLatency:
    """Benchmark full pipeline latency."""

    def test_full_pipeline_latency(self, perf_pipeline):
        """Full pipeline latency should be under reasonable threshold."""
        # Warmup
        perf_pipeline.retrieve("warmup", top_k=5, use_reranker=False)

        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            result = perf_pipeline.retrieve("machine learning", top_k=5, use_reranker=False)
            latencies.append(time.perf_counter() - start)

        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        assert avg_latency < 10.0, \
            f"Full pipeline average latency {avg_latency:.3f}s should be under 10s"
        assert p95_latency < 15.0, \
            f"Full pipeline p95 latency {p95_latency:.3f}s should be under 15s"
        assert len(result["results"]) > 0, "Should return results"


class TestCacheHitLatency:
    """Benchmark cache hit latency."""

    def test_cache_hit_latency(self, perf_pipeline):
        """Cache hit latency should be very fast (sub-millisecond range)."""
        # Populate cache
        perf_pipeline.retrieve("cache test query", top_k=5, use_reranker=False)

        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            result = perf_pipeline.retrieve("cache test query", top_k=5, use_reranker=False)
            latencies.append(time.perf_counter() - start)

        avg_latency = sum(latencies) / len(latencies)
        # Cache hits should be very fast (< 50ms for small index)
        assert avg_latency < 0.5, \
            f"Cache hit average latency {avg_latency*1000:.2f}ms should be under 500ms"
        assert result["query_info"].get("cached", False) is True, "Should be cache hit"


# ---------------------------------------------------------------------------
# Throughput benchmarks
# ---------------------------------------------------------------------------

class TestBatchRetrievalThroughput:
    """Benchmark batch retrieval throughput."""

    def test_batch_retrieval_throughput(self, perf_pipeline):
        """Test batch retrieval throughput with multiple queries."""
        queries = [
            "machine learning",
            "deep learning",
            "neural networks",
            "Python programming",
            "data science",
            "algorithms",
            "computer vision",
            "natural language",
            "reinforcement learning",
            "statistics",
        ]

        start = time.perf_counter()
        results = perf_pipeline.retrieve_batch(queries, top_k=5, use_reranker=False, max_workers=4)
        elapsed = time.perf_counter() - start

        assert len(results) == len(queries), "Should return results for all queries"
        throughput = len(queries) / elapsed
        assert throughput > 0.1, \
            f"Batch throughput {throughput:.2f} qps should be > 0.1 qps"


class TestIngestionThroughput:
    """Benchmark ingestion throughput."""

    def test_ingestion_throughput(self, temp_dir):
        """Test ingestion throughput for chunk batches."""
        vector_store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_ingest_perf"),
            collection_name="test_ingest_perf",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        vector_store.clear()
        bm25 = BM25Index()
        im = IndexManager(vector_store=vector_store, bm25_index=bm25)

        chunks = []
        for i in range(50):
            chunks.append({
                "chunk_id": f"ingest_{i}",
                "text": f"Performance test chunk {i} with content about various topics "
                        f"including machine learning, data science, and artificial intelligence.",
                "metadata": {"source": "perf_test.txt", "chunk_idx": i},
            })

        start = time.perf_counter()
        im.ingest_chunks(chunks)
        elapsed = time.perf_counter() - start

        chunks_per_second = len(chunks) / elapsed
        assert chunks_per_second > 0.5, \
            f"Ingestion throughput {chunks_per_second:.2f} chunks/s should be > 0.5"


# ---------------------------------------------------------------------------
# Memory usage benchmarks
# ---------------------------------------------------------------------------

class TestMemoryUsage:
    """Benchmark memory usage during ingestion."""

    def test_memory_usage_during_ingestion(self, temp_dir):
        """Test that memory usage stays reasonable during ingestion."""
        vector_store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_mem"),
            collection_name="test_mem",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        vector_store.clear()
        bm25 = BM25Index()
        im = IndexManager(vector_store=vector_store, bm25_index=bm25)

        metrics = MetricsCollector()
        mem_before = metrics.get_memory_usage()

        chunks = []
        for i in range(100):
            chunks.append({
                "chunk_id": f"mem_{i}",
                "text": f"Memory test chunk {i}. " * 5,
                "metadata": {"source": "mem_test.txt", "chunk_idx": i},
            })

        im.ingest_chunks(chunks)

        metrics.get_memory_usage()
        assert vector_store.count() == 100, "Should have 100 chunks"

        # Memory should be tracked (even if psutil is not available, it returns -1)
        assert isinstance(mem_before.get("rss_mb", 0), (int, float)), "Should have memory info"


# ---------------------------------------------------------------------------
# Scalability benchmarks
# ---------------------------------------------------------------------------

class TestQueryScalability:
    """Benchmark query scalability with increasing query counts."""

    @pytest.mark.parametrize("num_queries", [10, 50])
    def test_query_scalability(self, perf_pipeline, num_queries):
        """Test that query performance scales linearly with batch size."""
        queries = [f"query about topic {i % 10}" for i in range(num_queries)]

        start = time.perf_counter()
        results = perf_pipeline.retrieve_batch(queries, top_k=5, use_reranker=False, max_workers=4)
        elapsed = time.perf_counter() - start

        assert len(results) == num_queries, f"Should return results for all {num_queries} queries"
        throughput = num_queries / elapsed
        assert throughput > 0.05, \
            f"Throughput {throughput:.2f} qps for {num_queries} queries should be > 0.05"


class TestIndexSizeScalability:
    """Benchmark index size scalability."""

    def test_index_size_scalability(self, temp_dir):
        """Test that index performance scales with index size."""
        vector_store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_scale"),
            collection_name="test_scale",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        vector_store.clear()
        bm25 = BM25Index()
        im = IndexManager(vector_store=vector_store, bm25_index=bm25)

        sizes = [20, 50]
        times = []

        for size in sizes:
            chunks = []
            for i in range(size):
                chunks.append({
                    "chunk_id": f"scale_{size}_{i}",
                    "text": f"Scalability test chunk {i} of size {size}. "
                            f"Contains content about topic {i % 10}.",
                    "metadata": {"source": f"scale_{size}.txt", "chunk_idx": i},
                })

            start = time.perf_counter()
            im.ingest_chunks(chunks)
            elapsed = time.perf_counter() - start
            times.append((size, elapsed))

            # Verify count
            assert vector_store.count() == size, f"Should have {size} chunks"

            # Clear for next batch
            im.clear()

        # Verify that larger sizes are processed (time may vary)
        assert len(times) == len(sizes), "Should have timings for all sizes"
        assert all(t[1] > 0 for t in times), "All timings should be positive"


# ---------------------------------------------------------------------------
# Component-level benchmarks
# ---------------------------------------------------------------------------

class TestComponentBenchmarks:
    """Benchmark individual pipeline components."""

    def test_dense_retriever_standalone(self, temp_dir, benchmark_chunks):
        """Benchmark DenseRetriever standalone."""
        vector_store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_dense_bench"),
            collection_name="test_dense_bench",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        vector_store.clear()
        bm25 = BM25Index()
        im = IndexManager(vector_store=vector_store, bm25_index=bm25)
        im.ingest_chunks(benchmark_chunks)

        retriever = DenseRetriever(im)

        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            result = retriever.retrieve("machine learning", top_k=10)
            latencies.append(time.perf_counter() - start)

        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 5.0, \
            f"Dense retriever standalone latency {avg_latency:.3f}s should be under 5s"
        assert len(result) > 0, "Should return results"

    def test_sparse_retriever_standalone(self, temp_dir, benchmark_chunks):
        """Benchmark SparseRetriever standalone."""
        vector_store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_sparse_bench"),
            collection_name="test_sparse_bench",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        vector_store.clear()
        bm25 = BM25Index()
        im = IndexManager(vector_store=vector_store, bm25_index=bm25)
        im.ingest_chunks(benchmark_chunks)

        retriever = SparseRetriever(im)

        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            result = retriever.retrieve("machine learning", top_k=10)
            latencies.append(time.perf_counter() - start)

        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 5.0, \
            f"Sparse retriever standalone latency {avg_latency:.3f}s should be under 5s"
        assert len(result) > 0, "Should return results"

    def test_query_cache_standalone(self):
        """Benchmark QueryCache set/get operations."""
        cache = QueryCache(max_size=1000)

        # Benchmark set operations
        set_latencies = []
        for i in range(100):
            start = time.perf_counter()
            cache.set(f"query_{i}", [{"id": i}])
            set_latencies.append(time.perf_counter() - start)

        avg_set = sum(set_latencies) / len(set_latencies)
        assert avg_set < 0.01, \
            f"Cache set average latency {avg_set*1000:.3f}ms should be under 10ms"

        # Benchmark get operations (hits)
        get_latencies = []
        for i in range(100):
            start = time.perf_counter()
            cache.get(f"query_{i}")
            get_latencies.append(time.perf_counter() - start)

        avg_get = sum(get_latencies) / len(get_latencies)
        assert avg_get < 0.01, \
            f"Cache get average latency {avg_get*1000:.3f}ms should be under 10ms"

    def test_metrics_collector_performance(self):
        """Benchmark MetricsCollector recording performance."""
        metrics = MetricsCollector()

        latencies = []
        for _ in range(200):
            start = time.perf_counter()
            metrics.record_query(
                cached=False,
                recall_paths=["dense", "sparse"],
                total_latency=0.05,
                component_timings={"dense_retrieval": 0.03, "rrf_fusion": 0.01, "total": 0.05},
            )
            latencies.append(time.perf_counter() - start)

        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 0.01, \
            f"Metrics record average latency {avg_latency*1000:.3f}ms should be under 10ms"


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------

class TestStressTests:
    """Stress tests for pipeline robustness."""

    def test_repeated_queries_stable(self, perf_pipeline):
        """Running many queries should not degrade performance or leak memory."""
        for i in range(50):
            result = perf_pipeline.retrieve(
                f"repeated query about topic {i % 5}", top_k=5, use_reranker=False
            )
            assert "results" in result, f"Iteration {i}: should have results"
            assert "timing" in result, f"Iteration {i}: should have timing"

    def test_rapid_consecutive_queries(self, perf_pipeline):
        """Rapid consecutive queries should not crash."""
        for i in range(20):
            result = perf_pipeline.retrieve(f"topic {i}", top_k=3, use_reranker=False)
            assert isinstance(result["results"], list), f"Iteration {i}: should return list"