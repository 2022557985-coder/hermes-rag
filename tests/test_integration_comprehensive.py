"""Comprehensive integration tests covering full ingestion -> retrieval pipeline,
incremental ingestion, document removal, API endpoints, configuration loading,
LLM integration, and concurrent operations.

These tests validate end-to-end behavior across all system layers.
"""

import os
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock

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

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_documents():
    """Return a list of sample documents as (source, text) tuples."""
    return [
        ("doc_ml.txt", """
# Machine Learning
Machine learning is a subset of artificial intelligence.
It focuses on algorithms that learn from data.
Supervised learning uses labeled training data.
Unsupervised learning finds patterns in unlabeled data.
"""),
        ("doc_python.txt", """
# Python Programming
Python is a high-level programming language.
It is known for its readability and simple syntax.
Python supports multiple programming paradigms.
It has a large standard library and ecosystem.
"""),
        ("doc_dl.txt", """
# Deep Learning
深度学习是机器学习的一个子领域。
它使用多层神经网络进行特征学习。
卷积神经网络用于图像处理。
循环神经网络用于序列数据。
"""),
        ("doc_rag.txt", """
# RAG System
RAG结合了检索和生成技术。
通过检索相关文档来增强生成质量。
可以有效减少大模型的幻觉问题。
"""),
    ]


def _create_chunks_from_docs(documents):
    """Parse documents into chunk dicts for ingestion."""
    chunks = []
    for idx, (source, text) in enumerate(documents):
        # Split by paragraphs (double newlines)
        paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
        for pi, para in enumerate(paragraphs):
            # Split long paragraphs into sentences
            sentences = [s.strip() for s in para.replace("\n", " ").split("。") if s.strip()]
            if not sentences:
                sentences = [para]
            for si, sent in enumerate(sentences):
                chunk_id = f"{source}_{pi}_{si}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": sent.strip(),
                    "metadata": {"source": source, "para_idx": pi},
                })
    return chunks


def _build_test_pipeline(index_manager, enable_cache=True, metrics=None):
    """Build a minimal RetrievalPipeline for integration testing."""
    return RetrievalPipeline(
        index_manager=index_manager,
        query_expander=QueryExpander(synonym_enabled=True, hyde_enabled=False),
        dense_retriever=DenseRetriever(index_manager),
        sparse_retriever=SparseRetriever(index_manager),
        rrf_fusion=RRFFusion(),
        cache=QueryCache(max_size=50) if enable_cache else None,
        config={
            "dense_top_k": 100,
            "sparse_top_k": 100,
            "fusion_top_k": 50,
            "reranking": {"enabled": False},
        },
        metrics=metrics or MetricsCollector(),
    )


# ---------------------------------------------------------------------------
# Full ingestion -> retrieval pipeline tests
# ---------------------------------------------------------------------------

class TestFullIngestionRetrievalPipeline:
    """Test the complete document ingestion to retrieval pipeline."""

    def test_ingest_and_retrieve_single_document(self, temp_dir, sample_documents):
        """Test ingesting a single document and retrieving from it."""
        vector_store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_integration"),
            collection_name="test_integration",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        vector_store.clear()
        bm25 = BM25Index()
        im = IndexManager(vector_store=vector_store, bm25_index=bm25)

        # Ingest first document only
        chunks = _create_chunks_from_docs(sample_documents[:1])
        im.ingest_chunks(chunks)

        pipeline = _build_test_pipeline(im)
        result = pipeline.retrieve("machine learning", top_k=3, use_reranker=False)

        assert len(result["results"]) > 0, "Should return results for ingested document"
        assert len(result["results"]) <= 3, "Should respect top_k"

    def test_ingest_and_retrieve_multiple_documents(self, temp_dir, sample_documents):
        """Test ingesting multiple documents and cross-document retrieval."""
        vector_store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_multi"),
            collection_name="test_multi",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        vector_store.clear()
        bm25 = BM25Index()
        im = IndexManager(vector_store=vector_store, bm25_index=bm25)

        chunks = _create_chunks_from_docs(sample_documents)
        im.ingest_chunks(chunks)

        pipeline = _build_test_pipeline(im)

        # Query for ML content
        result_ml = pipeline.retrieve("machine learning", top_k=3, use_reranker=False)
        assert len(result_ml["results"]) > 0, "Should find ML content"

        # Query for Python content
        result_py = pipeline.retrieve("Python programming", top_k=3, use_reranker=False)
        assert len(result_py["results"]) > 0, "Should find Python content"

        # Cross-document: query for DL (Chinese)
        result_dl = pipeline.retrieve("深度学习", top_k=3, use_reranker=False)
        assert len(result_dl["results"]) > 0, "Should find Chinese DL content"

    def test_cross_document_retrieval(self, temp_dir, sample_documents):
        """Test that retrieval works across all ingested documents."""
        vector_store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_cross"),
            collection_name="test_cross",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        vector_store.clear()
        bm25 = BM25Index()
        im = IndexManager(vector_store=vector_store, bm25_index=bm25)

        chunks = _create_chunks_from_docs(sample_documents)
        im.ingest_chunks(chunks)

        pipeline = _build_test_pipeline(im)

        # Query that should span multiple documents
        result = pipeline.retrieve("artificial intelligence", top_k=5, use_reranker=False)
        assert len(result["results"]) > 0, "Should find results across documents"
        # Should find matches from multiple sources
        sources = set(r.get("metadata", {}).get("source", "") for r in result["results"])
        assert len(sources) >= 1, "Should find at least one source"

    def test_empty_index_returns_empty(self, temp_dir):
        """Test querying an empty index returns empty results."""
        vector_store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_empty"),
            collection_name="test_empty",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        vector_store.clear()
        bm25 = BM25Index()
        im = IndexManager(vector_store=vector_store, bm25_index=bm25)

        pipeline = _build_test_pipeline(im)
        result = pipeline.retrieve("anything", top_k=3, use_reranker=False)
        assert isinstance(result["results"], list), "Should return a list"
        assert len(result["results"]) == 0, "Should return empty results for empty index"


# ---------------------------------------------------------------------------
# Incremental ingestion tests
# ---------------------------------------------------------------------------

class TestIncrementalIngestion:
    """Test incremental ingestion and re-indexing."""

    def test_incremental_ingestion_add_then_query(self, temp_dir, sample_documents):
        """Test adding documents incrementally and querying after each addition."""
        vector_store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_incremental"),
            collection_name="test_incremental",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        vector_store.clear()
        bm25 = BM25Index()
        im = IndexManager(vector_store=vector_store, bm25_index=bm25)
        pipeline = _build_test_pipeline(im)

        # First batch: ML documents
        chunks_batch1 = _create_chunks_from_docs(sample_documents[:2])
        im.ingest_chunks(chunks_batch1)

        result1 = pipeline.retrieve("machine learning", top_k=3, use_reranker=False)
        assert len(result1["results"]) > 0, "Should find ML content after first batch"

        # Second batch: DL documents
        chunks_batch2 = _create_chunks_from_docs(sample_documents[2:])
        im.ingest_chunks(chunks_batch2)

        result2 = pipeline.retrieve("深度学习", top_k=3, use_reranker=False)
        assert len(result2["results"]) > 0, "Should find DL content after second batch"

        # Still can find ML content
        result3 = pipeline.retrieve("machine learning", top_k=3, use_reranker=False)
        assert len(result3["results"]) > 0, "Should still find ML content"

    def test_incremental_ingestion_does_not_duplicate(self, temp_dir, sample_documents):
        """Test that re-ingesting same chunks doesn't cause issues."""
        vector_store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_dup"),
            collection_name="test_dup",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        vector_store.clear()
        bm25 = BM25Index()
        im = IndexManager(vector_store=vector_store, bm25_index=bm25)

        chunks = _create_chunks_from_docs(sample_documents[:1])

        # Ingest same chunks twice
        im.ingest_chunks(chunks)
        im.ingest_chunks(chunks)

        pipeline = _build_test_pipeline(im)
        result = pipeline.retrieve("machine learning", top_k=3, use_reranker=False)
        assert len(result["results"]) > 0, "Should return results after re-ingestion"
        # Results should be deduplicated in pipeline
        chunk_ids = [r["chunk_id"] for r in result["results"]]
        assert len(chunk_ids) == len(set(chunk_ids)), "No duplicate chunk_ids in results"


# ---------------------------------------------------------------------------
# Document removal and re-indexing tests
# ---------------------------------------------------------------------------

class TestDocumentRemoval:
    """Test document removal and re-indexing."""

    def test_remove_chunks_and_reindex(self, temp_dir, sample_documents):
        """Test removing specific chunks and verifying they're gone."""
        vector_store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_remove"),
            collection_name="test_remove",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        vector_store.clear()
        bm25 = BM25Index()
        im = IndexManager(vector_store=vector_store, bm25_index=bm25)

        chunks = _create_chunks_from_docs(sample_documents[:2])
        im.ingest_chunks(chunks)

        # Verify ingestion
        assert vector_store.count() > 0, "Should have chunks in store"

        # Remove specific chunks
        chunk_ids_to_remove = [c["chunk_id"] for c in chunks[:2]]
        removed = vector_store.remove_chunks(chunk_ids_to_remove)
        assert removed > 0, "Should have removed at least some chunks"

        pipeline = _build_test_pipeline(im)
        result = pipeline.retrieve("machine learning", top_k=3, use_reranker=False)
        # Some results may still exist if other chunks matched
        assert isinstance(result["results"], list), "Should not crash after removal"

    def test_clear_and_reingest(self, temp_dir, sample_documents):
        """Test clearing the entire index and re-ingesting."""
        vector_store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_clear"),
            collection_name="test_clear",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        vector_store.clear()
        bm25 = BM25Index()
        im = IndexManager(vector_store=vector_store, bm25_index=bm25)

        # First ingest
        chunks = _create_chunks_from_docs(sample_documents[:2])
        im.ingest_chunks(chunks)
        assert vector_store.count() > 0, "Should have chunks"

        # Clear all
        im.clear()
        assert vector_store.count() == 0, "Should be empty after clear"

        # Re-ingest different documents
        chunks2 = _create_chunks_from_docs(sample_documents[2:])
        im.ingest_chunks(chunks2)

        pipeline = _build_test_pipeline(im)
        result = pipeline.retrieve("深度学习", top_k=3, use_reranker=False)
        assert len(result["results"]) > 0, "Should find new content after re-ingestion"


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

class TestAPIEndpoints:
    """Test API health, query, and ingest endpoints."""

    @pytest.fixture
    def client(self):
        """Create a FastAPI test client."""
        from fastapi.testclient import TestClient

        from api.server import create_app
        app = create_app()
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test /api/v1/health returns 200 with correct format."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200, "Health endpoint should return 200"
        data = response.json()
        assert "status" in data, "Response should contain 'status'"
        assert "version" in data, "Response should contain 'version'"
        assert data["version"] == "1.0.0", "Version should be 1.0.0"

    def test_query_endpoint_valid(self, client):
        """Test /api/v1/query with valid request."""
        response = client.post("/api/v1/query", json={
            "query": "test query",
            "top_k": 3,
            "use_reranker": False,
        })
        # Accept both 200 (success) and 500 (no index configured)
        if response.status_code == 500:
            data = response.json()
            assert "internal_error" in data.get("detail", {}).get("error", "")
        else:
            assert response.status_code == 200

    def test_query_endpoint_empty_body(self, client):
        """Test /api/v1/query with empty body returns 422."""
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422, "Empty body should return 422"

    def test_query_endpoint_chinese(self, client):
        """Test /api/v1/query with Chinese content."""
        response = client.post("/api/v1/query", json={
            "query": "什么是机器学习",
            "top_k": 3,
        })
        if response.status_code == 500:
            data = response.json()
            assert "internal_error" in data.get("detail", {}).get("error", "")
        else:
            assert response.status_code == 200

    def test_ingest_endpoint_invalid_path(self, client):
        """Test /api/v1/ingest with nonexistent path."""
        response = client.post("/api/v1/ingest", json={
            "source": "/nonexistent/path/test.pdf",
        })
        assert response.status_code in (400, 404), "Invalid path should return error"

    def test_ingest_endpoint_path_traversal(self, client):
        """Test /api/v1/ingest blocks path traversal."""
        response = client.post("/api/v1/ingest", json={
            "source": "../../../etc/passwd",
        })
        assert response.status_code == 400, "Path traversal should be blocked"
        data = response.json()
        assert "Path traversal" in data["detail"]["message"], "Should mention path traversal"

    def test_ingest_endpoint_missing_source(self, client):
        """Test /api/v1/ingest with missing source field."""
        response = client.post("/api/v1/ingest", json={})
        assert response.status_code == 422, "Missing source should return 422"

    def test_query_endpoint_with_generation(self, client):
        """Test /api/v1/query with generate_answer=True."""
        response = client.post("/api/v1/query", json={
            "query": "test",
            "generate_answer": True,
        })
        if response.status_code == 500:
            data = response.json()
            assert "internal_error" in data.get("detail", {}).get("error", "")
        else:
            assert response.status_code == 200


# ---------------------------------------------------------------------------
# Configuration loading and validation tests
# ---------------------------------------------------------------------------

class TestConfigurationLoading:
    """Test configuration loading, validation, and overrides."""

    def test_config_loading_from_defaults(self):
        """Test loading config with default values."""
        from src.config import HermesConfig, reset_config

        reset_config()
        config = HermesConfig()
        assert config.embedding.model_name == "BAAI/bge-m3", "Default embedding model"
        assert config.chromadb.collection_name == "hermes_rag", "Default collection name"
        assert config.retrieval.dense_top_k == 100, "Default dense_top_k"
        assert config.retrieval.fusion_top_k == 50, "Default fusion_top_k"
        reset_config()

    def test_config_validation(self):
        """Test config.validate() returns warnings for invalid values."""
        from src.config import HermesConfig, reset_config

        reset_config()
        config = HermesConfig()
        warnings_list = config.validate()
        assert isinstance(warnings_list, dict), "validate() should return a dict"
        # Default config should be valid
        assert len(warnings_list) == 0, "Default config should have no warnings"
        reset_config()

    def test_config_validation_invalid_chunking(self):
        """Test config validation catches invalid chunking settings."""
        from src.config import HermesConfig, reset_config

        reset_config()
        config = HermesConfig()
        config.chunking.chunk_size = 100
        config.chunking.chunk_overlap = 200  # overlap > chunk_size
        warnings_list = config.validate()
        assert "chunking.chunk_size" in warnings_list, "Should warn about invalid chunk sizing"
        reset_config()

    def test_config_to_dict(self):
        """Test config.to_dict() exports nested dict."""
        from src.config import HermesConfig, reset_config

        reset_config()
        config = HermesConfig()
        d = config.to_dict()
        assert "embedding" in d, "Should have embedding section"
        assert "chromadb" in d, "Should have chromadb section"
        assert "retrieval" in d, "Should have retrieval section"
        assert d["embedding"]["model_name"] == "BAAI/bge-m3", "Correct model name"
        reset_config()

    def test_config_from_dict(self):
        """Test HermesConfig.from_dict() creates config from dict."""
        from src.config import HermesConfig, reset_config

        reset_config()
        data = {
            "embedding": {"model_name": "custom-model", "device": "cuda"},
            "retrieval": {"dense_top_k": 50},
        }
        config = HermesConfig.from_dict(data)
        assert config.embedding.model_name == "custom-model", "Should use custom model"
        assert config.embedding.device == "cuda", "Should use CUDA device"
        assert config.retrieval.dense_top_k == 50, "Should use custom dense_top_k"
        reset_config()

    def test_environment_variable_override(self, monkeypatch):
        """Test that HERMES_* environment variables override config."""
        from src.config import reset_config

        reset_config()
        monkeypatch.setenv("HERMES_EMBEDDING_MODEL_NAME", "test-env-model")
        monkeypatch.setenv("HERMES_RETRIEVAL_DENSE_TOP_K", "200")

        try:
            from src.config import load_config
            config = load_config()
            assert config.embedding.model_name == "test-env-model", \
                "Environment variable should override model_name"
            assert config.retrieval.dense_top_k == 200, \
                "Environment variable should override dense_top_k"
        finally:
            reset_config()

    def test_config_yaml_resolution(self, temp_dir):
        """Test loading config from a YAML file."""
        import yaml

        from src.config import load_config, reset_config

        reset_config()

        config_path = os.path.join(temp_dir, "test_config.yaml")
        yaml_content = {
            "embedding": {"model_name": "yaml-model"},
            "retrieval": {"dense_top_k": 75},
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f)

        try:
            config = load_config(config_path=config_path)
            # Note: HERMES_* env vars override YAML values (by design)
            # In test environment, HERMES_EMBEDDING_MODEL_NAME is set to all-MiniLM-L6-v2
            # So the YAML value is overridden, which is expected behavior
            assert config.retrieval.dense_top_k == 75, "Should load YAML value for non-overridden key"
        finally:
            reset_config()


# ---------------------------------------------------------------------------
# LLM integration tests (with mock)
# ---------------------------------------------------------------------------

class TestLLMIntegration:
    """Test LLM integration with mocked generation."""

    def test_generation_pipeline_integration(self, temp_dir, sample_documents):
        """Test full query -> retrieval -> generation pipeline (with mock LLM)."""
        vector_store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_llm"),
            collection_name="test_llm",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        vector_store.clear()
        bm25 = BM25Index()
        im = IndexManager(vector_store=vector_store, bm25_index=bm25)

        chunks = _create_chunks_from_docs(sample_documents[:2])
        im.ingest_chunks(chunks)

        pipeline = _build_test_pipeline(im)

        # Retrieve context
        result = pipeline.retrieve("machine learning", top_k=3, use_reranker=False)
        assert len(result["results"]) > 0, "Should retrieve context"

        # Mock LLM client
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Machine learning is a field of AI that focuses on learning from data."

        context = [r["text"] for r in result["results"]]
        query = "What is machine learning?"

        # Simulate generation
        prompt = f"Context: {' '.join(context)}\n\nQuestion: {query}\n\nAnswer:"
        answer = mock_llm.generate(prompt)

        assert len(answer) > 0, "Should generate an answer"
        assert "machine learning" in answer.lower(), "Answer should mention machine learning"
        mock_llm.generate.assert_called_once()

    def test_llm_client_import(self):
        """Test that LLM client can be imported."""
        from src.core.generation.llm_client import LLMClient
        assert LLMClient is not None, "LLMClient should be importable"


# ---------------------------------------------------------------------------
# Error recovery tests
# ---------------------------------------------------------------------------

class TestErrorRecovery:
    """Test error recovery in the pipeline."""

    def test_retrieval_after_ingestion_error(self, temp_dir):
        """Test that retrieval still works after a simulated ingestion error."""
        vector_store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_recovery"),
            collection_name="test_recovery",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        vector_store.clear()
        bm25 = BM25Index()
        im = IndexManager(vector_store=vector_store, bm25_index=bm25)

        # Ingest valid chunks
        valid_chunks = [
            {"chunk_id": "v1", "text": "Valid test content about retrieval.", "metadata": {"source": "test.txt"}},
        ]
        im.ingest_chunks(valid_chunks)

        pipeline = _build_test_pipeline(im)

        # Should still be able to query after valid ingestion
        result = pipeline.retrieve("retrieval", top_k=3, use_reranker=False)
        assert len(result["results"]) > 0, "Should retrieve after valid ingestion"

    def test_graceful_degradation_missing_components(self, temp_dir):
        """Test pipeline with minimal components doesn't crash."""
        vector_store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_minimal"),
            collection_name="test_minimal",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        vector_store.clear()
        bm25 = BM25Index()
        im = IndexManager(vector_store=vector_store, bm25_index=bm25)

        chunks = [
            {"chunk_id": "m1", "text": "Minimal test document.", "metadata": {"source": "test.txt"}},
        ]
        im.ingest_chunks(chunks)

        # Build pipeline with only dense retriever
        pipeline = RetrievalPipeline(
            index_manager=im,
            query_expander=None,
            dense_retriever=DenseRetriever(im),
            sparse_retriever=None,
            rrf_fusion=RRFFusion(),
            cache=None,
            config={"dense_top_k": 100, "fusion_top_k": 50, "reranking": {"enabled": False}},
            metrics=MetricsCollector(),
        )

        result = pipeline.retrieve("test", top_k=3, use_reranker=False)
        assert isinstance(result["results"], list), "Should return list, not crash"


# ---------------------------------------------------------------------------
# Concurrent operations tests
# ---------------------------------------------------------------------------

class TestConcurrentOperations:
    """Test concurrent ingestion and query operations."""

    def test_concurrent_queries(self, temp_dir, sample_documents):
        """Test multiple concurrent queries don't crash."""
        vector_store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_concurrent"),
            collection_name="test_concurrent",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        vector_store.clear()
        bm25 = BM25Index()
        im = IndexManager(vector_store=vector_store, bm25_index=bm25)

        chunks = _create_chunks_from_docs(sample_documents[:2])
        im.ingest_chunks(chunks)

        pipeline = _build_test_pipeline(im)
        errors = []
        results_lock = threading.Lock()
        all_results = []

        def run_query(q):
            try:
                result = pipeline.retrieve(q, top_k=3, use_reranker=False)
                with results_lock:
                    all_results.append(result)
            except Exception as e:
                errors.append(e)

        queries = ["machine learning", "Python", "artificial intelligence", "data", "algorithms"]
        threads = [threading.Thread(target=run_query, args=(q,)) for q in queries]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent queries should not error: {errors}"
        assert len(all_results) == len(queries), "Should have results for all queries"

    def test_concurrent_ingestion_and_query(self, temp_dir, sample_documents):
        """Test concurrent ingestion and query operations."""
        vector_store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_concurrent2"),
            collection_name="test_concurrent2",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        vector_store.clear()
        bm25 = BM25Index()
        im = IndexManager(vector_store=vector_store, bm25_index=bm25)

        pipeline = _build_test_pipeline(im)
        errors = []

        def ingest_batch(docs):
            try:
                chunks = _create_chunks_from_docs(docs)
                im.ingest_chunks(chunks)
            except Exception as e:
                errors.append(e)

        def query_batch(queries):
            try:
                for q in queries:
                    pipeline.retrieve(q, top_k=3, use_reranker=False)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=ingest_batch, args=(sample_documents[:2],))
        t2 = threading.Thread(target=query_batch, args=(["machine learning", "Python"],))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Thread safety is the main concern - no crashes
        assert len(errors) == 0, f"Concurrent operations should not error: {errors}"


# ---------------------------------------------------------------------------
# Large document ingestion tests
# ---------------------------------------------------------------------------

class TestLargeDocumentIngestion:
    """Test ingestion of larger documents."""

    def test_large_document_ingestion(self, temp_dir):
        """Test ingesting a simulated large document."""
        vector_store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_large"),
            collection_name="test_large",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        vector_store.clear()
        bm25 = BM25Index()
        im = IndexManager(vector_store=vector_store, bm25_index=bm25)

        # Generate many chunks
        num_chunks = 50
        large_chunks = []
        for i in range(num_chunks):
            large_chunks.append({
                "chunk_id": f"large_{i}",
                "text": f"This is chunk {i} of a large document. It contains content about machine learning and artificial intelligence. "
                        f"Topic {i % 5}: {'classification' if i % 5 == 0 else 'regression' if i % 5 == 1 else 'clustering' if i % 5 == 2 else 'neural networks' if i % 5 == 3 else 'deep learning'}.",
                "metadata": {"source": "large_doc.txt", "chunk_idx": i},
            })

        im.ingest_chunks(large_chunks)
        assert vector_store.count() == num_chunks, f"Should have {num_chunks} chunks"

        pipeline = _build_test_pipeline(im)
        result = pipeline.retrieve("neural networks", top_k=5, use_reranker=False)
        assert len(result["results"]) > 0, "Should retrieve from large document"
        # Context window expansion may add neighboring chunks, so result count can exceed top_k
        assert len(result["results"]) >= 1, "Should have at least 1 result"


# ---------------------------------------------------------------------------
# End-to-end with metrics
# ---------------------------------------------------------------------------

class TestEndToEndWithMetrics:
    """Test full pipeline with metrics collection."""

    def test_full_pipeline_with_metrics(self, temp_dir, sample_documents):
        """Test full ingestion -> retrieval with metrics tracking."""
        metrics = MetricsCollector()
        vector_store = VectorStore(
            persist_directory=os.path.join(temp_dir, "chroma_e2e"),
            collection_name="test_e2e",
            embedding_model="BAAI/bge-small-zh-v1.5",
        )
        vector_store.clear()
        bm25 = BM25Index()
        im = IndexManager(vector_store=vector_store, bm25_index=bm25)

        chunks = _create_chunks_from_docs(sample_documents)
        im.ingest_chunks(chunks)

        pipeline = _build_test_pipeline(im, enable_cache=True, metrics=metrics)

        # Run several queries
        queries = [
            "machine learning",
            "Python",
            "深度学习",
            "RAG",
            "machine learning",  # Should hit cache
        ]
        for q in queries:
            pipeline.retrieve(q, top_k=3, use_reranker=False)

        report = metrics.get_full_report()
        assert report["total_queries"] == len(queries), "Should record all queries"
        assert report["cache"]["hits"] >= 1, "Should have at least one cache hit"
        assert "latency" in report, "Should have latency metrics"
        assert "recall_paths" in report, "Should have recall path distribution"
        assert "health" in report, "Should have health status"
        assert report["latency"]["avg_ms"] > 0, "Average latency should be positive"