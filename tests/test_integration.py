"""Comprehensive integration tests for the full retrieval pipeline with real data."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import get_config
from utils.cache import QueryCache
from utils.metrics import MetricsCollector


class TestRetrievalPipelineIntegration:
    """End-to-end tests for the retrieval pipeline with real documents."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup a minimal pipeline with test documents."""
        from core.indexing.bm25_index import BM25Index
        from core.indexing.index_manager import IndexManager
        from core.indexing.vector_store import VectorStore
        from core.retrieval.dense_retriever import DenseRetriever
        from core.retrieval.query_expander import QueryExpander
        from core.retrieval.retrieval_pipeline import RetrievalPipeline
        from core.retrieval.rrf_fusion import RRFFusion
        from core.retrieval.sparse_retriever import SparseRetriever

        self.tmpdir = tempfile.mkdtemp()
        config = get_config()

        # Create vector store with test data
        self.vs = VectorStore(
            persist_directory=os.path.join(self.tmpdir, "chroma"),
            collection_name="test_hermes",
            embedding_model=config.embedding.model_name,
            embedding_device="cpu",
        )
        self.vs.clear()

        # Create BM25 index
        self.bm25 = BM25Index()

        # Create test chunks
        self.test_chunks = [
            {"chunk_id": "c1", "text": "陈祖敬毕业于华南理工大学计算机科学与技术专业，拥有8年AI经验。", "metadata": {"source": "rencai.txt"}},
            {"chunk_id": "c2", "text": "李碧宣是2008年出生的天才少女，15岁进入麻省理工大学学习AI大模型开发。", "metadata": {"source": "rencai.txt"}},
            {"chunk_id": "c3", "text": "公司上班时间为早9:00到18:00，午休12:00-13:30。每月允许3次迟到。", "metadata": {"source": "company.txt"}},
            {"chunk_id": "c4", "text": "Python是一种高级编程语言，由Guido van Rossum于1991年创建。", "metadata": {"source": "tech.txt"}},
            {"chunk_id": "c5", "text": "机器学习是人工智能的一个分支，通过数据训练模型来做出预测。", "metadata": {"source": "tech.txt"}},
            {"chunk_id": "c6", "text": "深度学习使用多层神经网络来处理复杂模式识别任务。", "metadata": {"source": "tech.txt"}},
            {"chunk_id": "c7", "text": "RAG（检索增强生成）结合了信息检索和文本生成技术。", "metadata": {"source": "tech.txt"}},
            {"chunk_id": "c8", "text": "如何重置密码：进入设置页面，点击安全选项，选择重置密码。", "metadata": {"source": "guide.txt"}},
            {"chunk_id": "c9", "text": "安装Python环境：从官网下载安装包，运行安装程序，配置环境变量。", "metadata": {"source": "guide.txt"}},
            {"chunk_id": "c10", "text": "产品代码ABC-1234对应智能传感器模块，规格参数见产品手册。", "metadata": {"source": "product.txt"}},
        ]

        self.vs.add_chunks(self.test_chunks)
        self.bm25.add_chunks(self.test_chunks)

        # Build pipeline
        self.metrics = MetricsCollector()
        self.cache = QueryCache(max_size=100)

        self.im = IndexManager(vector_store=self.vs, bm25_index=self.bm25)

        self.pipeline = RetrievalPipeline(
            index_manager=self.im,
            query_expander=QueryExpander(synonym_enabled=True, hyde_enabled=False),
            dense_retriever=DenseRetriever(self.im),
            sparse_retriever=SparseRetriever(self.im),
            rrf_fusion=RRFFusion(),
            cache=self.cache,
            config={
                "dense_top_k": 100,
                "sparse_top_k": 100,
                "fusion_top_k": 50,
                "reranking": {"enabled": False},
            },
            metrics=self.metrics,
        )

        yield

        # Cleanup
        self.vs.clear()

    # ---- Basic Retrieval Tests ----

    def test_retrieve_person_name(self):
        result = self.pipeline.retrieve("陈祖敬是谁", top_k=5, use_reranker=False)
        assert len(result["results"]) > 0
        texts = [r["text"] for r in result["results"]]
        assert any("陈祖敬" in t for t in texts)

    def test_retrieve_another_person(self):
        result = self.pipeline.retrieve("李碧宣在哪毕业的", top_k=5, use_reranker=False)
        texts = [r["text"] for r in result["results"]]
        assert any("李碧宣" in t for t in texts)
        assert any("麻省理工" in t for t in texts)

    def test_retrieve_factual(self):
        result = self.pipeline.retrieve("公司几点上班", top_k=3, use_reranker=False)
        texts = [r["text"] for r in result["results"]]
        assert any("9:00" in t or "早9" in t for t in texts)

    def test_retrieve_tech_concept(self):
        result = self.pipeline.retrieve("什么是机器学习", top_k=3, use_reranker=False)
        texts = [r["text"] for r in result["results"]]
        assert any("机器学习" in t for t in texts)

    def test_retrieve_procedural(self):
        result = self.pipeline.retrieve("如何重置密码", top_k=3, use_reranker=False)
        texts = [r["text"] for r in result["results"]]
        assert any("重置密码" in t for t in texts)

    def test_retrieve_product_code(self):
        result = self.pipeline.retrieve("ABC-1234是什么", top_k=3, use_reranker=False)
        texts = [r["text"] for r in result["results"]]
        assert any("ABC-1234" in t for t in texts)

    # ---- Result Structure Tests ----

    def test_result_structure(self):
        result = self.pipeline.retrieve("Python", top_k=3, use_reranker=False)
        assert "results" in result
        assert "query_info" in result
        assert "timing" in result
        assert isinstance(result["results"], list)

    def test_result_chunk_structure(self):
        result = self.pipeline.retrieve("Python", top_k=3, use_reranker=False)
        for r in result["results"]:
            assert "chunk_id" in r
            assert "text" in r
            assert "score" in r
            assert "metadata" in r

    def test_query_info(self):
        result = self.pipeline.retrieve("Python", top_k=3, use_reranker=False)
        qi = result["query_info"]
        assert "original" in qi
        assert "expanded" in qi
        assert "query_type" in qi

    # ---- Top-K Tests ----

    def test_top_k_returned(self):
        for k in [1, 3, 5]:
            result = self.pipeline.retrieve("Python", top_k=k, use_reranker=False)
            assert len(result["results"]) <= k

    def test_top_k_exceeds_results(self):
        result = self.pipeline.retrieve("联合国气候变化框架公约", top_k=20, use_reranker=False)
        assert len(result["results"]) <= 20  # Should not crash

    # ---- Edge Cases ----

    def test_empty_query(self):
        result = self.pipeline.retrieve("", top_k=5, use_reranker=False)
        assert isinstance(result["results"], list)

    def test_whitespace_query(self):
        result = self.pipeline.retrieve("   ", top_k=5, use_reranker=False)
        assert isinstance(result["results"], list)

    def test_very_long_query(self):
        long_query = "这是一个非常长的查询" * 50
        result = self.pipeline.retrieve(long_query, top_k=3, use_reranker=False)
        assert isinstance(result["results"], list)

    def test_special_chars_query(self):
        result = self.pipeline.retrieve("!@#$%^&*()", top_k=3, use_reranker=False)
        assert isinstance(result["results"], list)

    # ---- Cache Tests ----

    def test_cache_hit(self):
        # First query (cold)
        self.pipeline.retrieve("Python", top_k=3, use_reranker=False)
        # Second query (warm - should hit cache)
        result2 = self.pipeline.retrieve("Python", top_k=3, use_reranker=False)
        assert result2["query_info"].get("cached", False) is True

    def test_cache_hit_rate_after_hits(self):
        self.pipeline.retrieve("Python", top_k=3, use_reranker=False)
        self.pipeline.retrieve("Python", top_k=3, use_reranker=False)
        self.pipeline.retrieve("different query", top_k=3, use_reranker=False)
        rate = self.cache.hit_rate()
        assert rate > 0

    # ---- Metrics Tests ----

    def test_metrics_recording(self):
        self.pipeline.retrieve("test", top_k=3, use_reranker=False)
        report = self.metrics.get_full_report()
        assert report["total_queries"] >= 1

    def test_metrics_cache_tracking(self):
        self.pipeline.retrieve("Python", top_k=3, use_reranker=False)
        self.pipeline.retrieve("Python", top_k=3, use_reranker=False)
        report = self.metrics.get_full_report()
        assert report["cache"]["hits"] >= 1

    # ---- Query Expansion Tests ----

    def test_query_expansion_applied(self):
        result = self.pipeline.retrieve("重置密码", top_k=3, use_reranker=False)
        assert result["query_info"]["expanded"] != result["query_info"]["original"]

    # ---- Deduplication Tests ----

    def test_no_duplicate_results(self):
        result = self.pipeline.retrieve("Python", top_k=5, use_reranker=False)
        chunk_ids = [r["chunk_id"] for r in result["results"]]
        assert len(chunk_ids) == len(set(chunk_ids))

    # ---- Score Tests ----

    def test_scores_descending(self):
        result = self.pipeline.retrieve("Python", top_k=5, use_reranker=False)
        scores = [r["score"] for r in result["results"]]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_scores_non_negative(self):
        result = self.pipeline.retrieve("Python", top_k=5, use_reranker=False)
        for r in result["results"]:
            assert r["score"] >= 0

    # ---- Timing Tests ----

    def test_timing_present(self):
        result = self.pipeline.retrieve("Python", top_k=3, use_reranker=False)
        assert result["timing"]["total"] > 0

    # ---- Error Handling ----

    def test_graceful_degradation_no_reranker(self):
        """Should work fine when reranker is disabled."""
        result = self.pipeline.retrieve("Python", top_k=3, use_reranker=False)
        assert len(result["results"]) > 0

    def test_repeated_queries_stable(self):
        """Multiple queries should not crash or leak."""
        for _ in range(10):
            result = self.pipeline.retrieve("Python", top_k=3, use_reranker=False)
            assert len(result["results"]) > 0