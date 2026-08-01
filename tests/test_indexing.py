"""Tests for indexing module."""

from src.core.indexing.bm25_index import BM25Index
from src.core.indexing.vector_store import VectorStore


class TestVectorStore:
    """Tests for VectorStore."""

    def test_initialization(self, temp_dir):
        store = VectorStore(
            persist_directory=temp_dir + "/chroma_test",
            collection_name="test_collection",
        )
        assert store.collection_name == "test_collection"
        assert store.count() == 0

    def test_add_and_count(self, temp_dir, sample_chunks):
        store = VectorStore(
            persist_directory=temp_dir + "/chroma_test2",
            collection_name="test_collection",
        )
        store.add_chunks(sample_chunks[:3])
        assert store.count() == 3

    def test_search_returns_results(self, temp_dir, sample_chunks):
        store = VectorStore(
            persist_directory=temp_dir + "/chroma_test3",
            collection_name="test_collection",
        )
        store.add_chunks(sample_chunks)
        results = store.search("machine learning", top_k=2)
        assert len(results) > 0
        assert len(results) <= 2
        for r in results:
            assert "chunk_id" in r
            assert "text" in r
            assert "score" in r

    def test_clear(self, temp_dir, sample_chunks):
        store = VectorStore(
            persist_directory=temp_dir + "/chroma_test4",
            collection_name="test_collection",
        )
        store.add_chunks(sample_chunks[:2])
        assert store.count() == 2
        store.clear()
        assert store.count() == 0


class TestBM25Index:
    """Tests for BM25Index."""

    def test_initialization(self):
        index = BM25Index()
        assert index.count() == 0

    def test_add_and_count(self, sample_chunks):
        index = BM25Index()
        index.add_chunks(sample_chunks[:3])
        assert index.count() == 3

    def test_search_basic(self, sample_chunks):
        index = BM25Index()
        index.add_chunks(sample_chunks)
        results = index.search("machine learning", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3
        for r in results:
            assert "chunk_id" in r
            assert "text" in r
            assert "score" in r

    def test_search_chinese(self):
        index = BM25Index()
        chunks = [
            {
                "chunk_id": "cn_0",
                "text": "机器学习是人工智能的一个子领域，专注于从数据中学习",
                "metadata": {"source": "cn.txt"},
            },
            {
                "chunk_id": "cn_1",
                "text": "深度学习使用多层神经网络进行特征提取",
                "metadata": {"source": "cn.txt"},
            },
        ]
        index.add_chunks(chunks)
        results = index.search("机器学习", top_k=2)
        assert len(results) > 0

    def test_clear(self, sample_chunks):
        index = BM25Index()
        index.add_chunks(sample_chunks)
        assert index.count() > 0
        index.clear()
        assert index.count() == 0

    def test_empty_search(self):
        index = BM25Index()
        results = index.search("nothing", top_k=10)
        assert results == []

    def test_tokenize(self):
        index = BM25Index()
        tokens_en = index._tokenize("Hello world, this is a test")
        assert len(tokens_en) > 0

        tokens_cn = index._tokenize("你好世界，这是一个测试")
        assert len(tokens_cn) > 0