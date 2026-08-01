"""End-to-end pipeline tests."""



class TestPipeline:
    """End-to-end pipeline tests."""

    def test_pipeline_import(self):
        """Test that all pipeline components can be imported."""
        from src.core.retrieval.retrieval_pipeline import RetrievalPipeline

        assert RetrievalPipeline is not None

    def test_config_import(self):
        """Test that config can be imported."""
        from src.config import HermesConfig
        assert HermesConfig is not None

    def test_full_pipeline_build(self, temp_dir, sample_chunks, sample_config):
        """Test building and running the full pipeline."""
        from src.core.indexing.bm25_index import BM25Index
        from src.core.indexing.index_manager import IndexManager
        from src.core.indexing.vector_store import VectorStore
        from src.core.reranking.cross_encoder import CrossEncoderReranker
        from src.core.retrieval.dense_retriever import DenseRetriever
        from src.core.retrieval.query_expander import QueryExpander
        from src.core.retrieval.retrieval_pipeline import RetrievalPipeline
        from src.core.retrieval.rrf_fusion import RRFFusion
        from src.core.retrieval.rule_retriever import RuleRetriever
        from src.core.retrieval.sparse_retriever import SparseRetriever
        from src.utils.cache import QueryCache

        # Build components
        vector_store = VectorStore(
            persist_directory=temp_dir + "/chroma_pipeline",
            collection_name="test_pipeline",
        )
        bm25_index = BM25Index()
        index_manager = IndexManager(
            vector_store=vector_store,
            bm25_index=bm25_index,
        )

        # Ingest sample chunks
        index_manager.ingest_chunks(sample_chunks[:3])

        query_expander = QueryExpander(synonym_enabled=True, hyde_enabled=False)
        dense_retriever = DenseRetriever(index_manager)
        sparse_retriever = SparseRetriever(index_manager)
        rule_retriever = RuleRetriever()
        rrf_fusion = RRFFusion()
        cross_encoder = CrossEncoderReranker()
        cache = QueryCache(max_size=10)

        pipeline = RetrievalPipeline(
            index_manager=index_manager,
            query_expander=query_expander,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            rule_retriever=rule_retriever,
            rrf_fusion=rrf_fusion,
            cross_encoder=cross_encoder,
            cache=cache,
            config=sample_config,
        )

        result = pipeline.retrieve(
            query="machine learning",
            top_k=2,
            use_reranker=False,
        )

        assert "results" in result
        assert "query_info" in result
        assert "timing" in result
        assert len(result["results"]) > 0
        assert len(result["results"]) >= 2  # Context window expansion may add neighbors

    def test_utils_imports(self):
        """Test that all utility modules can be imported."""
        from src.utils.text_utils import (
            clean_text,
            estimate_tokens,
            tokenize_text,
        )

        assert callable(tokenize_text)
        assert callable(clean_text)
        assert callable(estimate_tokens)

    def test_text_utils(self):
        """Test text utility functions."""
        from src.utils.text_utils import (
            clean_text,
            estimate_tokens,
            get_language_ratio,
            split_sentences,
            tokenize_text,
        )

        # Test clean_text
        cleaned = clean_text("  Hello   world  \n\n\nTest  ")
        assert "Hello world" in cleaned
        assert "\n\n\n" not in cleaned

        # Test estimate_tokens
        tokens = estimate_tokens("Hello world")
        assert tokens > 0

        # Test split_sentences
        sentences = split_sentences("Hello. World! How are you?")
        assert len(sentences) >= 3

        # Test language ratio
        ratio = get_language_ratio("Hello 你好")
        assert ratio["chinese_ratio"] > 0
        assert ratio["english_ratio"] > 0

        # Test tokenize
        tokens = tokenize_text("Hello world")
        assert len(tokens) > 0

    def test_stopwatch(self):
        """Test Stopwatch utility."""
        from src.utils.timer import Stopwatch

        sw = Stopwatch()
        import time
        time.sleep(0.01)
        elapsed = sw.lap("test")
        assert elapsed > 0

        summary = sw.summary()
        assert "test" in summary