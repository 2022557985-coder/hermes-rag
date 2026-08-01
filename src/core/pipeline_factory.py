"""Shared pipeline construction factory.

Extracts duplicate pipeline building code from api/routes.py and evaluation/eval.py
into a single reusable function.
"""

import logging
from typing import Optional

logger = logging.getLogger("hermes_rag")


def build_pipeline(
    config=None,
    *,
    query_expansion_enabled: bool = True,
    use_reranker: bool = True,
    use_sparse: bool = True,
    use_cache: bool = True,
):
    """Build and return a configured RetrievalPipeline.

    Args:
        config: HermesConfig instance. If None, loaded from get_config().
        query_expansion_enabled: Whether to enable query expansion.
        use_reranker: Whether to enable cross-encoder reranking.
        use_sparse: Whether to enable sparse (BM25) retrieval.
        use_cache: Whether to enable query cache.

    Returns:
        Configured RetrievalPipeline instance.
    """
    if config is None:
        from src.config import get_config
        config = get_config()

    from src.core.retrieval.retrieval_pipeline import RetrievalPipeline
    from src.core.retrieval.query_expander import QueryExpander
    from src.core.retrieval.dense_retriever import DenseRetriever
    from src.core.retrieval.sparse_retriever import SparseRetriever
    from src.core.retrieval.rule_retriever import RuleRetriever
    from src.core.retrieval.rrf_fusion import RRFFusion
    from src.core.reranking.cross_encoder import CrossEncoderReranker
    from src.core.indexing.vector_store import VectorStore
    from src.core.indexing.bm25_index import BM25Index
    from src.core.indexing.index_manager import IndexManager
    from src.utils.cache import QueryCache
    from src.utils.metrics import get_metrics

    # Vector store
    vector_store = VectorStore(
        persist_directory=config.chromadb.persist_directory,
        collection_name=config.chromadb.collection_name,
        hnsw_ef_construction=config.chromadb.hnsw_ef_construction,
        hnsw_M=config.chromadb.hnsw_M,
        hnsw_ef_search=config.chromadb.hnsw_ef_search,
        embedding_model=config.embedding.model_name,
        embedding_device=config.embedding.device,
    )

    # BM25 index
    bm25_index = BM25Index(
        b=config.bm25.b,
        k1=config.bm25.k1,
        max_index_entries=config.bm25.max_index_entries,
        fallback_db_path=config.bm25.fallback_db_path,
    )

    # Index manager
    index_manager = IndexManager(
        vector_store=vector_store,
        bm25_index=bm25_index,
    )

    # Query expander
    query_expander = QueryExpander(
        synonym_enabled=config.query_expansion.synonym_enabled and query_expansion_enabled,
        hyde_enabled=config.query_expansion.hyde_enabled and query_expansion_enabled,
        hyde_model=config.query_expansion.hyde_model,
        max_synonyms=config.query_expansion.max_synonyms,
    ) if query_expansion_enabled else None

    # Retrievers
    dense_retriever = DenseRetriever(index_manager)
    sparse_retriever = SparseRetriever(index_manager) if use_sparse else None
    rule_retriever = RuleRetriever()

    # RRF fusion
    rrf_fusion = RRFFusion(
        k=config.retrieval.rrf_k,
        default_dense_weight=config.retrieval.rrf_weights.default_dense,
        default_sparse_weight=config.retrieval.rrf_weights.default_sparse,
        product_code_dense_weight=config.retrieval.rrf_weights.product_code_dense,
        product_code_sparse_weight=config.retrieval.rrf_weights.product_code_sparse,
        colloquial_dense_weight=config.retrieval.rrf_weights.colloquial_dense,
        colloquial_sparse_weight=config.retrieval.rrf_weights.colloquial_sparse,
    )

    # Cross-encoder reranker
    cross_encoder = CrossEncoderReranker(
        model_name=config.reranking.model_name,
        device=config.reranking.device,
        batch_size=config.reranking.batch_size,
        max_candidates=config.reranking.max_candidates,
        timeout_seconds=config.reranking.timeout_seconds,
    ) if use_reranker else None

    # Query cache
    cache = QueryCache(
        max_size=config.cache.max_cache_size,
        similarity_threshold=config.cache.similarity_threshold,
        ttl_seconds=config.cache.ttl_seconds,
    ) if use_cache else None

    # Retrieval config
    retrieval_config = {
        "dense_top_k": config.retrieval.dense_top_k,
        "sparse_top_k": config.retrieval.sparse_top_k,
        "fusion_top_k": config.retrieval.fusion_top_k,
        "reranking": {
            "enabled": config.reranking.enabled and use_reranker,
            "timeout_seconds": config.reranking.timeout_seconds,
        },
    }

    logger.info("RetrievalPipeline built successfully")
    return RetrievalPipeline(
        index_manager=index_manager,
        query_expander=query_expander,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        rule_retriever=rule_retriever,
        rrf_fusion=rrf_fusion,
        cross_encoder=cross_encoder,
        cache=cache,
        config=retrieval_config,
        metrics=get_metrics(),
    )