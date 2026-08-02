"""Regression tests for index consistency and proper-noun retrieval.

Covers the defect that made "陈祖敬是谁？" return no usable hits even
though the knowledge base contained the person's profile:

1. Vector/BM25 indexes can silently diverge from the document store (the
   source of truth); ``ensure_indexes`` must detect and repair the gap.
2. When all stores are consistent, a natural-language person query must
   recall the person's own chunks in the top results.
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.core.pipeline_factory import build_pipeline


def _build_temp_pipeline(doc_names):
    """Build a real pipeline against a fresh temporary index."""
    cfg = load_config()
    temp_handle = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    temp_path = Path(temp_handle.name)
    cfg.chromadb.persist_directory = str(temp_path / "chroma")
    cfg.chromadb.document_store_path = str(temp_path / "document_store.db")
    cfg.bm25.fallback_db_path = str(temp_path / "bm25_fallback.db")

    from src.core.chunking.hierarchical_chunker import HierarchicalChunker
    from src.core.ingestion.parser_factory import ParserFactory

    pipeline = build_pipeline(config=cfg)
    index_manager = pipeline.index_manager
    docs_root = Path(__file__).parent.parent / "evaluation" / "data" / "sample_docs"
    chunker = HierarchicalChunker(
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
        semantic_threshold=cfg.chunking.semantic_threshold,
        min_chunk_size=cfg.chunking.min_chunk_size,
        max_section_size=cfg.chunking.max_section_size,
        embedding_model=cfg.embedding.model_name,
        embedding_device=cfg.embedding.device,
    )
    for name in doc_names:
        fp = docs_root / name
        parser = ParserFactory.get_parser(str(fp))
        parsed = parser.parse(str(fp))
        chunks = chunker.chunk(
            text=parsed["text"],
            source_name=name,
            headings=parsed.get("metadata", {}).get("headings"),
        )
        if chunks:
            index_manager.ingest_chunks(chunks)
    return pipeline, index_manager, temp_handle


class TestIndexConsistency:
    def test_ensure_indexes_repairs_divergent_bm25(self):
        pipeline, index_manager, handle = _build_temp_pipeline(
            ["ml_intro.md", "陈祖敬.txt"]
        )
        try:
            doc_ids = {c["chunk_id"] for c in index_manager.document_store.get_all_chunks()}
            assert doc_ids, "document store should be populated"
            assert set(index_manager.bm25_index.get_chunk_ids()) == doc_ids

            # Simulate a partial sparse index: drop one chunk from BM25 only.
            removed = sorted(doc_ids)[0]
            assert index_manager.bm25_index.remove_chunk(removed)

            result = index_manager.ensure_indexes()
            assert any("consistency_rebuild" in a for a in result["actions"]), result

            assert set(index_manager.bm25_index.get_chunk_ids()) == doc_ids
        finally:
            handle.cleanup()

    def test_proper_noun_query_recalls_person_chunks(self):
        pipeline, index_manager, handle = _build_temp_pipeline(
            ["ml_intro.md", "陈祖敬.txt"]
        )
        try:
            res = pipeline.retrieve("陈祖敬是谁？", top_k=5)
            hits = {r["chunk_id"] for r in res["results"]}
            assert any(cid.startswith("陈祖敬.txt_") for cid in hits), (
                "expected 陈祖敬 chunks in top-5, got %s" % sorted(hits)
            )
        finally:
            handle.cleanup()