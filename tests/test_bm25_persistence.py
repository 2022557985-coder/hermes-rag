"""Tests for persistent BM25 storage and the pure-Python fallback."""

import builtins

from src.core.indexing.bm25_index import BM25Index

CHUNKS = [
    {"chunk_id": "a_0", "text": "machine learning is a branch of artificial intelligence", "metadata": {"source": "a.md"}},
    {"chunk_id": "b_0", "text": "python is a programming language", "metadata": {"source": "b.md"}},
    {"chunk_id": "c_0", "text": "deep learning and neural networks belong to machine learning", "metadata": {"source": "c.md"}},
]


def test_persist_mode_survives_restart(tmp_path):
    db_path = str(tmp_path / "bm25.db")
    idx1 = BM25Index(fallback_db_path=db_path, persist=True)
    idx1.add_chunks(CHUNKS)
    assert idx1.count() == 3

    idx2 = BM25Index(fallback_db_path=db_path, persist=True)
    assert idx2.count() == 3
    results = idx2.search("machine learning", top_k=3)
    assert results, "persisted BM25 should return results"
    assert results[0]["chunk_id"] in {"a_0", "c_0"}


def test_persist_mode_remove_and_clear(tmp_path):
    db_path = str(tmp_path / "bm25.db")
    idx1 = BM25Index(fallback_db_path=db_path, persist=True)
    idx1.add_chunks(CHUNKS)
    assert idx1.remove_chunk("b_0") is True
    assert idx1.count() == 2
    idx1.clear()
    assert idx1.count() == 0

    idx2 = BM25Index(fallback_db_path=db_path, persist=True)
    assert idx2.count() == 0


def test_pure_python_fallback(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "rank_bm25":
            raise ImportError("rank_bm25 blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    idx = BM25Index()
    idx.add_chunks(CHUNKS)
    results = idx.search("machine learning", top_k=3)
    assert results, "pure-Python fallback should still return results"