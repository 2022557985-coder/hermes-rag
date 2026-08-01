"""Tests for the SQLite-backed document store."""

from src.core.indexing.document_store import DocumentStore

CHUNKS = [
    {"chunk_id": "a_0", "text": "first chunk", "metadata": {"source": "a.md"}},
    {"chunk_id": "b_0", "text": "second chunk", "metadata": {"source": "b.md"}},
]


def test_add_get_count_remove_clear(tmp_path):
    store = DocumentStore(str(tmp_path / "docs.db"))
    store.add_chunks(CHUNKS)
    assert store.count() == 2
    assert store.get_chunk("a_0")["text"] == "first chunk"

    assert store.remove_chunk("a_0") is True
    assert store.count() == 1
    assert store.get_chunk("a_0") is None

    store.clear()
    assert store.count() == 0


def test_upsert_replaces_existing(tmp_path):
    store = DocumentStore(str(tmp_path / "docs.db"))
    store.add_chunks(CHUNKS)
    store.add_chunks([{"chunk_id": "a_0", "text": "updated", "metadata": {"source": "a.md"}}])
    assert store.count() == 2
    assert store.get_chunk("a_0")["text"] == "updated"


def test_survives_restart(tmp_path):
    path = str(tmp_path / "docs.db")
    DocumentStore(path).add_chunks(CHUNKS)
    store2 = DocumentStore(path)
    assert store2.count() == 2