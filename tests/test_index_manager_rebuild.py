"""Tests for automatic index recovery through IndexManager."""

from src.core.indexing.index_manager import IndexManager
from src.core.indexing.vector_store import IndexDimensionMismatch


class FakeVectorStore:
    def __init__(self):
        self.raised = False
        self.chunks = []

    def count(self):
        if not self.raised:
            self.raised = True
            raise IndexDimensionMismatch(384, 1024, "test")
        return len(self.chunks)

    def rebuild_from_chunks(self, chunks):
        self.chunks = list(chunks)
        self.raised = True
        return len(chunks)

    def add_chunks(self, chunks):
        self.chunks.extend(chunks)

    def clear(self):
        self.chunks = []


class FakeBM25:
    def __init__(self):
        self.chunks = []

    def count(self):
        return len(self.chunks)

    def add_chunks(self, chunks):
        self.chunks.extend(chunks)

    def search(self, query, top_k=100):
        return []

    def clear(self):
        self.chunks = []


class FakeDocumentStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def count(self):
        return len(self.chunks)

    def get_all_chunks(self):
        return list(self.chunks)

    def add_chunks(self, chunks):
        self.chunks.extend(chunks)

    def clear(self):
        self.chunks = []


CHUNKS = [
    {"chunk_id": "a_0", "text": "text a", "metadata": {}},
    {"chunk_id": "b_0", "text": "text b", "metadata": {}},
]


def test_ensure_indexes_rebuilds_both_stores():
    vector = FakeVectorStore()
    bm25 = FakeBM25()
    docs = FakeDocumentStore(CHUNKS)
    manager = IndexManager(vector_store=vector, bm25_index=bm25, document_store=docs)

    result = manager.ensure_indexes()
    assert vector.chunks == CHUNKS
    assert bm25.chunks == CHUNKS
    assert any("vector_rebuilt" in action for action in result["actions"])
    assert "bm25_rebuilt" in result["actions"]


def test_rebuild_from_document_store():
    vector = FakeVectorStore()
    bm25 = FakeBM25()
    docs = FakeDocumentStore(CHUNKS)
    manager = IndexManager(vector_store=vector, bm25_index=bm25, document_store=docs)

    counts = manager.rebuild_from_document_store()
    assert counts["vector_count"] == 2
    assert counts["bm25_count"] == 2

def test_vector_store_count_propagates_dimension_mismatch():
    """A stale persisted collection must not be hidden by the count() guard."""
    from src.core.indexing.vector_store import VectorStore

    store = VectorStore(persist_directory=":memory:", collection_name="test")

    def raise_mismatch():
        raise IndexDimensionMismatch(384, 1024, "test")

    store._get_collection = raise_mismatch
    try:
        store.count()
    except IndexDimensionMismatch:
        pass
    else:
        raise AssertionError("IndexDimensionMismatch was swallowed by count()")


def test_compute_layout_signature_tracks_version_and_dimension():
    manager = IndexManager()
    base = manager.compute_layout_signature()
    assert base.startswith(IndexManager.INDEX_VERSION)
    assert manager.compute_layout_signature(1024) != manager.compute_layout_signature(384)

