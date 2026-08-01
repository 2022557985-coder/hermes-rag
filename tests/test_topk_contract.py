"""Regression tests for the top_k result-count contract."""

from src.core.retrieval.retrieval_pipeline import RetrievalPipeline


class FakeIndexManager:
    def __init__(self, neighbor_count: int = 4):
        self.neighbor_count = neighbor_count

    def get_neighbor_chunks(self, chunk_id: str, window: int = 1):
        return [
            {"chunk_id": f"neighbor_{i}", "text": "neighbor", "metadata": {}}
            for i in range(self.neighbor_count)
        ]


def _results(n: int):
    return [
        {"chunk_id": f"c{i}", "text": "text", "metadata": {}, "score": 1.0 - i * 0.01}
        for i in range(n)
    ]


def test_context_window_never_exceeds_top_k():
    pipeline = RetrievalPipeline(index_manager=FakeIndexManager())
    expanded = pipeline._expand_context_window(_results(5), top_k=5, window=1)
    assert len(expanded) == 5


def test_context_window_fills_missing_slots():
    pipeline = RetrievalPipeline(index_manager=FakeIndexManager(neighbor_count=4))
    expanded = pipeline._expand_context_window(_results(1), top_k=5, window=1)
    assert len(expanded) == 5
    assert expanded[0]["chunk_id"] == "c0"


def test_context_window_disabled_keeps_results():
    pipeline = RetrievalPipeline(index_manager=FakeIndexManager())
    results = _results(3)
    assert pipeline._expand_context_window(results, top_k=5, window=0) == results


def test_context_window_empty_results():
    pipeline = RetrievalPipeline(index_manager=FakeIndexManager())
    assert pipeline._expand_context_window([], top_k=5, window=1) == []