"""Comprehensive tests for RRFFusion."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from core.retrieval.rrf_fusion import RRFFusion


class TestRRFFusion:
    """Test RRF fusion algorithm correctness and edge cases."""

    def _make_result(self, chunk_id, text, score=0.0):
        return {"chunk_id": chunk_id, "text": text, "metadata": {}, "score": score}

    def test_fuse_both_empty(self):
        fusion = RRFFusion()
        result = fusion.fuse([], [])
        assert result == []

    def test_fuse_only_dense(self):
        fusion = RRFFusion()
        dense = [
            self._make_result("1", "text1"),
            self._make_result("2", "text2"),
        ]
        result = fusion.fuse(dense, [])
        assert len(result) == 2
        assert result[0]["chunk_id"] == "1"

    def test_fuse_only_sparse(self):
        fusion = RRFFusion()
        sparse = [
            self._make_result("a", "textA"),
            self._make_result("b", "textB"),
        ]
        result = fusion.fuse([], sparse)
        assert len(result) == 2
        assert result[0]["chunk_id"] == "a"

    def test_fuse_both_present(self):
        fusion = RRFFusion()
        dense = [
            self._make_result("1", "dense_text"),
            self._make_result("2", "dense_text2"),
        ]
        sparse = [
            self._make_result("1", "sparse_text"),  # Same chunk
            self._make_result("3", "sparse_text3"),
        ]
        result = fusion.fuse(dense, sparse)
        # Chunk "1" should appear once with combined score
        ids = [r["chunk_id"] for r in result]
        assert "1" in ids
        assert len(result) == 3

    def test_fuse_scores_descending(self):
        fusion = RRFFusion()
        dense = [self._make_result(str(i), f"text{i}") for i in range(10)]
        sparse = [self._make_result(str(i + 5), f"text{i+5}") for i in range(10)]
        result = fusion.fuse(dense, sparse)
        # Verify scores are descending
        for i in range(len(result) - 1):
            assert result[i]["score"] >= result[i + 1]["score"]

    def test_fuse_top_k_limit(self):
        fusion = RRFFusion()
        dense = [self._make_result(str(i), f"text{i}") for i in range(100)]
        sparse = [self._make_result(str(i + 50), f"text{i+50}") for i in range(100)]
        result = fusion.fuse(dense, sparse, top_k=10)
        assert len(result) == 10

    def test_dynamic_weights_product_code(self):
        fusion = RRFFusion()
        w_d, w_s = fusion._get_dynamic_weights("ABC-1234的价格")
        assert w_s > w_d  # Sparse weight should be higher

    def test_dynamic_weights_colloquial(self):
        fusion = RRFFusion()
        w_d, w_s = fusion._get_dynamic_weights("怎么安装Python")
        assert w_d > w_s  # Dense weight should be higher

    def test_dynamic_weights_default(self):
        fusion = RRFFusion()
        w_d, w_s = fusion._get_dynamic_weights("机器学习")
        # Short Chinese queries get slight sparse boost, so scores may differ from 0.5/0.5
        assert abs(w_d + w_s - 1.0) < 0.01, "Weights should sum to 1.0"
        assert 0.3 <= w_d <= 0.7, f"Dense weight {w_d} should be in [0.3, 0.7]"
        assert 0.3 <= w_s <= 0.7, f"Sparse weight {w_s} should be in [0.3, 0.7]"

    def test_fuse_sources_tracking(self):
        fusion = RRFFusion()
        dense = [self._make_result("1", "shared")]
        sparse = [self._make_result("1", "shared"), self._make_result("2", "sparse_only")]
        result = fusion.fuse(dense, sparse)
        for r in result:
            assert "sources" in r
            if r["chunk_id"] == "1":
                assert "dense" in r["sources"] and "sparse" in r["sources"]
            elif r["chunk_id"] == "2":
                assert "sparse" in r["sources"]

    def test_large_k_value(self):
        """Test with very large k value."""
        fusion = RRFFusion(k=1000)
        dense = [self._make_result("1", "text")]
        sparse = [self._make_result("1", "text")]
        result = fusion.fuse(dense, sparse)
        assert len(result) == 1
        assert result[0]["score"] > 0

    def test_single_result_each(self):
        fusion = RRFFusion()
        dense = [self._make_result("a", "d")]
        sparse = [self._make_result("b", "s")]
        result = fusion.fuse(dense, sparse)
        assert len(result) == 2