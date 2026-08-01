"""Quality gates for retrieval: dataset validity, chunking fidelity, doc metrics."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def _load_ground_truth():
    path = PROJECT_ROOT / "evaluation" / "data" / "ground_truth.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_benchmark_chunk_map():
    """Chunk the 11 benchmark docs with the production chunker (no embeddings)."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import get_config, reset_config
    reset_config()
    cfg = get_config()
    from src.core.chunking.hierarchical_chunker import HierarchicalChunker
    from src.core.ingestion.parser_factory import ParserFactory

    chunker = HierarchicalChunker(
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
        semantic_threshold=cfg.chunking.semantic_threshold,
        min_chunk_size=cfg.chunking.min_chunk_size,
        max_section_size=10**9,  # keep tests embedding-free
        embedding_model=cfg.embedding.model_name,
        embedding_device="cpu",
    )
    docs_root = PROJECT_ROOT / "evaluation" / "data" / "sample_docs"
    names = [
        "ml_arch.md", "ml_classification.md", "ml_clustering.md", "ml_cv.md",
        "ml_eval.md", "ml_intro.md", "ml_neural.md", "ml_regression.md",
        "ml_supervised.md", "py_basics.md", "py_features.md",
    ]
    chunk_map = {}
    for name in names:
        fp = docs_root / name
        if not fp.exists():
            continue
        parsed = ParserFactory.get_parser(str(fp)).parse(str(fp))
        chunks = chunker.chunk(
            text=parsed["text"],
            source_name=name,
            headings=parsed.get("metadata", {}).get("headings"),
        )
        for c in chunks:
            chunk_map[c["chunk_id"]] = c
    return chunk_map


class TestGroundTruth:
    def test_dataset_shape(self):
        data = _load_ground_truth()
        assert len(data) >= 35
        assert len({item["query"] for item in data}) == len(data)

    def test_entries_are_well_formed(self):
        data = _load_ground_truth()
        for item in data:
            assert item["query"].strip()
            assert item["difficulty"] in ("easy", "medium", "hard")
            assert item["relevant_chunk_ids"], f"empty labels for {item['query']}"
            derived_docs = sorted({cid.rsplit("_", 1)[0] for cid in item["relevant_chunk_ids"]})
            assert item["relevant_doc_ids"] == derived_docs
            for cid in item["relevant_chunk_ids"]:
                assert cid.count("_") >= 1 and cid.rsplit("_", 1)[1].isdigit()

    def test_all_relevant_chunks_exist(self):
        chunk_map = _build_benchmark_chunk_map()
        data = _load_ground_truth()
        missing = [
            cid
            for item in data
            for cid in item["relevant_chunk_ids"]
            if cid not in chunk_map
        ]
        assert not missing, f"ground truth references missing chunks: {missing[:10]}"

    def test_negative_samples_are_valid(self):
        chunk_map = _build_benchmark_chunk_map()
        data = _load_ground_truth()
        neg_queries = [item for item in data if item.get("negative_chunk_ids")]
        assert len(neg_queries) >= 3
        for item in neg_queries:
            for cid in item["negative_chunk_ids"]:
                assert cid in chunk_map
                assert cid not in item["relevant_chunk_ids"]


class TestChunkerQuality:
    def test_short_section_is_kept(self):
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.core.chunking.hierarchical_chunker import HierarchicalChunker

        chunker = HierarchicalChunker(min_chunk_size=20)
        text = "# 随机森林\n\n随机森林是多个决策树的集成，通过投票机制来决定最终分类结果。\n"
        chunks = chunker.chunk(text, source_name="test.md")
        assert len(chunks) == 1
        assert "随机森林" in chunks[0]["text"]

    def test_min_chunk_size_is_token_based(self):
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.core.chunking.hierarchical_chunker import HierarchicalChunker

        # 40 characters but only ~10 tokens: must be dropped at 50-token minimum.
        chunker = HierarchicalChunker(min_chunk_size=50)
        chunks = chunker.chunk("short text " * 4, source_name="tiny.txt")
        assert chunks == []

        # Same text passes a 10-token minimum.
        chunker2 = HierarchicalChunker(min_chunk_size=10)
        chunks2 = chunker2.chunk("short text " * 4, source_name="tiny.txt")
        assert len(chunks2) == 1

    def test_full_heading_path_is_indexed(self):
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.core.chunking.hierarchical_chunker import HierarchicalChunker

        chunker = HierarchicalChunker(min_chunk_size=10)
        text = (
            "# 监督学习详解\n\n"
            "监督学习是机器学习中最常用的一种方法。\n\n"
            "## 分类问题\n\n"
            "分类问题是指预测离散的类别标签。\n"
        )
        headings = [
            {"level": 1, "title": "监督学习详解"},
            {"level": 2, "title": "分类问题"},
        ]
        chunks = chunker.chunk(text, source_name="test.md", headings=headings)
        by_heading = {c["metadata"].get("heading_path", ""): c for c in chunks}
        assert "监督学习详解 > 分类问题" in by_heading
        assert by_heading["监督学习详解 > 分类问题"]["text"].startswith(
            "监督学习详解 > 分类问题 - "
        )


class TestDocLevelMetrics:
    def test_doc_hit_rate(self):
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from evaluation.eval import hit_rate_at_k, mrr

        results = [{"chunk_id": "a_2"}, {"chunk_id": "b_0"}, {"chunk_id": "c_5"}]
        docs = [{"chunk_id": r["chunk_id"].rsplit("_", 1)[0]} for r in results]
        assert hit_rate_at_k(docs, ["b"], 1) == 0.0
        assert hit_rate_at_k(docs, ["b"], 2) == 1.0
        assert mrr(docs, ["b"]) == 0.5
        assert hit_rate_at_k(results, ["a"], 1) == 0.0
