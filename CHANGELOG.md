# Changelog

All notable changes to Hermes-RAG are documented in this file.

## [Unreleased]

### Fixed

- Index stores could silently diverge from the document store (e.g. chunks
  present in the vector index but missing from the persisted BM25 index), so
  sparse retrieval served an incomplete index and RRF fusion pushed correct
  results out of Top-K. `ensure_indexes()` now compares chunk sets across the
  document store, vector store and BM25 index, and rebuilds divergent stores
  from the document store on startup.
- New documents dropped into the auto-ingest directory were never ingested
  when the stored sources were still a subset of the directory contents.
  `ingest` (CLI and startup auto-ingest) now records the processed source set
  (`ingested_sources`), and startup re-ingests when the directory contains
  files outside that set.
- Proper-noun questions such as "陈祖敬是谁？" returned zero usable hits after
  a partial re-ingest left the BM25 index missing the person's chunks while
  the vector store still had them. New `VectorStore.get_chunk_ids()` /
  `BM25Index.get_chunk_ids()` accessors plus the consistency check above repair
  the live index automatically.

### Added

- `tests/test_index_consistency.py`: regression coverage for cross-store
  divergence repair and for proper-noun retrieval ("陈祖敬是谁？" recalls
  `陈祖敬.txt` chunks in the top results).

## [2.4.0] - 2026-08-01

### Fixed

- Chunker dropped short sections: `min_chunk_size` was compared against character count instead of token count, so definitions under 50 characters (e.g. "随机森林", "分类问题详解" opening) never entered the index. It now uses `estimate_tokens()`; default lowered from 50 to 20 tokens.
- Section headings were invisible to retrieval: chunk text now prefixes the full heading path (e.g. `监督学习详解 > 分类问题 - ...`), so heading-only answers are recallable by dense and sparse paths.
- Query expansion polluted dense ranking: dense now uses the original query while sparse uses the expanded (synonym) query, keeping expansion gains without deranking the vector path.
- Heuristic reranker scale mismatch: the original RRF score was normalized to [0, 1] before combining with keyword/heading signals, so fusion order stays dominant.
- Document-level recall inflated by duplicate chunks from the same source: doc results are deduplicated before `recall_at_k`.
- Stale vector collections were served silently: `VectorStore.count()` swallowed `IndexDimensionMismatch`, so a persisted collection built with an old embedding model never triggered a rebuild and dense retrieval degraded to sparse-only. The exception now propagates and the collection is rebuilt from the document store.

### Changed

- Evaluation dataset re-labeled to answer-bearing chunks (40 queries), with `relevant_doc_ids` added; a quality-gate test verifies every label exists in the benchmark index.
- Benchmark adds document-level metrics (`doc_hit_rate@1/3/5`, `doc_mrr`, `doc_recall@5`) and `doc_failed_queries`.
- Default remains RRF fusion without reranking; the heuristic reranker stays opt-in (`reranking.enabled: false`).
- Default variant: Hit Rate@5 100% (40/40), MRR 0.8883, NDCG@10 0.9054 on the 40-query benchmark.

### Added

- Index version marker (`IndexManager.INDEX_VERSION`) persisted in DocumentStore; startup rebuilds vector/BM25 when chunk text/metadata layout changes.
- Layout signature (`index_version|embedding dimension`) persisted with the document store; CLI auto-ingest re-chunks stale stores from source when either the chunk layout or embedding schema changes.
- ML-domain synonym expansion for overfitting, random forest, decision tree, logistic regression, SVM, K-means, supervised/unsupervised learning, backpropagation, activation functions, decorators, and more.
- `tests/test_retrieval_quality.py`: ground-truth validity, chunker fidelity (short sections, heading prefix, token-based minimum), and doc-level metrics.
- Gradio evaluation dashboard now shows document-level H@5 and failed-query count.

## [2.3.0] - 2026-08-01

### Fixed

- `top_k` contract: context-window expansion no longer returns more results than requested (`retrieval_pipeline.py`).
- BM25 persistence: the sparse index now survives process restarts via SQLite (`bm25.persist: true`).
- Silent sparse-index degradation: a pure-Python BM25 fallback is used when `rank_bm25` is not installed.
- Vector index recovery: embedding-dimension mismatches rebuild from the persisted document store instead of silently deleting data.
- Auto-ingest: startup ingestion no longer silently fails (`ParserFactory.parse` misuse fixed).
- Corrupted SQLite recovery: the connection is closed before deleting the corrupt database file.

### Changed

- Reranking is conservative by default (`reranking.enabled: false`) because the previous heuristic reranker reduced Hit Rate@5 from 72.4% to 41.4%.
- The heuristic reranker now weighs the original RRF score much more heavily than keyword/position signals.
- Evaluation is reproducible: `python run_eval.py` builds a temporary in-process index and writes `docs/eval_results.json`.
- Documentation and configuration are aligned with `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3`.

### Added

- `DocumentStore`: SQLite-backed source of truth for chunks and automatic index recovery.
- `GET /api/v1/stats` and `POST /api/v1/rebuild` endpoints.
- Benchmark harness (`evaluation/benchmark.py`) with baseline/RRF/reranker variants and difficulty breakdowns.
- GitHub Actions CI, LICENSE, CONTRIBUTING, SECURITY, and requirements.txt.
- Multi-tab Gradio UI: chat, retrieval lab, evaluation dashboard, and system status.
### 2026-08-01 追加

- `BM25Index.close()` 与持久化重连：修复 Windows 下 SQLite 文件句柄未释放导致的 PermissionError，测试资源可被完整回收。
- 适配 Gradio 6.21：移除已废弃的 `Chatbot(type=...)`、`show_copy_button`、`bubble_full_width` 参数，主题参数移至 `launch()`。
- 全量测试基线：1197 passed / 1 skipped。
