#!/usr/bin/env python3
"""Hermes-RAG Command Line Interface.

Usage:
    python cli.py --ingest <path>     Ingest documents
    python cli.py --query <query>      Query the index
    python cli.py --serve              Start FastAPI server
    python cli.py --ui                 Launch Gradio UI
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def cmd_ingest(args):
    """Ingest documents into the index."""
    from src.config import get_config
    from src.core.chunking.hierarchical_chunker import HierarchicalChunker
    from src.core.ingestion.parser_factory import ParserFactory
    from src.core.pipeline_factory import build_pipeline
    from src.utils.logger import setup_logger
    from src.utils.security import (
        validate_file_extension,
        validate_file_path,
        validate_file_size,
    )
    from src.utils.timer import Stopwatch

    cfg = get_config()
    logger = setup_logger(
        level=cfg.logging.level,
        log_file=cfg.logging.file,
    )
    sw = Stopwatch()

    source_path = args.path

    if not os.path.exists(source_path) and not (source_path.startswith("http://") or source_path.startswith("https://")):
        logger.error(f"Source not found: {source_path}")
        sys.exit(1)

    # Use shared pipeline factory (ensures consistent index instances)
    pipeline = build_pipeline(config=cfg)
    index_manager = pipeline.index_manager

    chunker = HierarchicalChunker(
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
        semantic_threshold=cfg.chunking.semantic_threshold,
        min_chunk_size=cfg.chunking.min_chunk_size,
        max_section_size=cfg.chunking.max_section_size,
        embedding_model=cfg.embedding.model_name,
        embedding_device=cfg.embedding.device,
    )

    # Determine if source is a directory or single file
    if os.path.isdir(source_path):
        files = []
        for root, _, filenames in os.walk(source_path):
            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext in (".pdf", ".docx", ".pptx", ".ppt", ".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".html", ".htm", ".log"):
                    files.append(os.path.join(root, fname))
        logger.info(f"Found {len(files)} files to ingest")
    else:
        files = [source_path]

    total_chunks = 0
    ingested_basenames = set()
    for file_path in files:
        try:
            logger.info(f"Ingesting: {file_path}")

            # Security validation
            validate_file_path(file_path)
            validate_file_extension(file_path)
            validate_file_size(file_path)

            parser = ParserFactory.get_parser(file_path)
            parsed = parser.parse(file_path)

            source_name = Path(file_path).name
            chunks = chunker.chunk(
                text=parsed["text"],
                source_name=source_name,
                headings=parsed.get("metadata", {}).get("headings"),
            )
            ingested_basenames.add(source_name)

            if chunks:
                counts = index_manager.ingest_chunks(chunks)
                total_chunks += len(chunks)
                logger.info(
                    f"  {len(chunks)} chunks, "
                    f"vector: {counts['vector_count']}, "
                    f"bm25: {counts['bm25_count']}"
                )

        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")

    # Record which source files were processed so startup auto-ingest can
    # detect newly added documents without re-parsing the whole directory.
    try:
        index_manager.document_store.set_meta(
            "ingested_sources",
            json.dumps(sorted(ingested_basenames), ensure_ascii=False),
        )
    except Exception as e:
        logger.warning(f"Failed to persist ingested sources: {e}")

    elapsed = sw.lap("total")
    logger.info(f"Ingestion complete: {total_chunks} chunks in {elapsed:.2f}s")


def cmd_query(args):
    """Query the retrieval pipeline."""
    from api.routes import _get_pipeline

    pipeline = _get_pipeline()
    result = pipeline.retrieve(
        query=args.query,
        top_k=args.top_k,
        use_reranker=not args.no_reranker,
    )

    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"\nQuery: {result['query_info']['original']}")
        print(f"Expanded: {result['query_info']['expanded']}")
        print(f"Total time: {result['timing']['total']:.4f}s\n")
        print("=" * 80)

        for i, r in enumerate(result["results"], 1):
            score = r.get("score", 0)
            source = r.get("metadata", {}).get("source", "unknown")
            heading = r.get("metadata", {}).get("heading_path", "")
            page = r.get("metadata", {}).get("page", "")

            print(f"\n[{i}] Score: {score:.4f}")
            if heading:
                print(f"    Heading: {heading}")
            if page:
                print(f"    Page: {page}")
            print(f"    Source: {source}")
            print(f"    Text: {r.get('text', '')[:200]}...")
            print("-" * 80)


def cmd_serve(args):
    """Start the FastAPI server."""
    import uvicorn

    from src.config import get_config

    cfg = get_config()
    uvicorn.run(
        "api.server:app",
        host=cfg.api.host,
        port=cfg.api.port,
        workers=cfg.api.workers,
        reload=args.reload,
    )


def cmd_ui(args):
    """Launch the Gradio UI with auto-ingest support."""
    # Import FastAPI first: loading it before the heavy ML libraries
    # (torch / chromadb / onnxruntime) avoids a flaky _ssl DLL init
    # failure on Windows that crashes the process at import time.
    from api.routes import _get_pipeline  # noqa: E402,F401

    from src.config import get_config
    from src.utils.logger import setup_logger

    cfg = get_config()
    logger = setup_logger(level=cfg.logging.level, log_file=cfg.logging.file)
    print("[Hermes-RAG] 正在启动，请稍候...", flush=True)

    # Auto-ingest: check if knowledge base is empty on startup
    if cfg.auto_ingest.enabled:
        _auto_ingest_on_startup(cfg, logger)

    # Pre-build the shared retrieval pipeline at startup so the first user
    # query is not blocked by a slow lazy model load (60-90s on CPU).
    print("[Hermes-RAG] 正在构建检索管道（约需 1-2 分钟）...", flush=True)
    _get_pipeline()
    print("[Hermes-RAG] 检索管道就绪。", flush=True)

    # Warm up the LLM in the background so the first generated answer
    # streams without a long model-loading delay.
    import threading

    def _warm_llm():
        try:
            import requests as _req

            _base = cfg.generation.ollama.base_url
            _model = cfg.generation.ollama.model
            _req.post(
                f"{_base}/api/generate",
                json={
                    "model": _model,
                    "prompt": "你好",
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {"num_predict": 1},
                },
                timeout=600,
            )
        except Exception as _e:  # noqa: BLE001 - warm-up must never crash startup
            logger.warning(f"LLM warm-up skipped: {_e}")

    threading.Thread(target=_warm_llm, daemon=True).start()

    from ui.gradio_app import main
    print("[Hermes-RAG] 正在启动 Web 界面，请访问 http://localhost:7860 ...", flush=True)
    main()


def _auto_ingest_on_startup(cfg, logger):
    """Auto-ingest documents when the document store is empty."""
    from pathlib import Path

    doc_dir = cfg.auto_ingest.doc_dir
    if not Path(doc_dir).exists():
        logger.warning(f"Auto-ingest directory not found: {doc_dir}")
        return

    import os

    from src.core.chunking.hierarchical_chunker import HierarchicalChunker
    from src.core.indexing.document_store import DocumentStore
    from src.core.indexing.index_manager import IndexManager
    from src.core.ingestion.parser_factory import ParserFactory
    from src.utils.security import validate_file_extension, validate_file_path, validate_file_size
    from src.utils.timer import Stopwatch

    files = []
    for root, _, filenames in os.walk(doc_dir):
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in (".pdf", ".docx", ".pptx", ".ppt", ".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".html", ".htm", ".log"):
                files.append(os.path.join(root, fname))

    if not files:
        logger.warning(f"No supported files found in {doc_dir}")
        return

    # Lightweight pre-check before building the heavy pipeline: when the index
    # layout changed, clear only documents that came from doc_dir so startup
    # re-chunks them from source instead of rebuilding stale chunk text.
    try:
        doc_store = DocumentStore(cfg.chromadb.document_store_path)
        if doc_store.count() > 0:
            stored_version = doc_store.get_meta("index_version")
            stored_signature = doc_store.get_meta("layout_signature")
            expected_signature = IndexManager.compute_layout_signature()
            doc_basenames = {os.path.basename(f) for f in files}
            sources = {
                c.get("metadata", {}).get("source", "")
                for c in doc_store.get_all_chunks()
            }
            # Compare only the chunk-layout prefix: embedding dimension drift is
            # handled by IndexManager.ensure_indexes, which rebuilds the vector
            # collection from the document store when the persisted dimension
            # no longer matches. Comparing the full signature here would
            # re-ingest every document on each startup after a manual ingest.
            layout_stale = (
                stored_signature is None
                or stored_signature.split("|", 1)[0] != expected_signature.split("|", 1)[0]
            )
            # Detect newly added files: when the previous ingest recorded the
            # source set, re-ingest if the directory now contains files that
            # were not part of that set (e.g. documents dropped in later).
            new_files_present = False
            try:
                recorded = json.loads(doc_store.get_meta("ingested_sources") or "[]")
                if recorded:
                    new_files_present = bool(doc_basenames - set(recorded))
            except (TypeError, ValueError):
                new_files_present = False
            up_to_date = (
                stored_version == IndexManager.INDEX_VERSION
                and not layout_stale
                and not new_files_present
            )
            if up_to_date or not sources <= doc_basenames:
                logger.info(
                    f"Document store already has {doc_store.count()} chunks, skipping auto-ingest."
                )
                doc_store.close()
                return
            logger.info(
                "Index layout changed (version=%s signature=%s -> %s), clearing managed documents for re-ingest...",
                stored_version,
                stored_signature,
                expected_signature,
            )
            doc_store.clear()
            doc_store.set_meta("index_version", IndexManager.INDEX_VERSION)
            doc_store.set_meta("layout_signature", expected_signature)
        doc_store.close()
    except Exception as e:
        logger.warning(f"Document store check failed: {e}")

    from api.routes import _get_pipeline

    # Reuse the shared pipeline instance so the Gradio UI does not build
    # a second (slow) pipeline on the first user query.
    sw = Stopwatch()
    pipeline = _get_pipeline()
    index_manager = pipeline.index_manager

    # Remove any leftover vector/BM25 entries for the cleared documents.
    index_manager.clear()

    logger.info(f"Auto-ingesting documents from: {doc_dir}")
    chunker = HierarchicalChunker(
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
        semantic_threshold=cfg.chunking.semantic_threshold,
        min_chunk_size=cfg.chunking.min_chunk_size,
        max_section_size=cfg.chunking.max_section_size,
        embedding_model=cfg.embedding.model_name,
        embedding_device=cfg.embedding.device,
    )

    logger.info(f"Auto-ingest: found {len(files)} files")
    total_chunks = 0
    ingested_basenames = set()
    for file_path in files:
        try:
            validate_file_path(file_path)
            validate_file_extension(file_path)
            validate_file_size(file_path)
            parsed = ParserFactory.get_parser(file_path).parse(file_path)
            source_name = os.path.basename(file_path)  # Use basename for consistent chunk IDs
            chunks = chunker.chunk(
                text=parsed["text"],
                source_name=source_name,
                headings=parsed.get("metadata", {}).get("headings"),
            )
            ingested_basenames.add(source_name)
            if chunks:
                index_manager.ingest_chunks(chunks)
                total_chunks += len(chunks)
        except Exception as e:
            logger.warning(f"Auto-ingest skipped {file_path}: {e}")

    # Record which source files were processed so the next startup can detect
    # newly added documents without re-parsing the whole directory.
    try:
        index_manager.document_store.set_meta(
            "ingested_sources",
            json.dumps(sorted(ingested_basenames), ensure_ascii=False),
        )
    except Exception as e:
        logger.warning(f"Failed to persist ingested sources: {e}")

    elapsed = sw.elapsed()
    logger.info(f"Auto-ingest complete: {total_chunks} chunks in {elapsed:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Hermes-RAG: Lightweight RAG Retrieval Framework",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("path", help="Path to document or directory")
    ingest_parser.set_defaults(func=lambda a: cmd_ingest(a))

    # Query
    query_parser = subparsers.add_parser("query", help="Query the index")
    query_parser.add_argument("query", help="Query string")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    query_parser.add_argument("--no-reranker", action="store_true", help="Disable reranker")
    query_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    query_parser.set_defaults(func=lambda a: cmd_query(a))

    # Serve
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    serve_parser.set_defaults(func=lambda a: cmd_serve(a))

    # UI
    ui_parser = subparsers.add_parser("ui", help="Launch Gradio UI")
    ui_parser.set_defaults(func=lambda a: cmd_ui(a))

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Handle legacy --ingest and --query flags
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()