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
    from src.core.ingestion.parser_factory import ParserFactory
    from src.core.chunking.hierarchical_chunker import HierarchicalChunker
    from src.core.pipeline_factory import build_pipeline
    from src.utils.logger import setup_logger
    from src.utils.timer import Stopwatch
    from src.utils.security import (
        validate_file_path,
        validate_file_extension,
        validate_file_size,
    )

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
                if ext in (".pdf", ".docx", ".pptx", ".txt", ".md", ".markdown"):
                    files.append(os.path.join(root, fname))
        logger.info(f"Found {len(files)} files to ingest")
    else:
        files = [source_path]

    total_chunks = 0
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
    """Launch the Gradio UI."""
    from ui.gradio_app import main
    main()


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