"""API routes for Hermes-RAG."""

import asyncio
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fastapi import APIRouter, HTTPException, Query, Depends, Header
from pydantic import BaseModel

logger = logging.getLogger("hermes_rag")

router = APIRouter(prefix="/api/v1")

# Global pipeline instance (lazy-loaded, thread-safe)
_pipeline = None
_pipeline_lock = threading.Lock()

# Thread pool for CPU-bound operations
_executor = ThreadPoolExecutor(max_workers=2)

# API Key from environment (empty = no auth required)
_API_KEY = os.environ.get("HERMES_API_KEY", "")


def _verify_api_key(x_api_key: Optional[str] = Header(None)) -> bool:
    """Verify API key if authentication is configured."""
    if not _API_KEY:
        return True
    if not x_api_key:
        raise HTTPException(status_code=401, detail={"error": "unauthorized", "message": "API key required"})
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=403, detail={"error": "forbidden", "message": "Invalid API key"})
    return True


class IngestRequest(BaseModel):
    source: str  # File path or URL
    source_type: Optional[str] = None  # pdf, docx, pptx, txt, md, web


class IngestResponse(BaseModel):
    status: str
    chunks_count: int
    source: str
    vector_count: int
    bm25_count: int


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranker: bool = True
    generate_answer: bool = False


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: dict
    score: float
    source: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    results: list
    answer: Optional[str] = None
    timing: dict


class HealthResponse(BaseModel):
    status: str
    vector_count: int
    bm25_count: int
    version: str


def _get_pipeline():
    """Lazy-load the retrieval pipeline (thread-safe with double-checked locking)."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from src.core.pipeline_factory import build_pipeline
                _pipeline = build_pipeline()
    return _pipeline


async def _get_pipeline_async():
    """Async wrapper for lazy-loading the retrieval pipeline (thread-safe).

    Note: Currently unused but reserved for future async pipeline initialization
    where model loading should not block the event loop.
    """
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                loop = asyncio.get_running_loop()
                from src.core.pipeline_factory import build_pipeline
                _pipeline = await loop.run_in_executor(_executor, build_pipeline)
    return _pipeline


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest, _auth: bool = Depends(_verify_api_key)):
    """Ingest a document into the index."""
    try:
        from src.core.ingestion.parser_factory import ParserFactory
        from src.core.chunking.hierarchical_chunker import HierarchicalChunker
        from src.config import get_config
        from src.utils.security import (
            validate_file_path,
            validate_file_extension,
            validate_file_size,
            validate_url,
        )

        cfg = get_config()

        source = request.source

        # Security: validate URL or file path
        is_url = source.startswith("http://") or source.startswith("https://")
        if is_url:
            validate_url(source)
        else:
            validate_file_path(source)
            validate_file_extension(source)
            validate_file_size(source)

        # Parse document
        parser = ParserFactory.get_parser(source)
        parsed = parser.parse(source)

        # Chunk
        chunker = HierarchicalChunker(
            chunk_size=cfg.chunking.chunk_size,
            chunk_overlap=cfg.chunking.chunk_overlap,
            semantic_threshold=cfg.chunking.semantic_threshold,
            min_chunk_size=cfg.chunking.min_chunk_size,
            max_section_size=cfg.chunking.max_section_size,
            embedding_model=cfg.embedding.model_name,
            embedding_device=cfg.embedding.device,
        )

        source_name = Path(source).name if not is_url else "web"
        chunks = chunker.chunk(
            text=parsed["text"],
            source_name=source_name,
            headings=parsed.get("metadata", {}).get("headings"),
        )

        if not chunks:
            return IngestResponse(
                status="no_content",
                chunks_count=0,
                source=request.source,
                vector_count=0,
                bm25_count=0,
            )

        # Index
        pipeline = _get_pipeline()
        counts = pipeline.index_manager.ingest_chunks(chunks)

        return IngestResponse(
            status="success",
            chunks_count=len(chunks),
            source=request.source,
            vector_count=counts.get("vector_count", 0),
            bm25_count=counts.get("bm25_count", 0),
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail={"error": "file_not_found", "message": str(e)})
    except ConnectionError as e:
        raise HTTPException(status_code=502, detail={"error": "connection_error", "message": str(e)})
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": str(e)})
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail={"error": "internal_error", "message": str(e)})


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, _auth: bool = Depends(_verify_api_key)):
    """Query the retrieval pipeline."""
    try:
        pipeline = _get_pipeline()
        result = pipeline.retrieve(
            query=request.query,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
        )

        answer = None
        if request.generate_answer:
            from src.config import get_config
            from src.core.generation.llm_client import LLMClient

            cfg = get_config()
            provider = cfg.generation.provider
            if provider == "openai":
                llm = LLMClient(
                    provider="openai",
                    model=cfg.generation.openai.model,
                    base_url=cfg.generation.openai.base_url,
                    api_key=cfg.generation.openai.api_key,
                    temperature=cfg.generation.temperature,
                    max_context_tokens=cfg.generation.max_context_tokens,
                )
            else:
                llm = LLMClient(
                    provider="ollama",
                    model=cfg.generation.ollama.model,
                    base_url=cfg.generation.ollama.base_url,
                    temperature=cfg.generation.temperature,
                    max_context_tokens=cfg.generation.max_context_tokens,
                )
            answer = llm.generate(request.query, result["results"])

        return QueryResponse(
            query=request.query,
            results=result["results"],
            answer=answer,
            timing=result.get("timing", {}),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": "invalid_query", "message": str(e)})
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail={"error": "internal_error", "message": str(e)})


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    try:
        pipeline = _get_pipeline()
        return HealthResponse(
            status="healthy",
            vector_count=pipeline.index_manager.vector_store.count(),
            bm25_count=pipeline.index_manager.bm25_index.count(),
            version="1.0.0",
        )
    except Exception:
        return HealthResponse(
            status="initializing",
            vector_count=0,
            bm25_count=0,
            version="1.0.0",
        )