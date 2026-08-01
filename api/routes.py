"""API routes for Hermes-RAG."""

import asyncio
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Ensure project root is in sys.path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fastapi import APIRouter, Depends, Header, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

logger = logging.getLogger("hermes_rag")

router = APIRouter(prefix="/api/v1")

# Global pipeline instance (lazy-loaded, thread-safe)
_pipeline = None
_pipeline_lock = threading.Lock()

# Thread pool for CPU-bound operations
_executor = ThreadPoolExecutor(max_workers=2)

# API Key from environment (empty = no auth required)
_API_KEY = os.environ.get("HERMES_API_KEY", "")


def _verify_api_key(x_api_key: str | None = Header(None)) -> bool:
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
    source_type: str | None = None  # pdf, docx, pptx, txt, md, web


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

    model_config = {"extra": "forbid"}


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: dict
    score: float
    source: str | None = None


class QueryResponse(BaseModel):
    query: str
    results: list
    answer: str | None = None
    timing: dict


class HealthResponse(BaseModel):
    status: str
    vector_count: int
    bm25_count: int
    document_count: int
    version: str


class StatsResponse(BaseModel):
    index: dict
    cache: dict
    metrics: dict
    config: dict
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
        from src.config import get_config
        from src.core.chunking.hierarchical_chunker import HierarchicalChunker
        from src.core.ingestion.parser_factory import ParserFactory
        from src.utils.security import (
            validate_file_extension,
            validate_file_path,
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
        # Validate top_k
        if request.top_k < 1:
            raise HTTPException(status_code=422, detail={"error": "invalid_input", "message": "top_k must be >= 1"})
        if request.top_k > 1000:
            raise HTTPException(status_code=422, detail={"error": "invalid_input", "message": "top_k must be <= 1000"})

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

    except HTTPException:
        raise
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
            vector_count=pipeline.index_manager.vector_store.count() if pipeline.index_manager.vector_store else 0,
            bm25_count=pipeline.index_manager.bm25_index.count() if pipeline.index_manager.bm25_index else 0,
            document_count=pipeline.index_manager.document_store.count() if pipeline.index_manager.document_store else 0,
            version="1.0.0",
        )
    except Exception:
        return HealthResponse(
            status="initializing",
            vector_count=0,
            bm25_count=0,
            document_count=0,
            version="1.0.0",
        )


def _redact_config(data):
    """Remove secrets from config before exposing it through the API."""
    if isinstance(data, dict):
        redacted = {}
        for k, v in data.items():
            if isinstance(v, dict):
                redacted[k] = _redact_config(v)
            elif k.lower() in ("api_key", "password", "token", "secret"):
                redacted[k] = "***"
            else:
                redacted[k] = v
        return redacted
    return data


@router.get("/stats", response_model=StatsResponse)
async def stats(_auth: bool = Depends(_verify_api_key)):
    """Return index, cache, metrics, and redacted config statistics."""
    try:
        pipeline = _get_pipeline()
        from src.config import get_config
        from src.utils.metrics import get_metrics

        index_stats = {}
        try:
            index_stats = pipeline.index_manager.get_stats()
        except Exception as e:
            index_stats = {"error": str(e)}

        cache_stats = {"enabled": pipeline.cache is not None}
        if pipeline.cache is not None:
            try:
                cache_stats.update(pipeline.cache.get_stats())
            except Exception as e:
                cache_stats["error"] = str(e)

        metrics = {}
        try:
            metrics = get_metrics().get_full_report()
        except Exception as e:
            metrics = {"error": str(e)}

        return StatsResponse(
            index=index_stats,
            cache=cache_stats,
            metrics=metrics,
            config=_redact_config(get_config().to_dict()),
            version="1.0.0",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "stats_failed", "message": str(e)})


@router.post("/rebuild", response_model=dict)
async def rebuild(_auth: bool = Depends(_verify_api_key)):
    """Rebuild vector and BM25 indexes from the persisted document store."""
    try:
        pipeline = _get_pipeline()
        counts = pipeline.index_manager.rebuild_from_document_store()
        return {"status": "success", "counts": counts}
    except Exception as e:
        logger.exception(f"Rebuild failed: {e}")
        raise HTTPException(status_code=500, detail={"error": "rebuild_failed", "message": str(e)})