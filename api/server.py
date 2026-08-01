"""FastAPI application for Hermes-RAG."""

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import router

logger = logging.getLogger("hermes_rag")

# Rate limiting state (thread-safe)
_rate_limit_window: float = 1.0  # 1 second window
_rate_limit_max_requests: int = 100  # max requests per window
_request_timestamps: list = []
_rate_limit_lock = threading.Lock()


def _check_rate_limit() -> bool:
    """Simple sliding window rate limiter (thread-safe)."""
    now = time.time()
    global _request_timestamps
    with _rate_limit_lock:
        _request_timestamps = [t for t in _request_timestamps if now - t < _rate_limit_window]
        if len(_request_timestamps) >= _rate_limit_max_requests:
            return False
        _request_timestamps.append(now)
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with graceful shutdown."""
    logger.info("Hermes-RAG API starting up...")
    yield
    logger.info("Hermes-RAG API shutting down...")

    # Shutdown the global thread pool executor
    from .routes import _executor
    _executor.shutdown(wait=True)

    # Unload pipeline resources
    from .routes import _pipeline, _pipeline_lock
    with _pipeline_lock:
        if _pipeline is not None:
            if hasattr(_pipeline, 'cross_encoder') and _pipeline.cross_encoder is not None:
                try:
                    _pipeline.cross_encoder._unload_model()
                except Exception:
                    pass
            if hasattr(_pipeline, 'cache') and _pipeline.cache is not None:
                try:
                    _pipeline.cache.clear()
                except Exception:
                    pass
            _pipeline = None

    # Cleanup memory
    import gc
    gc.collect()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Hermes-RAG API",
        description="Lightweight, high-precision RAG retrieval optimization framework",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS: restrict in production, allow all in development
    allowed_origins = os.environ.get(
        "HERMES_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:7860,http://127.0.0.1:8000",
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    )

    # Rate limiting middleware
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        if request.url.path.startswith("/api/") and request.url.path != "/api/v1/health":
            if not _check_rate_limit():
                return JSONResponse(
                    status_code=429,
                    content={"error": "rate_limit_exceeded", "message": "Too many requests"},
                )
        response = await call_next(request)
        return response

    app.include_router(router)

    return app


app = create_app()