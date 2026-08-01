"""Memory monitoring utilities for resource-constrained environments."""

import gc
import logging
import os
import sys
from typing import Optional

logger = logging.getLogger("hermes_rag")


def get_memory_usage() -> dict:
    """Get current memory usage statistics.

    Returns:
        dict with keys: rss_mb, vms_mb, percent (if available).
    """
    try:
        import psutil

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        return {
            "rss_mb": mem.rss / (1024 * 1024),
            "vms_mb": mem.vms / (1024 * 1024),
            "percent": process.memory_percent(),
        }
    except ImportError:
        return {"rss_mb": -1, "vms_mb": -1, "percent": -1}


def log_memory(log: Optional[object] = None, tag: str = "") -> None:
    """Log current memory usage."""
    mem = get_memory_usage()
    msg = f"[MEMORY] {tag} RSS: {mem['rss_mb']:.1f}MB, VMS: {mem['vms_mb']:.1f}MB"
    target = log or logger
    target.info(msg)


def force_gc() -> None:
    """Force garbage collection to free memory."""
    gc.collect()


def set_memory_limit_mb(limit_mb: int = 7000) -> None:
    """Set a soft memory limit (supports Windows and Linux)."""
    try:
        if sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            limit_bytes = limit_mb * 1024 * 1024
            kernel32.SetProcessWorkingSetSize(-1, limit_bytes, limit_bytes)
        else:
            import resource

            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (limit_mb * 1024 * 1024, hard))
    except (ImportError, AttributeError, OSError):
        pass


def check_memory_pressure(threshold_mb: int = 7000) -> bool:
    """Check if memory usage is near the limit.

    Returns True if memory pressure is high.
    """
    mem = get_memory_usage()
    return mem["rss_mb"] > threshold_mb * 0.85