"""Logging utilities for Hermes-RAG."""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "hermes_rag", level: str = "INFO", log_file: str = "") -> logging.Logger:
    """Set up and return a configured logger.

    Args:
        name: Logger name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Path to log file. If empty, logs to stdout only.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Get the default Hermes-RAG logger."""
    return logging.getLogger("hermes_rag")