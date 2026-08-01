"""Security utilities for Hermes-RAG."""

import ipaddress
import logging
import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger("hermes_rag")

# Blocked IP ranges for SSRF protection
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),      # loopback
    ipaddress.ip_network("10.0.0.0/8"),       # private
    ipaddress.ip_network("172.16.0.0/12"),    # private
    ipaddress.ip_network("192.168.0.0/16"),   # private
    ipaddress.ip_network("169.254.0.0/16"),   # link-local
    ipaddress.ip_network("0.0.0.0/8"),        # reserved
    ipaddress.ip_network("::1/128"),          # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),         # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),        # IPv6 link-local
]

# Allowed URL schemes for web fetching
_ALLOWED_URL_SCHEMES = {"http", "https"}

# Max file size for ingestion (50MB)
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

# Allowed file extensions for local ingestion
_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md", ".csv", ".json"}


def validate_file_path(file_path: str, base_dir: Optional[str] = None) -> Path:
    """Validate and sanitize a file path to prevent path traversal.

    Args:
        file_path: The file path to validate.
        base_dir: Optional base directory to restrict to.

    Returns:
        Resolved absolute Path.

    Raises:
        ValueError: If the path is invalid or attempts path traversal.
    """
    resolved = Path(file_path).resolve()

    # Check for path traversal attempts
    if ".." in str(Path(file_path)):
        raise ValueError("Path traversal detected: '..' in path")

    if base_dir:
        base = Path(base_dir).resolve()
        try:
            resolved.relative_to(base)
        except ValueError:
            raise ValueError(f"File path escapes base directory: {file_path}")

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return resolved


def validate_file_extension(file_path: str) -> str:
    """Validate file extension against allowed list.

    Returns:
        Lowercase extension with dot.

    Raises:
        ValueError: If extension is not allowed.
    """
    ext = Path(file_path).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension '{ext}'. Allowed: {_ALLOWED_EXTENSIONS}")
    return ext


def validate_file_size(file_path: str, max_bytes: int = MAX_FILE_SIZE_BYTES) -> None:
    """Validate file size is within limits.

    Raises:
        ValueError: If file exceeds max size.
    """
    size = os.path.getsize(file_path)
    if size > max_bytes:
        raise ValueError(f"File too large: {size} bytes (max {max_bytes} bytes)")


def validate_url(url: str) -> str:
    """Validate URL for SSRF protection.

    Checks that:
    - URL scheme is http or https
    - Hostname does not resolve to internal/private IP
    - URL is well-formed

    Returns:
        Normalized URL string.

    Raises:
        ValueError: If URL is invalid or blocked.
    """
    parsed = urlparse(url)

    if parsed.scheme not in _ALLOWED_URL_SCHEMES:
        raise ValueError(f"URL scheme '{parsed.scheme}' not allowed. Use http or https.")

    if not parsed.hostname:
        raise ValueError(f"URL has no hostname: {url}")

    # Check for raw IP addresses
    try:
        addr = ipaddress.ip_address(parsed.hostname)
        for network in _BLOCKED_NETWORKS:
            if addr in network:
                raise ValueError(f"URL resolves to blocked network: {parsed.hostname}")
    except ValueError:
        # Not an IP address - resolve hostname
        import socket
        try:
            resolved_ips = socket.getaddrinfo(parsed.hostname, None)
            for info in resolved_ips:
                ip = ipaddress.ip_address(info[4][0])
                for network in _BLOCKED_NETWORKS:
                    if ip in network:
                        raise ValueError(f"URL resolves to blocked network: {parsed.hostname} -> {ip}")
        except socket.gaierror:
            raise ValueError(f"Unable to resolve hostname: {parsed.hostname}")

    return url


def validate_api_key(api_key: Optional[str], expected_key: Optional[str]) -> bool:
    """Validate API key if authentication is configured.

    Returns True if authentication passes or is not configured.
    """
    if expected_key is None:
        return True
    return api_key == expected_key


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal and injection."""
    # Remove path separators and null bytes
    filename = filename.replace("\\", "").replace("/", "").replace("\0", "")
    # Remove any path traversal attempts
    filename = re.sub(r"\.\.+", "", filename)
    # Keep only safe characters
    filename = re.sub(r"[^\w\-\. ]", "_", filename)
    return filename.strip() or "unnamed"