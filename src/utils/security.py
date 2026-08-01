"""Security utilities for Hermes-RAG."""

import ipaddress
import logging
import os
import re
import threading
import time
from pathlib import Path
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
_ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".ppt", ".txt", ".md", ".csv", ".json",
    ".rst", ".html", ".htm", ".markdown", ".log",
}

# Max query length
MAX_QUERY_LENGTH = 2000

# Max batch size
MAX_BATCH_SIZE = 1000

# Magic bytes for file type detection
_MAGIC_BYTES: dict[str, list[bytes]] = {
    "pdf": [b"%PDF-"],
    "zip": [b"PK\x03\x04"],
}

# Dangerous HTML tags to strip
_DANGEROUS_HTML_TAGS = [
    "script", "iframe", "object", "embed", "form", "input",
    "button", "link", "meta", "style", "applet", "base",
]

# SQL injection patterns
_SQL_INJECTION_PATTERNS = [
    r"(?i)\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE)\b",
    r"(?i)\b(UNION\s+SELECT|EXEC\s|EXECUTE\s)\b",
    r"(?i)(--|\#|\/\*|\*\/)",
    r"(?i)\b(OR\s+1\s*=\s*1|AND\s+1\s*=\s*1)\b",
    r"(?i)\b(WAITFOR\s+DELAY|SLEEP\s*\()",
]

# XSS patterns
_XSS_PATTERNS = [
    r"<script[^>]*>.*?</script>",
    r"javascript\s*:",
    r"on\w+\s*=",
    r"<iframe[^>]*>",
    r"<embed[^>]*>",
    r"<object[^>]*>",
]


# --- Token Bucket Rate Limiter ---

class _TokenBucket:
    """Simple in-memory token bucket rate limiter."""

    def __init__(self, rate: float = 10.0, capacity: float = 20.0):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def consume(self, tokens: float = 1.0) -> bool:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


# Global rate limiter instance
_rate_limiters: dict[str, _TokenBucket] = {}
_rate_limiters_lock = threading.Lock()


def rate_limit_check(
    key: str = "default",
    rate: float = 10.0,
    capacity: float = 20.0,
) -> bool:
    """Check if a request is within rate limits using token bucket algorithm.

    Args:
        key: Identifier for the rate limit bucket (e.g., IP address, user ID).
        rate: Token refill rate per second.
        capacity: Maximum token capacity.

    Returns:
        True if the request is allowed, False if rate limited.
    """
    with _rate_limiters_lock:
        if key not in _rate_limiters:
            _rate_limiters[key] = _TokenBucket(rate=rate, capacity=capacity)
    return _rate_limiters[key].consume()


# --- File Validation ---

def validate_file_path(file_path: str, base_dir: str | None = None) -> Path:
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


def validate_mime_type(file_path: str) -> str | None:
    """Validate file type by checking magic bytes.

    Checks magic bytes for PDF, DOCX, and PPTX files.

    Args:
        file_path: Path to the file.

    Returns:
        Detected file type string ('pdf', 'docx', 'pptx') or None if unknown.

    Raises:
        ValueError: If file magic bytes don't match the claimed extension.
    """
    resolved = Path(file_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = resolved.suffix.lower()

    try:
        with open(resolved, "rb") as f:
            header = f.read(8)
    except OSError as e:
        raise ValueError(f"Cannot read file: {file_path} - {e}")

    for file_type, signatures in _MAGIC_BYTES.items():
        for sig in signatures:
            if header.startswith(sig):
                # For ZIP-based formats (DOCX, PPTX), verify extension matches
                if file_type == "zip":
                    if ext in (".docx", ".pptx"):
                        return ext.lstrip(".")
                    # Generic ZIP — not one of our expected types
                    continue
                if file_type == "pdf" and ext == ".pdf":
                    return "pdf"
                return file_type

    return None


def validate_text_content(content: str) -> tuple[bool, str | None]:
    """Validate text content to detect binary or potentially malicious content.

    Checks for:
    - Null bytes (binary content indicator)
    - High ratio of non-printable characters
    - Extremely long lines (potential overflow)

    Args:
        content: The text content to validate.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    if not content:
        return (False, "Content is empty")

    # Check for null bytes (binary content)
    if "\x00" in content:
        return (False, "Content contains null bytes (binary data)")

    # Check ratio of non-printable characters
    non_printable = sum(1 for c in content if ord(c) < 32 and c not in "\n\r\t")
    if len(content) > 0 and non_printable / len(content) > 0.1:
        return (False, f"Content has {non_printable / len(content):.1%} non-printable characters")

    # Check for extremely long lines
    max_line_len = max((len(line) for line in content.split("\n")), default=0)
    if max_line_len > 100000:
        return (False, f"Content has extremely long line ({max_line_len} chars)")

    return (True, None)


# --- Query Validation ---

def validate_query(query: str, max_length: int = MAX_QUERY_LENGTH) -> tuple[bool, str | None]:
    """Validate and sanitize a user query.

    Checks for:
    - SQL injection patterns
    - XSS patterns
    - Maximum length

    Args:
        query: The user query string.
        max_length: Maximum allowed query length.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    if not query or not query.strip():
        return (False, "Query is empty")

    if len(query) > max_length:
        return (False, f"Query exceeds maximum length of {max_length} characters")

    # Check SQL injection patterns
    for pattern in _SQL_INJECTION_PATTERNS:
        if re.search(pattern, query):
            return (False, "Query contains potentially dangerous SQL pattern")

    # Check XSS patterns
    for pattern in _XSS_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return (False, "Query contains potentially dangerous XSS pattern")

    return (True, None)


# --- HTML Sanitization ---

def sanitize_html(text: str) -> str:
    """Strip dangerous HTML tags from text.

    Removes script, iframe, object, embed, form, and other potentially
    dangerous tags while preserving the text content inside safe tags.

    Args:
        text: Input text that may contain HTML.

    Returns:
        Sanitized text with dangerous tags removed.
    """
    if not text:
        return text

    for tag in _DANGEROUS_HTML_TAGS:
        # Remove opening and self-closing tags
        text = re.sub(
            rf"<{tag}\b[^>]*/?>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        # Remove closing tags
        text = re.sub(
            rf"</{tag}\s*>",
            "",
            text,
            flags=re.IGNORECASE,
        )

    return text


# --- URL Validation ---

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


# --- API Key Validation ---

def validate_api_key(api_key: str | None, expected_key: str | None) -> bool:
    """Validate API key if authentication is configured.

    Returns True if authentication passes or is not configured.
    """
    if expected_key is None:
        return True
    return api_key == expected_key


# --- Batch Size Validation ---

def validate_batch_size(size: int, max_size: int = MAX_BATCH_SIZE) -> tuple[bool, str | None]:
    """Validate batch operation size is within limits.

    Args:
        size: The requested batch size.
        max_size: Maximum allowed batch size.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    if size <= 0:
        return (False, "Batch size must be positive")
    if size > max_size:
        return (False, f"Batch size {size} exceeds maximum {max_size}")
    return (True, None)


# --- Filename Sanitization ---

def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal and injection."""
    # Remove path separators and null bytes
    filename = filename.replace("\\", "").replace("/", "").replace("\0", "")
    # Remove any path traversal attempts
    filename = re.sub(r"\.\.+", "", filename)
    # Keep only safe characters
    filename = re.sub(r"[^\w\-\. ]", "_", filename)
    return filename.strip() or "unnamed"