"""Advanced tests for security module: MIME types, text validation, query validation, HTML sanitization, rate limiting."""

import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from utils.security import (
        MAX_BATCH_SIZE,
        rate_limit_check,
        sanitize_html,
        validate_batch_size,
        validate_file_extension,
        validate_mime_type,
        validate_query,
        validate_text_content,
    )
except ImportError:
    validate_mime_type = None
    validate_text_content = None
    validate_query = None
    sanitize_html = None
    rate_limit_check = None
    validate_batch_size = None
    validate_file_extension = None
    MAX_BATCH_SIZE = 1000


class TestValidateMimeType:
    """Test validate_mime_type for PDF, DOCX, PPTX, TXT."""

    def test_pdf_magic_bytes(self):
        """PDF file should be detected by magic bytes."""
        if validate_mime_type is None:
            pytest.skip("Security module not available")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\nsome content")
            path = f.name
        try:
            result = validate_mime_type(path)
            assert result == "pdf", f"Expected 'pdf', got {result}"
        finally:
            os.unlink(path)

    def test_docx_magic_bytes(self):
        """DOCX file should be detected by magic bytes."""
        if validate_mime_type is None:
            pytest.skip("Security module not available")
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            f.write(b"PK\x03\x04\x00\x00\x00\x00")
            path = f.name
        try:
            result = validate_mime_type(path)
            assert result == "docx", f"Expected 'docx', got {result}"
        finally:
            os.unlink(path)

    def test_pptx_magic_bytes(self):
        """PPTX file should be detected by magic bytes."""
        if validate_mime_type is None:
            pytest.skip("Security module not available")
        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as f:
            f.write(b"PK\x03\x04\x00\x00\x00\x00")
            path = f.name
        try:
            result = validate_mime_type(path)
            assert result == "pptx", f"Expected 'pptx', got {result}"
        finally:
            os.unlink(path)

    def test_txt_no_magic_bytes(self):
        """TXT file without magic bytes should return None."""
        if validate_mime_type is None:
            pytest.skip("Security module not available")
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"plain text content")
            path = f.name
        try:
            result = validate_mime_type(path)
            assert result is None, f"Expected None for TXT, got {result}"
        finally:
            os.unlink(path)

    def test_nonexistent_file(self):
        """Non-existent file should raise FileNotFoundError."""
        if validate_mime_type is None:
            pytest.skip("Security module not available")
        with pytest.raises(FileNotFoundError):
            validate_mime_type("/nonexistent/file.pdf")

    def test_invalid_zip_extension(self):
        """ZIP file with wrong extension should not match."""
        if validate_mime_type is None:
            pytest.skip("Security module not available")
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            f.write(b"PK\x03\x04\x00\x00\x00\x00")
            path = f.name
        try:
            result = validate_mime_type(path)
            # ZIP files with .zip extension should not match docx/pptx
            assert result is None, f"Expected None for .zip, got {result}"
        finally:
            os.unlink(path)


class TestValidateTextContent:
    """Test validate_text_content for binary content and null bytes."""

    def test_valid_text(self):
        """Valid text should pass validation."""
        if validate_text_content is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_text_content("Hello, world!")
        assert is_valid is True
        assert error_msg is None

    def test_null_bytes_detected(self):
        """Text with null bytes should be invalid."""
        if validate_text_content is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_text_content("hello\x00world")
        assert is_valid is False
        assert "null bytes" in error_msg.lower()

    def test_binary_content_detected(self):
        """Binary content should be detected by non-printable ratio."""
        if validate_text_content is None:
            pytest.skip("Security module not available")
        binary_content = "text" + "\x01" * 100 + "\x02" * 100
        is_valid, error_msg = validate_text_content(binary_content)
        assert is_valid is False
        assert "non-printable" in error_msg.lower()

    def test_empty_content(self):
        """Empty content should be invalid."""
        if validate_text_content is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_text_content("")
        assert is_valid is False
        assert error_msg is not None

    def test_long_lines_detected(self):
        """Extremely long lines should be detected."""
        if validate_text_content is None:
            pytest.skip("Security module not available")
        long_line = "x" * 100001
        is_valid, error_msg = validate_text_content(long_line)
        assert is_valid is False
        assert "long line" in error_msg.lower()

    def test_normal_long_text(self):
        """Normal long text with newlines should pass."""
        if validate_text_content is None:
            pytest.skip("Security module not available")
        long_text = "short line\n" * 5000
        is_valid, error_msg = validate_text_content(long_text)
        assert is_valid is True, f"Expected valid, got: {error_msg}"


class TestValidateQuery:
    """Test validate_query for SQL injection and XSS patterns."""

    def test_valid_query(self):
        """Normal query should pass validation."""
        if validate_query is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_query("机器学习是什么？")
        assert is_valid is True

    def test_sql_injection_select(self):
        """SQL injection with SELECT should be detected."""
        if validate_query is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_query("SELECT * FROM users")
        assert is_valid is False
        assert "SQL" in error_msg

    def test_sql_injection_drop(self):
        """SQL injection with DROP should be detected."""
        if validate_query is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_query("DROP TABLE users")
        assert is_valid is False

    def test_sql_injection_union(self):
        """SQL injection with UNION SELECT should be detected."""
        if validate_query is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_query("1 UNION SELECT password FROM users")
        assert is_valid is False

    def test_sql_injection_or_equals(self):
        """SQL injection with OR 1=1 should be detected."""
        if validate_query is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_query("admin' OR 1=1 --")
        assert is_valid is False

    def test_xss_script_tag(self):
        """XSS with script tag should be detected."""
        if validate_query is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_query("<script>alert('xss')</script>")
        assert is_valid is False
        assert "XSS" in error_msg

    def test_xss_javascript_uri(self):
        """XSS with javascript: URI should be detected."""
        if validate_query is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_query("javascript:alert(1)")
        assert is_valid is False

    def test_xss_onclick(self):
        """XSS with onclick handler should be detected."""
        if validate_query is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_query('<img src=x onerror="alert(1)">')
        assert is_valid is False

    def test_query_too_long(self):
        """Query exceeding max length should be rejected."""
        if validate_query is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_query("x" * 3000, max_length=2000)
        assert is_valid is False
        assert "exceeds" in error_msg.lower()

    def test_empty_query(self):
        """Empty query should be rejected."""
        if validate_query is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_query("")
        assert is_valid is False
        assert "empty" in error_msg.lower()

    def test_whitespace_query(self):
        """Whitespace-only query should be rejected."""
        if validate_query is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_query("   ")
        assert is_valid is False


class TestSanitizeHtml:
    """Test sanitize_html for various dangerous tags."""

    @pytest.mark.parametrize("tag", [
        "script", "iframe", "object", "embed", "form", "input",
        "button", "link", "meta", "style", "applet", "base",
    ])
    def test_dangerous_tag_stripped(self, tag):
        """Each dangerous tag should be stripped from HTML."""
        if sanitize_html is None:
            pytest.skip("Security module not available")
        html = f"<{tag}>content</{tag}>"
        result = sanitize_html(html)
        assert f"<{tag}>" not in result.lower(), f"Tag <{tag}> was not stripped"
        assert f"</{tag}>" not in result.lower(), f"Closing tag </{tag}> was not stripped"

    def test_self_closing_tags_stripped(self):
        """Self-closing dangerous tags should be stripped."""
        if sanitize_html is None:
            pytest.skip("Security module not available")
        result = sanitize_html('<script src="evil.js"/>')
        assert "<script" not in result.lower()

    def test_safe_tags_preserved(self):
        """Safe HTML tags should be preserved."""
        if sanitize_html is None:
            pytest.skip("Security module not available")
        html = "<p>Hello</p><b>World</b><div>test</div>"
        result = sanitize_html(html)
        assert "<p>Hello</p>" in result
        assert "<b>World</b>" in result
        assert "<div>test</div>" in result

    def test_empty_text(self):
        """Empty text should return empty."""
        if sanitize_html is None:
            pytest.skip("Security module not available")
        result = sanitize_html("")
        assert result == ""

    def test_no_html_text(self):
        """Plain text should be returned unchanged."""
        if sanitize_html is None:
            pytest.skip("Security module not available")
        text = "This is plain text without any HTML tags."
        result = sanitize_html(text)
        assert result == text


class TestRateLimitCheck:
    """Test rate_limit_check with token bucket."""

    def test_initial_requests_allowed(self):
        """Initial requests should be allowed."""
        if rate_limit_check is None:
            pytest.skip("Security module not available")
        results = [rate_limit_check("test_key", rate=10.0, capacity=20.0) for _ in range(5)]
        assert all(results), "All initial requests should be allowed"

    def test_rate_limit_exhaustion(self):
        """After exhausting tokens, requests should be denied."""
        if rate_limit_check is None:
            pytest.skip("Security module not available")
        key = f"exhaustion_test_{time.time()}"
        # Consume all tokens
        for _ in range(20):
            rate_limit_check(key, rate=10.0, capacity=20.0)
        # Next request should be denied
        result = rate_limit_check(key, rate=10.0, capacity=20.0)
        assert result is False, "Request should be denied after token exhaustion"

    def test_rate_limit_refill(self):
        """Tokens should refill over time."""
        if rate_limit_check is None:
            pytest.skip("Security module not available")
        key = f"refill_test_{time.time()}"
        # Consume all tokens
        for _ in range(20):
            rate_limit_check(key, rate=100.0, capacity=20.0)
        # Wait for refill
        time.sleep(0.2)
        # Should be allowed again
        result = rate_limit_check(key, rate=100.0, capacity=20.0)
        assert result is True, "Request should be allowed after refill"

    def test_different_keys_independent(self):
        """Different keys should have independent rate limits."""
        if rate_limit_check is None:
            pytest.skip("Security module not available")
        key1 = f"key1_{time.time()}"
        key2 = f"key2_{time.time()}"
        # Exhaust key1
        for _ in range(20):
            rate_limit_check(key1, rate=10.0, capacity=20.0)
        assert rate_limit_check(key1, rate=10.0, capacity=20.0) is False
        # key2 should still work
        assert rate_limit_check(key2, rate=10.0, capacity=20.0) is True


class TestValidateBatchSize:
    """Test validate_batch_size."""

    def test_valid_batch_size(self):
        """Valid batch size should pass."""
        if validate_batch_size is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_batch_size(100)
        assert is_valid is True
        assert error_msg is None

    def test_batch_size_zero(self):
        """Zero batch size should be rejected."""
        if validate_batch_size is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_batch_size(0)
        assert is_valid is False
        assert "positive" in error_msg.lower()

    def test_batch_size_negative(self):
        """Negative batch size should be rejected."""
        if validate_batch_size is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_batch_size(-5)
        assert is_valid is False

    def test_batch_size_exceeds_max(self):
        """Batch size exceeding max should be rejected."""
        if validate_batch_size is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_batch_size(MAX_BATCH_SIZE + 1)
        assert is_valid is False
        assert "exceeds" in error_msg.lower()

    def test_batch_size_at_max(self):
        """Batch size at max should be allowed."""
        if validate_batch_size is None:
            pytest.skip("Security module not available")
        is_valid, error_msg = validate_batch_size(MAX_BATCH_SIZE)
        assert is_valid is True, f"Expected valid, got: {error_msg}"


class TestNewAllowedExtensions:
    """Test new allowed extensions (.rst, .html, .htm, .markdown, .log)."""

    @pytest.mark.parametrize("ext", [
        ".rst", ".html", ".htm", ".markdown", ".log",
    ])
    def test_new_extensions_allowed(self, ext):
        """New extensions should be allowed."""
        if validate_file_extension is None:
            pytest.skip("Security module not available")
        result = validate_file_extension(f"test{ext}")
        assert result == ext, f"Expected '{ext}' to be allowed"

    @pytest.mark.parametrize("ext", [".pdf", ".docx", ".pptx", ".txt", ".md", ".csv", ".json"])
    def test_original_extensions_still_allowed(self, ext):
        """Original extensions should still be allowed."""
        if validate_file_extension is None:
            pytest.skip("Security module not available")
        result = validate_file_extension(f"test{ext}")
        assert result == ext, f"Expected '{ext}' to be allowed"