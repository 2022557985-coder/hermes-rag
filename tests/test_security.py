"""Comprehensive tests for security module."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.security import (
    MAX_FILE_SIZE_BYTES,
    sanitize_filename,
    validate_api_key,
    validate_file_extension,
    validate_file_path,
    validate_file_size,
    validate_url,
)


class TestValidateFilePath:
    """Test path validation and traversal protection."""

    def test_valid_file(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"test")
            path = f.name
        try:
            result = validate_file_path(path)
            assert result.is_absolute()
        finally:
            os.unlink(path)

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            validate_file_path("/nonexistent/file.txt")

    def test_path_traversal(self):
        with pytest.raises(ValueError, match="Path traversal"):
            validate_file_path("../etc/passwd")

    def test_base_dir_restriction(self):
        base = tempfile.mkdtemp()
        inside = os.path.join(base, "test.txt")
        Path(inside).write_text("test")
        try:
            # Should work inside base dir
            result = validate_file_path(inside, base_dir=base)
            assert result.is_absolute()
        finally:
            os.unlink(inside)
            os.rmdir(base)

    def test_base_dir_escape(self):
        base = tempfile.mkdtemp()
        try:
            outside = os.path.join(tempfile.gettempdir(), "outside.txt")
            Path(outside).write_text("test")
            with pytest.raises(ValueError):
                validate_file_path(outside, base_dir=base)
        finally:
            os.rmdir(base)


class TestValidateFileExtension:
    """Test file extension validation."""

    def test_allowed_extensions(self):
        for ext in [".pdf", ".docx", ".pptx", ".txt", ".md", ".csv", ".json"]:
            assert validate_file_extension(f"file{ext}") == ext

    def test_uppercase_extension(self):
        assert validate_file_extension("file.PDF") == ".pdf"

    def test_disallowed_extension(self):
        with pytest.raises(ValueError, match="Unsupported"):
            validate_file_extension("file.exe")

    def test_no_extension(self):
        with pytest.raises(ValueError, match="Unsupported"):
            validate_file_extension("file")


class TestValidateFileSize:
    """Test file size validation."""

    def test_small_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"small file")
            path = f.name
        try:
            validate_file_size(path, max_bytes=1024)
        finally:
            os.unlink(path)

    def test_large_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * 100)
            path = f.name
        try:
            with pytest.raises(ValueError, match="File too large"):
                validate_file_size(path, max_bytes=10)
        finally:
            os.unlink(path)

    def test_default_max_size(self):
        """Default max size should be 50MB."""
        assert MAX_FILE_SIZE_BYTES == 50 * 1024 * 1024


class TestValidateURL:
    """Test URL validation and SSRF protection."""

    def test_valid_http_url(self):
        url = "https://example.com/doc.pdf"
        assert validate_url(url) == url

    def test_blocked_scheme(self):
        with pytest.raises(ValueError, match="not allowed"):
            validate_url("file:///etc/passwd")

    def test_no_hostname(self):
        with pytest.raises(ValueError, match="no hostname"):
            validate_url("http://")

    def test_localhost_blocked(self):
        with pytest.raises(ValueError, match="blocked network"):
            validate_url("http://127.0.0.1:8080/admin")

    def test_private_ip_blocked(self):
        with pytest.raises(ValueError, match="blocked network"):
            validate_url("http://192.168.1.1/config")

    def test_unresolvable_hostname(self):
        with pytest.raises(ValueError, match="Unable to resolve"):
            validate_url("http://thisdoesnotexist.invalid")


class TestValidateAPIKey:
    """Test API key validation."""

    def test_no_expected_key(self):
        assert validate_api_key("any_key", None) is True

    def test_matching_key(self):
        assert validate_api_key("secret", "secret") is True

    def test_mismatching_key(self):
        assert validate_api_key("wrong", "secret") is False

    def test_none_key_with_expected(self):
        assert validate_api_key(None, "secret") is False


class TestSanitizeFilename:
    """Test filename sanitization."""

    def test_normal_filename(self):
        assert sanitize_filename("document.txt") == "document.txt"

    def test_path_traversal(self):
        result = sanitize_filename("../../../etc/passwd")
        assert ".." not in result
        assert result != ""

    def test_null_bytes(self):
        result = sanitize_filename("test\0.txt")
        assert "\0" not in result

    def test_special_chars(self):
        result = sanitize_filename("my file!@#$%^&*.txt")
        assert "!" not in result
        assert "@" not in result

    def test_empty_result(self):
        result = sanitize_filename("!!!")
        # All special chars replaced with "_", result is "___"
        assert result == "___"

    def test_chinese_filename(self):
        result = sanitize_filename("文档.txt")
        assert "文档" in result