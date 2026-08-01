"""Advanced tests for LLMClient: count_tokens, truncate_context, _build_prompt, fallback, validation."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from src.core.generation.llm_client import LLMClient
except ImportError:
    LLMClient = None


class TestCountTokens:
    """Test count_tokens for Chinese, English, and mixed text."""

    @pytest.mark.parametrize("text, expected_min", [
        ("", 0),
        ("hello", 1),
        ("你好", 1),
        ("机器学习", 1),
        ("hello world", 2),
        ("机器学习与深度学习", 1),
        ("Python 3.11 安装教程", 3),
        ("a a a a a a a a a a", 5),
        ("中 中 中 中 中 中 中 中 中 中", 5),
    ])
    def test_count_tokens_basic(self, text, expected_min):
        """Test basic token counting for various text types."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient()
        count = client.count_tokens(text)
        assert count >= expected_min, (
            f"Text={text[:30]!r}: expected >= {expected_min}, got {count}"
        )

    def test_count_tokens_empty(self):
        """Empty text should return 0 tokens."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient()
        assert client.count_tokens("") == 0

    def test_count_tokens_none(self):
        """None text should return 0 tokens."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient()
        assert client.count_tokens(None) == 0

    def test_count_tokens_chinese(self):
        """Chinese text should estimate ~1.5 tokens per char."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient()
        text = "你好世界"
        count = client.count_tokens(text)
        # 4 Chinese chars * 1.5 = 6, min 1
        assert count >= 4, f"Chinese text: expected >= 4, got {count}"

    def test_count_tokens_mixed(self):
        """Mixed Chinese-English text should be estimated correctly."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient()
        text = "Python 机器学习"
        count = client.count_tokens(text)
        assert count >= 3, f"Mixed text: expected >= 3, got {count}"


class TestTruncateContext:
    """Test truncate_context when context exceeds max_tokens."""

    def test_truncate_within_limit(self):
        """Context within token limit should not be truncated."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient(max_context_tokens=9999)
        chunks = [
            {"text": "short text", "metadata": {}},
            {"text": "another short text", "metadata": {}},
        ]
        result = client.truncate_context(chunks, "test query")
        assert len(result) == 2

    def test_truncate_exceeds_limit(self):
        """Context exceeding token limit should be truncated."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient(max_context_tokens=10)
        # Use text with many separate words to increase token count
        chunks = [
            {"text": "word " * 50, "metadata": {}},
            {"text": "word " * 50, "metadata": {}},
            {"text": "word " * 50, "metadata": {}},
        ]
        result = client.truncate_context(chunks, "query")
        # Count tokens in "word" * 50 ~ 50 * 1.2 = 60 tokens, should truncate
        assert len(result) < 3, f"Expected truncated result, got {len(result)} chunks"

    def test_truncate_empty_context(self):
        """Empty context should return empty list."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient()
        result = client.truncate_context([], "test")
        assert result == []

    def test_truncate_query_exceeds_all(self):
        """Query that exceeds max_tokens should return empty context."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient(max_context_tokens=5)
        # Query with many separate words: each word ~1.2 tokens
        chunks = [{"text": "some text", "metadata": {}}]
        # "word " * 10 = 10 words * 1.2 = 12 tokens > 5
        result = client.truncate_context(chunks, "word " * 10)
        assert result == []

    def test_truncate_custom_max_tokens(self):
        """Custom max_tokens parameter should override instance default."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient(max_context_tokens=9999)
        chunks = [
            {"text": "word " * 50, "metadata": {}},
            {"text": "word " * 50, "metadata": {}},
        ]
        result = client.truncate_context(chunks, "query", max_tokens=5)
        assert len(result) == 0, "Custom max_tokens=5 should truncate all"


class TestBuildPrompt:
    """Test _build_prompt with various inputs."""

    def test_build_prompt_empty_context(self):
        """Prompt with empty context should still contain query."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient()
        prompt = client._build_prompt("测试问题", [])
        assert "测试问题" in prompt
        assert "参考文档" in prompt

    def test_build_prompt_special_characters(self):
        """Prompt with special characters should be preserved."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient()
        chunks = [
            {"text": "text with <tag> & special chars", "metadata": {"source": "doc.txt"}},
        ]
        prompt = client._build_prompt("query with ? and !", chunks)
        assert "<tag>" in prompt
        assert "?" in prompt
        assert "!" in prompt

    def test_build_prompt_citations(self):
        """Prompt should include citation markers."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient()
        chunks = [
            {"text": "text1", "metadata": {"source": "doc1.txt", "page": "5"}},
            {"text": "text2", "metadata": {"source": "doc2.txt", "heading_path": "Chapter 1"}},
        ]
        prompt = client._build_prompt("test", chunks)
        assert "[来源 1]" in prompt
        assert "(第5页)" in prompt
        assert "Chapter 1" in prompt
        assert "[来源 2]" in prompt

    def test_build_prompt_without_metadata(self):
        """Prompt should handle chunks without metadata."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient()
        chunks = [{"text": "plain text"}]
        prompt = client._build_prompt("test", chunks)
        assert "plain text" in prompt
        assert "[来源 1]" in prompt


class TestGenerateWithFallback:
    """Test generate_with_fallback configuration."""

    def test_fallback_not_configured(self):
        """Without fallback model, should return primary result."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient(fallback_model=None)
        result = client.generate_with_fallback("test", [{"text": "test", "metadata": {}}])
        assert isinstance(result, str)

    def test_fallback_configured(self):
        """Fallback model should be stored correctly."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient(fallback_model="qwen2.5:1.5b")
        assert client.fallback_model == "qwen2.5:1.5b"


class TestResponseValidation:
    """Test response validation (empty responses, error responses)."""

    def test_validate_empty_response(self):
        """Empty response should be invalid."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient()
        is_valid, error_msg = client._validate_response("")
        assert is_valid is False
        assert error_msg is not None

    def test_validate_whitespace_response(self):
        """Whitespace-only response should be invalid."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient()
        is_valid, error_msg = client._validate_response("   ")
        assert is_valid is False

    def test_validate_error_response(self):
        """Error-prefixed response should be invalid."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient()
        is_valid, error_msg = client._validate_response("[LLM Error: something went wrong]")
        assert is_valid is False

    def test_validate_error_prefix(self):
        """Response starting with 'Error:' should be invalid."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient()
        is_valid, error_msg = client._validate_response("Error: invalid input")
        assert is_valid is False

    def test_validate_valid_response(self):
        """Valid response should pass validation."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient()
        is_valid, error_msg = client._validate_response("This is a valid response.")
        assert is_valid is True
        assert error_msg is None


class TestProviderSwitching:
    """Test provider switching (ollama vs openai)."""

    def test_ollama_configuration(self):
        """Ollama provider should have correct defaults."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient(provider="ollama")
        assert client.provider == "ollama"
        assert client.base_url == "http://localhost:11434"

    def test_openai_configuration(self):
        """OpenAI provider should have correct configuration."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient(
            provider="openai",
            model="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
        )
        assert client.provider == "openai"
        assert client.model == "gpt-4"
        assert client.base_url == "https://api.openai.com/v1"
        assert client.api_key == "sk-test"


class TestTemperatureBoundaries:
    """Test temperature boundaries."""

    @pytest.mark.parametrize("temperature", [
        0.0, 0.1, 0.5, 1.0, 2.0,
    ])
    def test_temperature_values(self, temperature):
        """Various temperature values should be accepted."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient(temperature=temperature)
        assert client.temperature == temperature


class TestMaxContextTokensBoundaries:
    """Test max_context_tokens boundaries."""

    @pytest.mark.parametrize("max_tokens", [
        512, 1024, 2048, 4096, 8192,
    ])
    def test_max_context_tokens_values(self, max_tokens):
        """Various max_context_tokens values should be accepted."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        client = LLMClient(max_context_tokens=max_tokens)
        assert client.max_context_tokens == max_tokens


class TestErrorHandling:
    """Test error handling for malformed JSON."""

    @patch("requests.post")
    def test_openai_malformed_json(self, mock_post):
        """Malformed JSON from OpenAI should be handled."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("malformed JSON")
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = LLMClient(provider="openai", api_key="sk-test")
        result = client.generate("test", [{"text": "test", "metadata": {}}])
        assert "[LLM Error:" in result

    @patch("requests.post")
    def test_openai_missing_choices(self, mock_post):
        """Missing choices in OpenAI response should be handled."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": []}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = LLMClient(provider="openai", api_key="sk-test")
        result = client.generate("test", [{"text": "test", "metadata": {}}])
        assert "[LLM Error:" in result

    @patch("requests.post")
    def test_ollama_empty_response(self, mock_post):
        """Empty response from Ollama should be handled."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": ""}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = LLMClient(provider="ollama")
        result = client.generate("test", [{"text": "test", "metadata": {}}])
        assert "[LLM Error:" in result

    @patch("requests.post")
    def test_retry_with_backoff(self, mock_post):
        """Retry logic should work with exponential backoff."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        import requests
        # Fail twice then succeed
        mock_post.side_effect = [
            requests.ConnectionError("fail 1"),
            requests.ConnectionError("fail 2"),
            MagicMock(
                json=lambda: {"response": "success"},
                raise_for_status=MagicMock(),
            ),
        ]

        client = LLMClient(provider="ollama", max_retries=3, retry_base_delay=0.01)
        result = client.generate("test", [{"text": "test", "metadata": {}}])
        assert "success" in result

    @patch("requests.post")
    def test_retry_exhausted(self, mock_post):
        """All retries exhausted should return error."""
        if LLMClient is None:
            pytest.skip("LLMClient module not available")
        import requests
        mock_post.side_effect = requests.ConnectionError("always fail")

        client = LLMClient(provider="ollama", max_retries=2, retry_base_delay=0.01)
        result = client.generate("test", [{"text": "test", "metadata": {}}])
        assert "[LLM Error:" in result