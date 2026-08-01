"""Tests for LLM client with mocked HTTP responses."""

from unittest.mock import MagicMock, patch

from src.core.generation.llm_client import LLMClient


class TestLLMClient:
    """Tests for LLMClient."""

    def test_initialization_ollama(self):
        client = LLMClient(
            provider="ollama",
            model="qwen2.5:7b",
            base_url="http://localhost:11434",
        )
        assert client.provider == "ollama"
        assert client.model == "qwen2.5:7b"

    def test_initialization_openai(self):
        client = LLMClient(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="sk-test",
        )
        assert client.provider == "openai"
        assert client.api_key == "sk-test"

    def test_build_prompt_basic(self):
        client = LLMClient()
        chunks = [
            {"text": "Machine learning is a subset of AI.", "metadata": {"source": "doc1.txt", "page": "1"}},
            {"text": "Deep learning uses neural networks.", "metadata": {"source": "doc2.txt", "heading_path": "Neural Networks"}},
        ]
        prompt = client._build_prompt("What is ML?", chunks)
        assert "What is ML?" in prompt
        assert "Machine learning is a subset of AI" in prompt
        assert "Deep learning uses neural networks" in prompt
        assert "[来源 1]" in prompt
        assert "[来源 2]" in prompt

    def test_build_prompt_empty_context(self):
        client = LLMClient()
        prompt = client._build_prompt("test", [])
        assert "test" in prompt
        assert "参考文档" in prompt

    @patch("requests.post")
    def test_generate_ollama_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Machine learning is a field of AI."}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = LLMClient(provider="ollama")
        result = client.generate("What is ML?", [{"text": "ML is AI.", "metadata": {}}])
        assert "Machine learning" in result

    @patch("requests.post")
    def test_generate_ollama_connection_error(self, mock_post):
        import requests
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        client = LLMClient(provider="ollama")
        result = client.generate("test", [{"text": "test", "metadata": {}}])
        assert "[LLM Error:" in result and "Connection refused" in result

    @patch("requests.post")
    def test_generate_ollama_timeout(self, mock_post):
        import requests
        mock_post.side_effect = requests.Timeout("Request timed out")

        client = LLMClient(provider="ollama")
        result = client.generate("test", [{"text": "test", "metadata": {}}])
        assert "[LLM Error:" in result and "timed out" in result.lower()

    @patch("requests.post")
    def test_generate_ollama_http_error(self, mock_post):
        import requests
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error", response=mock_response)
        mock_post.return_value = mock_response

        client = LLMClient(provider="ollama")
        result = client.generate("test", [{"text": "test", "metadata": {}}])
        assert "[LLM Error:" in result and "500" in result

    @patch("requests.post")
    def test_generate_openai_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "AI is a broad field."}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = LLMClient(provider="openai", api_key="sk-test")
        result = client.generate("What is AI?", [{"text": "AI is artificial intelligence.", "metadata": {}}])
        assert "AI is a broad field" in result

    @patch("requests.post")
    def test_generate_openai_connection_error(self, mock_post):
        import requests
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        client = LLMClient(provider="openai", api_key="sk-test")
        result = client.generate("test", [{"text": "test", "metadata": {}}])
        assert "[LLM Error:" in result and "Connection refused" in result

    @patch("requests.post")
    def test_generate_openai_invalid_response(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {}  # Missing choices
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = LLMClient(provider="openai", api_key="sk-test")
        result = client.generate("test", [{"text": "test", "metadata": {}}])
        assert "[LLM Error:" in result and "Malformed JSON" in result

    @patch("requests.post")
    def test_generate_ollama_stream(self, mock_post):
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'{"response": "Hello", "done": false}',
            b'{"response": " World", "done": true}',
        ]
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = LLMClient(provider="ollama")
        result = client.generate("test", [{"text": "test", "metadata": {}}], stream=True)
        assert "Hello World" in result