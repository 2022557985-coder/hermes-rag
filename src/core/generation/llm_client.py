"""LLM client interface supporting OpenAI API and Ollama local deployment."""

from typing import List, Dict, Any, Optional, Iterator


class LLMClient:
    """Pluggable LLM client supporting OpenAI and Ollama."""

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        api_key: str = "",
        temperature: float = 0.1,
        max_context_tokens: int = 2048,
    ):
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_context_tokens = max_context_tokens

    def generate(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        stream: bool = False,
    ) -> str:
        """Generate an answer based on query and retrieved context.

        Args:
            query: User query.
            context_chunks: Retrieved context chunks.
            stream: Whether to stream the response.

        Returns:
            Generated answer string.
        """
        prompt = self._build_prompt(query, context_chunks)

        if self.provider == "openai":
            return self._generate_openai(prompt, stream)
        else:
            return self._generate_ollama(prompt, stream)

    def _build_prompt(
        self, query: str, context_chunks: List[Dict[str, Any]]
    ) -> str:
        """Build a prompt with context and citations."""
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            text = chunk.get("text", "")
            source = chunk.get("metadata", {}).get("source", "unknown")
            page = chunk.get("metadata", {}).get("page", "")
            heading = chunk.get("metadata", {}).get("heading_path", "")

            citation = f"[来源 {i}]"
            if heading:
                citation += f" {heading}"
            if page:
                citation += f" (第{page}页)"

            context_parts.append(f"{citation}\n{text}")

        context_text = "\n\n---\n\n".join(context_parts)

        prompt = f"""请基于以下参考文档回答用户的问题。如果参考文档中没有相关信息，请明确说明"根据现有资料无法回答"。

参考文档：
{context_text}

用户问题：{query}

请提供准确、简洁的回答，并在回答中引用来源编号（如 [来源 1]）。"""

        return prompt

    def _generate_ollama(self, prompt: str, stream: bool = False) -> str:
        """Generate using Ollama API."""
        import requests

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": stream,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_context_tokens,
                    },
                },
                timeout=60,
            )
            response.raise_for_status()

            if stream:
                # Collect streaming response
                full_text = []
                for line in response.iter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if "response" in data:
                            full_text.append(data["response"])
                        if data.get("done", False):
                            break
                return "".join(full_text)
            else:
                return response.json().get("response", "")

        except requests.ConnectionError as e:
            return f"[LLM Error: Connection failed - {e}]"
        except requests.Timeout as e:
            return f"[LLM Error: Request timed out - {e}]"
        except requests.HTTPError as e:
            return f"[LLM Error: HTTP {e.response.status_code} - {e}]"
        except (ValueError, KeyError) as e:
            return f"[LLM Error: Invalid response format - {e}]"

    def _generate_openai(self, prompt: str, stream: bool = False) -> str:
        """Generate using OpenAI-compatible API."""
        import requests

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "你是一个基于参考文档回答问题的助手。"},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_context_tokens,
                    "stream": stream,
                },
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()

            if stream:
                full_text = []
                for line in response.iter_lines():
                    if line and line.startswith(b"data: "):
                        data_str = line[6:].decode("utf-8")
                        if data_str == "[DONE]":
                            break
                        import json
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            full_text.append(delta["content"])
                return "".join(full_text)
            else:
                return response.json()["choices"][0]["message"]["content"]

        except requests.ConnectionError as e:
            return f"[LLM Error: Connection failed - {e}]"
        except requests.Timeout as e:
            return f"[LLM Error: Request timed out - {e}]"
        except requests.HTTPError as e:
            return f"[LLM Error: HTTP {e.response.status_code} - {e}]"
        except (ValueError, KeyError, IndexError) as e:
            return f"[LLM Error: Invalid response format - {e}]"