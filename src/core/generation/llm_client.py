"""LLM client interface supporting OpenAI API and Ollama local deployment."""

import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from collections.abc import Generator
from pathlib import Path
from typing import Any

logger = logging.getLogger("hermes_rag")

# Prompt template loaded from a file so prompt tweaks take effect without
# restarting the server. Falls back to an embedded default if missing.
_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompt_template.txt"
_PROMPT_TEMPLATE_FALLBACK = """你是一个专业、严谨的知识库问答助手。你必须基于下方「参考文档」回答问题，禁止编造任何信息。

## 核心规则
1. 直接作答，先给结论再展开细节。禁止用"根据现有资料无法回答"这类话术回避问题。
2. 只要参考文档足以推导出答案，就必须回答；允许跨文档聚合、数值计算（写出步骤并核对单位与数量级）、多跳推理与排序推导。
3. 排序/顺序题必须严格按参考文档列举的顺序作答；事件年份必须与原文精确对应，缺失时如实说明，禁止用其他年份事件顶替。
4. 引用真实性：每个 [来源 N] 标注的内容必须能在该片段原文中找到；找不到依据的信息不得标注来源，必须写"参考文档未提供该信息"。
5. 只有参考文档完全没有相关内容时才拒答，并具体说明缺少哪一项信息。
6. 数字、日期、人名、地名必须与文档一致；文档只给出年份时，禁止编造月份和日期。

## 参考文档
{context_text}

## 用户问题
{query}

## 回答
"""


def _load_prompt_template() -> str:
    """Read the prompt template from disk (hot-reloadable), else fallback."""
    try:
        if _PROMPT_TEMPLATE_PATH.exists():
            return _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8-sig")
    except OSError:
        pass
    return _PROMPT_TEMPLATE_FALLBACK


class LLMResponseCache:
    """Simple LRU cache for LLM responses to avoid redundant generation."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 1800):
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

    def _make_key(self, prompt: str, model: str = "") -> str:
        return hashlib.md5((prompt + model).encode("utf-8")).hexdigest()

    def get(self, prompt: str, model: str = "") -> str | None:
        key = self._make_key(prompt, model)
        if key in self._cache:
            response, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return response
            else:
                del self._cache[key]
        return None

    def set(self, prompt: str, response: str, model: str = "") -> None:
        key = self._make_key(prompt, model)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (response, time.time())
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class LLMClient:
    """Pluggable LLM client supporting OpenAI and Ollama."""

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "deepseek-r1:8b",
        base_url: str = "http://localhost:11434",
        api_key: str = "",
        temperature: float = 0.1,
        max_context_tokens: int = 2048,
        fallback_model: str | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        enable_response_cache: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_context_tokens = max_context_tokens
        self.fallback_model = fallback_model
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._response_cache = LLMResponseCache() if enable_response_cache else None

    def count_tokens(self, text: str) -> int:
        """Estimate token count for a text.

        Rough estimation:
        - Chinese characters: 1.5 tokens each
        - English words: 1.2 tokens each
        - Numbers/punctuation: 1 token each

        Args:
            text: Input text to estimate tokens for.

        Returns:
            Estimated token count as integer.
        """
        if not text:
            return 0

        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        # Remaining characters (numbers, punctuation, whitespace, etc.)
        remaining = len(text) - chinese_chars - sum(
            len(w) for w in re.findall(r'[a-zA-Z]+', text)
        )

        tokens = chinese_chars * 1.5 + english_words * 1.2 + remaining * 1.0
        return max(1, int(tokens))

    def truncate_context(
        self,
        context_chunks: list[dict[str, Any]],
        query: str,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """Truncate context chunks to fit within max_context_tokens.

        Prioritizes higher-scored chunks and compresses long chunks by
        keeping only the most relevant sentences.

        Args:
            context_chunks: List of context chunk dicts with 'score' and 'text'.
            query: The user query (to account for its token cost).
            max_tokens: Max tokens allowed. Defaults to self.max_context_tokens.

        Returns:
            Truncated list of context chunks.
        """
        if max_tokens is None:
            max_tokens = self.max_context_tokens

        # Reserve tokens for prompt template and query
        query_tokens = self.count_tokens(query)
        prompt_overhead = 500  # Approximate overhead for prompt template
        available_tokens = max_tokens - query_tokens - prompt_overhead

        if available_tokens <= 0:
            return []

        # Sort chunks by score descending (if scores exist)
        sorted_chunks = sorted(
            context_chunks,
            key=lambda x: x.get("score", 0),
            reverse=True,
        )

        truncated: list[dict[str, Any]] = []
        used_tokens = 0

        for chunk in sorted_chunks:
            chunk_text = chunk.get("text", "")
            chunk_tokens = self.count_tokens(chunk_text)

            if used_tokens + chunk_tokens <= available_tokens:
                # Chunk fits entirely
                truncated.append(chunk)
                used_tokens += chunk_tokens
            elif used_tokens < available_tokens:
                # Partial fit: compress the chunk by keeping first N sentences
                remaining = available_tokens - used_tokens
                if remaining > 50:  # Only include if we can fit meaningful content
                    compressed_text = self._compress_text(chunk_text, remaining)
                    compressed_chunk = dict(chunk)
                    compressed_chunk["text"] = compressed_text
                    compressed_chunk["_compressed"] = True
                    truncated.append(compressed_chunk)
                    used_tokens += self.count_tokens(compressed_text)
                break

        return truncated

    @staticmethod
    def _compress_text(text: str, max_tokens: int) -> str:
        """Compress text by keeping only the most informative sentences.

        Uses a simple heuristic: keep the first sentence and sentences
        that contain key terms (numbers, technical terms, proper nouns).

        Args:
            text: Input text to compress.
            max_tokens: Target max tokens.

        Returns:
            Compressed text string.
        """
        import re
        sentences = re.split(r'(?<=[。！？.!?\n])\s*', text)
        if not sentences:
            return text[:max_tokens * 2]  # Rough fallback

        # Always keep the first sentence
        result = [sentences[0]]
        token_count = len(sentences[0]) // 2  # Rough estimate

        # Score remaining sentences by informativeness
        for sent in sentences[1:]:
            sent_tokens = len(sent) // 2
            if token_count + sent_tokens > max_tokens:
                break
            # Keep sentences with numbers, English terms, or proper nouns
            has_info = bool(
                re.search(r'\d+', sent)
                or re.search(r'[A-Z][a-z]{2,}', sent)
                or re.search(r'《.*?》|".*?"', sent)
                or len(sent) > 20  # Non-trivial length
            )
            if has_info:
                result.append(sent)
                token_count += sent_tokens

        return "".join(result) if result else text[:max_tokens * 2]

    def _validate_response(self, response_text: str) -> tuple[bool, str | None]:
        """Validate LLM response for common issues.

        Args:
            response_text: The generated response text.

        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
        """
        if not response_text or not response_text.strip():
            return (False, "Empty response from LLM")

        # Check for error responses
        error_prefixes = ["[LLM Error:", "[Error:", "Error:"]
        for prefix in error_prefixes:
            if response_text.startswith(prefix):
                return (False, response_text)

        return (True, None)

    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """Execute a function with exponential backoff retry logic.

        Args:
            func: The function to call.
            *args: Positional arguments to pass to func.
            **kwargs: Keyword arguments to pass to func.

        Returns:
            The return value of func on success.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        f"LLM call attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

        raise RuntimeError(
            f"LLM call failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def generate(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
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
        # Truncate context to fit within token limits
        truncated_chunks = self.truncate_context(context_chunks, query)

        prompt = self._build_prompt(query, truncated_chunks)

        # Check response cache
        if self._response_cache is not None:
            cached = self._response_cache.get(prompt, model=self.model)
            if cached is not None:
                logger.debug("LLM response cache hit")
                return cached

        try:
            if self.provider == "openai":
                result = self._retry_with_backoff(
                    self._generate_openai, prompt, stream
                )
            else:
                result = self._retry_with_backoff(
                    self._generate_ollama, prompt, stream
                )
        except RuntimeError as e:
            return f"[LLM Error: {e}]"

        # Validate response
        is_valid, error_msg = self._validate_response(result)
        if not is_valid:
            return f"[LLM Error: {error_msg}]"

        # Cache valid response
        if self._response_cache is not None:
            self._response_cache.set(prompt, result, model=self.model)

        return result

    def generate_stream(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
    ) -> Generator[str, None, None]:
        """Generate an answer and yield tokens one at a time.

        Args:
            query: User query.
            context_chunks: Retrieved context chunks.

        Yields:
            Tokens (strings) one at a time.
        """
        truncated_chunks = self.truncate_context(context_chunks, query)
        prompt = self._build_prompt(query, truncated_chunks)

        try:
            if self.provider == "openai":
                yield from self._retry_with_backoff(
                    self._generate_openai_stream, prompt
                )
            else:
                yield from self._retry_with_backoff(
                    self._generate_ollama_stream, prompt
                )
        except RuntimeError as e:
            yield f"[LLM Error: {e}]"

    def generate_with_fallback(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
    ) -> str:
        """Generate using primary model, falling back to a simpler model on failure.

        Args:
            query: User query.
            context_chunks: Retrieved context chunks.

        Returns:
            Generated answer string.
        """
        # Try primary model first
        result = self.generate(query, context_chunks)

        is_valid, _ = self._validate_response(result)
        if is_valid:
            return result

        # If fallback model is configured, try it
        if self.fallback_model:
            logger.warning(
                f"Primary model '{self.model}' failed, "
                f"falling back to '{self.fallback_model}'"
            )
            original_model = self.model
            self.model = self.fallback_model
            try:
                result = self.generate(query, context_chunks)
            finally:
                self.model = original_model

        return result

    def _build_prompt(
        self, query: str, context_chunks: list[dict[str, Any]]
    ) -> str:
        """Build a prompt with context and citations."""
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            text = chunk.get("text", "")
            heading = chunk.get("metadata", {}).get("heading_path", "")
            page = chunk.get("metadata", {}).get("page", "")

            citation = f"[来源 {i}]"
            if heading:
                citation += f" {heading}"
            if page:
                citation += f" (第{page}页)"

            context_parts.append(f"{citation}\n{text}")

        context_text = "\n\n---\n\n".join(context_parts)
        return _load_prompt_template().format(
            context_text=context_text, query=query
        )
    def _generate_ollama(self, prompt: str, stream: bool = False) -> str:
        """Generate using Ollama API."""
        import requests

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
            full_text: list[str] = []
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        full_text.append(data["response"])
                    if data.get("done", False):
                        break
            return "".join(full_text)
        else:
            return response.json().get("response", "")

    def _generate_ollama_stream(self, prompt: str) -> Generator[str, None, None]:
        """Generate using Ollama API with streaming, yielding tokens."""
        import requests

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_context_tokens,
                },
            },
            timeout=120,
            stream=True,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "response" in data:
                    yield data["response"]
                if data.get("done", False):
                    break

    def _generate_openai(self, prompt: str, stream: bool = False) -> str:
        """Generate using OpenAI-compatible API."""
        import requests

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

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
            full_text: list[str] = []
            for line in response.iter_lines():
                if line and line.startswith(b"data: "):
                    data_str = line[6:].decode("utf-8")
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta:
                        full_text.append(delta["content"])
            return "".join(full_text)
        else:
            try:
                return response.json()["choices"][0]["message"]["content"]
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                raise ValueError(f"Malformed JSON response from OpenAI: {e}")

    def _generate_openai_stream(self, prompt: str) -> Generator[str, None, None]:
        """Generate using OpenAI-compatible API with streaming, yielding tokens."""
        import requests

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

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
                "stream": True,
            },
            headers=headers,
            timeout=120,
            stream=True,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line and line.startswith(b"data: "):
                data_str = line[6:].decode("utf-8")
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = data.get("choices", [{}])[0].get("delta", {})
                if "content" in delta:
                    yield delta["content"]