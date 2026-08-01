"""Configuration management using Pydantic Settings and YAML."""

import os
import re
import threading
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseSettings):
    model_name: str = "BAAI/bge-m3"  # Multilingual model (Chinese + English)
    model_name_zh: str = "BAAI/bge-m3"
    device: str = "cpu"
    normalize: bool = True
    batch_size: int = 32
    use_onnx: bool = False


class ChunkingConfig(BaseSettings):
    chunk_size: int = 512
    chunk_overlap: int = 128
    semantic_threshold: float = 0.65
    min_chunk_size: int = 50
    max_section_size: int = 512


class ChromaDBConfig(BaseSettings):
    persist_directory: str = "./data/chroma_db"
    collection_name: str = "hermes_rag"
    hnsw_ef_construction: int = 100
    hnsw_M: int = 8
    hnsw_ef_search: int = 50
    cache_max_entries: int = 10000


class BM25Config(BaseSettings):
    b: float = 0.75
    k1: float = 1.5
    max_index_entries: int = 100000
    fallback_db_path: str = "./data/bm25_fallback.db"


class RRFWeights(BaseSettings):
    default_dense: float = 0.5
    default_sparse: float = 0.5
    product_code_dense: float = 0.3
    product_code_sparse: float = 0.7
    colloquial_dense: float = 0.8
    colloquial_sparse: float = 0.2


class RetrievalConfig(BaseSettings):
    dense_top_k: int = 100
    sparse_top_k: int = 100
    rrf_k: int = 60
    fusion_top_k: int = 50
    final_top_k: int = 5
    rrf_weights: RRFWeights = Field(default_factory=RRFWeights)


class RerankingConfig(BaseSettings):
    enabled: bool = True
    model_name: str = "BAAI/bge-reranker-v2-m3"  # Multilingual reranker
    device: str = "cpu"
    batch_size: int = 16
    max_candidates: int = 50
    timeout_seconds: float = 1.5


class QueryExpansionConfig(BaseSettings):
    hyde_enabled: bool = False
    hyde_model: str = "google-t5/t5-small"
    synonym_enabled: bool = True
    max_synonyms: int = 3


class CacheConfig(BaseSettings):
    enabled: bool = True
    similarity_threshold: float = 0.95
    max_cache_size: int = 1000
    ttl_seconds: int = 3600


class OllamaConfig(BaseSettings):
    model: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434"


class OpenAIConfig(BaseSettings):
    model: str = "gpt-3.5-turbo"
    api_key: str = Field(default="", alias="OPENAI_API_KEY")
    base_url: str = "https://api.openai.com/v1"


class GenerationConfig(BaseSettings):
    provider: str = "ollama"
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    max_context_tokens: int = 2048
    temperature: float = 0.1


class APIConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1


class LoggingConfig(BaseSettings):
    level: str = "INFO"
    file: str = "./logs/hermes_rag.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class HermesConfig(BaseSettings):
    """Root configuration for Hermes-RAG."""

    model_config = SettingsConfigDict(env_prefix="HERMES_", extra="allow")

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    chromadb: ChromaDBConfig = Field(default_factory=ChromaDBConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    query_expansion: QueryExpansionConfig = Field(default_factory=QueryExpansionConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _load_yaml_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.environ.get(
            "HERMES_CONFIG_PATH",
            str(Path(__file__).parent.parent / "config.yaml"),
        )
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _resolve_env_vars(value):
    """Resolve ${ENV_VAR} patterns in string values."""
    if not isinstance(value, str):
        return value

    def _replace(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return _ENV_VAR_PATTERN.sub(_replace, value)


def _resolve_dict_env_vars(d: dict) -> dict:
    """Recursively resolve environment variables in all string values."""
    resolved = {}
    for k, v in d.items():
        if isinstance(v, dict):
            resolved[k] = _resolve_dict_env_vars(v)
        elif isinstance(v, str):
            resolved[k] = _resolve_env_vars(v)
        elif isinstance(v, list):
            resolved[k] = [
                _resolve_env_vars(item) if isinstance(item, str) else item
                for item in v
            ]
        else:
            resolved[k] = v
    return resolved


def _flatten_dict(d: dict, parent_key: str = "") -> dict:
    """Flatten nested dict for Pydantic model initialization."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items


def load_config(config_path: Optional[str] = None) -> HermesConfig:
    """Load and return Hermes-RAG configuration.

    First creates config with defaults (which reads from environment variables),
    then overlays YAML values. YAML values are processed for ${ENV_VAR} patterns
    so that config.yaml can reference environment variables.
    """
    yaml_data = _load_yaml_config(config_path)
    # Resolve ${ENV_VAR} patterns in YAML values
    yaml_data = _resolve_dict_env_vars(yaml_data)
    flat_data = _flatten_dict(yaml_data)

    config = HermesConfig()
    for key, value in flat_data.items():
        # Navigate nested config
        keys = key.split(".")
        obj = config
        for k in keys[:-1]:
            obj = getattr(obj, k)
        if hasattr(obj, keys[-1]):
            setattr(obj, keys[-1], value)

    return config


# Global config instance (lazy-loaded, thread-safe)
_config: Optional[HermesConfig] = None
_config_lock = threading.Lock()


def get_config() -> HermesConfig:
    """Get or create the global configuration instance (thread-safe)."""
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = load_config()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None