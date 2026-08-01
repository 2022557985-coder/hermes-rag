"""Configuration management using Pydantic Settings and YAML."""

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


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
    min_chunk_size: int = 20
    max_section_size: int = 512


class ChromaDBConfig(BaseSettings):
    persist_directory: str = "./data/chroma_db"
    collection_name: str = "hermes_rag"
    hnsw_ef_construction: int = 100
    hnsw_M: int = 8
    hnsw_ef_search: int = 50
    cache_max_entries: int = 10000
    document_store_path: str = "./data/document_store.db"


class BM25Config(BaseSettings):
    b: float = 0.75
    k1: float = 1.5
    max_index_entries: int = 100000
    fallback_db_path: str = "./data/bm25_fallback.db"
    persist: bool = True


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
    retrieval_timeout: float = 5.0
    retrieval_retries: int = 1
    min_score_threshold: float = 0.001
    context_window: int = 1  # Neighboring chunks to fetch (0=disabled)
    rrf_weights: RRFWeights = Field(default_factory=RRFWeights)


class RerankingConfig(BaseSettings):
    enabled: bool = False  # Master switch: conservative by default on CPU
    use_cross_encoder: bool = False  # Heavy cross-encoder model (requires ~1GB+ RAM)
    model_name: str = "BAAI/bge-reranker-v2-m3"  # Multilingual cross-encoder
    device: str = "cpu"
    batch_size: int = 16
    max_candidates: int = 50
    timeout_seconds: float = 3.0


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


class MonitoringConfig(BaseSettings):
    enable_metrics: bool = True
    metrics_window: int = 500


class SecurityConfig(BaseSettings):
    max_query_length: int = 2000
    rate_limit_per_minute: int = 60


class UIConfig(BaseSettings):
    dark_mode: bool = False


class AutoIngestConfig(BaseSettings):
    enabled: bool = True
    doc_dir: str = "evaluation/data/sample_docs"


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
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    auto_ingest: AutoIngestConfig = Field(default_factory=AutoIngestConfig)

    def validate(self) -> dict[str, str]:
        """Validate configuration values and return a dict of warnings.

        Returns:
            Dict mapping config path to warning message.
        """
        warnings_list: dict[str, str] = {}

        # chunk_size > chunk_overlap
        if self.chunking.chunk_size <= self.chunking.chunk_overlap:
            msg = (
                f"chunk_size ({self.chunking.chunk_size}) must be greater than "
                f"chunk_overlap ({self.chunking.chunk_overlap})"
            )
            warnings_list["chunking.chunk_size"] = msg
            logger.warning("Config validation: %s", msg)

        # min_chunk_size > 0 and <= chunk_size
        if self.chunking.min_chunk_size <= 0:
            msg = (
                f"min_chunk_size ({self.chunking.min_chunk_size}) must be > 0"
            )
            warnings_list["chunking.min_chunk_size"] = msg
            logger.warning("Config validation: %s", msg)
        elif self.chunking.min_chunk_size > self.chunking.chunk_size:
            msg = (
                f"min_chunk_size ({self.chunking.min_chunk_size}) must be <= "
                f"chunk_size ({self.chunking.chunk_size})"
            )
            warnings_list["chunking.min_chunk_size"] = msg
            logger.warning("Config validation: %s", msg)

        # semantic_threshold in [0, 1]
        if not (0 <= self.chunking.semantic_threshold <= 1):
            msg = (
                f"semantic_threshold ({self.chunking.semantic_threshold}) "
                f"must be in [0, 1]"
            )
            warnings_list["chunking.semantic_threshold"] = msg
            logger.warning("Config validation: %s", msg)

        # RRF weights sum to 1.0 (warn if they don't)
        rrw = self.retrieval.rrf_weights
        weight_pairs = [
            ("default", rrw.default_dense, rrw.default_sparse),
            ("product_code", rrw.product_code_dense, rrw.product_code_sparse),
            ("colloquial", rrw.colloquial_dense, rrw.colloquial_sparse),
        ]
        for name, dense, sparse in weight_pairs:
            total = dense + sparse
            if abs(total - 1.0) > 0.01:
                msg = (
                    f"rrf_weights.{name} dense ({dense}) + sparse ({sparse}) "
                    f"= {total}, expected 1.0"
                )
                warnings_list[f"retrieval.rrf_weights.{name}"] = msg
                logger.warning("Config validation: %s", msg)

        # temperature in [0, 2]
        if not (0 <= self.generation.temperature <= 2):
            msg = (
                f"temperature ({self.generation.temperature}) must be in [0, 2]"
            )
            warnings_list["generation.temperature"] = msg
            logger.warning("Config validation: %s", msg)

        # max_context_tokens > 0
        if self.generation.max_context_tokens <= 0:
            msg = (
                f"max_context_tokens ({self.generation.max_context_tokens}) "
                f"must be > 0"
            )
            warnings_list["generation.max_context_tokens"] = msg
            logger.warning("Config validation: %s", msg)

        # Non-critical warnings (do not block)
        if self.chunking.chunk_size < 128:
            logger.warning(
                "Config validation: chunk_size (%d) is small, consider >= 128",
                self.chunking.chunk_size,
            )
        if self.chunking.chunk_overlap < 16:
            logger.warning(
                "Config validation: chunk_overlap (%d) is small, consider >= 16",
                self.chunking.chunk_overlap,
            )
        if self.cache.similarity_threshold < 0.8:
            logger.warning(
                "Config validation: cache similarity_threshold (%f) is low, "
                "may produce false cache hits",
                self.cache.similarity_threshold,
            )

        return warnings_list

    def to_dict(self) -> dict[str, Any]:
        """Export the full configuration as a nested dictionary."""
        def _model_to_dict(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            elif hasattr(obj, "__dict__"):
                return {
                    k: _model_to_dict(v)
                    for k, v in obj.__dict__.items()
                    if not k.startswith("_")
                }
            return obj

        return _model_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HermesConfig":
        """Create a HermesConfig instance from a nested dictionary.

        Args:
            data: Nested dictionary with configuration values.

        Returns:
            A new HermesConfig instance.
        """
        config = cls()
        flat_data = _flatten_dict(data)
        for key, value in flat_data.items():
            keys = key.split(".")
            obj = config
            for k in keys[:-1]:
                obj = getattr(obj, k)
            if hasattr(obj, keys[-1]):
                setattr(obj, keys[-1], value)
        return config


def _load_yaml_config(config_path: str | None = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.environ.get(
            "HERMES_CONFIG_PATH",
            str(Path(__file__).parent.parent / "config.yaml"),
        )
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
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


def _resolve_env_overrides(flat_data: dict) -> dict:
    """Resolve HERMES_* environment variable overrides for all config keys.

    Environment variables like HERMES_EMBEDDING_MODEL_NAME will override
    the corresponding config key embedding.model_name.
    """
    env_prefix = "HERMES_"
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(env_prefix):
            continue
        # Convert HERMES_EMBEDDING_MODEL_NAME -> embedding.model_name
        config_key = env_key[len(env_prefix):].lower()
        # Map well-known components
        component_map = {
            "embedding": "embedding",
            "chunking": "chunking",
            "chromadb": "chromadb",
            "bm25": "bm25",
            "retrieval": "retrieval",
            "reranking": "reranking",
            "query_expansion": "query_expansion",
            "cache": "cache",
            "generation": "generation",
            "api": "api",
            "logging": "logging",
            "monitoring": "monitoring",
            "ui": "ui",
            "security": "security",
        }
        for prefix, component in component_map.items():
            if config_key.startswith(prefix + "_"):
                sub_key = config_key[len(prefix) + 1:]
                flat_key = f"{component}.{sub_key}"
                # Convert env value to appropriate type
                typed_value = _coerce_env_value(env_value)
                flat_data[flat_key] = typed_value
                logger.debug(
                    "Env override: %s -> %s = %s", env_key, flat_key, typed_value
                )
                break
        else:
            # Direct key without component prefix
            flat_data[config_key] = _coerce_env_value(env_value)

    return flat_data


def _coerce_env_value(value: str):
    """Coerce an environment variable string value to an appropriate Python type."""
    # Boolean
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False
    # Integer
    try:
        return int(value)
    except ValueError:
        pass
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    # String (default)
    return value


def load_config(config_path: str | None = None) -> HermesConfig:
    """Load and return Hermes-RAG configuration.

    First creates config with defaults (which reads from environment variables),
    then overlays YAML values. YAML values are processed for ${ENV_VAR} patterns
    so that config.yaml can reference environment variables.
    Finally, HERMES_* environment variables override all config values.
    """
    yaml_data = _load_yaml_config(config_path)
    # Resolve ${ENV_VAR} patterns in YAML values
    yaml_data = _resolve_dict_env_vars(yaml_data)
    flat_data = _flatten_dict(yaml_data)

    # Apply environment variable overrides (highest priority)
    flat_data = _resolve_env_overrides(flat_data)

    config = HermesConfig()
    for key, value in flat_data.items():
        # Navigate nested config
        keys = key.split(".")
        obj = config
        for k in keys[:-1]:
            obj = getattr(obj, k)
        if hasattr(obj, keys[-1]):
            setattr(obj, keys[-1], value)

    # Run validation and log warnings
    config.validate()

    return config


# Global config instance (lazy-loaded, thread-safe)
_config: HermesConfig | None = None
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