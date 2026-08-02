"""BM25 sparse index with optional SQLite persistence.

The index supports two modes:
- Memory mode: fast in-process indexing using rank_bm25 (with a pure-Python
  fallback when the package is unavailable).
- Persist mode: SQLite-backed storage with an inverted term index, so the
  sparse index survives process restarts at any collection size.
"""

import json
import logging
import math
import sqlite3
import threading
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger("hermes_rag")


class _BM25OkapiFallback:
    """Pure-Python BM25Okapi implementation used when rank_bm25 is missing."""

    def __init__(self, corpus, k1: float = 1.5, b: float = 0.75):
        self.corpus_size = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size else 0.0
        self.doc_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}
        self.k1 = k1
        self.b = b

        df: dict[str, int] = {}
        for doc in corpus:
            frequencies: dict[str, int] = {}
            for token in doc:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.doc_freqs.append(frequencies)
            for token in set(doc):
                df[token] = df.get(token, 0) + 1

        for token, freq in df.items():
            self.idf[token] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        for token in query:
            if token not in self.idf:
                continue
            idf = self.idf[token]
            for i, frequencies in enumerate(self.doc_freqs):
                tf = frequencies.get(token, 0)
                if tf == 0:
                    continue
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * self.doc_len[i] / max(self.avgdl, 1.0)
                )
                scores[i] += idf * (tf * (self.k1 + 1.0)) / denominator
        return scores


class BM25Index:
    """BM25-based sparse retrieval index."""

    def __init__(
        self,
        b: float = 0.75,
        k1: float = 1.5,
        max_index_entries: int = 100000,
        fallback_db_path: str = "./data/bm25_fallback.db",
        batch_size: int = 500,
        persist: bool = False,
    ):
        self.b = b
        self.k1 = k1
        self.max_index_entries = max_index_entries
        self.fallback_db_path = fallback_db_path
        self.batch_size = batch_size
        self.persist = persist
        self._bm25 = None
        self._chunks: list[dict[str, Any]] = []
        self._tokenized_corpus: list[list[str]] = []
        self._use_db = False
        self._db_conn: sqlite3.Connection | None = None
        self._stats_cache: dict[str, Any] | None = None
        self._doc_count = 0
        self._total_doc_length = 0
        self._db_lock = threading.RLock()

        if self.persist:
            self._db_conn = self._init_db_connection()
            self._use_db = True
            self._ensure_persist_schema()
            self._load_persisted_meta()

    def _init_db_connection(self) -> sqlite3.Connection:
        """Open a SQLite connection with WAL mode and performance pragmas."""
        Path(self.fallback_db_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            conn = sqlite3.connect(self.fallback_db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")
            conn.execute("PRAGMA mmap_size=268435456")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            return conn
        except sqlite3.DatabaseError:
            logger.warning(
                "Corrupted database detected at %s, attempting recovery...",
                self.fallback_db_path,
            )
            try:
                conn.close()
            except Exception:
                pass
            db_path = Path(self.fallback_db_path)
            db_path.unlink(missing_ok=True)
            for suffix in ("-wal", "-shm"):
                db_path.with_suffix(db_path.suffix + suffix).unlink(missing_ok=True)
            conn = sqlite3.connect(self.fallback_db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")
            conn.execute("PRAGMA mmap_size=268435456")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            return conn

    def _ensure_persist_schema(self) -> None:
        """Create the persistent schema when running in persist mode."""
        if self._db_conn is None:
            return
        cursor = self._db_conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS bm25_index (
                chunk_id TEXT PRIMARY KEY,
                text TEXT,
                metadata TEXT,
                tokens TEXT,
                doc_length INTEGER DEFAULT 0
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS bm25_stats (
                token TEXT PRIMARY KEY,
                df INTEGER DEFAULT 0
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS bm25_terms (
                token TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                tf INTEGER NOT NULL,
                PRIMARY KEY (token, chunk_id)
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_bm25_terms_chunk ON bm25_terms (chunk_id)"
        )
        self._db_conn.commit()

    def _load_persisted_meta(self) -> None:
        """Load collection metadata from the persistent database."""
        if self._db_conn is None:
            return
        cursor = self._db_conn.cursor()
        row = cursor.execute(
            "SELECT COALESCE(SUM(doc_length), 0), COUNT(*) FROM bm25_index"
        ).fetchone()
        if row:
            self._total_doc_length, self._doc_count = row
        self._stats_cache = None

    def _ensure_open(self) -> None:
        """Reconnect to the persistent database after it was closed."""
        if self.persist and self._db_conn is None:
            self._db_conn = self._init_db_connection()
            self._ensure_persist_schema()
            self._use_db = True
            self._load_persisted_meta()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text using jieba for Chinese, nltk for English."""
        tokens: list[str] = []

        has_chinese = any("\u4e00" <= c <= "\u9fff" for c in text)

        if has_chinese:
            try:
                import jieba

                self._ensure_jieba_dict()
                tokens = list(jieba.cut(text))
                tokens = [t.strip() for t in tokens if t.strip()]
            except ImportError:
                tokens = []
                chars = list(text)
                for i in range(len(chars)):
                    if i + 1 < len(chars):
                        tokens.append(chars[i] + chars[i + 1])
                    tokens.append(chars[i])
        else:
            try:
                import nltk

                try:
                    tokens = nltk.word_tokenize(text.lower())
                except LookupError:
                    nltk.download("punkt_tab", quiet=True)
                    tokens = nltk.word_tokenize(text.lower())
            except ImportError:
                tokens = text.lower().split()

        return [t.strip() for t in tokens if t.strip()]

    @staticmethod
    def _ensure_jieba_dict():
        """Add domain-specific terms to jieba's dictionary."""
        try:
            import jieba
        except ImportError:
            return

        if hasattr(BM25Index, "_jieba_dict_loaded"):
            return

        domain_terms = [
            ("机器学习", 100), ("深度学习", 100), ("神经网络", 100),
            ("自然语言处理", 100), ("计算机视觉", 100), ("强化学习", 100),
            ("迁移学习", 100), ("大语言模型", 100), ("注意力机制", 100),
            ("Transformer", 50), ("BERT", 50), ("GPT", 50),
            ("RAG", 50), ("检索增强生成", 100), ("向量数据库", 100),
            ("嵌入模型", 100), ("重排序", 100), ("语义搜索", 100),
            ("微服务", 100), ("容器化", 100), ("Kubernetes", 50),
            ("Docker", 50), ("CI/CD", 50), ("DevOps", 50),
            ("API网关", 100), ("负载均衡", 100), ("服务网格", 100),
            ("分布式系统", 100), ("消息队列", 100), ("缓存策略", 100),
            ("关系型数据库", 100), ("非关系型数据库", 100), ("NoSQL", 50),
            ("MySQL", 50), ("PostgreSQL", 50), ("MongoDB", 50),
            ("Redis", 50), ("Elasticsearch", 50), ("ChromaDB", 50),
            ("开源", 100), ("架构设计", 100), ("高可用", 100),
            ("可扩展", 100), ("性能优化", 100), ("故障排查", 100),
            ("数据安全", 100), ("隐私保护", 100), ("加密算法", 100),
            ("云计算", 100), ("边缘计算", 100), ("物联网", 100),
            ("区块链", 100), ("大数据", 100), ("数据挖掘", 100),
            ("特征工程", 100), ("模型训练", 100), ("模型部署", 100),
            ("推理加速", 100), ("模型压缩", 100), ("量化", 100),
            ("知识图谱", 100), ("图神经网络", 100), ("联邦学习", 100),
            ("多模态", 100), ("跨模态", 100), ("零样本学习", 100),
            ("提示工程", 100), ("思维链", 100), ("Agent", 50),
        ]
        for term, freq in domain_terms:
            jieba.add_word(term, freq)

        BM25Index._jieba_dict_loaded = True

    def _tokenize_with_ngrams(self, text: str) -> list[str]:
        """Tokenize text and generate bigrams/trigrams for better matching."""
        tokens = self._tokenize(text)

        has_chinese = any("\u4e00" <= c <= "\u9fff" for c in text)
        if has_chinese:
            chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
            for i in range(len(chars) - 1):
                tokens.append(chars[i] + chars[i + 1])
            for i in range(len(chars) - 2):
                tokens.append(chars[i] + chars[i + 1] + chars[i + 2])

        return tokens

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Add chunks to the BM25 index."""
        self._ensure_open()
        if self.persist and self._use_db:
            with self._db_lock:
                self._add_chunks_db(chunks)
            return

        for chunk in chunks:
            tokens = self._tokenize_with_ngrams(chunk["text"])
            self._chunks.append(chunk)
            self._tokenized_corpus.append(tokens)

        if len(self._chunks) > self.max_index_entries:
            self._migrate_to_db()

        self._build_bm25()

    def _add_chunks_db(self, chunks: list[dict[str, Any]]) -> None:
        """Add chunks to the persistent SQLite index (upsert semantics)."""
        if not chunks or self._db_conn is None:
            return

        cursor = self._db_conn.cursor()
        cursor.execute("BEGIN TRANSACTION")
        try:
            for chunk in chunks:
                chunk_id = chunk["chunk_id"]
                self._delete_chunk_db(cursor, chunk_id)

                tokens = self._tokenize_with_ngrams(chunk["text"])
                cursor.execute(
                    "INSERT INTO bm25_index VALUES (?, ?, ?, ?, ?)",
                    (
                        chunk_id,
                        chunk["text"],
                        json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
                        json.dumps(tokens),
                        len(tokens),
                    ),
                )
                tf_map = Counter(tokens)
                for token, tf in tf_map.items():
                    cursor.execute(
                        "INSERT OR REPLACE INTO bm25_terms VALUES (?, ?, ?)",
                        (token, chunk_id, tf),
                    )
                    cursor.execute(
                        "UPDATE bm25_stats SET df = df + 1 WHERE token = ?",
                        (token,),
                    )
                    if cursor.rowcount == 0:
                        cursor.execute("INSERT INTO bm25_stats VALUES (?, 1)", (token,))

            cursor.execute(
                "INSERT OR REPLACE INTO bm25_stats SELECT '__doc_count__', COUNT(*) FROM bm25_index"
            )
            cursor.execute(
                "INSERT OR REPLACE INTO bm25_stats SELECT '__total_dl__', COALESCE(SUM(doc_length), 0) FROM bm25_index"
            )
            self._db_conn.commit()
        except Exception:
            self._db_conn.rollback()
            raise
        finally:
            self._load_persisted_meta()

    @staticmethod
    def _delete_chunk_db(cursor, chunk_id: str) -> None:
        """Delete one chunk and its term rows using the given cursor."""
        cursor.execute("DELETE FROM bm25_terms WHERE chunk_id = ?", (chunk_id,))
        row = cursor.execute(
            "SELECT tokens FROM bm25_index WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row:
            try:
                old_tokens = json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                old_tokens = []
            for token in set(old_tokens):
                cursor.execute(
                    "UPDATE bm25_stats SET df = MAX(0, df - 1) WHERE token = ? AND token NOT LIKE '__%'",
                    (token,),
                )
        cursor.execute("DELETE FROM bm25_index WHERE chunk_id = ?", (chunk_id,))

    def _build_bm25(self) -> None:
        """Build or rebuild the BM25 model."""
        if self._use_db:
            self._bm25 = None
            return

        if not self._tokenized_corpus:
            return

        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)
        except ImportError:
            logger.warning(
                "rank_bm25 unavailable; using built-in pure-Python BM25 fallback"
            )
            self._bm25 = _BM25OkapiFallback(
                self._tokenized_corpus, k1=self.k1, b=self.b
            )

    def _migrate_to_db(self) -> None:
        """Migrate in-memory index to SQLite for large collections."""
        if self._use_db:
            return

        self._db_conn = self._init_db_connection()
        self._ensure_persist_schema()
        cursor = self._db_conn.cursor()

        doc_count = len(self._tokenized_corpus)
        sum(len(t) for t in self._tokenized_corpus) / max(doc_count, 1)

        df_map: dict[str, int] = {}
        for tokens in self._tokenized_corpus:
            for token in set(tokens):
                df_map[token] = df_map.get(token, 0) + 1

        cursor.execute("BEGIN TRANSACTION")
        df_items = list(df_map.items())
        for i in range(0, len(df_items), self.batch_size):
            batch = df_items[i : i + self.batch_size]
            cursor.executemany("INSERT OR REPLACE INTO bm25_stats VALUES (?, ?)", batch)

        cursor.execute(
            "INSERT OR REPLACE INTO bm25_stats VALUES (?, ?)",
            ("__doc_count__", doc_count),
        )
        cursor.execute(
            "INSERT OR REPLACE INTO bm25_stats VALUES (?, ?)",
            ("__total_dl__", sum(len(t) for t in self._tokenized_corpus)),
        )
        self._db_conn.commit()

        cursor.execute("BEGIN TRANSACTION")
        chunk_data = [
            (
                chunk["chunk_id"],
                chunk["text"],
                json.dumps(chunk.get("metadata", {})),
                json.dumps(tokens),
                len(tokens),
            )
            for chunk, tokens in zip(self._chunks, self._tokenized_corpus)
        ]
        for i in range(0, len(chunk_data), self.batch_size):
            batch = chunk_data[i : i + self.batch_size]
            cursor.executemany(
                "INSERT OR REPLACE INTO bm25_index VALUES (?, ?, ?, ?, ?)",
                batch,
            )

        term_rows = []
        for chunk, tokens in zip(self._chunks, self._tokenized_corpus):
            for token, tf in Counter(tokens).items():
                term_rows.append((token, chunk["chunk_id"], tf))
        for i in range(0, len(term_rows), self.batch_size):
            cursor.executemany(
                "INSERT OR REPLACE INTO bm25_terms VALUES (?, ?, ?)",
                term_rows[i : i + self.batch_size],
            )
        self._db_conn.commit()

        self._use_db = True
        self._load_persisted_meta()
        self._tokenized_corpus.clear()
        self._chunks.clear()
        self._bm25 = None

    def search(self, query: str, top_k: int = 100) -> list[dict[str, Any]]:
        """Search for relevant chunks using BM25."""
        query_tokens = self._tokenize_with_ngrams(query)

        if not query_tokens:
            return []

        self._ensure_open()
        if self._use_db and self._db_conn:
            with self._db_lock:
                return self._search_db(query_tokens, top_k)

        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(query_tokens)
        if len(scores) == 0:
            return []

        indices = list(range(len(scores)))
        indices.sort(key=lambda i: scores[i], reverse=True)
        top_indices = indices[:top_k]

        results = []
        for idx in top_indices:
            chunk = self._chunks[idx]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                    "score": float(scores[idx]),
                }
            )

        return results

    def _search_db(self, query_tokens: list[str], top_k: int) -> list[dict[str, Any]]:
        """Search in SQLite using the inverted term index."""
        if not self._db_conn:
            return []

        cursor = self._db_conn.cursor()
        doc_count = self._doc_count
        avg_dl = self._total_doc_length / max(doc_count, 1)

        if doc_count == 0:
            return []

        unique_tokens = list(set(query_tokens))
        df_cache: dict[str, int] = {}
        placeholders = ",".join("?" * len(unique_tokens))
        for token, df in cursor.execute(
            f"SELECT token, df FROM bm25_stats WHERE token IN ({placeholders})",
            unique_tokens,
        ).fetchall():
            if not token.startswith("__"):
                df_cache[token] = df

        candidates: set = set()
        for token in query_tokens:
            if token not in df_cache:
                continue
            rows = cursor.execute(
                "SELECT chunk_id FROM bm25_terms WHERE token = ?", (token,)
            ).fetchall()
            candidates.update(r[0] for r in rows)

        if not candidates:
            return []

        scored = []
        for chunk_id in candidates:
            row = cursor.execute(
                "SELECT text, metadata, tokens, doc_length FROM bm25_index WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            if not row:
                continue
            text, metadata_str, tokens_str, doc_length = row
            try:
                tokens = json.loads(tokens_str)
            except (json.JSONDecodeError, TypeError):
                continue

            bm25_score = 0.0
            for qt in query_tokens:
                df = df_cache.get(qt, 0)
                if df == 0:
                    continue
                idf = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
                tf = tokens.count(qt)
                if tf == 0:
                    continue
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * doc_length / max(avg_dl, 1.0)
                )
                bm25_score += idf * (tf * (self.k1 + 1.0)) / denominator

            if bm25_score > 0:
                scored.append(
                    {
                        "chunk_id": chunk_id,
                        "text": text,
                        "metadata": json.loads(metadata_str) if metadata_str else {},
                        "score": bm25_score,
                    }
                )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        """Return the number of indexed chunks."""
        self._ensure_open()
        if self._use_db and self._db_conn:
            with self._db_lock:
                cursor = self._db_conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM bm25_index")
                return cursor.fetchone()[0]
        return len(self._chunks)

    def get_chunk_ids(self) -> list[str]:
        """Return the chunk IDs currently indexed."""
        self._ensure_open()
        if self._use_db and self._db_conn:
            with self._db_lock:
                cursor = self._db_conn.cursor()
                cursor.execute("SELECT chunk_id FROM bm25_index")
                return [r[0] for r in cursor.fetchall()]
        return [c["chunk_id"] for c in self._chunks]

    def clear(self) -> None:
        """Clear all indexed data."""
        self._bm25 = None
        self._chunks.clear()
        self._tokenized_corpus.clear()
        self._doc_count = 0
        self._total_doc_length = 0
        self._stats_cache = None

        if self._db_conn:
            with self._db_lock:
                cursor = self._db_conn.cursor()
                cursor.execute("DELETE FROM bm25_terms")
                cursor.execute("DELETE FROM bm25_stats")
                cursor.execute("DELETE FROM bm25_index")
                self._db_conn.commit()
                if not self.persist:
                    self._db_conn.close()
                    self._db_conn = None
                self._use_db = self.persist

    def remove_chunk(self, chunk_id: str) -> bool:
        """Remove a single chunk from the index."""
        self._ensure_open()
        if self._use_db and self._db_conn:
            with self._db_lock:
                cursor = self._db_conn.cursor()
                cursor.execute("BEGIN TRANSACTION")
                try:
                    row = cursor.execute(
                        "SELECT chunk_id FROM bm25_index WHERE chunk_id = ?", (chunk_id,)
                    ).fetchone()
                    if not row:
                        cursor.execute("ROLLBACK")
                        return False
                    self._delete_chunk_db(cursor, chunk_id)
                    cursor.execute(
                        "INSERT OR REPLACE INTO bm25_stats SELECT '__doc_count__', COUNT(*) FROM bm25_index"
                    )
                    cursor.execute(
                        "INSERT OR REPLACE INTO bm25_stats SELECT '__total_dl__', COALESCE(SUM(doc_length), 0) FROM bm25_index"
                    )
                    self._db_conn.commit()
                except Exception:
                    self._db_conn.rollback()
                    raise
                self._load_persisted_meta()
            return True

        for i, chunk in enumerate(self._chunks):
            if chunk["chunk_id"] == chunk_id:
                del self._chunks[i]
                del self._tokenized_corpus[i]
                self._build_bm25()
                return True
        return False

    def close(self) -> None:
        """Close the SQLite connection and release all file handles."""
        with self._db_lock:
            if self._db_conn is not None:
                try:
                    self._db_conn.close()
                finally:
                    self._db_conn = None
            if self.persist:
                self._use_db = False

    def vacuum(self) -> None:
        """Optimize the SQLite database by reclaiming unused space."""
        self._ensure_open()
        if self._use_db and self._db_conn:
            self._db_conn.execute("VACUUM")
            logger.info("SQLite VACUUM completed for %s", self.fallback_db_path)

    def get_stats(self) -> dict[str, Any]:
        """Return index statistics."""
        if self._stats_cache is not None:
            return self._stats_cache

        self._ensure_open()
        if self._use_db and self._db_conn:
            with self._db_lock:
                cursor = self._db_conn.cursor()
                total_chunks = cursor.execute("SELECT COUNT(*) FROM bm25_index").fetchone()[0]
                total_tokens = cursor.execute(
                    "SELECT COALESCE(SUM(doc_length), 0) FROM bm25_index"
                ).fetchone()[0]
                avg_doc_length = total_tokens / max(total_chunks, 1)
                unique_tokens = cursor.execute(
                    "SELECT COUNT(*) FROM bm25_stats WHERE token NOT LIKE '__%'"
                ).fetchone()[0]
            stats = {
                "total_chunks": total_chunks,
                "total_tokens": total_tokens,
                "avg_doc_length": round(avg_doc_length, 2),
                "unique_tokens": unique_tokens,
                "mode": "sqlite",
                "db_path": self.fallback_db_path,
                "persist": self.persist,
            }
        else:
            total_chunks = len(self._chunks)
            total_tokens = sum(len(t) for t in self._tokenized_corpus)
            avg_doc_length = total_tokens / max(total_chunks, 1)
            unique_tokens_set: set = set()
            for tokens in self._tokenized_corpus:
                unique_tokens_set.update(tokens)
            stats = {
                "total_chunks": total_chunks,
                "total_tokens": total_tokens,
                "avg_doc_length": round(avg_doc_length, 2),
                "unique_tokens": len(unique_tokens_set),
                "mode": "memory",
                "db_path": None,
                "persist": self.persist,
            }

        self._stats_cache = stats
        return stats
