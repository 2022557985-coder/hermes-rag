"""BM25 sparse index wrapper with SQLite fallback for large indexes."""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional


class BM25Index:
    """BM25-based sparse retrieval index.

    Uses rank_bm25 for in-memory indexing with SQLite fallback
    when the index exceeds the configured threshold.
    """

    def __init__(
        self,
        b: float = 0.75,
        k1: float = 1.5,
        max_index_entries: int = 100000,
        fallback_db_path: str = "./data/bm25_fallback.db",
    ):
        self.b = b
        self.k1 = k1
        self.max_index_entries = max_index_entries
        self.fallback_db_path = fallback_db_path
        self._bm25 = None
        self._chunks: List[Dict[str, Any]] = []
        self._tokenized_corpus: List[List[str]] = []
        self._use_db = False
        self._db_conn: Optional[sqlite3.Connection] = None

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using jieba for Chinese, nltk for English."""
        tokens = []

        # Check if text contains Chinese
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)

        if has_chinese:
            try:
                import jieba
                tokens = list(jieba.cut(text))
            except ImportError:
                # Fallback: character-level bigrams for Chinese
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

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Add chunks to the BM25 index.

        Args:
            chunks: List of chunk dicts with 'chunk_id', 'text', 'metadata'.
        """
        for chunk in chunks:
            tokens = self._tokenize(chunk["text"])
            self._chunks.append(chunk)
            self._tokenized_corpus.append(tokens)

        # Check if we need to fallback to SQLite
        if len(self._chunks) > self.max_index_entries:
            self._migrate_to_db()

        # Rebuild BM25
        self._build_bm25()

    def _build_bm25(self) -> None:
        """Build or rebuild the BM25 model."""
        if self._use_db:
            self._bm25 = None
            return

        if not self._tokenized_corpus:
            return

        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(
                self._tokenized_corpus,
                k1=self.k1,
                b=self.b,
            )
        except ImportError:
            self._bm25 = None

    def _migrate_to_db(self) -> None:
        """Migrate index to SQLite for large collections with BM25 statistics."""
        if self._use_db:
            return

        Path(self.fallback_db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_conn = sqlite3.connect(self.fallback_db_path)
        cursor = self._db_conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bm25_index (
                chunk_id TEXT PRIMARY KEY,
                text TEXT,
                metadata TEXT,
                tokens TEXT,
                doc_length INTEGER DEFAULT 0
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bm25_stats (
                token TEXT PRIMARY KEY,
                df INTEGER DEFAULT 0
            )
        """)

        # Compute document frequencies for BM25
        doc_count = len(self._tokenized_corpus)
        avg_doc_length = sum(len(t) for t in self._tokenized_corpus) / max(doc_count, 1)

        # Build DF map
        df_map = {}
        for tokens in self._tokenized_corpus:
            for token in set(tokens):
                df_map[token] = df_map.get(token, 0) + 1

        # Insert stats
        for token, df in df_map.items():
            cursor.execute(
                "INSERT OR REPLACE INTO bm25_stats VALUES (?, ?)",
                (token, df),
            )

        # Store avg_doc_length and doc_count as metadata
        cursor.execute(
            "INSERT OR REPLACE INTO bm25_stats VALUES (?, ?)",
            ("__doc_count__", doc_count),
        )
        cursor.execute(
            "INSERT OR REPLACE INTO bm25_stats VALUES (?, ?)",
            ("__avg_dl__", int(avg_doc_length * 1000)),
        )

        # Insert chunks with document length
        for chunk, tokens in zip(self._chunks, self._tokenized_corpus):
            cursor.execute(
                "INSERT OR REPLACE INTO bm25_index VALUES (?, ?, ?, ?, ?)",
                (
                    chunk["chunk_id"],
                    chunk["text"],
                    json.dumps(chunk.get("metadata", {})),
                    json.dumps(tokens),
                    len(tokens),
                ),
            )

        self._db_conn.commit()
        self._use_db = True
        self._tokenized_corpus.clear()
        self._chunks.clear()  # Free memory after migration
        self._bm25 = None

    def search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Search for relevant chunks using BM25.

        Args:
            query: Query text.
            top_k: Number of results to return.

        Returns:
            List of result dicts with 'chunk_id', 'text', 'metadata', 'score'.
        """
        query_tokens = self._tokenize(query)

        if self._use_db:
            return self._search_db(query_tokens, top_k)

        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices
        if len(scores) == 0:
            return []

        indices = list(range(len(scores)))
        indices.sort(key=lambda i: scores[i], reverse=True)
        top_indices = indices[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = self._chunks[idx]
                results.append({
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                    "score": float(scores[idx]),
                })

        return results

    def _search_db(self, query_tokens: List[str], top_k: int) -> List[Dict[str, Any]]:
        """Search in SQLite fallback using proper BM25 scoring."""
        if not self._db_conn:
            return []

        cursor = self._db_conn.cursor()

        # Get BM25 stats
        cursor.execute("SELECT df FROM bm25_stats WHERE token = '__doc_count__'")
        doc_count_row = cursor.fetchone()
        doc_count = doc_count_row[0] if doc_count_row else 0

        cursor.execute("SELECT df FROM bm25_stats WHERE token = '__avg_dl__'")
        avg_dl_row = cursor.fetchone()
        avg_dl = (avg_dl_row[0] / 1000.0) if avg_dl_row else 1.0

        if doc_count == 0:
            return []

        # Pre-fetch DF for all query tokens
        df_cache = {}
        for token in set(query_tokens):
            cursor.execute("SELECT df FROM bm25_stats WHERE token = ? AND token NOT LIKE '__%'", (token,))
            row = cursor.fetchone()
            df_cache[token] = row[0] if row else 0

        # Find matching documents
        matching_docs = set()
        for token in query_tokens:
            safe_token = token.replace('"', '""').replace("%", "\\%").replace("_", "\\_")
            cursor.execute(
                "SELECT chunk_id FROM bm25_index WHERE tokens LIKE ? ESCAPE '\\'",
                (f'%"{safe_token}"%',),
            )
            for row in cursor.fetchall():
                matching_docs.add(row[0])

        if not matching_docs:
            return []

        # BM25 score each matching document
        scored = []
        for chunk_id in matching_docs:
            cursor.execute(
                "SELECT text, metadata, tokens, doc_length FROM bm25_index WHERE chunk_id = ?",
                (chunk_id,),
            )
            row = cursor.fetchone()
            if not row:
                continue
            text, metadata_str, tokens_str, doc_length = row
            tokens = json.loads(tokens_str)

            # BM25: sum over query terms of IDF * TF * (k1+1) / (TF + k1*(1-b+b*dl/avgdl))
            bm25_score = 0.0
            for qt in query_tokens:
                df = df_cache.get(qt, 0)
                if df == 0:
                    continue
                # IDF
                idf = max(0.0, (doc_count - df + 0.5) / (df + 0.5))
                idf = idf + 1.0  # BM25 IDF variant
                # TF
                tf = tokens.count(qt)
                if tf == 0:
                    continue
                # BM25 term score
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / max(avg_dl, 1))
                bm25_score += idf * numerator / denominator

            if bm25_score > 0:
                scored.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": json.loads(metadata_str) if metadata_str else {},
                    "score": bm25_score,
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        """Return the number of indexed chunks."""
        if self._use_db and self._db_conn:
            cursor = self._db_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM bm25_index")
            return cursor.fetchone()[0]
        return len(self._chunks)

    def clear(self) -> None:
        """Clear all indexed data."""
        self._bm25 = None
        self._chunks.clear()
        self._tokenized_corpus.clear()
        self._use_db = False
        if self._db_conn:
            self._db_conn.close()
            self._db_conn = None