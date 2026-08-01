"""SQLite-backed document store that acts as the source of truth for chunks.

The document store makes index recovery possible: if the vector collection is
rebuilt with a different embedding model, chunks can be re-embedded from this
store instead of being silently lost.
"""

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("hermes_rag")


class DocumentStore:
    """Persistent chunk repository backed by SQLite."""

    def __init__(self, db_path: str = "./data/document_store.db"):
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    def _init_db(self) -> None:
        conn = self._connect()
        with self._lock:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Persist chunks, replacing any chunk with the same id."""
        if not chunks:
            return
        conn = self._connect()
        with self._lock:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks (chunk_id, text, metadata, source)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        c["chunk_id"],
                        c.get("text", ""),
                        json.dumps(c.get("metadata", {}), ensure_ascii=False),
                        str(c.get("metadata", {}).get("source", "")),
                    )
                    for c in chunks
                ],
            )
            conn.commit()

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        with self._lock:
            row = conn.execute(
                "SELECT chunk_id, text, metadata FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "chunk_id": row[0],
            "text": row[1],
            "metadata": json.loads(row[2]) if row[2] else {},
        }

    def get_all_chunks(self) -> list[dict[str, Any]]:
        conn = self._connect()
        with self._lock:
            rows = conn.execute(
                "SELECT chunk_id, text, metadata FROM chunks ORDER BY chunk_id"
            ).fetchall()
        return [
            {
                "chunk_id": r[0],
                "text": r[1],
                "metadata": json.loads(r[2]) if r[2] else {},
            }
            for r in rows
        ]

    def count(self) -> int:
        conn = self._connect()
        with self._lock:
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0

    def remove_chunk(self, chunk_id: str) -> bool:
        conn = self._connect()
        with self._lock:
            cursor = conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (chunk_id,))
            conn.commit()
        return cursor.rowcount > 0

    def clear(self) -> None:
        conn = self._connect()
        with self._lock:
            conn.execute("DELETE FROM chunks")
            conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def get_meta(self, key: str) -> str | None:
        """Read a metadata value from the document store."""
        conn = self._connect()
        with self._lock:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        return str(row[0]) if row else None

    def set_meta(self, key: str, value: str) -> None:
        """Persist a metadata value (e.g. the index version marker)."""
        conn = self._connect()
        with self._lock:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, value),
            )
            conn.commit()

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_chunks": self.count(),
            "db_path": str(self.db_path),
        }
