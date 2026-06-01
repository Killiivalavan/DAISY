import asyncio
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


class SQLiteStore:
    def __init__(self, db_path: str):
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                created_at TEXT DEFAULT (datetime('now')),
                last_accessed TEXT DEFAULT (datetime('now'))
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                key, value, content=facts, content_rowid=id
            );

            CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
                INSERT INTO facts_fts(rowid, key, value) VALUES (new.id, new.key, new.value);
            END;

            CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, key, value) VALUES('delete', old.id, old.key, old.value);
            END;

            CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, key, value) VALUES('delete', old.id, old.key, old.value);
                INSERT INTO facts_fts(rowid, key, value) VALUES (new.id, new.key, new.value);
            END;

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT,
                started_at TEXT DEFAULT (datetime('now')),
                ended_at TEXT
            );
        """)
        self._conn.commit()

    async def store_fact(self, key: str, value: str, category: str = "general"):
        await asyncio.to_thread(self._store_fact_sync, key, value, category)

    def _store_fact_sync(self, key: str, value: str, category: str):
        with self._lock:
            self._conn.execute(
                """INSERT INTO facts(key, value, category) VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                       value=excluded.value,
                       category=excluded.category,
                       last_accessed=datetime('now')""",
                (key.lower(), value, category),
            )
            self._conn.commit()

    async def get_fact(self, key: str) -> Optional[dict]:
        return await asyncio.to_thread(self._get_fact_sync, key)

    def _get_fact_sync(self, key: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "UPDATE facts SET last_accessed=datetime('now') WHERE LOWER(key)=? RETURNING *",
                (key.lower(),),
            ).fetchone()
        return dict(row) if row else None

    async def search_facts(self, query: str, limit: int = 10) -> list[dict]:
        return await asyncio.to_thread(self._search_facts_sync, query, limit)

    def _search_facts_sync(self, query: str, limit: int) -> list[dict]:
        if not query or not query.strip():
            return []
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT f.* FROM facts f JOIN facts_fts fts ON f.id = fts.rowid "
                    "WHERE facts_fts MATCH ? ORDER BY f.last_accessed DESC LIMIT ?",
                    (query, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = self._conn.execute(
                    "SELECT * FROM facts WHERE key LIKE ? OR value LIKE ? "
                    "ORDER BY last_accessed DESC LIMIT ?",
                    (f"%{query}%", f"%{query}%", limit),
                ).fetchall()
        return [dict(r) for r in rows]

    async def get_all_facts(self, limit: int = 500) -> list[dict]:
        return await asyncio.to_thread(self._get_all_facts_sync, limit)

    def _get_all_facts_sync(self, limit: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM facts ORDER BY last_accessed DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    async def delete_fact(self, key: str) -> bool:
        return await asyncio.to_thread(self._delete_fact_sync, key)

    def _delete_fact_sync(self, key: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM facts WHERE key=?", (key.lower(),))
            self._conn.commit()
        return cur.rowcount > 0

    async def start_session(self) -> int:
        return await asyncio.to_thread(self._start_session_sync)

    def _start_session_sync(self) -> int:
        with self._lock:
            cur = self._conn.execute("INSERT INTO sessions DEFAULT VALUES")
            self._conn.commit()
        return cur.lastrowid

    async def end_session(self, session_id: int, summary: Optional[str] = None):
        await asyncio.to_thread(self._end_session_sync, session_id, summary)

    def _end_session_sync(self, session_id: int, summary: Optional[str]):
        with self._lock:
            if summary is not None:
                self._conn.execute(
                    "UPDATE sessions SET summary=?, ended_at=datetime('now') WHERE id=?",
                    (summary, session_id),
                )
            else:
                self._conn.execute(
                    "UPDATE sessions SET ended_at=datetime('now') WHERE id=?",
                    (session_id,),
                )
            self._conn.commit()

    async def get_last_session_summary(self) -> Optional[str]:
        return await asyncio.to_thread(self._get_last_session_summary_sync)

    def _get_last_session_summary_sync(self) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT summary FROM sessions WHERE summary IS NOT NULL "
                "ORDER BY ended_at DESC, id DESC LIMIT 1"
            ).fetchone()
        return row["summary"] if row else None

    def close(self):
        self._conn.close()
