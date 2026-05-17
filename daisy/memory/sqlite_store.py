import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


class SQLiteStore:
    def __init__(self, db_path: str):
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
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

    def store_fact(self, key: str, value: str, category: str = "general"):
        self._conn.execute(
            """INSERT INTO facts(key, value, category) VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                   value=excluded.value,
                   category=excluded.category,
                   last_accessed=datetime('now')""",
            (key.lower(), value, category),
        )
        self._conn.commit()

    def get_fact(self, key: str) -> Optional[dict]:
        row = self._conn.execute(
            "UPDATE facts SET last_accessed=datetime('now') WHERE LOWER(key)=? RETURNING *",
            (key.lower(),),
        ).fetchone()
        return dict(row) if row else None

    def search_facts(self, query: str, limit: int = 10) -> list[dict]:
        if not query or not query.strip():
            return []
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

    def get_all_facts(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM facts ORDER BY last_accessed DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_fact(self, key: str):
        self._conn.execute("DELETE FROM facts WHERE key=?", (key,))
        self._conn.commit()

    def start_session(self) -> int:
        cur = self._conn.execute("INSERT INTO sessions DEFAULT VALUES")
        self._conn.commit()
        return cur.lastrowid

    def end_session(self, session_id: int, summary: Optional[str] = None):
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

    def get_last_session_summary(self) -> Optional[str]:
        row = self._conn.execute(
            "SELECT summary FROM sessions WHERE summary IS NOT NULL "
            "ORDER BY ended_at DESC, id DESC LIMIT 1"
        ).fetchone()
        return row["summary"] if row else None

    def close(self):
        self._conn.close()
