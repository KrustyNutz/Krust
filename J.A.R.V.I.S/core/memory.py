import sqlite3
import time
import json
from pathlib import Path
from datetime import datetime
from config.settings import MEMORY_DB


class Memory:
    def __init__(self, db_path: Path = MEMORY_DB):
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL    NOT NULL,
                    session   TEXT,
                    user_msg  TEXT    NOT NULL,
                    bot_msg   TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp  REAL    NOT NULL,
                    content    TEXT    NOT NULL,
                    category   TEXT    DEFAULT 'general',
                    importance INTEGER DEFAULT 5,
                    tags       TEXT    DEFAULT ''
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts
                USING fts5(user_msg, bot_msg, content=conversations, content_rowid=id);

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(content, tags, content=memories, content_rowid=id);

                CREATE TRIGGER IF NOT EXISTS conversations_ai
                AFTER INSERT ON conversations BEGIN
                    INSERT INTO conversations_fts(rowid, user_msg, bot_msg)
                    VALUES (new.id, new.user_msg, new.bot_msg);
                END;

                CREATE TRIGGER IF NOT EXISTS memories_ai
                AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content, tags)
                    VALUES (new.id, new.content, new.tags);
                END;
            """)

    def store(self, user_msg: str, bot_msg: str, session: str = "") -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO conversations (timestamp, session, user_msg, bot_msg) VALUES (?,?,?,?)",
                (time.time(), session, user_msg, bot_msg),
            )
            return cur.lastrowid

    def remember(self, content: str, category: str = "general",
                 importance: int = 5, tags: list[str] | None = None) -> int:
        tags_str = " ".join(tags or [])
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO memories (timestamp, content, category, importance, tags) VALUES (?,?,?,?,?)",
                (time.time(), content, category, importance, tags_str),
            )
            return cur.lastrowid

    def recall(self, query: str, limit: int = 6) -> list[dict]:
        results = []
        with self._conn() as conn:
            # Search explicit memories first (higher weight)
            rows = conn.execute("""
                SELECT m.id, m.timestamp, m.content, m.category, m.importance, m.tags
                FROM memories_fts fts
                JOIN memories m ON m.id = fts.rowid
                WHERE memories_fts MATCH ?
                ORDER BY m.importance DESC, m.timestamp DESC
                LIMIT ?
            """, (query, limit // 2 + 1)).fetchall()

            for row in rows:
                results.append({
                    "source": "memory",
                    "timestamp": _fmt_ts(row["timestamp"]),
                    "content": row["content"],
                    "category": row["category"],
                })

            # Search conversation history
            rows = conn.execute("""
                SELECT c.id, c.timestamp, c.user_msg, c.bot_msg
                FROM conversations_fts fts
                JOIN conversations c ON c.id = fts.rowid
                WHERE conversations_fts MATCH ?
                ORDER BY c.timestamp DESC
                LIMIT ?
            """, (query, limit // 2 + 1)).fetchall()

            for row in rows:
                results.append({
                    "source": "conversation",
                    "timestamp": _fmt_ts(row["timestamp"]),
                    "content": f"You asked: {row['user_msg'][:100]} | I said: {row['bot_msg'][:150]}",
                })

        return results[:limit]

    def get_recent(self, limit: int = 10) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT timestamp, user_msg, bot_msg
                FROM conversations ORDER BY timestamp DESC LIMIT ?
            """, (limit,)).fetchall()
        return [
            {
                "timestamp": _fmt_ts(r["timestamp"]),
                "user": r["user_msg"],
                "bot": r["bot_msg"],
            }
            for r in reversed(rows)
        ]

    def forget(self, memory_id: int):
        with self._conn() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    def list_memories(self, category: str = "") -> list[dict]:
        with self._conn() as conn:
            if category:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE category = ? ORDER BY importance DESC, timestamp DESC",
                    (category,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories ORDER BY importance DESC, timestamp DESC"
                ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        with self._conn() as conn:
            conv_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        return {"conversations": conv_count, "explicit_memories": mem_count}


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
