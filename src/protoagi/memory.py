from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(slots=True)
class MemoryFact:
    id: int
    text: str
    tags: list[str]
    created_at: str


@dataclass(slots=True)
class TelegramChat:
    chat_id: str
    chat_type: str
    title: str | None
    username: str | None
    first_name: str | None
    last_name: str | None
    display_name: str
    reply_mode: str
    proactive_enabled: bool
    first_seen_at: str
    last_seen_at: str
    last_user_message_at: str | None
    last_bot_message_at: str | None
    last_initiative_at: str | None
    next_initiative_at: str | None
    metadata: dict[str, Any]


@dataclass(slots=True)
class TelegramMessage:
    chat_id: str
    message_id: int
    role: str
    sender_id: str | None
    sender_name: str
    text: str
    created_at: str
    metadata: dict[str, Any]


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    arguments TEXT NOT NULL,
                    result TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kv (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telegram_chats (
                    chat_id TEXT PRIMARY KEY,
                    chat_type TEXT NOT NULL,
                    title TEXT,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    display_name TEXT NOT NULL,
                    reply_mode TEXT NOT NULL DEFAULT 'smart',
                    proactive_enabled INTEGER NOT NULL DEFAULT 1,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    last_user_message_at TEXT,
                    last_bot_message_at TEXT,
                    last_initiative_at TEXT,
                    next_initiative_at TEXT,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telegram_messages (
                    chat_id TEXT NOT NULL,
                    message_id INTEGER NOT NULL,
                    persona_key TEXT NOT NULL DEFAULT 'mykola',
                    role TEXT NOT NULL,
                    sender_id TEXT,
                    sender_name TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (chat_id, message_id)
                )
                """
            )
            self._ensure_column(conn, "telegram_messages", "persona_key", "TEXT NOT NULL DEFAULT 'mykola'")
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
                    USING fts5(text, tags, content='facts', content_rowid='id')
                    """
                )
            except sqlite3.OperationalError:
                pass

    def remember(self, text: str, tags: list[str] | None = None) -> int:
        text = text.strip()
        if not text:
            raise ValueError("memory text cannot be empty")
        tags = tags or []
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO facts(text, tags, created_at) VALUES (?, ?, ?)",
                (text, json.dumps(tags, ensure_ascii=False), utc_now()),
            )
            rowid = int(cur.lastrowid)
            try:
                conn.execute(
                    "INSERT INTO facts_fts(rowid, text, tags) VALUES (?, ?, ?)",
                    (rowid, text, " ".join(tags)),
                )
            except sqlite3.OperationalError:
                pass
            return rowid

    def search(self, query: str, *, limit: int = 5) -> list[MemoryFact]:
        query = query.strip()
        if not query:
            return []
        with self.connect() as conn:
            rows: list[sqlite3.Row]
            try:
                fts_query = self._make_fts_query(query)
                rows = conn.execute(
                    """
                    SELECT facts.id, facts.text, facts.tags, facts.created_at
                    FROM facts_fts
                    JOIN facts ON facts.id = facts_fts.rowid
                    WHERE facts_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = conn.execute(
                    """
                    SELECT id, text, tags, created_at
                    FROM facts
                    WHERE text LIKE ? OR tags LIKE ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (f"%{query}%", f"%{query}%", limit),
                ).fetchall()
        return [
            MemoryFact(
                id=int(row["id"]),
                text=str(row["text"]),
                tags=json.loads(row["tags"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def search_tagged(self, query: str, tag: str, *, limit: int = 5) -> list[MemoryFact]:
        return self.search_tagged_all(query, [tag], limit=limit)

    def search_tagged_all(self, query: str, tags: list[str], *, limit: int = 5) -> list[MemoryFact]:
        required_tags = [tag for tag in tags if tag]
        if not required_tags:
            return self.search(query, limit=limit)
        hits = self.search(query, limit=max(limit * 4, 20))
        scoped = [fact for fact in hits if all(tag in fact.tags for tag in required_tags)]
        if len(scoped) >= limit:
            return scoped[:limit]
        tag_clauses = " AND ".join("tags LIKE ?" for _ in required_tags)
        params: list[Any] = [f"%{tag}%" for tag in required_tags]
        params.extend([f"%{query}%", f"%{query}%", max(limit * 4, 20)])
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, text, tags, created_at
                FROM facts
                WHERE {tag_clauses} AND (text LIKE ? OR tags LIKE ?)
                ORDER BY id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        seen = {fact.id for fact in scoped}
        for row in rows:
            if int(row["id"]) in seen:
                continue
            tags = json.loads(row["tags"])
            if not all(tag in tags for tag in required_tags):
                continue
            scoped.append(
                MemoryFact(
                    id=int(row["id"]),
                    text=str(row["text"]),
                    tags=tags,
                    created_at=str(row["created_at"]),
                )
            )
            if len(scoped) >= limit:
                break
        return scoped[:limit]

    def recent_tagged_all(self, tags: list[str], *, limit: int = 5) -> list[MemoryFact]:
        required_tags = [tag for tag in tags if tag]
        if not required_tags:
            return []
        tag_clauses = " AND ".join("tags LIKE ?" for _ in required_tags)
        params: list[Any] = [f"%{tag}%" for tag in required_tags]
        params.append(max(limit * 4, 20))
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, text, tags, created_at
                FROM facts
                WHERE {tag_clauses}
                ORDER BY id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        facts: list[MemoryFact] = []
        for row in rows:
            tags_value = json.loads(row["tags"])
            if not all(tag in tags_value for tag in required_tags):
                continue
            facts.append(
                MemoryFact(
                    id=int(row["id"]),
                    text=str(row["text"]),
                    tags=tags_value,
                    created_at=str(row["created_at"]),
                )
            )
            if len(facts) >= limit:
                break
        return facts

    def log_message(self, thread_id: str, role: str, content: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO messages(thread_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (thread_id, role, content, utc_now()),
            )

    def recent_messages(self, thread_id: str, *, limit: int = 12) -> list[dict[str, str]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE thread_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (thread_id, limit),
            ).fetchall()
        return [{"role": str(row["role"]), "content": str(row["content"])} for row in reversed(rows)]

    def log_tool_event(
        self,
        thread_id: str,
        name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO tool_events(thread_id, name, arguments, result, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    name,
                    json.dumps(arguments, ensure_ascii=False),
                    json.dumps(result, ensure_ascii=False),
                    utc_now(),
                ),
            )

    def get_kv(self, key: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_kv(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO kv(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, value, utc_now()),
            )

    def upsert_telegram_chat(
        self,
        chat: dict[str, Any],
        user: dict[str, Any] | None = None,
        *,
        reply_mode: str = "smart",
    ) -> TelegramChat:
        now = utc_now()
        chat_id = str(chat["id"])
        display_name = self._telegram_display_name(chat, user)
        metadata = {"chat": chat, "user": user or {}}
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO telegram_chats(
                    chat_id, chat_type, title, username, first_name, last_name, display_name,
                    reply_mode, proactive_enabled, first_seen_at, last_seen_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    chat_type = excluded.chat_type,
                    title = excluded.title,
                    username = excluded.username,
                    first_name = excluded.first_name,
                    last_name = excluded.last_name,
                    display_name = excluded.display_name,
                    last_seen_at = excluded.last_seen_at,
                    metadata = excluded.metadata
                """,
                (
                    chat_id,
                    str(chat.get("type", "unknown")),
                    chat.get("title"),
                    chat.get("username") or (user or {}).get("username"),
                    chat.get("first_name") or (user or {}).get("first_name"),
                    chat.get("last_name") or (user or {}).get("last_name"),
                    display_name,
                    reply_mode,
                    now,
                    now,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
        found = self.get_telegram_chat(chat_id)
        if found is None:
            raise RuntimeError("telegram chat upsert failed")
        return found

    def get_telegram_chat(self, chat_id: str | int) -> TelegramChat | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM telegram_chats WHERE chat_id = ?",
                (str(chat_id),),
            ).fetchone()
        return None if row is None else self._telegram_chat_from_row(row)

    def list_due_telegram_chats(self, now: str, *, limit: int = 20) -> list[TelegramChat]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM telegram_chats
                WHERE proactive_enabled = 1
                  AND last_user_message_at IS NOT NULL
                  AND (next_initiative_at IS NULL OR next_initiative_at <= ?)
                ORDER BY COALESCE(next_initiative_at, first_seen_at) ASC
                LIMIT ?
                """,
                (now, limit),
            ).fetchall()
        return [self._telegram_chat_from_row(row) for row in rows]

    def set_telegram_proactive(self, chat_id: str | int, enabled: bool) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE telegram_chats
                SET proactive_enabled = ?, last_seen_at = ?
                WHERE chat_id = ?
                """,
                (1 if enabled else 0, utc_now(), str(chat_id)),
            )

    def set_telegram_reply_mode(self, chat_id: str | int, reply_mode: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE telegram_chats
                SET reply_mode = ?, last_seen_at = ?
                WHERE chat_id = ?
                """,
                (reply_mode, utc_now(), str(chat_id)),
            )

    def mark_telegram_user_message(self, chat_id: str | int) -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE telegram_chats
                SET last_user_message_at = ?, last_seen_at = ?
                WHERE chat_id = ?
                """,
                (now, now, str(chat_id)),
            )

    def mark_telegram_bot_message(self, chat_id: str | int, *, initiative: bool = False) -> None:
        now = utc_now()
        with self.connect() as conn:
            if initiative:
                conn.execute(
                    """
                    UPDATE telegram_chats
                    SET last_bot_message_at = ?, last_initiative_at = ?, last_seen_at = ?
                    WHERE chat_id = ?
                    """,
                    (now, now, now, str(chat_id)),
                )
            else:
                conn.execute(
                    """
                    UPDATE telegram_chats
                    SET last_bot_message_at = ?, last_seen_at = ?
                    WHERE chat_id = ?
                    """,
                    (now, now, str(chat_id)),
                )

    def schedule_telegram_initiative(self, chat_id: str | int, next_at: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE telegram_chats
                SET next_initiative_at = ?, last_seen_at = ?
                WHERE chat_id = ?
                """,
                (next_at, utc_now(), str(chat_id)),
            )

    def log_telegram_message(
        self,
        *,
        chat_id: str | int,
        message_id: int,
        role: str,
        sender_id: str | int | None,
        sender_name: str,
        text: str,
        persona_key: str = "mykola",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO telegram_messages(
                    chat_id, message_id, persona_key, role, sender_id, sender_name, text, created_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, message_id) DO UPDATE SET
                    persona_key = excluded.persona_key,
                    role = excluded.role,
                    sender_id = excluded.sender_id,
                    sender_name = excluded.sender_name,
                    text = excluded.text,
                    metadata = excluded.metadata
                """,
                (
                    str(chat_id),
                    int(message_id),
                    persona_key,
                    role,
                    None if sender_id is None else str(sender_id),
                    sender_name,
                    text,
                    utc_now(),
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )

    def recent_telegram_messages(
        self,
        chat_id: str | int,
        *,
        limit: int = 12,
        persona_key: str | None = None,
    ) -> list[dict[str, Any]]:
        where = "chat_id = ?"
        params: list[Any] = [str(chat_id)]
        if persona_key:
            where += " AND persona_key = ?"
            params.append(persona_key)
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT chat_id, message_id, persona_key, role, sender_id, sender_name, text, created_at, metadata
                FROM telegram_messages
                WHERE {where}
                ORDER BY message_id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [
            {
                "chat_id": str(row["chat_id"]),
                "message_id": int(row["message_id"]),
                "persona_key": str(row["persona_key"]),
                "role": str(row["role"]),
                "sender_id": row["sender_id"],
                "sender_name": str(row["sender_name"]),
                "text": str(row["text"]),
                "created_at": str(row["created_at"]),
                "metadata": json.loads(row["metadata"] or "{}"),
            }
            for row in reversed(rows)
        ]

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @staticmethod
    def _make_fts_query(query: str) -> str:
        tokens = ["".join(ch for ch in part if ch.isalnum() or ch in "_-") for part in query.split()]
        tokens = [token for token in tokens if token]
        if not tokens:
            return json.dumps(query)
        return " OR ".join(f'"{token}"' for token in tokens[:12])

    @staticmethod
    def _telegram_display_name(chat: dict[str, Any], user: dict[str, Any] | None) -> str:
        if chat.get("title"):
            return str(chat["title"])
        first = chat.get("first_name") or (user or {}).get("first_name") or ""
        last = chat.get("last_name") or (user or {}).get("last_name") or ""
        username = chat.get("username") or (user or {}).get("username")
        name = " ".join(part for part in (first, last) if part).strip()
        if name:
            return name
        if username:
            return f"@{username}"
        return str(chat.get("id", "unknown"))

    @staticmethod
    def _telegram_chat_from_row(row: sqlite3.Row) -> TelegramChat:
        return TelegramChat(
            chat_id=str(row["chat_id"]),
            chat_type=str(row["chat_type"]),
            title=row["title"],
            username=row["username"],
            first_name=row["first_name"],
            last_name=row["last_name"],
            display_name=str(row["display_name"]),
            reply_mode=str(row["reply_mode"]),
            proactive_enabled=bool(row["proactive_enabled"]),
            first_seen_at=str(row["first_seen_at"]),
            last_seen_at=str(row["last_seen_at"]),
            last_user_message_at=row["last_user_message_at"],
            last_bot_message_at=row["last_bot_message_at"],
            last_initiative_at=row["last_initiative_at"],
            next_initiative_at=row["next_initiative_at"],
            metadata=json.loads(row["metadata"] or "{}"),
        )
