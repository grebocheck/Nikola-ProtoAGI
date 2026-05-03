"""SQLite memory store for ProtoAGI.

This module implements a typed memory model:

- ``users`` — known principals (Telegram user, agent caller).
- ``memory_items`` — typed memory entries (semantic / episodic / procedural /
  persona_self / fact) with scope, importance, supersession, and access
  metadata.
- ``memory_tags`` — normalized tag table indexed for exact matching.
- ``memory_embeddings`` — optional BLOB vectors for semantic recall.
- ``messages`` / ``tool_events`` / ``kv`` — agent loop logs and small KV state.
- ``telegram_chats`` / ``telegram_messages`` — Telegram-specific state.
- ``reminders`` — scheduled prompts the bot should surface later.

Older API (``remember``, ``search``, ``search_tagged``) is preserved so the
rest of the codebase and existing tests can keep working while new code
prefers the v2 methods (``store_memory``, ``recall``, etc.).
"""

from __future__ import annotations

import array
import json
import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Dataclasses


# Memory item kinds. ``fact`` is the legacy bucket for unsorted writes coming
# from older callers that don't classify their memories.
KIND_FACT = "fact"
KIND_SEMANTIC = "semantic"
KIND_EPISODIC = "episodic"
KIND_PROCEDURAL = "procedural"
KIND_PERSONA_SELF = "persona_self"
ALL_KINDS = (KIND_FACT, KIND_SEMANTIC, KIND_EPISODIC, KIND_PROCEDURAL, KIND_PERSONA_SELF)

# Memory scopes describe how broadly a memory applies.
SCOPE_GLOBAL = "global"
SCOPE_USER = "user"
SCOPE_CHAT = "chat"
SCOPE_PERSONA = "persona"
ALL_SCOPES = (SCOPE_GLOBAL, SCOPE_USER, SCOPE_CHAT, SCOPE_PERSONA)


@dataclass(slots=True)
class MemoryFact:
    """Backwards-compatible view of a memory item used by older callers."""

    id: int
    text: str
    tags: list[str]
    created_at: str
    importance: float = 0.5
    kind: str = KIND_FACT
    scope: str = SCOPE_GLOBAL


@dataclass(slots=True)
class MemoryItem:
    id: int
    kind: str
    text: str
    scope: str
    user_id: str | None
    chat_id: str | None
    persona_key: str | None
    importance: float
    confidence: float
    source: str | None
    supersedes_id: int | None
    superseded_by: int | None
    pinned: bool
    created_at: str
    last_accessed_at: str | None
    access_count: int
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RecallResult:
    item: MemoryItem
    score: float
    bm25: float = 0.0
    cosine: float = 0.0


@dataclass(slots=True)
class UserProfile:
    user_id: str
    display_name: str
    language: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(slots=True)
class Reminder:
    id: int
    user_id: str | None
    chat_id: str | None
    persona_key: str | None
    trigger_at: str
    text: str
    status: str
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


# ---------------------------------------------------------------------------
# Embedding helpers


def pack_embedding(vector: Sequence[float]) -> bytes:
    """Pack a float vector into a compact little-endian float32 BLOB."""
    arr = array.array("f", (float(value) for value in vector))
    if hasattr(arr, "byteswap") and array.array("f").itemsize == 4:
        # Force little-endian on big-endian hosts.
        import sys

        if sys.byteorder == "big":
            arr.byteswap()
    return arr.tobytes()


def unpack_embedding(blob: bytes) -> list[float]:
    arr = array.array("f")
    arr.frombytes(blob)
    import sys

    if sys.byteorder == "big":
        arr.byteswap()
    return list(arr)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    norm_left = 0.0
    norm_right = 0.0
    for l_value, r_value in zip(left, right):
        dot += l_value * r_value
        norm_left += l_value * l_value
        norm_right += r_value * r_value
    if norm_left <= 0.0 or norm_right <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_left) * math.sqrt(norm_right))


# ---------------------------------------------------------------------------
# Storage class


class MemoryStore:
    """SQLite-backed memory storage with v2 schema and legacy-compatible API.

    The store keeps a single long-lived connection in WAL mode for low-latency
    reads and concurrent polling. Use ``connect()`` for bulk operations that
    need an explicit transaction.
    """

    SCHEMA_VERSION = 2

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # WAL mode is persistent at the file level; set it once on init and
        # rely on per-call connections from then on. ``sqlite3.connect`` as a
        # context manager only commits/rollbacks; we close explicitly so
        # Windows releases the file handle (otherwise tempdir teardown in
        # tests fails on Windows).
        bootstrap = sqlite3.connect(self.path)
        try:
            bootstrap.execute("PRAGMA journal_mode=WAL")
            bootstrap.execute("PRAGMA synchronous=NORMAL")
            bootstrap.commit()
        finally:
            bootstrap.close()
        self._init_db()

    def close(self) -> None:
        # Kept for API symmetry with the previous persistent-connection
        # variant. There is nothing to close because connections are
        # short-lived.
        return None

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Schema

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL DEFAULT '',
                    language TEXT,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL DEFAULT 'fact',
                    text TEXT NOT NULL,
                    scope TEXT NOT NULL DEFAULT 'global',
                    user_id TEXT,
                    chat_id TEXT,
                    persona_key TEXT,
                    importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 0.7,
                    source TEXT,
                    supersedes_id INTEGER,
                    superseded_by INTEGER,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_memory_items_scope_user
                    ON memory_items(scope, user_id);
                CREATE INDEX IF NOT EXISTS idx_memory_items_chat
                    ON memory_items(chat_id);
                CREATE INDEX IF NOT EXISTS idx_memory_items_persona
                    ON memory_items(persona_key);
                CREATE INDEX IF NOT EXISTS idx_memory_items_kind
                    ON memory_items(kind);
                CREATE INDEX IF NOT EXISTS idx_memory_items_active
                    ON memory_items(superseded_by);
                CREATE TABLE IF NOT EXISTS memory_tags (
                    memory_id INTEGER NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (memory_id, tag),
                    FOREIGN KEY (memory_id) REFERENCES memory_items(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag);
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id INTEGER PRIMARY KEY,
                    model TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memory_items(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, id);
                CREATE TABLE IF NOT EXISTS tool_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    arguments TEXT NOT NULL,
                    result TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS kv (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
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
                );
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
                );
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    chat_id TEXT,
                    persona_key TEXT,
                    trigger_at TEXT NOT NULL,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_reminders_due
                    ON reminders(status, trigger_at);
                """
            )
            try:
                # Self-contained FTS5 (no external-content) so DELETE / INSERT
                # work with plain SQL during update_memory. The rowid still
                # mirrors memory_items.id.
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_fts
                    USING fts5(text, tags)
                    """
                )
            except sqlite3.OperationalError:
                pass
            conn.execute(
                "INSERT INTO kv(key, value, updated_at) VALUES('schema_version', ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (str(self.SCHEMA_VERSION), utc_now()),
            )

    # ------------------------------------------------------------------
    # Memory v2 API

    def store_memory(
        self,
        text: str,
        *,
        kind: str = KIND_FACT,
        scope: str = SCOPE_GLOBAL,
        tags: Iterable[str] | None = None,
        user_id: str | None = None,
        chat_id: str | None = None,
        persona_key: str | None = None,
        importance: float = 0.5,
        confidence: float = 0.7,
        source: str | None = None,
        pinned: bool = False,
        supersedes_id: int | None = None,
        embedding: Sequence[float] | None = None,
        embedding_model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        text = text.strip()
        if not text:
            raise ValueError("memory text cannot be empty")
        if kind not in ALL_KINDS:
            kind = KIND_FACT
        if scope not in ALL_SCOPES:
            scope = SCOPE_GLOBAL
        importance = max(0.0, min(1.0, float(importance)))
        confidence = max(0.0, min(1.0, float(confidence)))
        tag_set = sorted({tag.strip() for tag in (tags or []) if tag and tag.strip()})

        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO memory_items(
                    kind, text, scope, user_id, chat_id, persona_key,
                    importance, confidence, source, supersedes_id,
                    pinned, created_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    text,
                    scope,
                    user_id,
                    None if chat_id is None else str(chat_id),
                    persona_key,
                    importance,
                    confidence,
                    source,
                    supersedes_id,
                    1 if pinned else 0,
                    utc_now(),
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            rowid = int(cur.lastrowid)
            for tag in tag_set:
                conn.execute(
                    "INSERT OR IGNORE INTO memory_tags(memory_id, tag) VALUES(?, ?)",
                    (rowid, tag),
                )
            try:
                conn.execute(
                    "INSERT INTO memory_items_fts(rowid, text, tags) VALUES (?, ?, ?)",
                    (rowid, text, " ".join(tag_set)),
                )
            except sqlite3.OperationalError:
                pass
            if supersedes_id is not None:
                conn.execute(
                    "UPDATE memory_items SET superseded_by = ? WHERE id = ?",
                    (rowid, supersedes_id),
                )
            if embedding is not None:
                conn.execute(
                    """
                    INSERT INTO memory_embeddings(memory_id, model, dim, vector)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(memory_id) DO UPDATE SET
                        model = excluded.model,
                        dim = excluded.dim,
                        vector = excluded.vector
                    """,
                    (
                        rowid,
                        embedding_model or "unknown",
                        len(embedding),
                        pack_embedding(embedding),
                    ),
                )
        return rowid

    def attach_embedding(
        self,
        memory_id: int,
        vector: Sequence[float],
        *,
        model: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_embeddings(memory_id, model, dim, vector)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    model = excluded.model,
                    dim = excluded.dim,
                    vector = excluded.vector
                """,
                (memory_id, model, len(vector), pack_embedding(vector)),
            )

    def get_memory(self, memory_id: int) -> MemoryItem | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_items WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                return None
            tags = self._tags_for(conn, memory_id)
        return self._memory_from_row(row, tags)

    def supersede(self, old_id: int, new_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE memory_items SET superseded_by = ? WHERE id = ?",
                (new_id, old_id),
            )
            conn.execute(
                "UPDATE memory_items SET supersedes_id = ? WHERE id = ?",
                (old_id, new_id),
            )

    def delete_memory(self, memory_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM memory_items WHERE id = ?", (memory_id,))
            try:
                conn.execute("DELETE FROM memory_items_fts WHERE rowid = ?", (memory_id,))
            except sqlite3.OperationalError:
                pass

    def update_memory(
        self,
        memory_id: int,
        *,
        text: str | None = None,
        importance: float | None = None,
        tags: Iterable[str] | None = None,
        pinned: bool | None = None,
    ) -> MemoryItem | None:
        """Update a memory item in place.

        Only fields that are not ``None`` are touched. Tag updates replace
        the full tag set. The FTS row is rebuilt to match the new content.
        Returns the refreshed ``MemoryItem`` or ``None`` if the row does not
        exist.
        """

        with self.connect() as conn:
            row = conn.execute(
                "SELECT id, text FROM memory_items WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                return None
            updates: list[str] = []
            params: list[Any] = []
            if text is not None:
                cleaned = text.strip()
                if not cleaned:
                    raise ValueError("memory text cannot be empty")
                updates.append("text = ?")
                params.append(cleaned)
            if importance is not None:
                updates.append("importance = ?")
                params.append(max(0.0, min(1.0, float(importance))))
            if pinned is not None:
                updates.append("pinned = ?")
                params.append(1 if pinned else 0)
            if updates:
                params.append(memory_id)
                conn.execute(
                    f"UPDATE memory_items SET {', '.join(updates)} WHERE id = ?",
                    tuple(params),
                )
            tag_set: list[str] | None = None
            if tags is not None:
                tag_set = sorted({str(tag).strip() for tag in tags if str(tag).strip()})
                conn.execute("DELETE FROM memory_tags WHERE memory_id = ?", (memory_id,))
                for tag in tag_set:
                    conn.execute(
                        "INSERT OR IGNORE INTO memory_tags(memory_id, tag) VALUES(?, ?)",
                        (memory_id, tag),
                    )
            if text is not None or tag_set is not None:
                # Rebuild the FTS row so search reflects the new state.
                effective_tags = tag_set if tag_set is not None else self._tags_for(conn, memory_id)
                effective_text = (
                    text.strip()
                    if text is not None
                    else str(
                        conn.execute(
                            "SELECT text FROM memory_items WHERE id = ?", (memory_id,)
                        ).fetchone()["text"]
                    )
                )
                try:
                    conn.execute(
                        "DELETE FROM memory_items_fts WHERE rowid = ?", (memory_id,)
                    )
                    conn.execute(
                        "INSERT INTO memory_items_fts(rowid, text, tags) VALUES (?, ?, ?)",
                        (memory_id, effective_text, " ".join(effective_tags)),
                    )
                except sqlite3.OperationalError:
                    pass
        return self.get_memory(memory_id)

    def set_pinned(self, memory_id: int, pinned: bool) -> MemoryItem | None:
        return self.update_memory(memory_id, pinned=pinned)

    def mark_accessed(self, memory_ids: Iterable[int]) -> None:
        ids = list({int(value) for value in memory_ids})
        if not ids:
            return
        now = utc_now()
        with self.connect() as conn:
            conn.executemany(
                "UPDATE memory_items SET last_accessed_at = ?, access_count = access_count + 1 "
                "WHERE id = ?",
                [(now, item_id) for item_id in ids],
            )

    def list_memories(
        self,
        *,
        scope: str | None = None,
        user_id: str | None = None,
        chat_id: str | None = None,
        persona_key: str | None = None,
        kind: str | None = None,
        include_superseded: bool = False,
        limit: int = 50,
    ) -> list[MemoryItem]:
        clauses: list[str] = []
        params: list[Any] = []
        if scope:
            clauses.append("scope = ?")
            params.append(scope)
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if chat_id is not None:
            clauses.append("chat_id = ?")
            params.append(str(chat_id))
        if persona_key is not None:
            clauses.append("persona_key = ?")
            params.append(persona_key)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if not include_superseded:
            clauses.append("superseded_by IS NULL")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM memory_items {where} ORDER BY id DESC LIMIT ?",
                tuple(params),
            ).fetchall()
            results: list[MemoryItem] = []
            for row in rows:
                results.append(
                    self._memory_from_row(row, self._tags_for(conn, int(row["id"])))
                )
        return results

    def fts_candidates(
        self,
        query: str,
        *,
        limit: int = 50,
        require_tags: Sequence[str] | None = None,
    ) -> list[MemoryItem]:
        query = query.strip()
        with self.connect() as conn:
            rows: list[sqlite3.Row]
            if query:
                fts_query = self._make_fts_query(query)
                try:
                    rows = conn.execute(
                        """
                        SELECT memory_items.*
                        FROM memory_items_fts
                        JOIN memory_items ON memory_items.id = memory_items_fts.rowid
                        WHERE memory_items_fts MATCH ?
                          AND memory_items.superseded_by IS NULL
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (fts_query, limit * 2),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = conn.execute(
                        """
                        SELECT * FROM memory_items
                        WHERE text LIKE ? AND superseded_by IS NULL
                        ORDER BY id DESC LIMIT ?
                        """,
                        (f"%{query}%", limit * 2),
                    ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memory_items WHERE superseded_by IS NULL "
                    "ORDER BY id DESC LIMIT ?",
                    (limit * 2,),
                ).fetchall()
            results: list[MemoryItem] = []
            for row in rows:
                tags = self._tags_for(conn, int(row["id"]))
                if require_tags and not all(tag in tags for tag in require_tags):
                    continue
                results.append(self._memory_from_row(row, tags))
                if len(results) >= limit:
                    break
        return results

    def all_embeddings(self, *, model: str | None = None) -> list[tuple[int, list[float]]]:
        with self.connect() as conn:
            if model is None:
                rows = conn.execute(
                    """
                    SELECT memory_id, vector FROM memory_embeddings
                    JOIN memory_items ON memory_items.id = memory_embeddings.memory_id
                    WHERE memory_items.superseded_by IS NULL
                    """
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT memory_id, vector FROM memory_embeddings
                    JOIN memory_items ON memory_items.id = memory_embeddings.memory_id
                    WHERE memory_items.superseded_by IS NULL AND memory_embeddings.model = ?
                    """,
                    (model,),
                ).fetchall()
        return [(int(row["memory_id"]), unpack_embedding(row["vector"])) for row in rows]

    def get_memories(self, memory_ids: Iterable[int]) -> dict[int, MemoryItem]:
        ids = list({int(value) for value in memory_ids})
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM memory_items WHERE id IN ({placeholders})",
                tuple(ids),
            ).fetchall()
            mapping: dict[int, MemoryItem] = {}
            for row in rows:
                tags = self._tags_for(conn, int(row["id"]))
                mapping[int(row["id"])] = self._memory_from_row(row, tags)
        return mapping

    def all_active_memory_ids(self) -> list[int]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id FROM memory_items WHERE superseded_by IS NULL"
            ).fetchall()
        return [int(row["id"]) for row in rows]

    # ------------------------------------------------------------------
    # User profiles

    def upsert_user(
        self,
        user_id: str,
        *,
        display_name: str | None = None,
        language: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UserProfile:
        now = utc_now()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row is None:
                merged_meta = metadata or {}
                conn.execute(
                    """
                    INSERT INTO users(user_id, display_name, language, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        display_name or "",
                        language,
                        json.dumps(merged_meta, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
            else:
                merged_meta = json.loads(row["metadata"] or "{}")
                if metadata:
                    merged_meta.update(metadata)
                conn.execute(
                    """
                    UPDATE users
                    SET display_name = COALESCE(?, display_name),
                        language = COALESCE(?, language),
                        metadata = ?,
                        updated_at = ?
                    WHERE user_id = ?
                    """,
                    (
                        display_name,
                        language,
                        json.dumps(merged_meta, ensure_ascii=False),
                        now,
                        user_id,
                    ),
                )
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        return self._user_from_row(row)

    def get_user(self, user_id: str) -> UserProfile | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        return None if row is None else self._user_from_row(row)

    # ------------------------------------------------------------------
    # Reminders

    def add_reminder(
        self,
        *,
        text: str,
        trigger_at: str,
        user_id: str | None = None,
        chat_id: str | None = None,
        persona_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO reminders(user_id, chat_id, persona_key, trigger_at, text, status, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    user_id,
                    None if chat_id is None else str(chat_id),
                    persona_key,
                    trigger_at,
                    text.strip(),
                    utc_now(),
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid)

    def due_reminders(self, now: str, *, limit: int = 20) -> list[Reminder]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM reminders
                WHERE status = 'pending' AND trigger_at <= ?
                ORDER BY trigger_at ASC
                LIMIT ?
                """,
                (now, limit),
            ).fetchall()
        return [self._reminder_from_row(row) for row in rows]

    def mark_reminder(self, reminder_id: int, status: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE reminders SET status = ? WHERE id = ?",
                (status, reminder_id),
            )

    # ------------------------------------------------------------------
    # Backwards-compatible legacy API

    def remember(self, text: str, tags: list[str] | None = None) -> int:
        kind, scope, user_id, chat_id, persona_key = self._infer_legacy_dimensions(tags or [])
        return self.store_memory(
            text,
            kind=kind,
            scope=scope,
            tags=tags,
            user_id=user_id,
            chat_id=chat_id,
            persona_key=persona_key,
            source="legacy_remember",
        )

    def search(self, query: str, *, limit: int = 5) -> list[MemoryFact]:
        query = query.strip()
        if not query:
            return []
        items = self.fts_candidates(query, limit=limit)
        return [self._fact_from_item(item) for item in items[:limit]]

    def search_tagged(self, query: str, tag: str, *, limit: int = 5) -> list[MemoryFact]:
        return self.search_tagged_all(query, [tag], limit=limit)

    def search_tagged_all(
        self, query: str, tags: list[str], *, limit: int = 5
    ) -> list[MemoryFact]:
        required_tags = [tag for tag in tags if tag]
        if not required_tags:
            return self.search(query, limit=limit)
        items = self.fts_candidates(query, limit=max(limit * 4, 20), require_tags=required_tags)
        results: list[MemoryFact] = []
        seen: set[int] = set()
        for item in items:
            if item.id in seen:
                continue
            results.append(self._fact_from_item(item))
            seen.add(item.id)
            if len(results) >= limit:
                break
        if len(results) >= limit:
            return results
        # Fallback: scan recent items with required tags even when FTS missed.
        recent = self.recent_tagged_all(required_tags, limit=max(limit * 4, 20))
        for fact in recent:
            if fact.id in seen:
                continue
            results.append(fact)
            seen.add(fact.id)
            if len(results) >= limit:
                break
        return results[:limit]

    def recent_tagged_all(self, tags: list[str], *, limit: int = 5) -> list[MemoryFact]:
        required_tags = [tag for tag in tags if tag]
        if not required_tags:
            return []
        placeholders = ",".join("?" for _ in required_tags)
        params: list[Any] = list(required_tags)
        params.extend([len(required_tags), max(limit * 4, 20)])
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT memory_items.* FROM memory_items
                JOIN memory_tags ON memory_tags.memory_id = memory_items.id
                WHERE memory_tags.tag IN ({placeholders})
                  AND memory_items.superseded_by IS NULL
                GROUP BY memory_items.id
                HAVING COUNT(DISTINCT memory_tags.tag) = ?
                ORDER BY memory_items.id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
            facts: list[MemoryFact] = []
            for row in rows:
                tags_value = self._tags_for(conn, int(row["id"]))
                if not all(tag in tags_value for tag in required_tags):
                    continue
                item = self._memory_from_row(row, tags_value)
                facts.append(self._fact_from_item(item))
                if len(facts) >= limit:
                    break
        return facts

    # ------------------------------------------------------------------
    # Conversation logs

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

    # ------------------------------------------------------------------
    # Telegram state

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
                "SELECT * FROM telegram_chats WHERE chat_id = ?", (str(chat_id),)
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

    # ------------------------------------------------------------------
    # Helpers

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

    @staticmethod
    def _user_from_row(row: sqlite3.Row) -> UserProfile:
        return UserProfile(
            user_id=str(row["user_id"]),
            display_name=str(row["display_name"] or ""),
            language=row["language"],
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _reminder_from_row(row: sqlite3.Row) -> Reminder:
        return Reminder(
            id=int(row["id"]),
            user_id=row["user_id"],
            chat_id=row["chat_id"],
            persona_key=row["persona_key"],
            trigger_at=str(row["trigger_at"]),
            text=str(row["text"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _memory_from_row(row: sqlite3.Row, tags: list[str]) -> MemoryItem:
        return MemoryItem(
            id=int(row["id"]),
            kind=str(row["kind"]),
            text=str(row["text"]),
            scope=str(row["scope"]),
            user_id=row["user_id"],
            chat_id=row["chat_id"],
            persona_key=row["persona_key"],
            importance=float(row["importance"]),
            confidence=float(row["confidence"]),
            source=row["source"],
            supersedes_id=row["supersedes_id"],
            superseded_by=row["superseded_by"],
            pinned=bool(row["pinned"]),
            created_at=str(row["created_at"]),
            last_accessed_at=row["last_accessed_at"],
            access_count=int(row["access_count"]),
            tags=tags,
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _fact_from_item(item: MemoryItem) -> MemoryFact:
        return MemoryFact(
            id=item.id,
            text=item.text,
            tags=list(item.tags),
            created_at=item.created_at,
            importance=item.importance,
            kind=item.kind,
            scope=item.scope,
        )

    @staticmethod
    def _tags_for(conn: sqlite3.Connection, memory_id: int) -> list[str]:
        rows = conn.execute(
            "SELECT tag FROM memory_tags WHERE memory_id = ? ORDER BY tag",
            (memory_id,),
        ).fetchall()
        return [str(row["tag"]) for row in rows]

    @staticmethod
    def _infer_legacy_dimensions(
        tags: list[str],
    ) -> tuple[str, str, str | None, str | None, str | None]:
        """Map legacy free-form tags into typed dimensions.

        Older callers tag with substrings like ``persona:solomiya`` or
        ``source_chat:123``; v2 stores those dimensions in dedicated columns.
        We still keep the original tag set so legacy ``search_tagged`` queries
        match identically.
        """

        kind = KIND_FACT
        scope = SCOPE_GLOBAL
        user_id: str | None = None
        chat_id: str | None = None
        persona_key: str | None = None
        for tag in tags:
            if tag.startswith("persona:"):
                persona_key = tag.split(":", 1)[1].strip() or None
            elif tag.startswith("source_chat:"):
                chat_id = tag.split(":", 1)[1].strip() or None
            elif tag.startswith("user:"):
                user_id = tag.split(":", 1)[1].strip() or None
            elif tag.startswith("telegram_chat_"):
                chat_id = tag[len("telegram_chat_") :] or None
            elif tag in {"telegram_persona_self", "persona_self"}:
                kind = KIND_PERSONA_SELF
                scope = SCOPE_PERSONA
        if chat_id and scope == SCOPE_GLOBAL and kind != KIND_PERSONA_SELF:
            scope = SCOPE_CHAT
        return kind, scope, user_id, chat_id, persona_key
