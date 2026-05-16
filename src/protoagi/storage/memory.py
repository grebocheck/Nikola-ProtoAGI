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

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .models import (
    ALL_CONFLICT_STATUSES,
    ALL_GOAL_STATUSES,
    ALL_KINDS,
    ALL_SCOPES,
    CONFLICT_STATUS_DISMISSED,
    CONFLICT_STATUS_KEPT_BOTH,
    CONFLICT_STATUS_SUPERSEDED,
    CONFLICT_STATUS_UNRESOLVED,
    GOAL_STATUS_ABANDONED,
    GOAL_STATUS_COMPLETED,
    GOAL_STATUS_OPEN,
    Goal,
    KIND_EPISODIC,
    KIND_FACT,
    KIND_PERSONA_SELF,
    KIND_PROCEDURAL,
    KIND_SEMANTIC,
    MediaBlob,
    MemoryConflict,
    MemoryFact,
    MemoryItem,
    RecallResult,
    Reminder,
    SCOPE_CHAT,
    SCOPE_GLOBAL,
    SCOPE_PERSONA,
    SCOPE_USER,
    StickerDescription,
    TelegramChat,
    TelegramMessage,
    UserProfile,
    UserState,
    cosine_similarity,
    pack_embedding,
    unpack_embedding,
    utc_now,
)

__all__ = [
    "ALL_CONFLICT_STATUSES",
    "ALL_GOAL_STATUSES",
    "ALL_KINDS",
    "ALL_SCOPES",
    "CONFLICT_STATUS_DISMISSED",
    "CONFLICT_STATUS_KEPT_BOTH",
    "CONFLICT_STATUS_SUPERSEDED",
    "CONFLICT_STATUS_UNRESOLVED",
    "GOAL_STATUS_ABANDONED",
    "GOAL_STATUS_COMPLETED",
    "GOAL_STATUS_OPEN",
    "Goal",
    "KIND_EPISODIC",
    "KIND_FACT",
    "KIND_PERSONA_SELF",
    "KIND_PROCEDURAL",
    "KIND_SEMANTIC",
    "MediaBlob",
    "MemoryConflict",
    "MemoryFact",
    "MemoryItem",
    "MemoryStore",
    "RecallResult",
    "Reminder",
    "SCOPE_CHAT",
    "SCOPE_GLOBAL",
    "SCOPE_PERSONA",
    "SCOPE_USER",
    "StickerDescription",
    "TelegramChat",
    "TelegramMessage",
    "UserProfile",
    "UserState",
    "cosine_similarity",
    "pack_embedding",
    "unpack_embedding",
    "utc_now",
]


def _tag_suffix(tags: Iterable[str], prefix: str) -> str | None:
    for raw in tags:
        tag = str(raw)
        if tag.startswith(prefix):
            value = tag[len(prefix) :].strip()
            if value:
                return value
    return None


# ---------------------------------------------------------------------------
# Storage class


class MemoryStore:
    """SQLite-backed memory storage with v2 schema and legacy-compatible API.

    The database is opened in WAL mode once at init, then every operation
    uses a short-lived per-call connection through ``connect()``. WAL mode
    is persistent at the file level, so a fresh connection still benefits
    from concurrent readers + single-writer semantics. Per-call connections
    keep Windows tempdir cleanup in tests trivial (no lingering file
    handles).
    """

    SCHEMA_VERSION = 8

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
                CREATE TABLE IF NOT EXISTS media_blobs (
                    file_id TEXT PRIMARY KEY,
                    mime TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    bytes BLOB NOT NULL,
                    caption TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL DEFAULT 'fact',
                    text TEXT NOT NULL,
                    scope TEXT NOT NULL DEFAULT 'global',
                    user_id TEXT,
                    chat_id TEXT,
                    persona_key TEXT,
                    media_id TEXT,
                    importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 0.7,
                    source TEXT,
                    supersedes_id INTEGER,
                    superseded_by INTEGER,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
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
                CREATE INDEX IF NOT EXISTS idx_memory_items_media
                    ON memory_items(media_id);
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
                CREATE TABLE IF NOT EXISTS importance_cache (
                    key TEXT PRIMARY KEY,
                    importance REAL NOT NULL,
                    kind TEXT NOT NULL,
                    reasoning TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_importance_cache_accessed
                    ON importance_cache(last_accessed_at);
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
                CREATE TABLE IF NOT EXISTS goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    persona_key TEXT NOT NULL,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    priority REAL NOT NULL DEFAULT 0.5,
                    user_id TEXT,
                    chat_id TEXT,
                    origin_message_id INTEGER,
                    due_at TEXT,
                    last_touched_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    closed_at TEXT,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_goals_persona_status
                    ON goals(persona_key, status);
                CREATE INDEX IF NOT EXISTS idx_goals_chat_status
                    ON goals(chat_id, status);
                CREATE INDEX IF NOT EXISTS idx_goals_due_open
                    ON goals(due_at) WHERE status = 'open';
                CREATE TABLE IF NOT EXISTS sticker_descriptions (
                    sticker_id TEXT PRIMARY KEY,
                    set_name TEXT NOT NULL,
                    emoji TEXT NOT NULL DEFAULT '',
                    description TEXT NOT NULL DEFAULT '',
                    embedding BLOB,
                    embedding_model TEXT,
                    failure_reason TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_used_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_sticker_descriptions_set
                    ON sticker_descriptions(set_name);
                CREATE INDEX IF NOT EXISTS idx_sticker_descriptions_desc_present
                    ON sticker_descriptions(set_name) WHERE description != '';
                CREATE TABLE IF NOT EXISTS user_state (
                    user_id TEXT NOT NULL,
                    persona_key TEXT NOT NULL,
                    mood TEXT NOT NULL DEFAULT '',
                    themes TEXT NOT NULL DEFAULT '[]',
                    open_questions TEXT NOT NULL DEFAULT '[]',
                    preferences TEXT NOT NULL DEFAULT '{}',
                    summary TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    last_updated_at TEXT NOT NULL,
                    messages_at_last_update INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (user_id, persona_key)
                );
                CREATE INDEX IF NOT EXISTS idx_user_state_persona_age
                    ON user_state(persona_key, last_updated_at);
                CREATE TABLE IF NOT EXISTS memory_conflicts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_a_id INTEGER NOT NULL,
                    memory_b_id INTEGER NOT NULL,
                    similarity REAL NOT NULL,
                    persona_key TEXT,
                    detected_at TEXT NOT NULL,
                    resolution_status TEXT NOT NULL DEFAULT 'unresolved',
                    resolution_winner_id INTEGER,
                    resolved_at TEXT,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (memory_a_id) REFERENCES memory_items(id) ON DELETE CASCADE,
                    FOREIGN KEY (memory_b_id) REFERENCES memory_items(id) ON DELETE CASCADE,
                    CHECK (memory_a_id < memory_b_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_conflicts_pair
                    ON memory_conflicts(memory_a_id, memory_b_id);
                CREATE INDEX IF NOT EXISTS idx_memory_conflicts_unresolved
                    ON memory_conflicts(resolution_status, detected_at)
                    WHERE resolution_status = 'unresolved';
                CREATE INDEX IF NOT EXISTS idx_memory_conflicts_persona
                    ON memory_conflicts(persona_key, resolution_status);
                """
            )
            self._ensure_column(conn, "memory_items", "media_id", "TEXT")
            self._ensure_column(conn, "memory_items", "updated_at", "TEXT")
            self._ensure_column(conn, "memory_items", "origin_message_id", "TEXT")
            self._ensure_column(conn, "memory_items", "expires_at", "TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_items_media ON memory_items(media_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_items_origin "
                "ON memory_items(origin_message_id) WHERE origin_message_id IS NOT NULL"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_items_expires "
                "ON memory_items(expires_at) WHERE expires_at IS NOT NULL"
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
        media_id: str | None = None,
        pinned: bool = False,
        supersedes_id: int | None = None,
        embedding: Sequence[float] | None = None,
        embedding_model: str | None = None,
        metadata: dict[str, Any] | None = None,
        origin_message_id: str | int | None = None,
        expires_at: str | None = None,
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

        now = utc_now()
        origin_value: str | None
        if origin_message_id is None:
            origin_value = None
        else:
            stripped = str(origin_message_id).strip()
            origin_value = stripped or None
        expires_value: str | None = None
        if expires_at is not None:
            stripped = str(expires_at).strip()
            expires_value = stripped or None
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO memory_items(
                    kind, text, scope, user_id, chat_id, persona_key, media_id,
                    importance, confidence, source, supersedes_id,
                    pinned, created_at, updated_at, metadata, origin_message_id,
                    expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    text,
                    scope,
                    user_id,
                    None if chat_id is None else str(chat_id),
                    persona_key,
                    media_id,
                    importance,
                    confidence,
                    source,
                    supersedes_id,
                    1 if pinned else 0,
                    now,
                    now,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    origin_value,
                    expires_value,
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

    def store_media_blob(
        self,
        *,
        file_id: str,
        mime: str,
        data: bytes,
        caption: str = "",
    ) -> MediaBlob:
        file_id = str(file_id or "").strip()
        if not file_id:
            raise ValueError("media file_id cannot be empty")
        if not data:
            raise ValueError("media bytes cannot be empty")
        digest = hashlib.sha256(data).hexdigest()
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO media_blobs(file_id, mime, sha256, bytes, caption, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    mime = excluded.mime,
                    sha256 = excluded.sha256,
                    bytes = excluded.bytes,
                    caption = excluded.caption
                """,
                (file_id, mime or "application/octet-stream", digest, data, caption, now),
            )
        found = self.get_media_blob(file_id)
        if found is None:
            raise RuntimeError("media blob store failed")
        return found

    def get_media_blob(self, file_id: str) -> MediaBlob | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM media_blobs WHERE file_id = ?", (str(file_id),)
            ).fetchone()
        return None if row is None else self._media_from_row(row)

    def prune_orphan_media(self, *, older_than_days: float = 60.0) -> int:
        """Delete old media blobs no memory item still references."""

        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat(
            timespec="seconds"
        )
        with self.connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM media_blobs
                WHERE created_at < ?
                  AND NOT EXISTS (
                    SELECT 1 FROM memory_items
                    WHERE memory_items.media_id = media_blobs.file_id
                  )
                """,
                (cutoff,),
            )
        return max(0, int(cur.rowcount))

    def get_importance_cache(self, key: str) -> dict[str, Any] | None:
        key = str(key or "").strip()
        if not key:
            return None
        now = utc_now()
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT key, importance, kind, reasoning, created_at, last_accessed_at, access_count
                FROM importance_cache
                WHERE key = ?
                """,
                (key,),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                """
                UPDATE importance_cache
                SET last_accessed_at = ?, access_count = access_count + 1
                WHERE key = ?
                """,
                (now, key),
            )
        return {
            "key": str(row["key"]),
            "importance": float(row["importance"]),
            "kind": str(row["kind"]),
            "reasoning": str(row["reasoning"] or ""),
            "created_at": str(row["created_at"]),
            "last_accessed_at": str(row["last_accessed_at"]),
            "access_count": int(row["access_count"]),
        }

    def set_importance_cache(
        self,
        key: str,
        *,
        importance: float,
        kind: str,
        reasoning: str = "",
    ) -> None:
        key = str(key or "").strip()
        if not key:
            return
        if kind not in ALL_KINDS:
            kind = KIND_FACT
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO importance_cache(
                    key, importance, kind, reasoning, created_at, last_accessed_at, access_count
                )
                VALUES (?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(key) DO UPDATE SET
                    importance = excluded.importance,
                    kind = excluded.kind,
                    reasoning = excluded.reasoning,
                    last_accessed_at = excluded.last_accessed_at
                """,
                (
                    key,
                    max(0.0, min(1.0, float(importance))),
                    kind,
                    str(reasoning or "")[:500],
                    now,
                    now,
                ),
            )

    def prune_importance_cache(
        self,
        *,
        older_than_days: float = 60.0,
        max_entries: int = 10000,
    ) -> dict[str, int]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat(
            timespec="seconds"
        )
        deleted_old = 0
        deleted_overflow = 0
        with self.connect() as conn:
            cur = conn.execute(
                "DELETE FROM importance_cache WHERE created_at < ?",
                (cutoff,),
            )
            deleted_old = max(0, int(cur.rowcount))
            total = int(conn.execute("SELECT COUNT(*) FROM importance_cache").fetchone()[0])
            overflow = max(0, total - max(0, int(max_entries)))
            if overflow:
                rows = conn.execute(
                    """
                    SELECT key
                    FROM importance_cache
                    ORDER BY last_accessed_at ASC, created_at ASC
                    LIMIT ?
                    """,
                    (overflow,),
                ).fetchall()
                keys = [str(row["key"]) for row in rows]
                if keys:
                    placeholders = ",".join("?" for _ in keys)
                    cur = conn.execute(
                        f"DELETE FROM importance_cache WHERE key IN ({placeholders})",
                        tuple(keys),
                    )
                    deleted_overflow = max(0, int(cur.rowcount))
            remaining = int(conn.execute("SELECT COUNT(*) FROM importance_cache").fetchone()[0])
        return {
            "deleted": deleted_old + deleted_overflow,
            "deleted_old": deleted_old,
            "deleted_overflow": deleted_overflow,
            "remaining": remaining,
        }

    def importance_cache_count(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM importance_cache").fetchone()[0])

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
                "UPDATE memory_items SET superseded_by = ?, updated_at = ? WHERE id = ?",
                (new_id, utc_now(), old_id),
            )
            conn.execute(
                "UPDATE memory_items SET supersedes_id = ?, updated_at = ? WHERE id = ?",
                (old_id, utc_now(), new_id),
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
                updates.append("updated_at = ?")
                params.append(utc_now())
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
                if not updates:
                    conn.execute(
                        "UPDATE memory_items SET updated_at = ? WHERE id = ?",
                        (utc_now(), memory_id),
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
        # Hide expired working-memory rows from regular listings. The
        # reflection sweep is what actually deletes them; until then,
        # the row stays in storage but stops surfacing.
        clauses.append("(expires_at IS NULL OR expires_at > ?)")
        params.append(utc_now())
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

    def rescope_telegram_memories(
        self,
        *,
        to_scope: str = SCOPE_USER,
        dry_run: bool = False,
    ) -> dict[str, int | bool | str]:
        """Move legacy Telegram global rows into a narrower typed scope.

        The first supported migration targets privacy-mode deployments:
        rows tagged ``user:<id>`` become ``scope=user`` with ``user_id`` set.
        ``source_chat:<id>`` is copied into ``chat_id`` when the row does not
        already have one.
        """

        if to_scope != SCOPE_USER:
            raise ValueError("only --to user is currently supported")
        matched = 0
        updated = 0
        skipped_no_user_tag = 0
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM memory_items
                WHERE scope = ?
                """,
                (SCOPE_GLOBAL,),
            ).fetchall()
            for row in rows:
                tags = self._tags_for(conn, int(row["id"]))
                user_id = _tag_suffix(tags, "user:")
                if not user_id:
                    skipped_no_user_tag += 1
                    continue
                matched += 1
                chat_id = row["chat_id"] or _tag_suffix(tags, "source_chat:")
                if dry_run:
                    continue
                conn.execute(
                    """
                    UPDATE memory_items
                    SET scope = ?, user_id = ?, chat_id = COALESCE(chat_id, ?), updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        SCOPE_USER,
                        user_id,
                        None if chat_id is None else str(chat_id),
                        utc_now(),
                        int(row["id"]),
                    ),
                )
                updated += 1
        return {
            "to": to_scope,
            "dry_run": dry_run,
            "matched": matched,
            "updated": updated,
            "skipped_no_user_tag": skipped_no_user_tag,
        }

    def fts_candidates(
        self,
        query: str,
        *,
        limit: int = 50,
        require_tags: Sequence[str] | None = None,
    ) -> list[MemoryItem]:
        query = query.strip()
        now = utc_now()
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
                          AND (memory_items.expires_at IS NULL OR memory_items.expires_at > ?)
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (fts_query, now, limit * 2),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = conn.execute(
                        """
                        SELECT * FROM memory_items
                        WHERE text LIKE ? AND superseded_by IS NULL
                          AND (expires_at IS NULL OR expires_at > ?)
                        ORDER BY id DESC LIMIT ?
                        """,
                        (f"%{query}%", now, limit * 2),
                    ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memory_items WHERE superseded_by IS NULL "
                    "AND (expires_at IS NULL OR expires_at > ?) "
                    "ORDER BY id DESC LIMIT ?",
                    (now, limit * 2),
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
    # Goals (persistent commitments / open threads)

    def open_goal(
        self,
        *,
        persona_key: str,
        text: str,
        priority: float = 0.5,
        user_id: str | None = None,
        chat_id: str | int | None = None,
        origin_message_id: int | None = None,
        due_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("goal text cannot be empty")
        persona = (persona_key or "").strip()
        if not persona:
            raise ValueError("goal persona_key cannot be empty")
        priority_value = max(0.0, min(1.0, float(priority)))
        chat_id_value = None if chat_id is None else str(chat_id)
        now = utc_now()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO goals(
                    persona_key, text, status, priority, user_id, chat_id,
                    origin_message_id, due_at, last_touched_at, created_at,
                    updated_at, metadata
                )
                VALUES (?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    persona,
                    cleaned,
                    priority_value,
                    user_id,
                    chat_id_value,
                    origin_message_id,
                    due_at,
                    now,
                    now,
                    now,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid)

    def get_goal(self, goal_id: int) -> Goal | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM goals WHERE id = ?", (int(goal_id),)
            ).fetchone()
        return None if row is None else self._goal_from_row(row)

    def list_open_goals(
        self,
        *,
        persona_key: str,
        chat_id: str | int | None = None,
        user_id: str | None = None,
        limit: int = 20,
    ) -> list[Goal]:
        clauses: list[str] = ["persona_key = ?", "status = 'open'"]
        params: list[Any] = [persona_key]
        if chat_id is not None:
            clauses.append("chat_id = ?")
            params.append(str(chat_id))
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        params.append(int(limit))
        where = " AND ".join(clauses)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM goals
                WHERE {where}
                ORDER BY
                    CASE WHEN due_at IS NULL THEN 1 ELSE 0 END,
                    due_at ASC,
                    priority DESC,
                    last_touched_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [self._goal_from_row(row) for row in rows]

    def list_due_goals(
        self,
        *,
        persona_key: str,
        now: str,
        lookahead_hours: float = 24.0,
        chat_id: str | int | None = None,
        limit: int = 10,
    ) -> list[Goal]:
        """Open goals with a ``due_at`` at or before ``now + lookahead_hours``.

        Goals without a ``due_at`` are intentionally excluded: this method is
        for the initiative loop, which needs a real time anchor to decide
        whether to act now.
        """

        cutoff = (
            datetime.fromisoformat(now.replace("Z", "+00:00"))
            + timedelta(hours=float(lookahead_hours))
        ).isoformat(timespec="seconds")
        clauses: list[str] = [
            "persona_key = ?",
            "status = 'open'",
            "due_at IS NOT NULL",
            "due_at <= ?",
        ]
        params: list[Any] = [persona_key, cutoff]
        if chat_id is not None:
            clauses.append("chat_id = ?")
            params.append(str(chat_id))
        params.append(int(limit))
        where = " AND ".join(clauses)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM goals
                WHERE {where}
                ORDER BY due_at ASC, priority DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [self._goal_from_row(row) for row in rows]

    def update_goal(
        self,
        goal_id: int,
        *,
        status: str | None = None,
        priority: float | None = None,
        text: str | None = None,
        due_at: str | None | type(...) = ...,
        metadata_patch: dict[str, Any] | None = None,
    ) -> Goal | None:
        """Partial update. ``due_at=...`` means "leave as-is"; pass ``None`` to clear it."""

        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM goals WHERE id = ?", (int(goal_id),)
            ).fetchone()
            if row is None:
                return None
            updates: list[str] = []
            params: list[Any] = []
            now = utc_now()
            if status is not None:
                if status not in ALL_GOAL_STATUSES:
                    raise ValueError(f"unknown goal status: {status}")
                updates.append("status = ?")
                params.append(status)
                if status != GOAL_STATUS_OPEN:
                    updates.append("closed_at = ?")
                    params.append(now)
                else:
                    updates.append("closed_at = NULL")
            if priority is not None:
                updates.append("priority = ?")
                params.append(max(0.0, min(1.0, float(priority))))
            if text is not None:
                cleaned = text.strip()
                if not cleaned:
                    raise ValueError("goal text cannot be empty")
                updates.append("text = ?")
                params.append(cleaned)
            if due_at is not ...:
                updates.append("due_at = ?")
                params.append(due_at)
            if metadata_patch is not None:
                existing = json.loads(row["metadata"] or "{}")
                existing.update(metadata_patch)
                updates.append("metadata = ?")
                params.append(json.dumps(existing, ensure_ascii=False))
            if not updates:
                return self._goal_from_row(row)
            updates.append("updated_at = ?")
            params.append(now)
            updates.append("last_touched_at = ?")
            params.append(now)
            params.append(int(goal_id))
            conn.execute(
                f"UPDATE goals SET {', '.join(updates)} WHERE id = ?",
                tuple(params),
            )
        return self.get_goal(goal_id)

    def touch_goal(self, goal_id: int) -> None:
        """Bump ``last_touched_at`` so list ordering reflects recent attention."""

        with self.connect() as conn:
            conn.execute(
                "UPDATE goals SET last_touched_at = ? WHERE id = ?",
                (utc_now(), int(goal_id)),
            )

    def count_goals(
        self,
        *,
        persona_key: str | None = None,
        status: str | None = None,
    ) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if persona_key is not None:
            clauses.append("persona_key = ?")
            params.append(persona_key)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM goals {where}", tuple(params)
            ).fetchone()
        return int(row["n"])

    # ------------------------------------------------------------------
    # User state (persona-specific working model of a user)

    def get_user_state(self, user_id: str, persona_key: str) -> UserState | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM user_state WHERE user_id = ? AND persona_key = ?",
                (str(user_id), str(persona_key)),
            ).fetchone()
        return None if row is None else self._user_state_from_row(row)

    def upsert_user_state(
        self,
        *,
        user_id: str,
        persona_key: str,
        mood: str = "",
        themes: list[str] | None = None,
        open_questions: list[str] | None = None,
        preferences: dict[str, Any] | None = None,
        summary: str = "",
        confidence: float = 0.5,
        messages_at_last_update: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> UserState:
        user_id = str(user_id or "").strip()
        persona = str(persona_key or "").strip()
        if not user_id or not persona:
            raise ValueError("user_state needs user_id and persona_key")
        themes_clean = [str(t).strip() for t in (themes or []) if str(t).strip()][:10]
        questions_clean = [str(q).strip() for q in (open_questions or []) if str(q).strip()][:10]
        confidence_value = max(0.0, min(1.0, float(confidence)))
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO user_state(
                    user_id, persona_key, mood, themes, open_questions,
                    preferences, summary, confidence, last_updated_at,
                    messages_at_last_update, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, persona_key) DO UPDATE SET
                    mood = excluded.mood,
                    themes = excluded.themes,
                    open_questions = excluded.open_questions,
                    preferences = excluded.preferences,
                    summary = excluded.summary,
                    confidence = excluded.confidence,
                    last_updated_at = excluded.last_updated_at,
                    messages_at_last_update = excluded.messages_at_last_update,
                    metadata = excluded.metadata
                """,
                (
                    user_id,
                    persona,
                    str(mood or "").strip()[:200],
                    json.dumps(themes_clean, ensure_ascii=False),
                    json.dumps(questions_clean, ensure_ascii=False),
                    json.dumps(preferences or {}, ensure_ascii=False),
                    str(summary or "").strip()[:1000],
                    confidence_value,
                    now,
                    int(messages_at_last_update),
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
        found = self.get_user_state(user_id, persona)
        if found is None:
            raise RuntimeError("user_state upsert failed")
        return found

    def stale_user_states(
        self,
        *,
        persona_key: str,
        older_than: str,
        limit: int = 20,
    ) -> list[UserState]:
        """Return rows whose ``last_updated_at`` is older than ``older_than``.

        Used by the reflection pass to pick which states to refresh. The
        caller computes ``older_than`` (e.g. ``now - 24h``).
        """

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM user_state
                WHERE persona_key = ? AND last_updated_at < ?
                ORDER BY last_updated_at ASC
                LIMIT ?
                """,
                (str(persona_key), str(older_than), int(limit)),
            ).fetchall()
        return [self._user_state_from_row(row) for row in rows]

    def list_user_state_user_ids(self, persona_key: str) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT user_id FROM user_state WHERE persona_key = ?",
                (str(persona_key),),
            ).fetchall()
        return [str(row["user_id"]) for row in rows]

    def count_user_messages(
        self,
        *,
        chat_id: str | int | None = None,
        user_id: str | None = None,
        since: str | None = None,
    ) -> int:
        clauses: list[str] = ["role = 'user'"]
        params: list[Any] = []
        if chat_id is not None:
            clauses.append("chat_id = ?")
            params.append(str(chat_id))
        if user_id is not None:
            clauses.append("sender_id = ?")
            params.append(str(user_id))
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(since)
        where = " AND ".join(clauses)
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM telegram_messages WHERE {where}",
                tuple(params),
            ).fetchone()
        return int(row["n"])

    def recent_user_message_texts(
        self,
        *,
        user_id: str,
        limit: int = 20,
    ) -> list[dict[str, str]]:
        """Last N Telegram messages where the user was the sender.

        Used by the user_state refresh routine. Returns oldest-first so
        the LLM sees the conversation in temporal order.
        """

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT chat_id, message_id, text, created_at
                FROM telegram_messages
                WHERE role = 'user' AND sender_id = ?
                ORDER BY created_at DESC, message_id DESC
                LIMIT ?
                """,
                (str(user_id), int(limit)),
            ).fetchall()
        items = [
            {
                "chat_id": str(row["chat_id"]),
                "text": str(row["text"] or ""),
                "created_at": str(row["created_at"]),
            }
            for row in rows
            if str(row["text"] or "").strip()
        ]
        items.reverse()
        return items

    # ------------------------------------------------------------------
    # Sticker descriptions (vision-model captions cached per sticker)

    def get_sticker_description(self, sticker_id: str) -> StickerDescription | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM sticker_descriptions WHERE sticker_id = ?",
                (str(sticker_id),),
            ).fetchone()
        return None if row is None else self._sticker_description_from_row(row)

    def upsert_sticker_description(
        self,
        *,
        sticker_id: str,
        set_name: str,
        emoji: str = "",
        description: str = "",
        embedding: Sequence[float] | None = None,
        embedding_model: str | None = None,
        failure_reason: str | None = None,
    ) -> StickerDescription:
        sticker_id = str(sticker_id or "").strip()
        set_name = str(set_name or "").strip()
        if not sticker_id or not set_name:
            raise ValueError("sticker description needs sticker_id and set_name")
        emoji_clean = str(emoji or "").strip()
        description_clean = str(description or "").strip()
        failure_clean = str(failure_reason or "").strip() or None
        now = utc_now()
        vector_blob = pack_embedding(embedding) if embedding else None
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT attempt_count, created_at FROM sticker_descriptions "
                "WHERE sticker_id = ?",
                (sticker_id,),
            ).fetchone()
            attempts = int(existing["attempt_count"]) + 1 if existing else 1
            created_at = str(existing["created_at"]) if existing else now
            conn.execute(
                """
                INSERT INTO sticker_descriptions(
                    sticker_id, set_name, emoji, description, embedding,
                    embedding_model, failure_reason, attempt_count,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sticker_id) DO UPDATE SET
                    set_name = excluded.set_name,
                    emoji = excluded.emoji,
                    description = excluded.description,
                    embedding = excluded.embedding,
                    embedding_model = excluded.embedding_model,
                    failure_reason = excluded.failure_reason,
                    attempt_count = excluded.attempt_count,
                    updated_at = excluded.updated_at
                """,
                (
                    sticker_id,
                    set_name,
                    emoji_clean,
                    description_clean,
                    vector_blob,
                    embedding_model,
                    failure_clean,
                    attempts,
                    created_at,
                    now,
                ),
            )
        found = self.get_sticker_description(sticker_id)
        if found is None:
            raise RuntimeError("sticker description upsert failed")
        return found

    def list_sticker_descriptions(
        self,
        *,
        set_name: str | None = None,
        only_described: bool = False,
        limit: int = 1000,
    ) -> list[StickerDescription]:
        clauses: list[str] = []
        params: list[Any] = []
        if set_name is not None:
            clauses.append("set_name = ?")
            params.append(set_name)
        if only_described:
            clauses.append("description != ''")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(limit))
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM sticker_descriptions {where} LIMIT ?",
                tuple(params),
            ).fetchall()
        return [self._sticker_description_from_row(row) for row in rows]

    def list_undescribed_stickers(
        self,
        *,
        set_name: str | None = None,
        max_attempts: int = 3,
        limit: int = 100,
    ) -> list[StickerDescription]:
        """Stickers we haven't successfully captioned yet.

        ``attempt_count`` is bumped on every upsert, so persistently
        failing stickers stop getting retried after ``max_attempts``.
        """

        clauses = ["description = ''", "attempt_count < ?"]
        params: list[Any] = [int(max_attempts)]
        if set_name is not None:
            clauses.append("set_name = ?")
            params.append(set_name)
        where = " AND ".join(clauses)
        params.append(int(limit))
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM sticker_descriptions
                WHERE {where}
                ORDER BY attempt_count ASC, created_at ASC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [self._sticker_description_from_row(row) for row in rows]

    def mark_sticker_used(self, sticker_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE sticker_descriptions SET last_used_at = ? WHERE sticker_id = ?",
                (utc_now(), str(sticker_id)),
            )

    def count_sticker_descriptions(
        self,
        *,
        set_name: str | None = None,
        only_described: bool = False,
    ) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if set_name is not None:
            clauses.append("set_name = ?")
            params.append(set_name)
        if only_described:
            clauses.append("description != ''")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM sticker_descriptions {where}",
                tuple(params),
            ).fetchone()
        return int(row["n"])

    # ------------------------------------------------------------------
    # Memory conflicts (pairs of semantically-similar facts the
    # consolidate pass did not auto-merge — candidates for the persona
    # to review or for belief revision down the line).

    def record_conflict(
        self,
        memory_a_id: int,
        memory_b_id: int,
        *,
        similarity: float,
        persona_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int | None:
        """Insert a conflict pair (normalized so smaller id is ``a``).

        Returns the new row id, or ``None`` when the pair already exists
        (we treat the existing row as canonical). Refuses degenerate
        same-id pairs since they cannot conflict.
        """

        a = int(memory_a_id)
        b = int(memory_b_id)
        if a == b:
            return None
        if a > b:
            a, b = b, a
        sim_value = max(0.0, min(1.0, float(similarity)))
        now = utc_now()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO memory_conflicts(
                    memory_a_id, memory_b_id, similarity, persona_key,
                    detected_at, resolution_status, metadata
                )
                VALUES (?, ?, ?, ?, ?, 'unresolved', ?)
                """,
                (
                    a,
                    b,
                    sim_value,
                    persona_key,
                    now,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            if cur.rowcount == 0:
                return None
            return int(cur.lastrowid)

    def get_conflict(self, conflict_id: int) -> MemoryConflict | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_conflicts WHERE id = ?", (int(conflict_id),)
            ).fetchone()
        return None if row is None else self._conflict_from_row(row)

    def list_unresolved_conflicts(
        self,
        *,
        persona_key: str | None = None,
        limit: int = 20,
    ) -> list[MemoryConflict]:
        clauses: list[str] = ["resolution_status = 'unresolved'"]
        params: list[Any] = []
        if persona_key is not None:
            clauses.append("persona_key = ?")
            params.append(persona_key)
        params.append(int(limit))
        where = " AND ".join(clauses)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memory_conflicts
                WHERE {where}
                ORDER BY detected_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [self._conflict_from_row(row) for row in rows]

    def resolve_conflict(
        self,
        conflict_id: int,
        *,
        status: str,
        winner_id: int | None = None,
        metadata_patch: dict[str, Any] | None = None,
    ) -> MemoryConflict | None:
        if status not in ALL_CONFLICT_STATUSES:
            raise ValueError(f"unknown conflict status: {status}")
        if status != CONFLICT_STATUS_UNRESOLVED and winner_id is None and status == CONFLICT_STATUS_SUPERSEDED:
            raise ValueError("status=superseded requires winner_id")
        now = utc_now()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_conflicts WHERE id = ?", (int(conflict_id),)
            ).fetchone()
            if row is None:
                return None
            updates: list[str] = ["resolution_status = ?"]
            params: list[Any] = [status]
            if winner_id is not None:
                updates.append("resolution_winner_id = ?")
                params.append(int(winner_id))
            if status == CONFLICT_STATUS_UNRESOLVED:
                updates.append("resolved_at = NULL")
            else:
                updates.append("resolved_at = ?")
                params.append(now)
            if metadata_patch:
                existing = json.loads(row["metadata"] or "{}")
                existing.update(metadata_patch)
                updates.append("metadata = ?")
                params.append(json.dumps(existing, ensure_ascii=False))
            params.append(int(conflict_id))
            conn.execute(
                f"UPDATE memory_conflicts SET {', '.join(updates)} WHERE id = ?",
                tuple(params),
            )
        return self.get_conflict(conflict_id)

    def expire_working_memory(self, *, grace_seconds: int = 0) -> int:
        """Hard-delete memory_items past their ``expires_at``.

        Called from the reflection pass. The grace window lets a row sit
        around for a bit after expiry so a concurrent reader doesn't see
        it vanish mid-query. Returns the number of rows deleted.
        """

        now = datetime.now(timezone.utc) - timedelta(seconds=max(0, int(grace_seconds)))
        cutoff = now.isoformat(timespec="seconds")
        with self.connect() as conn:
            cur = conn.execute(
                "DELETE FROM memory_items WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (cutoff,),
            )
            try:
                # Keep FTS in sync — the trigger-free FTS5 setup needs
                # manual cleanup. ``rowid NOT IN (SELECT id FROM memory_items)``
                # would scan twice; the per-row delete above already
                # cascades for our usage.
                conn.execute(
                    "DELETE FROM memory_items_fts "
                    "WHERE rowid NOT IN (SELECT id FROM memory_items)"
                )
            except sqlite3.OperationalError:
                pass
        return max(0, int(cur.rowcount))

    def count_memories(
        self,
        *,
        scope: str | None = None,
        persona_key: str | None = None,
        include_superseded: bool = False,
    ) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope)
        if persona_key is not None:
            clauses.append("persona_key = ?")
            params.append(persona_key)
        if not include_superseded:
            clauses.append("superseded_by IS NULL")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM memory_items {where}",
                tuple(params),
            ).fetchone()
        return int(row["n"])

    def count_user_states(self, *, persona_key: str | None = None) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if persona_key is not None:
            clauses.append("persona_key = ?")
            params.append(persona_key)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM user_state {where}",
                tuple(params),
            ).fetchone()
        return int(row["n"])

    def count_conflicts(
        self,
        *,
        persona_key: str | None = None,
        status: str | None = None,
    ) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if persona_key is not None:
            clauses.append("persona_key = ?")
            params.append(persona_key)
        if status is not None:
            clauses.append("resolution_status = ?")
            params.append(status)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM memory_conflicts {where}",
                tuple(params),
            ).fetchone()
        return int(row["n"])

    def conflict_partners_for(
        self,
        memory_ids: Iterable[int],
        *,
        only_unresolved: bool = True,
    ) -> dict[int, list[tuple[int, float]]]:
        """For each id in ``memory_ids``, return its conflict partners.

        The partner is the OTHER side of the (a, b) pair. Items not in
        a conflict get no entry in the result (rather than an empty
        list) so callers can detect "no conflict" with a single ``in``
        check.
        """

        ids = sorted({int(value) for value in memory_ids})
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        clauses = [
            f"(memory_a_id IN ({placeholders}) OR memory_b_id IN ({placeholders}))"
        ]
        params: list[Any] = list(ids) + list(ids)
        if only_unresolved:
            clauses.append("resolution_status = 'unresolved'")
        where = " AND ".join(clauses)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT memory_a_id, memory_b_id, similarity
                FROM memory_conflicts
                WHERE {where}
                ORDER BY similarity DESC
                """,
                tuple(params),
            ).fetchall()
        partners: dict[int, list[tuple[int, float]]] = {}
        ids_set = set(ids)
        for row in rows:
            a = int(row["memory_a_id"])
            b = int(row["memory_b_id"])
            sim = float(row["similarity"])
            if a in ids_set:
                partners.setdefault(a, []).append((b, sim))
            if b in ids_set:
                partners.setdefault(b, []).append((a, sim))
        return partners

    def conflicts_for_memory(
        self,
        memory_id: int,
        *,
        only_unresolved: bool = True,
    ) -> list[MemoryConflict]:
        clauses = ["(memory_a_id = ? OR memory_b_id = ?)"]
        params: list[Any] = [int(memory_id), int(memory_id)]
        if only_unresolved:
            clauses.append("resolution_status = 'unresolved'")
        where = " AND ".join(clauses)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM memory_conflicts WHERE {where} ORDER BY detected_at DESC",
                tuple(params),
            ).fetchall()
        return [self._conflict_from_row(row) for row in rows]

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
    def _conflict_from_row(row: sqlite3.Row) -> MemoryConflict:
        return MemoryConflict(
            id=int(row["id"]),
            memory_a_id=int(row["memory_a_id"]),
            memory_b_id=int(row["memory_b_id"]),
            similarity=float(row["similarity"]),
            persona_key=row["persona_key"],
            detected_at=str(row["detected_at"]),
            resolution_status=str(row["resolution_status"]),
            resolution_winner_id=row["resolution_winner_id"],
            resolved_at=row["resolved_at"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _sticker_description_from_row(row: sqlite3.Row) -> StickerDescription:
        embedding_blob = row["embedding"] if "embedding" in row.keys() else None
        embedding = unpack_embedding(embedding_blob) if embedding_blob else None
        return StickerDescription(
            sticker_id=str(row["sticker_id"]),
            set_name=str(row["set_name"]),
            emoji=str(row["emoji"] or ""),
            description=str(row["description"] or ""),
            embedding=embedding,
            embedding_model=row["embedding_model"],
            failure_reason=row["failure_reason"],
            attempt_count=int(row["attempt_count"]),
            last_used_at=row["last_used_at"],
            created_at=str(row["created_at"]),
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _user_state_from_row(row: sqlite3.Row) -> UserState:
        return UserState(
            user_id=str(row["user_id"]),
            persona_key=str(row["persona_key"]),
            mood=str(row["mood"] or ""),
            themes=list(json.loads(row["themes"] or "[]")),
            open_questions=list(json.loads(row["open_questions"] or "[]")),
            preferences=dict(json.loads(row["preferences"] or "{}")),
            summary=str(row["summary"] or ""),
            confidence=float(row["confidence"]),
            last_updated_at=str(row["last_updated_at"]),
            messages_at_last_update=int(row["messages_at_last_update"]),
            metadata=dict(json.loads(row["metadata"] or "{}")),
        )

    @staticmethod
    def _goal_from_row(row: sqlite3.Row) -> Goal:
        return Goal(
            id=int(row["id"]),
            persona_key=str(row["persona_key"]),
            text=str(row["text"]),
            status=str(row["status"]),
            priority=float(row["priority"]),
            user_id=row["user_id"],
            chat_id=row["chat_id"],
            origin_message_id=row["origin_message_id"],
            due_at=row["due_at"],
            last_touched_at=str(row["last_touched_at"]),
            created_at=str(row["created_at"]),
            updated_at=row["updated_at"],
            closed_at=row["closed_at"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _memory_from_row(row: sqlite3.Row, tags: list[str]) -> MemoryItem:
        # Newer columns (``origin_message_id``, ``expires_at``) are read
        # defensively because legacy databases predate them. ``sqlite3.Row``
        # has ``.keys()`` so we probe rather than catching IndexError.
        keys = row.keys()
        origin = row["origin_message_id"] if "origin_message_id" in keys else None
        expires = row["expires_at"] if "expires_at" in keys else None
        return MemoryItem(
            id=int(row["id"]),
            kind=str(row["kind"]),
            text=str(row["text"]),
            scope=str(row["scope"]),
            user_id=row["user_id"],
            chat_id=row["chat_id"],
            persona_key=row["persona_key"],
            media_id=row["media_id"],
            importance=float(row["importance"]),
            confidence=float(row["confidence"]),
            source=row["source"],
            supersedes_id=row["supersedes_id"],
            superseded_by=row["superseded_by"],
            pinned=bool(row["pinned"]),
            created_at=str(row["created_at"]),
            updated_at=row["updated_at"],
            last_accessed_at=row["last_accessed_at"],
            access_count=int(row["access_count"]),
            tags=tags,
            metadata=json.loads(row["metadata"] or "{}"),
            origin_message_id=origin,
            expires_at=expires,
        )

    @staticmethod
    def _media_from_row(row: sqlite3.Row) -> MediaBlob:
        return MediaBlob(
            file_id=str(row["file_id"]),
            mime=str(row["mime"]),
            sha256=str(row["sha256"]),
            bytes=bytes(row["bytes"]),
            caption=str(row["caption"] or ""),
            created_at=str(row["created_at"]),
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
    def _ensure_column(
        conn: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")}
        if column in columns:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

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
