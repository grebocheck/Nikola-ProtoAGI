"""Tests for short-lived ("working memory") notes.

Memory items can carry an ``expires_at`` ISO timestamp. After expiry
they stop appearing in recall / list_memories results, and the next
reflection pass hard-deletes them. The Decision JSON now has a
``temporary_notes`` field the persona uses for ephemeral observations
("user sounds tired tonight") — those become memory rows with a 4h
default expiry.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from protoagi.config import AgentConfig
from protoagi.storage.memory import MemoryStore
from protoagi.storage.service import MemoryService
from protoagi.telegram import NikolaBot, TelegramConfig


def _iso_offset(hours: float) -> str:
    return (
        datetime.now(timezone.utc) + timedelta(hours=hours)
    ).isoformat(timespec="seconds")


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def get_me(self):
        return {"id": 1, "username": "WorkingMemBot"}

    def delete_webhook(self, *, drop_pending_updates=False):
        return True

    def get_updates(self, **kwargs):
        return []

    def send_chat_action(self, *args, **kwargs):
        return True

    def send_message(self, chat_id, text, *, reply_to_message_id=None, disable_notification=False):
        self.sent.append({"chat_id": str(chat_id), "text": text})
        return {"message_id": len(self.sent)}

    def send_sticker(self, *args, **kwargs):
        return {"message_id": 99}

    def get_sticker_set(self, name):
        return {"stickers": []}

    def get_file(self, file_id):
        return {"file_path": ""}

    def download_file(self, file_path, *, max_bytes):
        return b""


class FakeLLM:
    def __init__(self, content: str = '{"should_reply": false}') -> None:
        self.content = content

    def chat_completion(self, messages, **kwargs):
        return {"choices": [{"message": {"content": self.content}}]}


def _build_bot(memory: MemoryStore, llm: FakeLLM, db_path: Path) -> NikolaBot:
    bot = NikolaBot(
        telegram=FakeTelegram(),
        llm=llm,
        memory=memory,
        telegram_config=TelegramConfig(token="t", persona_key="solomiya"),
        agent_config=AgentConfig(database_path=db_path),
    )
    bot.bootstrap()
    return bot


class ExpiresAtStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name) / "memory.sqlite3")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_store_persists_expires_at(self) -> None:
        future = _iso_offset(4)
        mid = self.store.store_memory("temporary", expires_at=future)
        item = self.store.get_memory(mid)
        assert item is not None
        self.assertEqual(item.expires_at, future)

    def test_default_expires_at_is_none(self) -> None:
        mid = self.store.store_memory("permanent")
        item = self.store.get_memory(mid)
        assert item is not None
        self.assertIsNone(item.expires_at)

    def test_list_memories_hides_expired(self) -> None:
        live = self.store.store_memory("alive", expires_at=_iso_offset(1))
        expired = self.store.store_memory("dead", expires_at=_iso_offset(-1))
        permanent = self.store.store_memory("forever")
        listed_ids = {item.id for item in self.store.list_memories()}
        self.assertIn(live, listed_ids)
        self.assertIn(permanent, listed_ids)
        self.assertNotIn(expired, listed_ids)

    def test_fts_candidates_hide_expired(self) -> None:
        live = self.store.store_memory("чай ромашковий", expires_at=_iso_offset(1))
        expired = self.store.store_memory("чай зелений", expires_at=_iso_offset(-1))
        candidates = self.store.fts_candidates("чай")
        ids = {item.id for item in candidates}
        self.assertIn(live, ids)
        self.assertNotIn(expired, ids)

    def test_expire_working_memory_hard_deletes_expired(self) -> None:
        permanent = self.store.store_memory("keep me")
        soon = self.store.store_memory("dies", expires_at=_iso_offset(-1))
        deleted = self.store.expire_working_memory()
        self.assertEqual(deleted, 1)
        # Hard delete: cannot be fetched anymore.
        self.assertIsNone(self.store.get_memory(soon))
        self.assertIsNotNone(self.store.get_memory(permanent))

    def test_expire_skips_future_rows(self) -> None:
        future = self.store.store_memory("alive", expires_at=_iso_offset(2))
        deleted = self.store.expire_working_memory()
        self.assertEqual(deleted, 0)
        self.assertIsNotNone(self.store.get_memory(future))


class TemporaryNotesEndToEndTests(unittest.TestCase):
    def test_temporary_notes_persisted_with_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            llm = FakeLLM(
                json.dumps(
                    {
                        "should_reply": True,
                        "reply": "ок",
                        "temporary_notes": [
                            "людина зараз втомлена і трохи дратівлива",
                        ],
                    }
                )
            )
            bot = _build_bot(memory, llm, path)
            bot.process_update(
                {
                    "update_id": 1,
                    "message": {
                        "message_id": 5,
                        "chat": {"id": 555, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 555, "first_name": "Vadim"},
                        "text": "багато всього на роботі",
                    },
                }
            )
            items = memory.list_memories(limit=20)
            temp = [i for i in items if "втомлена" in i.text]
            self.assertEqual(len(temp), 1)
            self.assertIsNotNone(temp[0].expires_at)
            self.assertIn("working_memory", temp[0].tags)

    def test_empty_temporary_notes_creates_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            llm = FakeLLM(
                json.dumps({"should_reply": True, "reply": "ок", "temporary_notes": []})
            )
            bot = _build_bot(memory, llm, path)
            bot.process_update(
                {
                    "update_id": 1,
                    "message": {
                        "message_id": 5,
                        "chat": {"id": 555, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 555, "first_name": "Vadim"},
                        "text": "ок",
                    },
                }
            )
            items = memory.list_memories(limit=20)
            self.assertFalse(
                any("working_memory" in item.tags for item in items)
            )


if __name__ == "__main__":
    unittest.main()
