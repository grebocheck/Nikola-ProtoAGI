"""Test that user_state is created automatically once a user is engaged.

Without bootstrap, ``_run_user_state_refresh`` only refreshes rows that
already exist — meaning new users would never get a state model. The
bootstrap path watches ``count_user_messages`` and pre-allocates an
empty row (with a backdated ``last_updated_at``) so the next reflection
pass picks it up.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from protoagi.config import AgentConfig
from protoagi.storage.memory import MemoryStore
from protoagi.telegram import NikolaBot, TelegramConfig


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def get_me(self):
        return {"id": 1, "username": "BootstrapBot"}

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


def _send(bot: NikolaBot, chat_id: int, message_id: int, text: str) -> None:
    bot.process_update(
        {
            "update_id": message_id,
            "message": {
                "message_id": message_id,
                "chat": {"id": chat_id, "type": "private", "first_name": "Vadim"},
                "from": {"id": chat_id, "first_name": "Vadim"},
                "text": text,
            },
        }
    )


class UserStateBootstrapTests(unittest.TestCase):
    def test_no_state_after_first_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            bot = _build_bot(memory, FakeLLM(), path)
            _send(bot, 555, 1, "першa")
            state = memory.get_user_state("telegram:555", "solomiya")
            self.assertIsNone(state)

    def test_state_created_at_third_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            bot = _build_bot(memory, FakeLLM(), path)
            _send(bot, 555, 1, "першa")
            _send(bot, 555, 2, "друга")
            self.assertIsNone(memory.get_user_state("telegram:555", "solomiya"))
            _send(bot, 555, 3, "третя — тут має зʼявитись")
            state = memory.get_user_state("telegram:555", "solomiya")
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.summary, "")
            self.assertEqual(state.confidence, 0.0)
            self.assertTrue(state.metadata.get("bootstrap"))

    def test_bootstrap_does_not_overwrite_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            # Seed a real, non-empty state.
            memory.upsert_user_state(
                user_id="telegram:555",
                persona_key="solomiya",
                summary="вже відомий контекст",
                confidence=0.7,
            )
            bot = _build_bot(memory, FakeLLM(), path)
            _send(bot, 555, 1, "a")
            _send(bot, 555, 2, "b")
            _send(bot, 555, 3, "c")
            state = memory.get_user_state("telegram:555", "solomiya")
            assert state is not None
            self.assertEqual(state.summary, "вже відомий контекст")
            self.assertEqual(state.confidence, 0.7)

    def test_bootstrap_backdates_for_reflection_pickup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            bot = _build_bot(memory, FakeLLM(), path)
            for i in range(1, 4):
                _send(bot, 555, i, f"msg {i}")
            state = memory.get_user_state("telegram:555", "solomiya")
            assert state is not None
            # ``last_updated_at`` should be backdated > 24h so the next
            # reflection pass finds it via stale_user_states.
            updated = datetime.fromisoformat(state.last_updated_at)
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            self.assertGreater((now - updated).total_seconds(), 24 * 3600)

    def test_stale_pickup_returns_bootstrap_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            bot = _build_bot(memory, FakeLLM(), path)
            for i in range(1, 4):
                _send(bot, 555, i, f"msg {i}")
            cutoff = (
                datetime.now(timezone.utc) - timedelta(hours=24)
            ).isoformat(timespec="seconds")
            stale = memory.stale_user_states(persona_key="solomiya", older_than=cutoff)
            self.assertEqual([s.user_id for s in stale], ["telegram:555"])


if __name__ == "__main__":
    unittest.main()
