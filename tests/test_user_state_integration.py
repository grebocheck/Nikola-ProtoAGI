"""End-to-end tests for the user_state pipeline.

Covers: refresh_user_state() actually upserts a row from an LLM
response; the reflection pass picks up stale rows; decide_incoming
injects ``known_user_state`` into the prompt payload; cross-persona
isolation is preserved.
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
        return {"id": 1, "username": "StateBot"}

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


class QueuedLLM:
    """Returns each queued content in order, looping the last value."""

    def __init__(self, contents: list[str]) -> None:
        if not contents:
            raise ValueError("QueuedLLM requires at least one content")
        self.contents = list(contents)
        self.calls: list[list[dict]] = []

    def chat_completion(self, messages, **kwargs):
        self.calls.append(list(messages))
        content = self.contents.pop(0) if len(self.contents) > 1 else self.contents[0]
        return {"choices": [{"message": {"content": content}}]}


def _build_bot(memory: MemoryStore, llm: QueuedLLM, db_path: Path, *, persona: str = "solomiya") -> NikolaBot:
    bot = NikolaBot(
        telegram=FakeTelegram(),
        llm=llm,
        memory=memory,
        telegram_config=TelegramConfig(token="t", persona_key=persona),
        agent_config=AgentConfig(database_path=db_path),
    )
    bot.bootstrap()
    return bot


def _seed_user_messages(memory: MemoryStore, chat_id: str, user_id: str, texts: list[str]) -> None:
    for idx, text in enumerate(texts, start=1):
        memory.log_telegram_message(
            chat_id=chat_id,
            message_id=idx,
            persona_key="solomiya",
            role="user",
            sender_id=user_id,
            sender_name="Vadim",
            text=text,
        )


class RefreshUserStateTests(unittest.TestCase):
    def test_refresh_writes_state_from_llm_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            _seed_user_messages(
                memory, "555", "u1",
                ["сьогодні було важко", "не висипаюсь", "хочеться спокою"],
            )
            llm = QueuedLLM(
                [
                    json.dumps(
                        {
                            "mood": "втомлена",
                            "themes": ["сон", "відновлення"],
                            "open_questions": ["як виспатись"],
                            "preferences": {"tone": "мʼяка"},
                            "summary": "Людина виснажена, шукає спокою і сну.",
                            "confidence": 0.7,
                        }
                    )
                ]
            )
            bot = _build_bot(memory, llm, path)
            state = bot.refresh_user_state("u1")
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.mood, "втомлена")
            self.assertEqual(state.themes, ["сон", "відновлення"])
            self.assertEqual(state.open_questions, ["як виспатись"])
            self.assertEqual(state.preferences, {"tone": "мʼяка"})
            self.assertEqual(state.confidence, 0.7)
            self.assertEqual(state.messages_at_last_update, 3)

    def test_refresh_returns_none_when_no_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            llm = QueuedLLM([json.dumps({"summary": "should not be called"})])
            bot = _build_bot(memory, llm, path)
            result = bot.refresh_user_state("u_unknown")
            self.assertIsNone(result)
            # No LLM call should be made when there's nothing to summarize.
            self.assertEqual(llm.calls, [])

    def test_empty_summary_keeps_previous_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            _seed_user_messages(memory, "555", "u1", ["привіт"])
            memory.upsert_user_state(
                user_id="u1",
                persona_key="solomiya",
                summary="попередній знімок",
                mood="спокійна",
            )
            llm = QueuedLLM([json.dumps({"summary": ""})])
            bot = _build_bot(memory, llm, path)
            result = bot.refresh_user_state("u1")
            assert result is not None
            # Should still be the previous version, not overwritten by empty.
            self.assertEqual(result.summary, "попередній знімок")
            self.assertEqual(result.mood, "спокійна")

    def test_state_is_persona_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            _seed_user_messages(memory, "555", "u1", ["one", "two"])
            llm = QueuedLLM(
                [
                    json.dumps({"summary": "Solomiya's view of u1", "confidence": 0.6}),
                    json.dumps({"summary": "Mykola's view of u1", "confidence": 0.4}),
                ]
            )
            bot_solo = _build_bot(memory, llm, path, persona="solomiya")
            bot_solo.refresh_user_state("u1")
            bot_myk = _build_bot(memory, llm, path, persona="mykola")
            bot_myk.refresh_user_state("u1")
            solo = memory.get_user_state("u1", "solomiya")
            myk = memory.get_user_state("u1", "mykola")
            assert solo is not None and myk is not None
            self.assertEqual(solo.summary, "Solomiya's view of u1")
            self.assertEqual(myk.summary, "Mykola's view of u1")


class ReflectionUserStateTests(unittest.TestCase):
    def test_reflection_pass_refreshes_stale_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            _seed_user_messages(memory, "555", "u1", ["a", "b", "c"])
            # Seed a stale row.
            memory.upsert_user_state(
                user_id="u1", persona_key="solomiya", summary="старий знімок"
            )
            old_ts = (
                datetime.now(timezone.utc) - timedelta(hours=48)
            ).isoformat(timespec="seconds")
            with memory.connect() as conn:
                conn.execute(
                    "UPDATE user_state SET last_updated_at = ? WHERE user_id = 'u1'",
                    (old_ts,),
                )
            # fictional_self_enabled defaults to False, so the reflection
            # pass skips the self-memory LLM call. Only refresh_user_state
            # actually fires here.
            llm = QueuedLLM(
                [json.dumps({"summary": "оновлений знімок", "confidence": 0.6})]
            )
            bot = _build_bot(memory, llm, path)
            result = bot.run_reflection_pass()
            self.assertGreaterEqual(result.get("user_states_refreshed", 0), 1)
            refreshed = memory.get_user_state("u1", "solomiya")
            assert refreshed is not None
            self.assertEqual(refreshed.summary, "оновлений знімок")


class DecideIncomingUserStateTests(unittest.TestCase):
    def test_known_user_state_appears_in_context_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            # The bot prefixes Telegram user ids with "telegram:" before
            # passing them to decide_incoming, so the row key must match.
            memory.upsert_user_state(
                user_id="telegram:555",
                persona_key="solomiya",
                mood="втомлена",
                themes=["сон"],
                open_questions=["як виспатись"],
                summary="бачу що ти зараз втомлений",
                confidence=0.7,
            )
            llm = QueuedLLM([json.dumps({"should_reply": True, "reply": "ок"})])
            bot = _build_bot(memory, llm, path)
            bot.process_update(
                {
                    "update_id": 1,
                    "message": {
                        "message_id": 1,
                        "chat": {"id": 555, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 555, "first_name": "Vadim"},
                        "text": "привіт",
                    },
                }
            )
            self.assertTrue(llm.calls)
            user_payload = json.loads(llm.calls[0][1]["content"])
            self.assertIn("known_user_state", user_payload)
            state_view = user_payload["known_user_state"]
            self.assertIsNotNone(state_view)
            self.assertEqual(state_view["mood"], "втомлена")
            self.assertEqual(state_view["themes"], ["сон"])
            self.assertEqual(state_view["summary"], "бачу що ти зараз втомлений")
            self.assertIn("age_hours", state_view)

    def test_known_user_state_is_none_when_no_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            llm = QueuedLLM([json.dumps({"should_reply": False})])
            bot = _build_bot(memory, llm, path)
            bot.process_update(
                {
                    "update_id": 1,
                    "message": {
                        "message_id": 1,
                        "chat": {"id": 999, "type": "private", "first_name": "X"},
                        "from": {"id": 999, "first_name": "X"},
                        "text": "привіт",
                    },
                }
            )
            user_payload = json.loads(llm.calls[0][1]["content"])
            self.assertIn("known_user_state", user_payload)
            self.assertIsNone(user_payload["known_user_state"])


if __name__ == "__main__":
    unittest.main()
