"""End-to-end tests for the decision/initiative `goals` field.

These tests exercise the Phase 2/3 wiring: the model JSON returns
``goals: [...]``, the orchestrator persists those actions into the
``goals`` table, and the next-turn context payload exposes ``open_goals``
back to the model. The storage layer itself is covered in test_goals.py;
this file focuses on the integration boundary.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from protoagi.config import AgentConfig
from protoagi.storage.memory import MemoryStore
from protoagi.storage.models import (
    GOAL_STATUS_ABANDONED,
    GOAL_STATUS_COMPLETED,
    GOAL_STATUS_OPEN,
)
from protoagi.telegram import (
    NikolaBot,
    TelegramConfig,
    normalize_goal_actions,
)


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def get_me(self):
        return {"id": 1, "username": "GoalsBot"}

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
    """Returns a queued sequence of contents, falling back to the last entry."""

    def __init__(self, contents: list[str]) -> None:
        if not contents:
            raise ValueError("FakeLLM needs at least one content")
        self.contents = list(contents)
        self.calls: list[list[dict]] = []

    def chat_completion(self, messages, **kwargs):
        self.calls.append(list(messages))
        if len(self.contents) > 1:
            content = self.contents.pop(0)
        else:
            content = self.contents[0]
        return {"choices": [{"message": {"content": content}}]}


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


class GoalActionNormalizerTests(unittest.TestCase):
    def test_open_requires_text(self) -> None:
        self.assertEqual(normalize_goal_actions([{"action": "open"}]), [])
        actions = normalize_goal_actions([{"action": "open", "text": "  test  "}])
        self.assertEqual(actions, [{"action": "open", "text": "test"}])

    def test_complete_and_abandon_require_goal_id(self) -> None:
        self.assertEqual(normalize_goal_actions([{"action": "complete"}]), [])
        self.assertEqual(
            normalize_goal_actions([{"action": "complete", "goal_id": 7}]),
            [{"action": "complete", "goal_id": 7}],
        )

    def test_update_requires_goal_id(self) -> None:
        self.assertEqual(
            normalize_goal_actions([{"action": "update", "text": "x"}]), []
        )
        self.assertEqual(
            normalize_goal_actions(
                [{"action": "update", "goal_id": 4, "priority": 1.5}]
            ),
            [{"action": "update", "goal_id": 4, "priority": 1.0}],
        )

    def test_unknown_action_dropped(self) -> None:
        self.assertEqual(
            normalize_goal_actions(
                [{"action": "celebrate", "text": "x"}, {"action": "open", "text": "y"}]
            ),
            [{"action": "open", "text": "y"}],
        )

    def test_cap_at_five(self) -> None:
        items = [{"action": "open", "text": f"g{i}"} for i in range(10)]
        self.assertEqual(len(normalize_goal_actions(items)), 5)


class DecisionGoalEndToEndTests(unittest.TestCase):
    def test_open_action_creates_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            llm = FakeLLM(
                [
                    json.dumps(
                        {
                            "should_reply": True,
                            "reply": "ok",
                            "goals": [
                                {
                                    "action": "open",
                                    "text": "повернутись до теми про чай",
                                    "priority": 0.7,
                                }
                            ],
                        }
                    )
                ]
            )
            bot = _build_bot(memory, llm, path)
            _send(bot, 555, 10, "поговоримо ще завтра")
            goals = memory.list_open_goals(persona_key="solomiya")
            self.assertEqual(len(goals), 1)
            self.assertEqual(goals[0].text, "повернутись до теми про чай")
            self.assertEqual(goals[0].priority, 0.7)
            self.assertEqual(goals[0].chat_id, "555")
            self.assertEqual(goals[0].origin_message_id, 10)
            self.assertEqual(goals[0].status, GOAL_STATUS_OPEN)

    def test_complete_action_closes_existing_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            gid = memory.open_goal(
                persona_key="solomiya", text="допити чай", chat_id="555"
            )
            llm = FakeLLM(
                [
                    json.dumps(
                        {
                            "should_reply": True,
                            "reply": "ок",
                            "goals": [{"action": "complete", "goal_id": gid}],
                        }
                    )
                ]
            )
            bot = _build_bot(memory, llm, path)
            _send(bot, 555, 11, "так і зробила")
            updated = memory.get_goal(gid)
            assert updated is not None
            self.assertEqual(updated.status, GOAL_STATUS_COMPLETED)
            self.assertIsNotNone(updated.closed_at)

    def test_abandon_action_marks_abandoned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            gid = memory.open_goal(
                persona_key="solomiya", text="зайти в спортзал", chat_id="555"
            )
            llm = FakeLLM(
                [
                    json.dumps(
                        {
                            "should_reply": True,
                            "reply": "ок",
                            "goals": [{"action": "abandon", "goal_id": gid}],
                        }
                    )
                ]
            )
            bot = _build_bot(memory, llm, path)
            _send(bot, 555, 12, "забили")
            updated = memory.get_goal(gid)
            assert updated is not None
            self.assertEqual(updated.status, GOAL_STATUS_ABANDONED)

    def test_update_action_revises_text_and_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            gid = memory.open_goal(
                persona_key="solomiya", text="старий текст", chat_id="555"
            )
            llm = FakeLLM(
                [
                    json.dumps(
                        {
                            "should_reply": True,
                            "reply": "ок",
                            "goals": [
                                {
                                    "action": "update",
                                    "goal_id": gid,
                                    "text": "новий текст",
                                    "priority": 0.9,
                                }
                            ],
                        }
                    )
                ]
            )
            bot = _build_bot(memory, llm, path)
            _send(bot, 555, 13, "уточнюю")
            updated = memory.get_goal(gid)
            assert updated is not None
            self.assertEqual(updated.text, "новий текст")
            self.assertEqual(updated.priority, 0.9)

    def test_unknown_goal_id_silently_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            llm = FakeLLM(
                [
                    json.dumps(
                        {
                            "should_reply": True,
                            "reply": "ок",
                            "goals": [{"action": "complete", "goal_id": 9999}],
                        }
                    )
                ]
            )
            bot = _build_bot(memory, llm, path)
            _send(bot, 555, 14, "ок")
            # No goals created, no crash.
            self.assertEqual(memory.count_goals(), 0)

    def test_other_persona_goal_not_closable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            gid = memory.open_goal(persona_key="mykola", text="чужа ціль")
            llm = FakeLLM(
                [
                    json.dumps(
                        {
                            "should_reply": True,
                            "reply": "ок",
                            "goals": [{"action": "complete", "goal_id": gid}],
                        }
                    )
                ]
            )
            bot = _build_bot(memory, llm, path)
            _send(bot, 555, 15, "ок")
            # solomiya cannot close mykola's goal.
            other = memory.get_goal(gid)
            assert other is not None
            self.assertEqual(other.status, GOAL_STATUS_OPEN)

    def test_open_goals_injected_into_context_on_next_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            memory.open_goal(
                persona_key="solomiya",
                text="повернутись до теми про чай",
                chat_id="555",
                priority=0.7,
            )
            llm = FakeLLM(
                [json.dumps({"should_reply": True, "reply": "ок"})]
            )
            bot = _build_bot(memory, llm, path)
            _send(bot, 555, 16, "привіт")
            self.assertTrue(llm.calls, "expected at least one LLM invocation")
            user_payload = json.loads(llm.calls[0][1]["content"])
            self.assertIn("open_goals", user_payload)
            self.assertEqual(len(user_payload["open_goals"]), 1)
            goal_view = user_payload["open_goals"][0]
            self.assertEqual(goal_view["text"], "повернутись до теми про чай")
            self.assertEqual(goal_view["priority"], 0.7)
            self.assertIn("age_days", goal_view)


class InitiativeGoalEndToEndTests(unittest.TestCase):
    def test_initiative_payload_includes_open_and_due_goals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            memory.open_goal(
                persona_key="solomiya",
                text="дописати листа",
                chat_id="777",
                priority=0.6,
            )
            due_at = (
                datetime.now(timezone.utc) + timedelta(hours=2)
            ).isoformat(timespec="seconds")
            memory.open_goal(
                persona_key="solomiya",
                text="зідзвонитись з мамою",
                chat_id="777",
                priority=0.8,
                due_at=due_at,
            )
            memory.upsert_telegram_chat(
                {"id": 777, "type": "private", "first_name": "Vadim"},
                {"id": 777, "first_name": "Vadim"},
            )
            chat = memory.get_telegram_chat(777)
            assert chat is not None
            llm = FakeLLM(
                [json.dumps({"send": False, "message": "", "next_check_minutes": 300})]
            )
            bot = _build_bot(memory, llm, path)
            bot.decide_initiative(chat)
            self.assertTrue(llm.calls)
            user_payload = json.loads(llm.calls[0][1]["content"])
            self.assertIn("open_goals", user_payload)
            self.assertIn("due_goals", user_payload)
            self.assertEqual(len(user_payload["open_goals"]), 2)
            # Only the goal with due_at should appear in due_goals.
            self.assertEqual(len(user_payload["due_goals"]), 1)
            self.assertEqual(
                user_payload["due_goals"][0]["text"], "зідзвонитись з мамою"
            )


if __name__ == "__main__":
    unittest.main()
