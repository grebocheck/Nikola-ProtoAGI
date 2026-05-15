"""Tests for LLM-driven conflict resolution.

The detect-only phase (B2) records pairs in ``memory_conflicts`` but
leaves them ``unresolved`` forever. This phase closes that loop: the
reflection pass picks up unresolved conflicts and asks the persona to
adjudicate — supersede / keep-both / dismiss — with an honest
confidence signal. Low-confidence verdicts are deferred to the next
pass; only confident ones apply changes.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from protoagi.config import AgentConfig
from protoagi.storage.memory import MemoryStore
from protoagi.storage.models import (
    CONFLICT_STATUS_DISMISSED,
    CONFLICT_STATUS_KEPT_BOTH,
    CONFLICT_STATUS_SUPERSEDED,
    CONFLICT_STATUS_UNRESOLVED,
)
from protoagi.telegram import NikolaBot, TelegramConfig


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def get_me(self):
        return {"id": 1, "username": "ResolverBot"}

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
    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls: list[list[dict]] = []

    def chat_completion(self, messages, **kwargs):
        self.calls.append(list(messages))
        content = self.contents.pop(0) if len(self.contents) > 1 else (
            self.contents[0] if self.contents else "{}"
        )
        return {"choices": [{"message": {"content": content}}]}


def _build_bot(memory: MemoryStore, llm: QueuedLLM, db_path: Path) -> NikolaBot:
    bot = NikolaBot(
        telegram=FakeTelegram(),
        llm=llm,
        memory=memory,
        telegram_config=TelegramConfig(token="t", persona_key="solomiya"),
        agent_config=AgentConfig(database_path=db_path),
    )
    bot.bootstrap()
    return bot


def _seed_pair(memory: MemoryStore) -> tuple[int, int, int]:
    a = memory.store_memory("людина любить ромашковий чай", persona_key="solomiya")
    b = memory.store_memory("людина перейшла на каву зранку", persona_key="solomiya")
    cid = memory.record_conflict(a, b, similarity=0.85, persona_key="solomiya")
    assert cid is not None
    return a, b, cid


class TryResolveConflictTests(unittest.TestCase):
    def test_superseded_verdict_applies_supersession(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            a, b, cid = _seed_pair(memory)
            llm = QueuedLLM(
                [json.dumps(
                    {"verdict": "superseded", "winner_id": b, "confidence": 0.85}
                )]
            )
            bot = _build_bot(memory, llm, path)
            verdict = bot.try_resolve_conflict(cid)
            self.assertIsNotNone(verdict)
            assert verdict is not None
            self.assertEqual(verdict.verdict, "superseded")
            # Loser is now superseded; winner kept active.
            item_a = memory.get_memory(a)
            item_b = memory.get_memory(b)
            assert item_a is not None and item_b is not None
            self.assertEqual(item_a.superseded_by, b)
            self.assertIsNone(item_b.superseded_by)
            conflict = memory.get_conflict(cid)
            assert conflict is not None
            self.assertEqual(conflict.resolution_status, CONFLICT_STATUS_SUPERSEDED)
            self.assertEqual(conflict.resolution_winner_id, b)

    def test_kept_both_marks_resolved_no_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            a, b, cid = _seed_pair(memory)
            llm = QueuedLLM(
                [json.dumps({"verdict": "kept_both", "confidence": 0.75})]
            )
            bot = _build_bot(memory, llm, path)
            bot.try_resolve_conflict(cid)
            conflict = memory.get_conflict(cid)
            assert conflict is not None
            self.assertEqual(conflict.resolution_status, CONFLICT_STATUS_KEPT_BOTH)
            # Neither side superseded.
            self.assertIsNone(memory.get_memory(a).superseded_by)
            self.assertIsNone(memory.get_memory(b).superseded_by)

    def test_dismissed_marks_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            a, b, cid = _seed_pair(memory)
            llm = QueuedLLM(
                [json.dumps({"verdict": "dismissed", "confidence": 0.7})]
            )
            bot = _build_bot(memory, llm, path)
            bot.try_resolve_conflict(cid)
            conflict = memory.get_conflict(cid)
            assert conflict is not None
            self.assertEqual(conflict.resolution_status, CONFLICT_STATUS_DISMISSED)

    def test_low_confidence_leaves_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            a, b, cid = _seed_pair(memory)
            llm = QueuedLLM(
                [json.dumps(
                    {"verdict": "superseded", "winner_id": b, "confidence": 0.4}
                )]
            )
            bot = _build_bot(memory, llm, path)
            verdict = bot.try_resolve_conflict(cid)
            self.assertIsNotNone(verdict)
            conflict = memory.get_conflict(cid)
            assert conflict is not None
            self.assertEqual(conflict.resolution_status, CONFLICT_STATUS_UNRESOLVED)
            # Last attempt is recorded so we can see persona was undecided.
            self.assertIn("last_verdict", conflict.metadata)
            self.assertEqual(conflict.metadata["last_verdict"], "superseded")
            # Loser not actually superseded.
            self.assertIsNone(memory.get_memory(a).superseded_by)

    def test_invalid_winner_id_leaves_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            a, b, cid = _seed_pair(memory)
            llm = QueuedLLM(
                [json.dumps(
                    {"verdict": "superseded", "winner_id": 99999, "confidence": 0.9}
                )]
            )
            bot = _build_bot(memory, llm, path)
            bot.try_resolve_conflict(cid)
            conflict = memory.get_conflict(cid)
            assert conflict is not None
            self.assertEqual(conflict.resolution_status, CONFLICT_STATUS_UNRESOLVED)
            self.assertIn("last_reasoning", conflict.metadata)

    def test_stale_pair_auto_dismissed_when_side_superseded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            a, b, cid = _seed_pair(memory)
            # External supersession before resolver runs.
            memory.supersede(a, b)
            llm = QueuedLLM([])  # should NOT be called
            bot = _build_bot(memory, llm, path)
            result = bot.try_resolve_conflict(cid)
            self.assertIsNone(result)
            self.assertEqual(llm.calls, [])
            conflict = memory.get_conflict(cid)
            assert conflict is not None
            self.assertEqual(conflict.resolution_status, CONFLICT_STATUS_DISMISSED)

    def test_already_resolved_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            a, b, cid = _seed_pair(memory)
            memory.resolve_conflict(cid, status=CONFLICT_STATUS_KEPT_BOTH)
            llm = QueuedLLM([])
            bot = _build_bot(memory, llm, path)
            self.assertIsNone(bot.try_resolve_conflict(cid))
            self.assertEqual(llm.calls, [])


class ResolutionInReflectionPassTests(unittest.TestCase):
    def test_reflection_resolves_unresolved_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            _, b, cid = _seed_pair(memory)
            llm = QueuedLLM(
                [json.dumps(
                    {"verdict": "superseded", "winner_id": b, "confidence": 0.8}
                )]
            )
            bot = _build_bot(memory, llm, path)
            result = bot.run_reflection_pass()
            self.assertGreaterEqual(result.get("conflicts_resolved", 0), 1)
            conflict = memory.get_conflict(cid)
            assert conflict is not None
            self.assertEqual(conflict.resolution_status, CONFLICT_STATUS_SUPERSEDED)


if __name__ == "__main__":
    unittest.main()
