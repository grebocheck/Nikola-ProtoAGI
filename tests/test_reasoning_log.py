import json
import tempfile
import unittest
from pathlib import Path

from protoagi.storage.memory import MemoryStore
from protoagi.telegram.reasoning_log import (
    REASONING_KV_PREFIX,
    ReasoningLog,
    ReasoningLogConfig,
    extract_reasoning_text,
)


def _temp_memory() -> tuple[MemoryStore, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory()
    store = MemoryStore(Path(tmp.name) / "memory.sqlite3")
    return store, tmp


class ExtractReasoningTextTests(unittest.TestCase):
    def test_prefers_reasoning_content_field(self) -> None:
        message = {"reasoning_content": "I think A then B", "content": "{}"}
        self.assertEqual(extract_reasoning_text(message), "I think A then B")

    def test_falls_back_to_think_tags_in_content(self) -> None:
        message = {"content": "<think>step one\nstep two</think>final"}
        self.assertEqual(extract_reasoning_text(message), "step one\nstep two")

    def test_extracts_harmony_analysis_channel(self) -> None:
        # Production gpt-oss output: skip-chat-parsing leaks Harmony tokens
        # straight into content. The reasoning lives in the analysis channel.
        raw = (
            "<|channel|>analysis<|message|>The user said hi. Reply briefly."
            "<|end|><|start|>assistant<|channel|>final<|message|>привіт"
        )
        self.assertIn(
            "user said hi",
            extract_reasoning_text({"content": raw}),
        )

    def test_returns_empty_for_missing_payload(self) -> None:
        self.assertEqual(extract_reasoning_text({}), "")
        self.assertEqual(extract_reasoning_text({"content": "no thinking here"}), "")

    def test_returns_empty_when_message_is_not_dict(self) -> None:
        self.assertEqual(extract_reasoning_text(None), "")  # type: ignore[arg-type]
        self.assertEqual(extract_reasoning_text("string"), "")  # type: ignore[arg-type]


class ReasoningLogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory, self._tmp = _temp_memory()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_disabled_log_does_not_persist(self) -> None:
        log = ReasoningLog(self.memory, ReasoningLogConfig(enabled=False))
        log.record(
            chat_id="42",
            message_id=1,
            captured_at="2026-05-07T10:00:00+00:00",
            decision_kind="decision",
            incoming_text="hello",
            reasoning="thinking...",
            reply_excerpt="hi",
        )
        self.assertEqual(log.list_for_chat("42"), [])

    def test_enabled_log_records_entry(self) -> None:
        log = ReasoningLog(self.memory, ReasoningLogConfig(enabled=True))
        log.record(
            chat_id="42",
            message_id=1,
            captured_at="2026-05-07T10:00:00+00:00",
            decision_kind="decision",
            incoming_text="hello",
            reasoning="step one then two",
            reply_excerpt="hi",
        )
        entries = log.list_for_chat("42")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["reasoning"], "step one then two")
        self.assertEqual(entries[0]["chat_id"], "42")
        self.assertEqual(entries[0]["message_id"], 1)

    def test_empty_reasoning_is_skipped(self) -> None:
        log = ReasoningLog(self.memory, ReasoningLogConfig(enabled=True))
        log.record(
            chat_id="42",
            message_id=1,
            captured_at="2026-05-07T10:00:00+00:00",
            decision_kind="decision",
            incoming_text="hello",
            reasoning="   ",
            reply_excerpt="hi",
        )
        self.assertEqual(log.list_for_chat("42"), [])

    def test_ring_buffer_drops_oldest(self) -> None:
        log = ReasoningLog(self.memory, ReasoningLogConfig(enabled=True, max_entries_per_chat=3))
        for index in range(5):
            log.record(
                chat_id="42",
                message_id=index,
                captured_at=f"2026-05-07T10:00:0{index}+00:00",
                decision_kind="decision",
                incoming_text=f"in{index}",
                reasoning=f"thought-{index}",
                reply_excerpt="",
            )
        entries = log.list_for_chat("42")
        self.assertEqual(len(entries), 3)
        self.assertEqual([entry["message_id"] for entry in entries], [2, 3, 4])

    def test_long_text_is_clipped(self) -> None:
        log = ReasoningLog(
            self.memory,
            ReasoningLogConfig(enabled=True, max_chars_per_entry=20),
        )
        big = "X" * 1000
        log.record(
            chat_id="9",
            message_id=1,
            captured_at="2026-05-07T10:00:00+00:00",
            decision_kind="decision",
            incoming_text=big,
            reasoning=big,
            reply_excerpt=big,
        )
        entry = log.list_for_chat("9")[0]
        self.assertLessEqual(len(entry["reasoning"]), 20)
        self.assertTrue(entry["reasoning"].endswith("…"))

    def test_list_chats_returns_overview(self) -> None:
        log = ReasoningLog(self.memory, ReasoningLogConfig(enabled=True))
        log.record(
            chat_id="11",
            message_id=1,
            captured_at="2026-05-07T10:00:00+00:00",
            decision_kind="decision",
            incoming_text="hello",
            reasoning="thinking",
            reply_excerpt="hi",
        )
        log.record(
            chat_id="22",
            message_id=1,
            captured_at="2026-05-07T11:00:00+00:00",
            decision_kind="decision",
            incoming_text="привіт",
            reasoning="думаю",
            reply_excerpt="ага",
        )
        rows = log.list_chats()
        ids = {row["chat_id"] for row in rows}
        self.assertEqual(ids, {"11", "22"})

    def test_kv_key_prefix(self) -> None:
        log = ReasoningLog(self.memory, ReasoningLogConfig(enabled=True))
        log.record(
            chat_id="42",
            message_id=1,
            captured_at="2026-05-07T10:00:00+00:00",
            decision_kind="decision",
            incoming_text="hello",
            reasoning="thinking",
            reply_excerpt="hi",
        )
        raw = self.memory.get_kv(REASONING_KV_PREFIX + "42")
        self.assertIsNotNone(raw)
        payload = json.loads(raw)
        self.assertEqual(len(payload["entries"]), 1)


if __name__ == "__main__":
    unittest.main()
