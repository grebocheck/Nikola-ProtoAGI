"""Storage-layer tests for user_state.

user_state is a small per-(user × persona) blob that captures the
persona's current working model of a Telegram user: mood, themes,
open questions, preferences, and a free-form summary. Each row is
upserted by an LLM-driven refresh routine; this file just exercises
the SQL primitives.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from protoagi.storage.memory import MemoryStore


def _iso_offset(hours: float) -> str:
    return (
        datetime.now(timezone.utc) + timedelta(hours=hours)
    ).isoformat(timespec="seconds")


class UserStateStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name) / "memory.sqlite3")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_get_returns_none_for_missing(self) -> None:
        self.assertIsNone(self.store.get_user_state("u1", "solomiya"))

    def test_upsert_inserts_and_reads_back(self) -> None:
        state = self.store.upsert_user_state(
            user_id="u1",
            persona_key="solomiya",
            mood="спокійна",
            themes=["робота", "сон"],
            open_questions=["чи їхати у відпустку"],
            preferences={"tone": "тепла"},
            summary="Людина працює над великим проектом, втомлюється, думає про відпочинок.",
            confidence=0.7,
            messages_at_last_update=12,
        )
        self.assertEqual(state.user_id, "u1")
        self.assertEqual(state.persona_key, "solomiya")
        self.assertEqual(state.mood, "спокійна")
        self.assertEqual(state.themes, ["робота", "сон"])
        self.assertEqual(state.open_questions, ["чи їхати у відпустку"])
        self.assertEqual(state.preferences, {"tone": "тепла"})
        self.assertEqual(state.confidence, 0.7)
        self.assertEqual(state.messages_at_last_update, 12)

        roundtrip = self.store.get_user_state("u1", "solomiya")
        assert roundtrip is not None
        self.assertEqual(roundtrip.themes, ["робота", "сон"])
        self.assertEqual(roundtrip.summary, state.summary)

    def test_upsert_replaces_existing(self) -> None:
        self.store.upsert_user_state(
            user_id="u1",
            persona_key="solomiya",
            mood="тривожна",
            summary="перша версія",
        )
        self.store.upsert_user_state(
            user_id="u1",
            persona_key="solomiya",
            mood="піднесена",
            summary="друга версія",
        )
        state = self.store.get_user_state("u1", "solomiya")
        assert state is not None
        self.assertEqual(state.mood, "піднесена")
        self.assertEqual(state.summary, "друга версія")

    def test_state_is_isolated_per_persona(self) -> None:
        self.store.upsert_user_state(
            user_id="u1", persona_key="solomiya", summary="warm view"
        )
        self.store.upsert_user_state(
            user_id="u1", persona_key="mykola", summary="practical view"
        )
        solo = self.store.get_user_state("u1", "solomiya")
        myk = self.store.get_user_state("u1", "mykola")
        assert solo is not None and myk is not None
        self.assertEqual(solo.summary, "warm view")
        self.assertEqual(myk.summary, "practical view")

    def test_confidence_clamped(self) -> None:
        s_high = self.store.upsert_user_state(
            user_id="u1", persona_key="p", confidence=5.0
        )
        s_low = self.store.upsert_user_state(
            user_id="u2", persona_key="p", confidence=-0.4
        )
        self.assertEqual(s_high.confidence, 1.0)
        self.assertEqual(s_low.confidence, 0.0)

    def test_themes_and_questions_capped(self) -> None:
        many_themes = [f"t{i}" for i in range(20)]
        many_questions = [f"q{i}" for i in range(20)]
        state = self.store.upsert_user_state(
            user_id="u1",
            persona_key="p",
            themes=many_themes,
            open_questions=many_questions,
        )
        self.assertEqual(len(state.themes), 10)
        self.assertEqual(len(state.open_questions), 10)

    def test_empty_strings_in_themes_are_dropped(self) -> None:
        state = self.store.upsert_user_state(
            user_id="u1",
            persona_key="p",
            themes=["робота", "   ", "", "сон"],
        )
        self.assertEqual(state.themes, ["робота", "сон"])

    def test_rejects_empty_user_or_persona(self) -> None:
        with self.assertRaises(ValueError):
            self.store.upsert_user_state(user_id="", persona_key="p")
        with self.assertRaises(ValueError):
            self.store.upsert_user_state(user_id="u", persona_key="")

    def test_stale_user_states_returns_old_rows(self) -> None:
        self.store.upsert_user_state(user_id="u_fresh", persona_key="p")
        # Manually backdate one row.
        old_ts = _iso_offset(-48)
        with self.store.connect() as conn:
            conn.execute(
                "UPDATE user_state SET last_updated_at = ? WHERE user_id = 'u_fresh'",
                (old_ts,),
            )
        # Add a fresh one that should not appear.
        self.store.upsert_user_state(user_id="u_new", persona_key="p")
        cutoff = _iso_offset(-24)
        stale = self.store.stale_user_states(persona_key="p", older_than=cutoff)
        self.assertEqual([s.user_id for s in stale], ["u_fresh"])

    def test_stale_states_scope_by_persona(self) -> None:
        old_ts = _iso_offset(-48)
        self.store.upsert_user_state(user_id="u1", persona_key="solomiya")
        self.store.upsert_user_state(user_id="u1", persona_key="mykola")
        with self.store.connect() as conn:
            conn.execute(
                "UPDATE user_state SET last_updated_at = ? WHERE persona_key = 'solomiya'",
                (old_ts,),
            )
        cutoff = _iso_offset(-24)
        solo_stale = self.store.stale_user_states(persona_key="solomiya", older_than=cutoff)
        myk_stale = self.store.stale_user_states(persona_key="mykola", older_than=cutoff)
        self.assertEqual(len(solo_stale), 1)
        self.assertEqual(myk_stale, [])

    def test_recent_user_message_texts_in_temporal_order(self) -> None:
        # Log a few Telegram user messages and read them back oldest-first.
        for idx, text in enumerate(["спочатку", "потім", "наостанок"], start=1):
            self.store.log_telegram_message(
                chat_id="999",
                message_id=idx,
                persona_key="solomiya",
                role="user",
                sender_id="u1",
                sender_name="Vadim",
                text=text,
            )
        items = self.store.recent_user_message_texts(user_id="u1", limit=10)
        self.assertEqual([item["text"] for item in items], ["спочатку", "потім", "наостанок"])

    def test_count_user_messages_filters(self) -> None:
        for idx in range(3):
            self.store.log_telegram_message(
                chat_id="100",
                message_id=idx + 1,
                persona_key="p",
                role="user",
                sender_id="u1",
                sender_name="A",
                text=f"msg {idx}",
            )
        self.store.log_telegram_message(
            chat_id="200",
            message_id=99,
            persona_key="p",
            role="user",
            sender_id="u2",
            sender_name="B",
            text="other user",
        )
        self.assertEqual(self.store.count_user_messages(), 4)
        self.assertEqual(self.store.count_user_messages(user_id="u1"), 3)
        self.assertEqual(self.store.count_user_messages(chat_id="100"), 3)

    def test_schema_version_is_recorded(self) -> None:
        self.assertEqual(self.store.get_kv("schema_version"), "8")


if __name__ == "__main__":
    unittest.main()
