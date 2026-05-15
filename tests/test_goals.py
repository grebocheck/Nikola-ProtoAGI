"""Storage-layer tests for the goals table.

Goals are a separate concept from memory items: they have a discrete
lifecycle (open/completed/abandoned), a persona owner, an optional due
date, and are queried by status rather than by free-text recall. This
file exercises the MemoryStore methods directly so the Telegram wiring
in test_telegram_bot.py can stay focused on prompt/decision behavior.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from protoagi.storage.memory import MemoryStore
from protoagi.storage.models import (
    GOAL_STATUS_ABANDONED,
    GOAL_STATUS_COMPLETED,
    GOAL_STATUS_OPEN,
)


def _iso_offset(hours: float) -> str:
    return (
        datetime.now(timezone.utc) + timedelta(hours=hours)
    ).isoformat(timespec="seconds")


class GoalStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name) / "memory.sqlite3")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_open_goal_inserts_with_defaults(self) -> None:
        goal_id = self.store.open_goal(persona_key="solomiya", text="domiraty pro chai")
        goal = self.store.get_goal(goal_id)
        self.assertIsNotNone(goal)
        assert goal is not None
        self.assertEqual(goal.persona_key, "solomiya")
        self.assertEqual(goal.text, "domiraty pro chai")
        self.assertEqual(goal.status, GOAL_STATUS_OPEN)
        self.assertEqual(goal.priority, 0.5)
        self.assertIsNone(goal.due_at)
        self.assertIsNone(goal.closed_at)
        self.assertEqual(goal.metadata, {})

    def test_open_goal_rejects_empty_text(self) -> None:
        with self.assertRaises(ValueError):
            self.store.open_goal(persona_key="solomiya", text="   ")

    def test_open_goal_rejects_empty_persona(self) -> None:
        with self.assertRaises(ValueError):
            self.store.open_goal(persona_key="", text="something")

    def test_priority_is_clamped(self) -> None:
        gid_high = self.store.open_goal(persona_key="m", text="t1", priority=2.0)
        gid_low = self.store.open_goal(persona_key="m", text="t2", priority=-1.0)
        high = self.store.get_goal(gid_high)
        low = self.store.get_goal(gid_low)
        assert high is not None and low is not None
        self.assertEqual(high.priority, 1.0)
        self.assertEqual(low.priority, 0.0)

    def test_list_open_goals_orders_due_first_then_priority(self) -> None:
        # No due, high priority
        gid_high_no_due = self.store.open_goal(
            persona_key="solomiya", text="no due high", priority=0.9
        )
        # Due in 1h, low priority — should come first due to due_at NOT NULL
        gid_soon_low = self.store.open_goal(
            persona_key="solomiya",
            text="soon low",
            priority=0.2,
            due_at=_iso_offset(1),
        )
        # Due in 3h, mid priority
        gid_later_mid = self.store.open_goal(
            persona_key="solomiya",
            text="later mid",
            priority=0.5,
            due_at=_iso_offset(3),
        )
        goals = self.store.list_open_goals(persona_key="solomiya")
        ids = [g.id for g in goals]
        self.assertEqual(ids, [gid_soon_low, gid_later_mid, gid_high_no_due])

    def test_list_open_goals_scopes_by_persona(self) -> None:
        self.store.open_goal(persona_key="solomiya", text="solo")
        self.store.open_goal(persona_key="mykola", text="myk")
        solo = self.store.list_open_goals(persona_key="solomiya")
        myk = self.store.list_open_goals(persona_key="mykola")
        self.assertEqual([g.text for g in solo], ["solo"])
        self.assertEqual([g.text for g in myk], ["myk"])

    def test_list_open_goals_excludes_closed(self) -> None:
        gid_open = self.store.open_goal(persona_key="m", text="open")
        gid_done = self.store.open_goal(persona_key="m", text="done")
        self.store.update_goal(gid_done, status=GOAL_STATUS_COMPLETED)
        listed = self.store.list_open_goals(persona_key="m")
        self.assertEqual([g.id for g in listed], [gid_open])

    def test_list_open_goals_filters_by_chat(self) -> None:
        gid_chat_a = self.store.open_goal(persona_key="m", text="a", chat_id="123")
        self.store.open_goal(persona_key="m", text="b", chat_id="456")
        listed = self.store.list_open_goals(persona_key="m", chat_id="123")
        self.assertEqual([g.id for g in listed], [gid_chat_a])

    def test_list_due_goals_excludes_no_due(self) -> None:
        self.store.open_goal(persona_key="m", text="no due")
        gid_due = self.store.open_goal(
            persona_key="m", text="due soon", due_at=_iso_offset(1)
        )
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        due = self.store.list_due_goals(persona_key="m", now=now, lookahead_hours=24)
        self.assertEqual([g.id for g in due], [gid_due])

    def test_list_due_goals_respects_lookahead(self) -> None:
        gid_near = self.store.open_goal(persona_key="m", text="near", due_at=_iso_offset(2))
        self.store.open_goal(persona_key="m", text="far", due_at=_iso_offset(48))
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        due = self.store.list_due_goals(persona_key="m", now=now, lookahead_hours=6)
        self.assertEqual([g.id for g in due], [gid_near])

    def test_list_due_goals_excludes_closed(self) -> None:
        gid = self.store.open_goal(persona_key="m", text="due", due_at=_iso_offset(1))
        self.store.update_goal(gid, status=GOAL_STATUS_COMPLETED)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        due = self.store.list_due_goals(persona_key="m", now=now)
        self.assertEqual(due, [])

    def test_update_goal_sets_status_and_closed_at(self) -> None:
        gid = self.store.open_goal(persona_key="m", text="t")
        updated = self.store.update_goal(gid, status=GOAL_STATUS_COMPLETED)
        assert updated is not None
        self.assertEqual(updated.status, GOAL_STATUS_COMPLETED)
        self.assertIsNotNone(updated.closed_at)
        self.assertIsNotNone(updated.updated_at)

    def test_update_goal_reopen_clears_closed_at(self) -> None:
        gid = self.store.open_goal(persona_key="m", text="t")
        self.store.update_goal(gid, status=GOAL_STATUS_ABANDONED)
        reopened = self.store.update_goal(gid, status=GOAL_STATUS_OPEN)
        assert reopened is not None
        self.assertEqual(reopened.status, GOAL_STATUS_OPEN)
        self.assertIsNone(reopened.closed_at)

    def test_update_goal_rejects_invalid_status(self) -> None:
        gid = self.store.open_goal(persona_key="m", text="t")
        with self.assertRaises(ValueError):
            self.store.update_goal(gid, status="finished")

    def test_update_goal_clears_due_when_explicit_none(self) -> None:
        gid = self.store.open_goal(persona_key="m", text="t", due_at=_iso_offset(1))
        cleared = self.store.update_goal(gid, due_at=None)
        assert cleared is not None
        self.assertIsNone(cleared.due_at)

    def test_update_goal_preserves_due_when_omitted(self) -> None:
        due = _iso_offset(1)
        gid = self.store.open_goal(persona_key="m", text="t", due_at=due)
        bumped = self.store.update_goal(gid, priority=0.9)
        assert bumped is not None
        self.assertEqual(bumped.due_at, due)
        self.assertEqual(bumped.priority, 0.9)

    def test_update_goal_merges_metadata(self) -> None:
        gid = self.store.open_goal(persona_key="m", text="t", metadata={"a": 1})
        merged = self.store.update_goal(gid, metadata_patch={"b": 2})
        assert merged is not None
        self.assertEqual(merged.metadata, {"a": 1, "b": 2})

    def test_touch_goal_updates_last_touched(self) -> None:
        gid = self.store.open_goal(persona_key="m", text="t")
        first = self.store.get_goal(gid)
        assert first is not None
        # touch should bump last_touched_at to a later (or equal) timestamp
        self.store.touch_goal(gid)
        second = self.store.get_goal(gid)
        assert second is not None
        self.assertGreaterEqual(second.last_touched_at, first.last_touched_at)

    def test_count_goals_filters(self) -> None:
        self.store.open_goal(persona_key="m", text="a")
        gid = self.store.open_goal(persona_key="m", text="b")
        self.store.open_goal(persona_key="solomiya", text="c")
        self.store.update_goal(gid, status=GOAL_STATUS_COMPLETED)
        self.assertEqual(self.store.count_goals(), 3)
        self.assertEqual(self.store.count_goals(persona_key="m"), 2)
        self.assertEqual(
            self.store.count_goals(persona_key="m", status=GOAL_STATUS_OPEN), 1
        )

    def test_schema_version_is_recorded(self) -> None:
        self.assertEqual(self.store.get_kv("schema_version"), "7")


if __name__ == "__main__":
    unittest.main()
