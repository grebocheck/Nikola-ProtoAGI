"""Tests for MemoryService.memory_health introspection.

This is the lightweight admin/audit endpoint: count of active and
superseded memories, open goals, tracked user states, unresolved
conflicts. Useful both for a future admin panel and for plain CLI
"how full is the brain" checks.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from protoagi.storage.memory import MemoryStore
from protoagi.storage.models import (
    CONFLICT_STATUS_DISMISSED,
    GOAL_STATUS_COMPLETED,
)
from protoagi.storage.service import MemoryService


class MemoryHealthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name) / "memory.sqlite3")
        self.service = MemoryService(store=self.store)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_empty_store_reports_zero(self) -> None:
        health = self.service.memory_health()
        self.assertEqual(health["memories_active"], 0)
        self.assertEqual(health["memories_superseded"], 0)
        self.assertEqual(health["memories_total"], 0)
        self.assertEqual(health["open_goals"], 0)
        self.assertEqual(health["user_states_tracked"], 0)
        self.assertEqual(health["unresolved_conflicts"], 0)

    def test_counts_active_and_superseded_memories(self) -> None:
        a = self.store.store_memory("fact a")
        b = self.store.store_memory("fact b")
        self.store.supersede(a, b)
        health = self.service.memory_health()
        self.assertEqual(health["memories_active"], 1)
        self.assertEqual(health["memories_superseded"], 1)
        self.assertEqual(health["memories_total"], 2)

    def test_counts_open_goals_only(self) -> None:
        self.store.open_goal(persona_key="solomiya", text="g1")
        gid = self.store.open_goal(persona_key="solomiya", text="g2")
        self.store.update_goal(gid, status=GOAL_STATUS_COMPLETED)
        health = self.service.memory_health(persona_key="solomiya")
        self.assertEqual(health["open_goals"], 1)

    def test_counts_user_states_per_persona(self) -> None:
        self.store.upsert_user_state(user_id="u1", persona_key="solomiya")
        self.store.upsert_user_state(user_id="u2", persona_key="solomiya")
        self.store.upsert_user_state(user_id="u1", persona_key="mykola")
        solo = self.service.memory_health(persona_key="solomiya")
        myk = self.service.memory_health(persona_key="mykola")
        self.assertEqual(solo["user_states_tracked"], 2)
        self.assertEqual(myk["user_states_tracked"], 1)

    def test_counts_only_unresolved_conflicts(self) -> None:
        a = self.store.store_memory("a")
        b = self.store.store_memory("b")
        c = self.store.store_memory("c")
        cid1 = self.store.record_conflict(a, b, similarity=0.85, persona_key="solomiya")
        self.store.record_conflict(a, c, similarity=0.85, persona_key="solomiya")
        self.store.resolve_conflict(cid1, status=CONFLICT_STATUS_DISMISSED)
        health = self.service.memory_health(persona_key="solomiya")
        self.assertEqual(health["unresolved_conflicts"], 1)

    def test_persona_scope_isolates_counts(self) -> None:
        self.store.store_memory("solo fact", persona_key="solomiya")
        self.store.store_memory("myk fact", persona_key="mykola")
        self.store.open_goal(persona_key="solomiya", text="solo goal")
        self.store.open_goal(persona_key="mykola", text="myk goal")
        solo = self.service.memory_health(persona_key="solomiya")
        myk = self.service.memory_health(persona_key="mykola")
        self.assertEqual(solo["memories_active"], 1)
        self.assertEqual(myk["memories_active"], 1)
        self.assertEqual(solo["open_goals"], 1)
        self.assertEqual(myk["open_goals"], 1)


if __name__ == "__main__":
    unittest.main()
