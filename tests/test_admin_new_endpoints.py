"""Tests for the new admin endpoints introduced for the React UI:
goals, conflicts, user_state, health.
"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from protoagi.admin_panel.server import serve
from protoagi.storage.memory import MemoryStore
from protoagi.storage.models import (
    CONFLICT_STATUS_SUPERSEDED,
    GOAL_STATUS_COMPLETED,
)
from protoagi.storage.service import MemoryService


class AdminNewEndpointsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / "memory.sqlite3"
        cls.memory = MemoryStore(path)
        cls.service = MemoryService(store=cls.memory)
        cls.server = serve(cls.memory, cls.service, host="127.0.0.1", port=0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.tmp.cleanup()

    def setUp(self) -> None:
        with self.memory.connect() as conn:
            for table in ("goals", "memory_conflicts", "user_state", "memory_items"):
                conn.execute(f"DELETE FROM {table}")

    # ---------- helpers ----------

    def _get(self, path: str) -> dict | list:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}{path}", timeout=2
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ---------- /api/health ----------

    def test_health_summary_counts(self) -> None:
        self.memory.store_memory("a", persona_key="solomiya")
        self.memory.open_goal(persona_key="solomiya", text="g1")
        self.memory.upsert_user_state(user_id="u1", persona_key="solomiya")
        data = self._get("/api/health?persona=solomiya")
        self.assertEqual(data["memories_active"], 1)
        self.assertEqual(data["open_goals"], 1)
        self.assertEqual(data["user_states_tracked"], 1)

    # ---------- /api/goals ----------

    def test_list_open_goals(self) -> None:
        self.memory.open_goal(persona_key="solomiya", text="open one")
        gid_closed = self.memory.open_goal(persona_key="solomiya", text="closed one")
        self.memory.update_goal(gid_closed, status=GOAL_STATUS_COMPLETED)
        data = self._get("/api/goals?status=open&persona=solomiya")
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["text"], "open one")
        self.assertEqual(data[0]["status"], "open")

    def test_list_goals_all(self) -> None:
        self.memory.open_goal(persona_key="solomiya", text="g1")
        gid = self.memory.open_goal(persona_key="solomiya", text="g2")
        self.memory.update_goal(gid, status=GOAL_STATUS_COMPLETED)
        data = self._get("/api/goals?status=all&persona=solomiya")
        self.assertEqual(len(data), 2)

    def test_update_goal_closes_via_post(self) -> None:
        gid = self.memory.open_goal(persona_key="solomiya", text="finish me")
        body = self._post(f"/api/goals/{gid}/update", {"status": "completed"})
        self.assertEqual(body["status"], "completed")
        self.assertIsNotNone(body["closed_at"])

    def test_update_goal_revises_text_and_priority(self) -> None:
        gid = self.memory.open_goal(persona_key="solomiya", text="old", priority=0.4)
        body = self._post(f"/api/goals/{gid}/update", {"text": "new", "priority": 0.9})
        self.assertEqual(body["text"], "new")
        self.assertEqual(body["priority"], 0.9)

    def test_update_goal_invalid_status(self) -> None:
        gid = self.memory.open_goal(persona_key="solomiya", text="x")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post(f"/api/goals/{gid}/update", {"status": "finished"})
        self.assertEqual(ctx.exception.code, 400)

    # ---------- /api/conflicts ----------

    def test_list_unresolved_conflicts_with_sides(self) -> None:
        a = self.memory.store_memory("чай ромашковий", persona_key="solomiya")
        b = self.memory.store_memory("каву зранку", persona_key="solomiya")
        self.memory.record_conflict(a, b, similarity=0.85, persona_key="solomiya")
        data = self._get("/api/conflicts?status=unresolved&persona=solomiya")
        self.assertEqual(len(data), 1)
        item = data[0]
        self.assertIn("memory_a", item)
        self.assertIn("memory_b", item)
        self.assertEqual(
            {item["memory_a"]["text"], item["memory_b"]["text"]},
            {"чай ромашковий", "каву зранку"},
        )

    def test_resolve_conflict_superseded_marks_loser(self) -> None:
        a = self.memory.store_memory("чай", persona_key="solomiya")
        b = self.memory.store_memory("кава", persona_key="solomiya")
        cid = self.memory.record_conflict(a, b, similarity=0.85, persona_key="solomiya")
        body = self._post(
            f"/api/conflicts/{cid}/resolve",
            {"status": "superseded", "winner_id": b},
        )
        self.assertEqual(body["status"], "superseded")
        self.assertEqual(body["winner_id"], b)
        # Server-side: the loser should now have superseded_by set so
        # recall stops returning it.
        loser = self.memory.get_memory(a)
        assert loser is not None
        self.assertEqual(loser.superseded_by, b)

    def test_resolve_conflict_dismissed_no_supersession(self) -> None:
        a = self.memory.store_memory("чай", persona_key="solomiya")
        b = self.memory.store_memory("кава", persona_key="solomiya")
        cid = self.memory.record_conflict(a, b, similarity=0.85, persona_key="solomiya")
        body = self._post(
            f"/api/conflicts/{cid}/resolve",
            {"status": "dismissed"},
        )
        self.assertEqual(body["status"], "dismissed")
        self.assertIsNone(self.memory.get_memory(a).superseded_by)

    def test_resolve_requires_status(self) -> None:
        a = self.memory.store_memory("x", persona_key="solomiya")
        b = self.memory.store_memory("y", persona_key="solomiya")
        cid = self.memory.record_conflict(a, b, similarity=0.85, persona_key="solomiya")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post(f"/api/conflicts/{cid}/resolve", {})
        self.assertEqual(ctx.exception.code, 400)

    # ---------- /api/user_state ----------

    def test_list_user_states(self) -> None:
        self.memory.upsert_user_state(
            user_id="u1", persona_key="solomiya", summary="warm view"
        )
        self.memory.upsert_user_state(
            user_id="u2", persona_key="mykola", summary="mykola view"
        )
        data = self._get("/api/user_state?persona=solomiya")
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["summary"], "warm view")

    # ---------- /api/memories filters ----------

    def test_memories_search_filter(self) -> None:
        self.memory.store_memory("чай", persona_key="solomiya")
        self.memory.store_memory("кава", persona_key="solomiya")
        encoded = urllib.parse.quote("чай")
        data = self._get(f"/api/memories?search={encoded}&limit=10")
        texts = {item["text"] for item in data}
        self.assertIn("чай", texts)
        self.assertNotIn("кава", texts)

    def test_memories_pinned_filter(self) -> None:
        a = self.memory.store_memory("a", persona_key="solomiya")
        self.memory.store_memory("b", persona_key="solomiya")
        self.memory.set_pinned(a, True)
        data = self._get("/api/memories?pinned=true&limit=10")
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["id"], a)


if __name__ == "__main__":
    unittest.main()
