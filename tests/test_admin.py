import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from protoagi.admin_panel.server import serve
from protoagi.storage.memory import MemoryStore
from protoagi.storage.service import MemoryService


def _free_port() -> int:
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class AdminServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "memory.sqlite3"
        self.memory = MemoryStore(path)
        self.service = MemoryService(self.memory)
        self.service.remember("alpha fact about coffee", tags=["preference"], importance=0.6)
        self.service.remember("beta fact about tea", tags=["preference"], importance=0.4)
        self.port = _free_port()
        self.server = serve(self.memory, self.service, port=self.port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.tmp.cleanup()

    def _get_json(self, path: str):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=2) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_json(self, path: str, payload: dict | None = None):
        body = json.dumps(payload or {}).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_root_serves_spa_when_built(self) -> None:
        # The legacy inline dashboard is gone — once admin_panel/web/dist
        # exists the server hands out the React bundle. In dev without
        # a build the request is a 404 (we don't synthesise a placeholder
        # so the missing build is obvious).
        import urllib.error  # local import; not always exercised

        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/", timeout=2
            ) as resp:
                self.assertEqual(resp.status, 200)
                self.assertIn("text/html", resp.headers.get("Content-Type", ""))
        except urllib.error.HTTPError as exc:
            # No SPA build available — that's an acceptable state.
            self.assertEqual(exc.code, 404)

    def test_api_404_returns_json(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/api/no-such-thing", timeout=2
            )
        self.assertEqual(ctx.exception.code, 404)

    def test_stats_endpoint(self) -> None:
        data = self._get_json("/api/stats")
        self.assertEqual(data["memories_active"], 2)
        self.assertEqual(data["memories_superseded"], 0)
        self.assertIn("telegram_chats", data)
        self.assertIn("media_blobs", data)
        self.assertIn("importance_cache", data)
        self.assertIn("telegram_decision_avg_llm_calls", data)

    def test_memories_endpoint(self) -> None:
        data = self._get_json("/api/memories?limit=10")
        texts = [item["text"] for item in data]
        self.assertIn("alpha fact about coffee", texts)

    def test_memory_graph_endpoint(self) -> None:
        data = self._get_json("/api/memory-graph?limit=20")
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        node_ids = {node["id"] for node in data["nodes"]}
        self.assertTrue(any(node_id.startswith("memory:") for node_id in node_ids))
        self.assertTrue(any(node_id == "tag:preference" for node_id in node_ids))
        self.assertTrue(any(edge["kind"] == "tagged" for edge in data["edges"]))

    def test_memory_graph_filters_scope_and_persona(self) -> None:
        self.memory.store_memory(
            "solomiya graph memory",
            scope="persona",
            persona_key="solomiya",
            tags=["persona-graph"],
        )
        data = self._get_json("/api/memory-graph?scope=persona&persona=solomiya&limit=20")
        self.assertEqual(data["filters"]["scope"], "persona")
        self.assertEqual(data["filters"]["persona_key"], "solomiya")
        memory_nodes = [node for node in data["nodes"] if node["kind"] == "memory"]
        self.assertTrue(memory_nodes)
        self.assertTrue(all(node["scope"] == "persona" for node in memory_nodes))

    def test_style_endpoint_reports_chat_and_aggregate(self) -> None:
        self.memory.upsert_telegram_chat(
            {"id": 123, "type": "private", "first_name": "Vadim"},
            {"id": 123, "first_name": "Vadim"},
        )
        self.memory.set_kv(
            "telegram:style:123",
            json.dumps(
                {
                    "last_choice": "expressive",
                    "arms": {
                        "balanced": {"trials": 1, "successes": 0.0},
                        "expressive": {"trials": 3, "successes": 2.0},
                    },
                    "signals": {"reply": 2, "reaction": 1},
                    "updated_at": "2026-05-03T00:00:00+00:00",
                }
            ),
        )
        data = self._get_json("/api/style")
        self.assertEqual(data["chats"][0]["chat_id"], "123")
        self.assertEqual(data["chats"][0]["active_arm"], "expressive")
        self.assertEqual(data["aggregate"]["expressive"]["trials"], 3)
        self.assertEqual(data["signals"]["reply"], 2)

    def test_media_endpoint_returns_blob_bytes(self) -> None:
        self.memory.store_media_blob(
            file_id="photo-admin",
            mime="image/jpeg",
            data=b"admin-image",
            caption="admin caption",
        )
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}/api/media/photo-admin",
            timeout=2,
        ) as resp:
            body = resp.read()
            content_type = resp.headers.get("Content-Type")
        self.assertEqual(body, b"admin-image")
        self.assertEqual(content_type, "image/jpeg")

    def test_delete_memory_endpoint(self) -> None:
        data = self._get_json("/api/memories?limit=10")
        target_id = data[0]["id"]
        deleted = self._post_json(f"/api/memories/{target_id}/delete")
        self.assertEqual(deleted["deleted"], target_id)
        self.assertIsNone(self.memory.get_memory(target_id))

    def test_pin_memory_toggles(self) -> None:
        data = self._get_json("/api/memories?limit=10")
        target_id = data[0]["id"]
        first = self._post_json(f"/api/memories/{target_id}/pin")
        self.assertTrue(first["pinned"])
        again = self._post_json(f"/api/memories/{target_id}/pin")
        self.assertFalse(again["pinned"])

    def test_pin_memory_explicit_value(self) -> None:
        data = self._get_json("/api/memories?limit=10")
        target_id = data[0]["id"]
        result = self._post_json(f"/api/memories/{target_id}/pin", {"pinned": True})
        self.assertTrue(result["pinned"])
        item = self.memory.get_memory(target_id)
        assert item is not None
        self.assertTrue(item.pinned)

    def test_edit_memory_updates_text_and_importance(self) -> None:
        data = self._get_json("/api/memories?limit=10")
        target_id = data[0]["id"]
        result = self._post_json(
            f"/api/memories/{target_id}/edit",
            {"text": "оновлений факт про каву", "importance": 0.9, "tags": ["coffee"]},
        )
        self.assertEqual(result["text"], "оновлений факт про каву")
        self.assertAlmostEqual(result["importance"], 0.9, places=4)
        self.assertEqual(result["tags"], ["coffee"])
        # FTS row should reflect the new text — search by the new word.
        hits = self.memory.search("оновлений", limit=5)
        self.assertTrue(any(hit.id == target_id for hit in hits))

    def test_edit_rejects_empty_text(self) -> None:
        data = self._get_json("/api/memories?limit=10")
        target_id = data[0]["id"]
        try:
            self._post_json(f"/api/memories/{target_id}/edit", {"text": "   "})
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)
            return
        self.fail("expected HTTP 400 for empty text")

    def test_prune_preview_returns_plan_without_deleting(self) -> None:
        stored = self.service.remember("old low value note", importance=0.01)
        assert stored is not None
        old = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat(timespec="seconds")
        with self.memory.connect() as conn:
            conn.execute("UPDATE memory_items SET created_at = ? WHERE id = ?", (old, stored.memory_id))
        result = self._post_json(
            "/api/memories/prune/preview",
            {"score_threshold": 0.9, "keep_newer_than_days": 30},
        )
        self.assertEqual(result["deleted"], 1)
        self.assertEqual(result["plan"][0]["dropped"]["id"], stored.memory_id)
        self.assertIsNotNone(self.memory.get_memory(stored.memory_id))

    def test_reasoning_endpoint_returns_overview_and_entries(self) -> None:
        from protoagi.telegram.reasoning_log import ReasoningLog, ReasoningLogConfig

        log = ReasoningLog(self.memory, ReasoningLogConfig(enabled=True))
        log.record(
            chat_id="555",
            message_id=42,
            captured_at="2026-05-07T10:00:00+00:00",
            decision_kind="decision",
            incoming_text="привіт",
            reasoning="step one\nstep two",
            reply_excerpt="ага",
        )
        overview = self._get_json("/api/reasoning")
        self.assertTrue(any(row["chat_id"] == "555" for row in overview))
        entries = self._get_json("/api/reasoning/555")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["message_id"], 42)
        self.assertIn("step one", entries[0]["reasoning"])

    def test_consolidate_preview_returns_supersession_plan(self) -> None:
        first = self.service.remember("duplicate admin preview", importance=0.3)
        second = self.service.remember("duplicate admin preview", importance=0.8)
        assert first is not None and second is not None
        result = self._post_json("/api/memories/consolidate/preview", {})
        self.assertEqual(result["merged"], 1)
        self.assertEqual(result["plan"][0]["kept"]["id"], second.memory_id)
        self.assertEqual(result["plan"][0]["dropped"]["id"], first.memory_id)
        self.assertIsNone(self.memory.get_memory(first.memory_id).superseded_by)


if __name__ == "__main__":
    unittest.main()
