import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from protoagi.admin import serve
from protoagi.memory import MemoryStore
from protoagi.memory_service import MemoryService


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

    def test_dashboard_renders(self) -> None:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=2) as resp:
            body = resp.read().decode("utf-8")
        self.assertIn("ProtoAGI", body)
        self.assertIn("alpha fact", body)

    def test_stats_endpoint(self) -> None:
        data = self._get_json("/api/stats")
        self.assertEqual(data["memories_active"], 2)
        self.assertEqual(data["memories_superseded"], 0)
        self.assertIn("telegram_chats", data)

    def test_memories_endpoint(self) -> None:
        data = self._get_json("/api/memories?limit=10")
        texts = [item["text"] for item in data]
        self.assertIn("alpha fact about coffee", texts)

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


if __name__ == "__main__":
    unittest.main()
