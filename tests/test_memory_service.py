import tempfile
import unittest
from pathlib import Path

from protoagi.embedding import EmbeddingClient, EmbeddingConfig
from protoagi.memory import (
    KIND_PERSONA_SELF,
    KIND_SEMANTIC,
    SCOPE_CHAT,
    SCOPE_GLOBAL,
    SCOPE_PERSONA,
    SCOPE_USER,
    MemoryStore,
)
from protoagi.memory_service import MemoryService, RecallQuery


class StubEmbeddingClient:
    def __init__(self, model: str = "stub") -> None:
        self.config = EmbeddingConfig(base_url="http://stub", model=model)
        self._lookup: dict[str, list[float]] = {}
        self._media_lookup: dict[tuple[bytes, str], list[float]] = {}

    def define(self, text: str, vector: list[float]) -> None:
        self._lookup[text] = vector

    def embed(self, text: str) -> list[float] | None:
        return self._lookup.get(text)

    def define_media(self, data: bytes, mime: str, vector: list[float]) -> None:
        self._media_lookup[(data, mime)] = vector

    def embed_media(self, data: bytes, *, mime: str, caption: str = "") -> list[float] | None:
        return self._media_lookup.get((data, mime))


class StubImportanceClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_completion(self, messages, **kwargs):
        self.calls += 1
        payload = messages[1]["content"]
        if "allergy" in payload or "алергі" in payload:
            content = '{"importance": 0.93, "kind": "semantic", "reasoning": "stable safety fact"}'
        else:
            content = '{"importance": 0.22, "kind": "episodic", "reasoning": "transient mood"}'
        return {"choices": [{"message": {"content": content}}]}


class MemoryServiceTests(unittest.TestCase):
    def _make(self, *, with_embeddings: bool = False) -> tuple[MemoryStore, MemoryService]:
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "memory.sqlite3"
        store = MemoryStore(path)
        client = StubEmbeddingClient() if with_embeddings else None
        service = MemoryService(store, embedding_client=client)
        return store, service

    def tearDown(self) -> None:
        tmp = getattr(self, "tmp", None)
        if tmp is not None:
            tmp.cleanup()

    def test_store_memory_assigns_dimensions(self) -> None:
        store, service = self._make()
        stored = service.remember(
            "User loves coffee in the morning",
            kind=KIND_SEMANTIC,
            scope=SCOPE_GLOBAL,
            tags=["preference"],
            user_id="telegram:1",
            importance=0.7,
        )
        self.assertIsNotNone(stored)
        item = store.get_memory(stored.memory_id)
        self.assertIsNotNone(item)
        self.assertEqual(item.kind, KIND_SEMANTIC)
        self.assertEqual(item.scope, SCOPE_GLOBAL)
        self.assertAlmostEqual(item.importance, 0.7)
        self.assertEqual(item.user_id, "telegram:1")

    def test_recall_returns_fts_hits_without_embeddings(self) -> None:
        store, service = self._make()
        service.remember("user prefers chamomile tea late evenings", scope=SCOPE_GLOBAL)
        service.remember("user enjoys morning coffee with cardamom", scope=SCOPE_GLOBAL)
        results = service.recall(RecallQuery(text="coffee", limit=3))
        self.assertEqual(len(results), 1)
        self.assertIn("coffee", results[0].item.text)

    def test_recall_combines_cosine_and_fts(self) -> None:
        store, service = self._make(with_embeddings=True)
        client = service.embedding_client
        assert isinstance(client, StubEmbeddingClient)
        client.define("user loves a quiet evening with chamomile tea", [1.0, 0.0, 0.0])
        client.define("user enjoys morning coffee with cardamom spice", [0.0, 1.0, 0.0])
        client.define("чай", [1.0, 0.0, 0.0])
        service.remember("user loves a quiet evening with chamomile tea", scope=SCOPE_GLOBAL)
        service.remember("user enjoys morning coffee with cardamom spice", scope=SCOPE_GLOBAL)
        results = service.recall(RecallQuery(text="чай", limit=2))
        self.assertGreater(len(results), 0)
        self.assertIn("chamomile", results[0].item.text)
        self.assertGreater(results[0].cosine, 0.0)

    def test_image_media_embedding_can_drive_recall(self) -> None:
        store, service = self._make(with_embeddings=True)
        client = service.embedding_client
        assert isinstance(client, StubEmbeddingClient)
        image_bytes = b"gamepad-image"
        store.store_media_blob(
            file_id="photo-1",
            mime="image/jpeg",
            data=image_bytes,
            caption="black gamepad on a desk",
        )
        client.define_media(image_bytes, "image/jpeg", [1.0, 0.0, 0.0])
        client.define("controller picture", [1.0, 0.0, 0.0])
        stored = service.remember(
            "Telegram image in chat 1: black gamepad on a desk",
            media_id="photo-1",
            tags=["media", "image"],
        )
        self.assertIsNotNone(stored)
        results = service.recall(RecallQuery(text="controller picture", limit=1))
        self.assertEqual(results[0].item.media_id, "photo-1")
        self.assertGreater(results[0].cosine, 0.0)

    def test_persona_scope_blocks_other_personas(self) -> None:
        store, service = self._make()
        service.remember(
            "Соломія любить каву",
            kind=KIND_PERSONA_SELF,
            scope=SCOPE_PERSONA,
            persona_key="solomiya",
            tags=["telegram_persona_self"],
        )
        service.remember(
            "Микола любить чай",
            kind=KIND_PERSONA_SELF,
            scope=SCOPE_PERSONA,
            persona_key="mykola",
            tags=["telegram_persona_self"],
        )
        results = service.recall(
            RecallQuery(text="любить", persona_key="solomiya", limit=5)
        )
        texts = [result.item.text for result in results]
        self.assertIn("Соломія любить каву", texts)
        self.assertNotIn("Микола любить чай", texts)

    def test_consolidate_supersedes_near_duplicates(self) -> None:
        store, service = self._make()
        service.remember("user loves coffee", scope=SCOPE_GLOBAL, importance=0.5)
        service.remember("user loves coffee", scope=SCOPE_GLOBAL, importance=0.8)
        merges = service.consolidate(scope=SCOPE_GLOBAL)
        self.assertEqual(merges, 1)
        active = store.list_memories(scope=SCOPE_GLOBAL, include_superseded=False)
        self.assertEqual(len(active), 1)

    def test_chat_scope_filters_when_chat_id_set(self) -> None:
        store, service = self._make()
        service.remember(
            "loved chat 1 detail",
            scope=SCOPE_CHAT,
            chat_id="1",
        )
        service.remember(
            "loved chat 2 detail",
            scope=SCOPE_CHAT,
            chat_id="2",
        )
        results = service.recall(RecallQuery(text="loved", chat_id="1", limit=5))
        texts = [result.item.text for result in results]
        self.assertIn("loved chat 1 detail", texts)
        self.assertNotIn("loved chat 2 detail", texts)

    def test_user_scope_requires_matching_user(self) -> None:
        store, service = self._make()
        service.remember("alice coffee preference", scope=SCOPE_USER, user_id="telegram:1")
        service.remember("bob coffee preference", scope=SCOPE_USER, user_id="telegram:2")
        results = service.recall(RecallQuery(text="coffee", user_id="telegram:1", limit=5))
        texts = [result.item.text for result in results]
        self.assertIn("alice coffee preference", texts)
        self.assertNotIn("bob coffee preference", texts)
        self.assertEqual(service.recall(RecallQuery(text="coffee", limit=5)), [])

    def test_llm_importance_scoring_is_opt_in_and_cached(self) -> None:
        store, _ = self._make()
        client = StubImportanceClient()
        service = MemoryService(
            store,
            importance_client=client,
            llm_importance=True,
        )
        critical = service.remember("user has a nut allergy", scope=SCOPE_GLOBAL)
        self.assertIsNotNone(critical)
        assert critical is not None
        self.assertGreater(critical.item.importance, 0.85)
        self.assertEqual(critical.item.kind, KIND_SEMANTIC)

        mood = service.remember("user feels sleepy tonight", scope=SCOPE_GLOBAL)
        self.assertIsNotNone(mood)
        assert mood is not None
        self.assertLess(mood.item.importance, 0.3)

        before = client.calls
        for _ in range(10):
            service.remember("user has a nut allergy", scope=SCOPE_GLOBAL)
        self.assertEqual(client.calls, before)
        self.assertEqual(store.importance_cache_count(), 2)
        self.assertIsNone(store.get_kv("memory:importance"))

    def test_consolidate_dry_run_returns_plan_without_superseding(self) -> None:
        store, service = self._make()
        first = service.remember("user loves coffee", scope=SCOPE_GLOBAL, importance=0.5)
        second = service.remember("user loves coffee", scope=SCOPE_GLOBAL, importance=0.8)
        assert first is not None and second is not None
        result = service.consolidate(scope=SCOPE_GLOBAL, dry_run=True)
        self.assertIsInstance(result, dict)
        assert isinstance(result, dict)
        self.assertEqual(result["merged"], 1)
        self.assertEqual(result["plan"][0]["kept"]["id"], second.memory_id)
        self.assertEqual(result["plan"][0]["dropped"]["id"], first.memory_id)
        active = store.list_memories(scope=SCOPE_GLOBAL, include_superseded=False)
        self.assertEqual(len(active), 2)


if __name__ == "__main__":
    unittest.main()
