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
    MemoryStore,
)
from protoagi.memory_service import MemoryService, RecallQuery


class StubEmbeddingClient:
    def __init__(self, model: str = "stub") -> None:
        self.config = EmbeddingConfig(base_url="http://stub", model=model)
        self._lookup: dict[str, list[float]] = {}

    def define(self, text: str, vector: list[float]) -> None:
        self._lookup[text] = vector

    def embed(self, text: str) -> list[float] | None:
        return self._lookup.get(text)


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


if __name__ == "__main__":
    unittest.main()
