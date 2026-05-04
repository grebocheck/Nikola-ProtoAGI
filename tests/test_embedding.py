from pathlib import Path
import tempfile
import unittest

from protoagi.embedding import EmbeddingClient, EmbeddingConfig, EmbeddingIndex
from protoagi.storage.memory import MemoryStore, cosine_similarity, pack_embedding, unpack_embedding


class EmbeddingPackTests(unittest.TestCase):
    def test_pack_unpack_round_trip(self) -> None:
        vector = [0.1, -0.2, 0.5, 1.0, -1.5]
        blob = pack_embedding(vector)
        unpacked = unpack_embedding(blob)
        for left, right in zip(vector, unpacked):
            self.assertAlmostEqual(left, right, places=4)

    def test_cosine_basics(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertAlmostEqual(cosine_similarity([], [1.0]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)


class EmbeddingClientTests(unittest.TestCase):
    def test_disabled_client_returns_none(self) -> None:
        client = EmbeddingClient(EmbeddingConfig())
        self.assertIsNone(client.embed("anything"))

    def test_cache_eviction_is_lru(self) -> None:
        client = EmbeddingClient(
            EmbeddingConfig(base_url="http://embedding.local/v1", model="stub", cache_size=2)
        )
        calls: list[str] = []

        def fake_request(text: str) -> list[float]:
            calls.append(text)
            return [float(len(text))]

        client._request = fake_request  # type: ignore[method-assign]
        self.assertEqual(client.embed("hot"), [3.0])
        self.assertEqual(client.embed("cold"), [4.0])
        self.assertEqual(client.embed("hot"), [3.0])
        self.assertEqual(client.embed("new"), [3.0])
        self.assertIn("hot", client._cache)
        self.assertNotIn("cold", client._cache)
        self.assertEqual(calls, ["hot", "cold", "new"])


class EmbeddingIndexTests(unittest.TestCase):
    def test_lsh_backend_returns_exact_vector_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            first = memory.store_memory(
                "vector alpha",
                embedding=[1.0, 0.0, 0.0, 0.0],
                embedding_model="stub",
            )
            memory.store_memory(
                "vector beta",
                embedding=[0.0, 1.0, 0.0, 0.0],
                embedding_model="stub",
            )
            index = EmbeddingIndex(memory, model="stub", backend="lsh")
            hits = index.similar([1.0, 0.0, 0.0, 0.0], limit=1)
            self.assertEqual(hits[0][0], first)


if __name__ == "__main__":
    unittest.main()
