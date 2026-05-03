from pathlib import Path
import tempfile
import unittest

from protoagi.embedding import EmbeddingClient, EmbeddingConfig, EmbeddingIndex
from protoagi.memory import cosine_similarity, pack_embedding, unpack_embedding
from protoagi.memory import MemoryStore


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
