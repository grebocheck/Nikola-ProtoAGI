import unittest

from protoagi.embedding import EmbeddingClient, EmbeddingConfig
from protoagi.memory import cosine_similarity, pack_embedding, unpack_embedding


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


if __name__ == "__main__":
    unittest.main()
