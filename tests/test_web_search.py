import json
import tempfile
import unittest
from pathlib import Path

from protoagi.storage.memory import MemoryStore
from protoagi.web_search import (
    CACHE_KV_PREFIX,
    WebSearchClient,
    WebSearchConfig,
    WebSearchUnavailable,
)


def _temp_memory() -> tuple[MemoryStore, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory()
    store = MemoryStore(Path(tmp.name) / "memory.sqlite3")
    return store, tmp


class WebSearchClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory, self._tmp = _temp_memory()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _stub_fetcher(self, payload: dict) -> tuple[list[str], object]:
        urls: list[str] = []

        def fetcher(url: str, max_chars: int) -> tuple[str, bytes]:
            urls.append(url)
            return ("application/json", json.dumps(payload).encode("utf-8"))

        return urls, fetcher

    def test_disabled_when_base_url_empty(self) -> None:
        client = WebSearchClient(WebSearchConfig())
        self.assertFalse(client.enabled)
        with self.assertRaises(WebSearchUnavailable):
            client.search("anything")

    def test_returns_trimmed_results(self) -> None:
        payload = {
            "results": [
                {"title": "A", "url": "https://example.com/a", "content": "answer one"},
                {"title": "B", "url": "https://example.com/b", "content": "answer two"},
                {"title": "C", "url": "https://example.com/c", "snippet": "answer three"},
            ]
        }
        urls, fetcher = self._stub_fetcher(payload)
        client = WebSearchClient(
            WebSearchConfig(base_url="https://search.example/api", max_results=2),
            memory=self.memory,
            fetcher=fetcher,
        )
        results = client.search("курс долара")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "A")
        self.assertEqual(results[0].url, "https://example.com/a")
        self.assertEqual(results[0].snippet, "answer one")
        self.assertEqual(len(urls), 1)
        self.assertIn("q=", urls[0])

    def test_drops_results_without_url(self) -> None:
        payload = {"results": [{"title": "no url"}, {"url": "https://example.com/x"}]}
        _, fetcher = self._stub_fetcher(payload)
        client = WebSearchClient(
            WebSearchConfig(base_url="https://search.example/api"),
            memory=self.memory,
            fetcher=fetcher,
        )
        results = client.search("foo")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://example.com/x")

    def test_caches_subsequent_queries(self) -> None:
        payload = {"results": [{"title": "A", "url": "https://example.com/a", "content": "one"}]}
        urls, fetcher = self._stub_fetcher(payload)
        client = WebSearchClient(
            WebSearchConfig(base_url="https://search.example/api", cache_seconds=600),
            memory=self.memory,
            fetcher=fetcher,
            clock=lambda: 1000.0,
        )
        first = client.search("кава")
        second = client.search("кава")
        self.assertEqual(len(urls), 1)
        self.assertEqual([item.url for item in first], [item.url for item in second])

    def test_cache_expires_after_window(self) -> None:
        payload = {"results": [{"title": "A", "url": "https://example.com/a"}]}
        urls, fetcher = self._stub_fetcher(payload)
        clock_value = {"now": 1000.0}
        client = WebSearchClient(
            WebSearchConfig(base_url="https://search.example/api", cache_seconds=60),
            memory=self.memory,
            fetcher=fetcher,
            clock=lambda: clock_value["now"],
        )
        client.search("кава")
        clock_value["now"] = 9999.0
        client.search("кава")
        self.assertEqual(len(urls), 2)

    def test_truncates_long_snippets(self) -> None:
        long_text = "a" * 1000
        payload = {"results": [{"title": "A", "url": "https://example.com/a", "content": long_text}]}
        _, fetcher = self._stub_fetcher(payload)
        client = WebSearchClient(
            WebSearchConfig(base_url="https://search.example/api", snippet_max_chars=100),
            memory=self.memory,
            fetcher=fetcher,
        )
        results = client.search("foo")
        self.assertLessEqual(len(results[0].snippet), 100)
        self.assertTrue(results[0].snippet.endswith("…"))

    def test_cache_key_uses_kv_prefix(self) -> None:
        payload = {"results": [{"title": "A", "url": "https://example.com/a"}]}
        _, fetcher = self._stub_fetcher(payload)
        client = WebSearchClient(
            WebSearchConfig(base_url="https://search.example/api", cache_seconds=60),
            memory=self.memory,
            fetcher=fetcher,
        )
        client.search("foo")
        with self.memory.connect() as conn:
            keys = [
                row[0]
                for row in conn.execute(
                    "SELECT key FROM kv WHERE key LIKE ?", (CACHE_KV_PREFIX + "%",)
                )
            ]
        self.assertEqual(len(keys), 1)


if __name__ == "__main__":
    unittest.main()
