"""Optional embedding client backed by an OpenAI-compatible /v1/embeddings endpoint.

The client is built so that ProtoAGI degrades gracefully when no embedding
server is configured: callers receive ``None`` and recall falls back to FTS
only. When an embedding server is configured (for example a llama-server
running ``bge-m3`` or ``nomic-embed-text`` in embedding mode) every memory
write attaches a vector and recall combines BM25 with cosine similarity.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .memory import MemoryStore, cosine_similarity, pack_embedding, unpack_embedding


class EmbeddingError(RuntimeError):
    pass


@dataclass(slots=True)
class EmbeddingConfig:
    base_url: str = ""
    model: str = ""
    timeout_seconds: int = 30
    cache_size: int = 1024
    request_dimensions: int | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)


class EmbeddingClient:
    """Tiny dependency-free embedding client with a tiny LRU cache."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._cache: dict[str, list[float]] = {}
        self._cache_keys: list[str] = []

    def embed(self, text: str) -> list[float] | None:
        text = text.strip()
        if not text or not self.config.enabled:
            return None
        cached = self._cache.get(text)
        if cached is not None:
            return list(cached)
        try:
            vector = self._request(text)
        except EmbeddingError:
            return None
        self._cache_set(text, vector)
        return vector

    def embed_many(self, texts: Sequence[str]) -> list[list[float] | None]:
        return [self.embed(text) for text in texts]

    def _request(self, text: str) -> list[float]:
        url = self.config.base_url.rstrip("/") + "/embeddings"
        payload = {"model": self.config.model, "input": text}
        if self.config.request_dimensions:
            payload["dimensions"] = self.config.request_dimensions
        body = json.dumps(payload).encode("utf-8")
        request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise EmbeddingError(f"embedding HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise EmbeddingError(f"embedding network error: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise EmbeddingError("embedding endpoint returned non-JSON") from exc
        items = data.get("data") or []
        if not items:
            raise EmbeddingError("embedding endpoint returned no data")
        vector = items[0].get("embedding") or []
        if not isinstance(vector, list) or not vector:
            raise EmbeddingError("embedding endpoint returned invalid vector")
        return [float(value) for value in vector]

    def _cache_set(self, text: str, vector: list[float]) -> None:
        if self.config.cache_size <= 0:
            return
        if text in self._cache:
            return
        self._cache[text] = list(vector)
        self._cache_keys.append(text)
        while len(self._cache_keys) > self.config.cache_size:
            evict = self._cache_keys.pop(0)
            self._cache.pop(evict, None)


class EmbeddingIndex:
    """In-memory cache of all active embeddings for fast cosine recall.

    SQLite returns a few hundred kB of vectors quickly enough for an
    experiment-sized memory, but we still avoid hitting disk on the hot path
    by caching everything once and refreshing lazily.
    """

    def __init__(self, store: MemoryStore, *, model: str | None = None) -> None:
        self.store = store
        self.model = model
        self._loaded_at = 0.0
        self._vectors: dict[int, list[float]] = {}

    def add(self, memory_id: int, vector: Sequence[float]) -> None:
        self._vectors[int(memory_id)] = list(vector)

    def remove(self, memory_id: int) -> None:
        self._vectors.pop(int(memory_id), None)

    def refresh(self) -> None:
        self._vectors = {
            memory_id: vector
            for memory_id, vector in self.store.all_embeddings(model=self.model)
        }
        self._loaded_at = time.time()

    def ensure_loaded(self) -> None:
        if not self._vectors and self._loaded_at == 0.0:
            self.refresh()

    def similar(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 20,
        candidate_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        self.ensure_loaded()
        if not self._vectors:
            return []
        scored: list[tuple[int, float]] = []
        for memory_id, vector in self._vectors.items():
            if candidate_ids is not None and memory_id not in candidate_ids:
                continue
            score = cosine_similarity(query_vector, vector)
            if score <= 0:
                continue
            scored.append((memory_id, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]


__all__ = [
    "EmbeddingClient",
    "EmbeddingConfig",
    "EmbeddingError",
    "EmbeddingIndex",
]
