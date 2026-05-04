"""Optional embedding client backed by an OpenAI-compatible /v1/embeddings endpoint.

The client is built so that ProtoAGI degrades gracefully when no embedding
server is configured: callers receive ``None`` and recall falls back to FTS
only. When an embedding server is configured (for example a llama-server
running ``bge-m3`` or ``nomic-embed-text`` in embedding mode) every memory
write attaches a vector and recall combines BM25 with cosine similarity.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .memory import MemoryStore, cosine_similarity


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
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    def embed(self, text: str) -> list[float] | None:
        text = text.strip()
        if not text or not self.config.enabled:
            return None
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return list(cached)
        try:
            vector = self._request(text)
        except EmbeddingError:
            return None
        self._cache_set(text, vector)
        return vector

    def embed_many(self, texts: Sequence[str]) -> list[list[float] | None]:
        return [self.embed(text) for text in texts]

    def embed_media(self, data: bytes, *, mime: str, caption: str = "") -> list[float] | None:
        if not data or not self.config.enabled:
            return None
        digest = hashlib.sha256(data).hexdigest()
        cache_key = f"media:{mime}:{digest}:{caption.strip()}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return list(cached)
        try:
            vector = self._request_media(data, mime=mime, caption=caption)
        except EmbeddingError:
            return None
        self._cache_set(cache_key, vector)
        return vector

    def _request(self, text: str) -> list[float]:
        url = self.config.base_url.rstrip("/") + "/embeddings"
        payload = {"model": self.config.model, "input": text}
        return self._request_payload(url, payload)

    def _request_media(self, data: bytes, *, mime: str, caption: str = "") -> list[float]:
        url = self.config.base_url.rstrip("/") + "/embeddings"
        image_payload = {
            "type": "image",
            "mime_type": mime or "application/octet-stream",
            "data": base64.b64encode(data).decode("ascii"),
        }
        if caption.strip():
            image_payload["caption"] = caption.strip()
        payload = {"model": self.config.model, "input": [image_payload]}
        return self._request_payload(url, payload)

    def _request_payload(self, url: str, payload: dict[str, object]) -> list[float]:
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
            self._cache[text] = list(vector)
            self._cache.move_to_end(text)
            return
        self._cache[text] = list(vector)
        while len(self._cache) > self.config.cache_size:
            self._cache.popitem(last=False)


class EmbeddingBackend:
    """Backend interface for vector search over stored embeddings."""

    def add(self, memory_id: int, vector: Sequence[float]) -> None:
        raise NotImplementedError

    def remove(self, memory_id: int) -> None:
        raise NotImplementedError

    def refresh(self) -> None:
        raise NotImplementedError

    def similar(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
        candidate_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        raise NotImplementedError


class FlatEmbeddingBackend(EmbeddingBackend):
    """Exact cosine scan, kept as the deterministic baseline."""

    def __init__(self, store: MemoryStore, *, model: str | None = None) -> None:
        self.store = store
        self.model = model
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

    def similar(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
        candidate_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
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


@dataclass(slots=True)
class LSHSettings:
    planes: int = 14
    min_candidates: int = 512
    max_candidates: int = 4096
    seed: str = "protoagi-lsh-v1"
    _plane_cache: dict[tuple[int, int], list[float]] = field(default_factory=dict)


class LSHEmbeddingBackend(FlatEmbeddingBackend):
    """Small pure-Python approximate backend based on random-hyperplane LSH.

    This is not a full HNSW implementation, but it gives ``EmbeddingIndex`` a
    real backend boundary and a dependency-free path that avoids scanning every
    vector once memory grows beyond experiment size. Exact re-ranking still
    happens within the selected candidate bucket.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        model: str | None = None,
        settings: LSHSettings | None = None,
    ) -> None:
        super().__init__(store, model=model)
        self.settings = settings or LSHSettings()
        self._buckets: dict[int, set[int]] = {}

    def add(self, memory_id: int, vector: Sequence[float]) -> None:
        super().add(memory_id, vector)
        bucket = self._hash(vector)
        self._buckets.setdefault(bucket, set()).add(int(memory_id))

    def remove(self, memory_id: int) -> None:
        memory_id = int(memory_id)
        vector = self._vectors.get(memory_id)
        if vector is not None:
            bucket = self._hash(vector)
            values = self._buckets.get(bucket)
            if values is not None:
                values.discard(memory_id)
                if not values:
                    self._buckets.pop(bucket, None)
        super().remove(memory_id)

    def refresh(self) -> None:
        super().refresh()
        self._buckets = {}
        for memory_id, vector in self._vectors.items():
            self._buckets.setdefault(self._hash(vector), set()).add(memory_id)

    def similar(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
        candidate_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if not self._vectors:
            return []
        candidates = self._candidate_ids(query_vector)
        if candidate_ids is not None:
            candidates &= candidate_ids
        if not candidates:
            return []
        return super().similar(query_vector, limit=limit, candidate_ids=candidates)

    def _candidate_ids(self, query_vector: Sequence[float]) -> set[int]:
        primary = self._hash(query_vector)
        candidates = set(self._buckets.get(primary, set()))
        if len(candidates) >= self.settings.min_candidates:
            return set(list(candidates)[: self.settings.max_candidates])
        # Add nearby buckets by flipping one bit. This keeps recall sane for
        # small stores without falling all the way back to a global scan.
        for bit in range(self.settings.planes):
            candidates.update(self._buckets.get(primary ^ (1 << bit), set()))
            if len(candidates) >= self.settings.max_candidates:
                break
        if not candidates and len(self._vectors) <= self.settings.max_candidates:
            return set(self._vectors)
        return set(list(candidates)[: self.settings.max_candidates])

    def _hash(self, vector: Sequence[float]) -> int:
        bits = 0
        dim = len(vector)
        for plane_index in range(self.settings.planes):
            plane = self._plane(dim, plane_index)
            dot = sum(float(left) * right for left, right in zip(vector, plane))
            if dot >= 0:
                bits |= 1 << plane_index
        return bits

    def _plane(self, dim: int, plane_index: int) -> list[float]:
        key = (dim, plane_index)
        cached = self.settings._plane_cache.get(key)
        if cached is not None:
            return cached
        values: list[float] = []
        for axis in range(dim):
            digest = hashlib.sha256(
                f"{self.settings.seed}:{dim}:{plane_index}:{axis}".encode("utf-8")
            ).digest()
            # Deterministic pseudo-random hyperplane component in [-1, 1].
            raw = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
            values.append((raw * 2.0) - 1.0)
        self.settings._plane_cache[key] = values
        return values


class EmbeddingIndex:
    """In-memory cache of all active embeddings for fast cosine recall.

    SQLite returns a few hundred kB of vectors quickly enough for an
    experiment-sized memory, but we still avoid hitting disk on the hot path
    by caching everything once and refreshing lazily.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        model: str | None = None,
        backend: str | EmbeddingBackend = "flat",
    ) -> None:
        self.store = store
        self.model = model
        self._loaded_at = 0.0
        self.backend = backend if isinstance(backend, EmbeddingBackend) else self._make_backend(backend)

    def add(self, memory_id: int, vector: Sequence[float]) -> None:
        self.backend.add(memory_id, vector)

    def remove(self, memory_id: int) -> None:
        self.backend.remove(memory_id)

    def refresh(self) -> None:
        self.backend.refresh()
        self._loaded_at = time.time()

    def ensure_loaded(self) -> None:
        if self._loaded_at == 0.0:
            self.refresh()

    def similar(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 20,
        candidate_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        self.ensure_loaded()
        return self.backend.similar(
            query_vector,
            limit=limit,
            candidate_ids=candidate_ids,
        )

    def _make_backend(self, backend: str) -> EmbeddingBackend:
        normalized = backend.strip().lower()
        if normalized in {"lsh", "hnsw", "auto"}:
            return LSHEmbeddingBackend(self.store, model=self.model)
        return FlatEmbeddingBackend(self.store, model=self.model)


__all__ = [
    "EmbeddingClient",
    "EmbeddingConfig",
    "EmbeddingError",
    "EmbeddingBackend",
    "EmbeddingIndex",
    "FlatEmbeddingBackend",
    "LSHEmbeddingBackend",
    "LSHSettings",
]
