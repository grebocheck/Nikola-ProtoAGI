"""Optional web-search tool backed by a SearxNG-style JSON endpoint.

The Telegram persona occasionally needs fresh facts ("що сьогодні з курсом …")
but the local model has no internet of its own. ``WebSearchClient`` issues a
GET against an operator-controlled endpoint that follows the SearxNG
``?q=&format=json`` contract, reuses the SSRF guard from
``protoagi.agent_tools.core``, caches results in the existing ``kv`` table,
and returns a small list of trimmed snippets.

Disabled deployments (no ``base_url``) raise ``WebSearchUnavailable`` from
``search``; callers are expected to gate the tool schema on
``WebSearchClient.enabled``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import URLError
from urllib.parse import quote_plus

from .agent_tools.core import _fetch_public_url
from .storage.memory import MemoryStore


CACHE_KV_PREFIX = "web_search:cache:"


class WebSearchUnavailable(RuntimeError):
    """Raised when the tool is invoked but no endpoint is configured."""


@dataclass(slots=True, frozen=True)
class WebSearchConfig:
    base_url: str = ""
    timeout_seconds: int = 10
    max_results: int = 5
    cache_seconds: int = 900
    snippet_max_chars: int = 320

    @property
    def enabled(self) -> bool:
        return bool(self.base_url.strip())


@dataclass(slots=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str

    def as_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


class WebSearchClient:
    def __init__(
        self,
        config: WebSearchConfig,
        *,
        memory: MemoryStore | None = None,
        clock: Callable[[], float] | None = None,
        fetcher: Callable[[str, int], tuple[str, bytes]] | None = None,
    ) -> None:
        self.config = config
        self.memory = memory
        self._clock = clock or time.time
        self._fetcher = fetcher or self._default_fetch

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def search(self, query: str, *, limit: int | None = None) -> list[WebSearchResult]:
        if not self.config.enabled:
            raise WebSearchUnavailable("web_search is disabled (no PROTOAGI_WEB_SEARCH_URL)")
        clean_query = (query or "").strip()
        if not clean_query:
            return []
        wanted = limit if isinstance(limit, int) and limit > 0 else self.config.max_results
        wanted = max(1, min(wanted, self.config.max_results))
        cached = self._read_cache(clean_query)
        if cached is not None:
            return cached[:wanted]
        url = self._build_url(clean_query)
        max_chars = max(8192, wanted * 2048)
        try:
            content_type, raw = self._fetcher(url, max_chars)
        except URLError as exc:
            raise WebSearchUnavailable(f"web_search fetch failed: {exc}") from exc
        results = self._parse_response(content_type, raw, wanted)
        self._write_cache(clean_query, results)
        return results

    def _default_fetch(self, url: str, max_chars: int) -> tuple[str, bytes]:
        return _fetch_public_url(url, max_chars=max_chars)

    def _build_url(self, query: str) -> str:
        base = self.config.base_url.rstrip("/")
        separator = "&" if "?" in base else "?"
        return f"{base}{separator}q={quote_plus(query)}&format=json"

    def _parse_response(self, content_type: str, raw: bytes, wanted: int) -> list[WebSearchResult]:
        text = raw.decode("utf-8", errors="replace").strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise WebSearchUnavailable(f"web_search returned non-JSON ({content_type})") from exc
        items = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return []
        results: list[WebSearchResult] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            url = str(entry.get("url") or entry.get("link") or "").strip()
            if not url:
                continue
            title = str(entry.get("title") or "").strip()
            snippet = str(
                entry.get("content") or entry.get("snippet") or entry.get("description") or ""
            ).strip()
            if len(snippet) > self.config.snippet_max_chars:
                snippet = snippet[: self.config.snippet_max_chars - 1].rstrip() + "…"
            results.append(WebSearchResult(title=title, url=url, snippet=snippet))
            if len(results) >= wanted:
                break
        return results

    def _cache_key(self, query: str) -> str:
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:24]
        return CACHE_KV_PREFIX + digest

    def _read_cache(self, query: str) -> list[WebSearchResult] | None:
        if self.memory is None or self.config.cache_seconds <= 0:
            return None
        raw = self.memory.get_kv(self._cache_key(query))
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        stored_at = float(payload.get("at", 0.0))
        if self._clock() - stored_at > self.config.cache_seconds:
            return None
        items = payload.get("results")
        if not isinstance(items, list):
            return None
        out: list[WebSearchResult] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            out.append(
                WebSearchResult(
                    title=str(entry.get("title", "")),
                    url=str(entry.get("url", "")),
                    snippet=str(entry.get("snippet", "")),
                )
            )
        return out

    def _write_cache(self, query: str, results: list[WebSearchResult]) -> None:
        if self.memory is None or self.config.cache_seconds <= 0:
            return
        payload: dict[str, Any] = {
            "at": self._clock(),
            "results": [item.as_dict() for item in results],
        }
        self.memory.set_kv(self._cache_key(query), json.dumps(payload, ensure_ascii=False))


__all__ = [
    "CACHE_KV_PREFIX",
    "WebSearchClient",
    "WebSearchConfig",
    "WebSearchResult",
    "WebSearchUnavailable",
]
