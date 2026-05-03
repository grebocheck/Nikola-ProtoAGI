"""High-level memory facade used by the agent and Telegram bot.

The service owns three responsibilities that the raw ``MemoryStore`` should
not care about:

1. Hybrid recall: combine FTS (``BM25``-style) signals with cosine similarity
   over optional embeddings, then re-rank by importance and recency.
2. Importance scoring: a transparent heuristic that gives sensible defaults
   without a model call. Callers can override the score when the model already
   produced one.
3. Consolidation hooks: dedupe near-duplicates and supersede contradicting
   facts. The simple version we ship here is heuristic; a richer LLM-driven
   pass can be added on top later.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Sequence

from .embedding import EmbeddingClient, EmbeddingIndex
from .memory import (
    ALL_KINDS,
    ALL_SCOPES,
    KIND_FACT,
    KIND_PERSONA_SELF,
    MemoryItem,
    MemoryStore,
    RecallResult,
    SCOPE_CHAT,
    SCOPE_GLOBAL,
    SCOPE_PERSONA,
    SCOPE_USER,
)


PRONOUN_TOKENS = {
    "я", "мене", "мені", "мій", "моя", "моє", "мої",
    "ти", "тебе", "тобі", "твій", "твоя", "твоє", "твої",
    "i", "me", "my", "mine", "you", "your", "yours",
}


@dataclass(slots=True)
class StoredMemory:
    memory_id: int
    item: MemoryItem


@dataclass(slots=True)
class RecallQuery:
    text: str
    user_id: str | None = None
    chat_id: str | None = None
    persona_key: str | None = None
    require_tags: tuple[str, ...] = ()
    limit: int = 6
    include_global: bool = True
    kinds: tuple[str, ...] = ()


class MemoryService:
    def __init__(
        self,
        store: MemoryStore,
        *,
        embedding_client: EmbeddingClient | None = None,
        embedding_index: EmbeddingIndex | None = None,
    ) -> None:
        self.store = store
        self.embedding_client = embedding_client
        self.embedding_index = embedding_index
        if embedding_client and embedding_client.config.enabled and embedding_index is None:
            self.embedding_index = EmbeddingIndex(store, model=embedding_client.config.model)

    # ------------------------------------------------------------------
    # Writes

    def remember(
        self,
        text: str,
        *,
        kind: str = KIND_FACT,
        scope: str = SCOPE_GLOBAL,
        tags: Iterable[str] | None = None,
        user_id: str | None = None,
        chat_id: str | None = None,
        persona_key: str | None = None,
        importance: float | None = None,
        confidence: float = 0.7,
        source: str | None = None,
        pinned: bool = False,
        metadata: dict | None = None,
        embed: bool = True,
    ) -> StoredMemory | None:
        text = text.strip()
        if not text:
            return None
        if kind not in ALL_KINDS:
            kind = KIND_FACT
        if scope not in ALL_SCOPES:
            scope = SCOPE_GLOBAL

        score = importance if importance is not None else self.score_importance(text, kind=kind)
        vector = None
        embed_model: str | None = None
        if embed and self.embedding_client and self.embedding_client.config.enabled:
            vector = self.embedding_client.embed(text)
            if vector is not None:
                embed_model = self.embedding_client.config.model

        memory_id = self.store.store_memory(
            text,
            kind=kind,
            scope=scope,
            tags=tags,
            user_id=user_id,
            chat_id=chat_id,
            persona_key=persona_key,
            importance=score,
            confidence=confidence,
            source=source,
            pinned=pinned,
            embedding=vector,
            embedding_model=embed_model,
            metadata=metadata,
        )
        if vector is not None and self.embedding_index is not None:
            self.embedding_index.add(memory_id, vector)
        item = self.store.get_memory(memory_id)
        if item is None:
            return None
        return StoredMemory(memory_id=memory_id, item=item)

    # ------------------------------------------------------------------
    # Reads

    def recall(self, query: RecallQuery) -> list[RecallResult]:
        text = query.text.strip()
        require_tags = list(query.require_tags)
        candidates_limit = max(query.limit * 6, 30)
        fts_items = self.store.fts_candidates(
            text,
            limit=candidates_limit,
            require_tags=tuple(require_tags) or None,
        )

        candidate_map: dict[int, MemoryItem] = {item.id: item for item in fts_items}
        cosine_scores: dict[int, float] = {}
        if self.embedding_client and self.embedding_index and self.embedding_client.config.enabled and text:
            query_vector = self.embedding_client.embed(text)
            if query_vector is not None:
                hits = self.embedding_index.similar(query_vector, limit=candidates_limit)
                if hits:
                    missing_ids = [mid for mid, _ in hits if mid not in candidate_map]
                    if missing_ids:
                        loaded = self.store.get_memories(missing_ids)
                        for mid, item in loaded.items():
                            if item.superseded_by is not None:
                                continue
                            if require_tags and not all(tag in item.tags for tag in require_tags):
                                continue
                            candidate_map[mid] = item
                    for mid, score in hits:
                        if mid in candidate_map:
                            cosine_scores[mid] = score

        # FTS contributes a positional score ~ 1/(rank+1).
        fts_scores: dict[int, float] = {item.id: 1.0 / (index + 1) for index, item in enumerate(fts_items)}

        scored: list[RecallResult] = []
        now = datetime.now(timezone.utc)
        for memory_id, item in candidate_map.items():
            if not self._scope_matches(item, query):
                continue
            if query.kinds and item.kind not in query.kinds:
                continue
            bm25 = fts_scores.get(memory_id, 0.0)
            cosine = cosine_scores.get(memory_id, 0.0)
            recency = self._recency_score(item.created_at, now)
            importance = item.importance
            pinned_bonus = 0.2 if item.pinned else 0.0
            blended = (
                0.45 * cosine
                + 0.30 * bm25
                + 0.15 * importance
                + 0.10 * recency
                + pinned_bonus
            )
            scored.append(
                RecallResult(item=item, score=blended, bm25=bm25, cosine=cosine)
            )
        scored.sort(key=lambda result: result.score, reverse=True)
        results = scored[: query.limit]
        if results:
            self.store.mark_accessed(result.item.id for result in results)
        return results

    def recent(
        self,
        *,
        scope: str | None = None,
        user_id: str | None = None,
        chat_id: str | None = None,
        persona_key: str | None = None,
        kind: str | None = None,
        limit: int = 5,
    ) -> list[MemoryItem]:
        return self.store.list_memories(
            scope=scope,
            user_id=user_id,
            chat_id=chat_id,
            persona_key=persona_key,
            kind=kind,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # Maintenance

    def reembed_missing(self) -> int:
        """Generate embeddings for items that don't yet have one."""

        if not self.embedding_client or not self.embedding_client.config.enabled:
            return 0
        active_ids = self.store.all_active_memory_ids()
        if not active_ids:
            return 0
        embedded = {memory_id for memory_id, _ in self.store.all_embeddings()}
        missing = [item for item in active_ids if item not in embedded]
        if not missing:
            return 0
        items = self.store.get_memories(missing)
        added = 0
        model = self.embedding_client.config.model
        for memory_id, item in items.items():
            vector = self.embedding_client.embed(item.text)
            if vector is None:
                continue
            self.store.attach_embedding(memory_id, vector, model=model)
            if self.embedding_index is not None:
                self.embedding_index.add(memory_id, vector)
            added += 1
        return added

    def consolidate(
        self,
        *,
        scope: str | None = None,
        chat_id: str | None = None,
        persona_key: str | None = None,
        similarity_threshold: float = 0.92,
        max_items: int = 200,
    ) -> int:
        """Heuristic consolidation: merge near-duplicate items.

        Walks recent memory items in the requested scope, keeps the latest
        higher-importance version, and supersedes the older duplicate.
        Returns the number of supersessions performed.
        """

        items = self.store.list_memories(
            scope=scope,
            chat_id=chat_id,
            persona_key=persona_key,
            limit=max_items,
        )
        items.sort(key=lambda item: item.id)  # oldest first
        normalized = [(item, _normalize_text(item.text)) for item in items]
        merges = 0
        active: list[tuple[MemoryItem, str]] = []
        for item, norm in normalized:
            duplicate = None
            for kept_item, kept_norm in active:
                if not _signature_match(norm, kept_norm):
                    continue
                if (
                    self.embedding_client
                    and self.embedding_index
                    and self.embedding_client.config.enabled
                ):
                    vec_a = self.embedding_client.embed(item.text)
                    vec_b = self.embedding_client.embed(kept_item.text)
                    if vec_a and vec_b:
                        from .memory import cosine_similarity

                        if cosine_similarity(vec_a, vec_b) < similarity_threshold:
                            continue
                duplicate = kept_item
                break
            if duplicate is None:
                active.append((item, norm))
                continue
            # Choose which one wins: higher importance, or newer if equal.
            if item.importance >= duplicate.importance:
                self.store.supersede(duplicate.id, item.id)
                if self.embedding_index is not None:
                    self.embedding_index.remove(duplicate.id)
                active = [
                    (existing_item, existing_norm)
                    for existing_item, existing_norm in active
                    if existing_item.id != duplicate.id
                ]
                active.append((item, norm))
            else:
                self.store.supersede(item.id, duplicate.id)
                if self.embedding_index is not None:
                    self.embedding_index.remove(item.id)
            merges += 1
        return merges

    # ------------------------------------------------------------------
    # Importance heuristic

    def score_importance(self, text: str, *, kind: str = KIND_FACT) -> float:
        text = (text or "").strip()
        if not text:
            return 0.1
        words = text.split()
        score = 0.4
        if kind == KIND_PERSONA_SELF:
            score += 0.1
        if any(token in PRONOUN_TOKENS for token in (word.lower().strip(".,!?:;") for word in words)):
            score += 0.05
        if any(symbol in text for symbol in ("@", "+38", "паспорт", "адреса", "ключ")):
            score -= 0.2
        if any(token in text.lower() for token in ("люблю", "ненавиджу", "обіцяв", "домовились", "зрозумів", "пам", "always", "never")):
            score += 0.15
        if len(words) < 3:
            score -= 0.1
        if len(words) > 24:
            score += 0.05
        return max(0.05, min(1.0, score))

    # ------------------------------------------------------------------
    # Helpers

    @staticmethod
    def _scope_matches(item: MemoryItem, query: RecallQuery) -> bool:
        if query.user_id and item.user_id and item.user_id != query.user_id:
            # User-scoped memory must belong to the right principal.
            if item.scope == SCOPE_USER:
                return False
        if query.chat_id and item.scope == SCOPE_CHAT:
            if item.chat_id and str(item.chat_id) != str(query.chat_id):
                return False
        if query.persona_key and item.scope == SCOPE_PERSONA:
            if item.persona_key and item.persona_key != query.persona_key:
                return False
        if not query.include_global and item.scope == SCOPE_GLOBAL:
            return False
        return True

    @staticmethod
    def _recency_score(created_at: str, now: datetime) -> float:
        try:
            created = datetime.fromisoformat(created_at)
        except ValueError:
            return 0.0
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        delta_days = max(0.0, (now - created).total_seconds() / 86400.0)
        # 90-day half life.
        return math.exp(-delta_days / 90.0)


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\"'`.,!?:;()\[\]\{\}]")


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _signature_match(left: str, right: str) -> bool:
    """Cheap pre-filter for consolidation candidates."""

    if not left or not right:
        return False
    if left == right:
        return True
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return False
    overlap = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(overlap) / len(union) >= 0.7


__all__ = [
    "MemoryService",
    "RecallQuery",
    "StoredMemory",
]
