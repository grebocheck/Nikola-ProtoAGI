"""Memory recall evaluation harness.

The harness loads a corpus of facts and a list of probe queries, plays them
back against ``MemoryService.recall``, and reports recall@k, MRR, and a
per-query breakdown. The default corpus is bundled in
``config/memory_eval/golden.json`` so ``protoagi memory-eval`` works out of
the box even without a project-specific dataset.
"""

from __future__ import annotations

import json
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..config import PROJECT_ROOT
from ..embedding import EmbeddingClient
from ..storage.memory import MemoryStore
from ..storage.service import MemoryService, RecallQuery, StoredMemory


DEFAULT_CORPUS_PATH = PROJECT_ROOT / "config" / "memory_eval" / "golden.json"


@dataclass(slots=True)
class EvalFact:
    text: str
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    section: str = "friendly"
    supersedes: list[str] = field(default_factory=list)
    media: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalQuery:
    query: str
    expected_substrings: list[str]
    require_tags: tuple[str, ...] = ()
    description: str = ""
    section: str = "friendly"


@dataclass(slots=True)
class QueryReport:
    query: str
    description: str
    section: str
    rank: int | None
    hit_at_k: dict[int, bool]
    retrieved: list[str]


@dataclass(slots=True)
class EvalReport:
    queries: list[QueryReport]
    recall_at_k: dict[int, float]
    mrr: float
    section_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "queries": len(self.queries),
                "mrr": round(self.mrr, 4),
                "recall_at_k": {str(k): round(v, 4) for k, v in self.recall_at_k.items()},
            },
            "sections": self.section_metrics,
            "queries": [
                {
                    "query": item.query,
                    "description": item.description,
                    "section": item.section,
                    "rank": item.rank,
                    "hit_at_k": {str(k): hit for k, hit in item.hit_at_k.items()},
                    "retrieved": item.retrieved,
                }
                for item in self.queries
            ],
        }


def load_corpus(path: Path | None = None) -> tuple[list[EvalFact], list[EvalQuery]]:
    target = path or DEFAULT_CORPUS_PATH
    data = json.loads(target.read_text(encoding="utf-8"))
    facts: list[EvalFact] = []
    queries: list[EvalQuery] = []

    def add_fact(item: dict[str, Any], default_section: str) -> None:
        text = str(item.get("text", "")).strip()
        if not text:
            return
        facts.append(
            EvalFact(
                text=text,
                tags=[str(tag) for tag in item.get("tags", [])],
                importance=float(item.get("importance", 0.5)),
                section=str(item.get("section") or default_section or "friendly"),
                supersedes=[str(value) for value in item.get("supersedes", [])],
                media=dict(item.get("media") or {}) if isinstance(item.get("media"), dict) else {},
            )
        )

    def add_query(item: dict[str, Any], default_section: str) -> None:
        query = str(item.get("query", "")).strip()
        if not query:
            return
        queries.append(
            EvalQuery(
                query=query,
                expected_substrings=[str(value) for value in item.get("expected_substrings", [])],
                require_tags=tuple(str(tag) for tag in item.get("require_tags", [])),
                description=str(item.get("description", "")),
                section=str(item.get("section") or default_section or "friendly"),
            )
        )

    for item in data.get("facts", []):
        if isinstance(item, dict):
            add_fact(item, "friendly")
    for item in data.get("queries", []):
        if isinstance(item, dict):
            add_query(item, "friendly")
    for section in data.get("sections", []):
        if not isinstance(section, dict):
            continue
        name = str(section.get("name") or "adversarial")
        for item in section.get("facts", []):
            if isinstance(item, dict):
                add_fact(item, name)
        for item in section.get("queries", []):
            if isinstance(item, dict):
                add_query(item, name)
    return facts, queries


def build_eval_service(
    facts: Iterable[EvalFact],
    *,
    embedding_client: EmbeddingClient | None = None,
    db_path: Path | None = None,
) -> tuple[MemoryStore, MemoryService]:
    if db_path is None:
        tmp = tempfile.mkdtemp(prefix="protoagi_eval_")
        db_path = Path(tmp) / "memory.sqlite3"
    store = MemoryStore(db_path)
    service = MemoryService(store, embedding_client=embedding_client)
    stored_items: list[StoredMemory] = []
    for fact in facts:
        media_id = _store_eval_media(store, fact)
        stored = service.remember(
            fact.text,
            tags=fact.tags,
            importance=fact.importance,
            media_id=media_id,
        )
        if stored is None:
            continue
        if fact.supersedes:
            lowered_needles = [needle.lower() for needle in fact.supersedes if needle]
            for previous in stored_items:
                if any(needle in previous.item.text.lower() for needle in lowered_needles):
                    store.supersede(previous.memory_id, stored.memory_id)
        stored_items.append(stored)
    return store, service


def _store_eval_media(store: MemoryStore, fact: EvalFact) -> str | None:
    if not fact.media:
        return None
    file_id = str(fact.media.get("file_id") or "").strip()
    data_b64 = str(fact.media.get("bytes_b64") or "").strip()
    if not file_id or not data_b64:
        return None
    import base64

    try:
        data = base64.b64decode(data_b64)
    except ValueError:
        return None
    if not data:
        return None
    store.store_media_blob(
        file_id=file_id,
        mime=str(fact.media.get("mime") or "application/octet-stream"),
        data=data,
        caption=str(fact.media.get("caption") or ""),
    )
    return file_id


def evaluate(
    queries: Iterable[EvalQuery],
    service: MemoryService,
    *,
    k_values: tuple[int, ...] = (1, 3, 5),
    limit: int | None = None,
) -> EvalReport:
    k_values = tuple(sorted(set(k_values)))
    request_limit = max(limit or 0, max(k_values))
    reports: list[QueryReport] = []
    for query in queries:
        recall_results = service.recall(
            RecallQuery(
                text=query.query,
                require_tags=query.require_tags,
                limit=request_limit,
            )
        )
        retrieved_texts = [result.item.text for result in recall_results]
        rank: int | None = None
        for index, text in enumerate(retrieved_texts, start=1):
            if any(needle.lower() in text.lower() for needle in query.expected_substrings):
                rank = index
                break
        hit_at_k = {k: (rank is not None and rank <= k) for k in k_values}
        reports.append(
            QueryReport(
                query=query.query,
                description=query.description,
                section=query.section,
                rank=rank,
                hit_at_k=hit_at_k,
                retrieved=retrieved_texts[:max(k_values)],
            )
        )
    if reports:
        recall_at_k = {
            k: statistics.mean(1.0 if report.hit_at_k.get(k) else 0.0 for report in reports)
            for k in k_values
        }
        mrr = statistics.mean(
            (1.0 / report.rank) if report.rank else 0.0 for report in reports
        )
    else:
        recall_at_k = {k: 0.0 for k in k_values}
        mrr = 0.0
    return EvalReport(
        queries=reports,
        recall_at_k=recall_at_k,
        mrr=mrr,
        section_metrics=_section_metrics(reports, k_values),
    )


def _section_metrics(
    reports: Iterable[QueryReport],
    k_values: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[QueryReport]] = {}
    for report in reports:
        grouped.setdefault(report.section or "friendly", []).append(report)
    metrics: dict[str, dict[str, Any]] = {}
    for section, items in sorted(grouped.items()):
        if not items:
            continue
        metrics[section] = {
            "queries": len(items),
            "mrr": round(
                statistics.mean((1.0 / item.rank) if item.rank else 0.0 for item in items),
                4,
            ),
            "recall_at_k": {
                str(k): round(
                    statistics.mean(1.0 if item.hit_at_k.get(k) else 0.0 for item in items),
                    4,
                )
                for k in k_values
            },
        }
    return metrics


def run_eval(
    *,
    corpus_path: Path | None = None,
    embedding_client: EmbeddingClient | None = None,
    k_values: tuple[int, ...] = (1, 3, 5),
) -> EvalReport:
    facts, queries = load_corpus(corpus_path)
    _, service = build_eval_service(facts, embedding_client=embedding_client)
    return evaluate(queries, service, k_values=k_values)


__all__ = [
    "DEFAULT_CORPUS_PATH",
    "EvalFact",
    "EvalQuery",
    "EvalReport",
    "QueryReport",
    "build_eval_service",
    "evaluate",
    "load_corpus",
    "run_eval",
]
