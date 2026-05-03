"""Memory recall evaluation harness.

The harness loads a corpus of facts and a list of probe queries, plays them
back against ``MemoryService.recall``, and reports recall@k, MRR, and a
per-query breakdown. The default corpus is bundled in
``config/memory_eval/golden.json`` so ``protoagi memory-eval`` works out of
the box even without a project-specific dataset.
"""

from __future__ import annotations

import json
import math
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import PROJECT_ROOT
from .embedding import EmbeddingClient
from .memory import MemoryStore
from .memory_service import MemoryService, RecallQuery


DEFAULT_CORPUS_PATH = PROJECT_ROOT / "config" / "memory_eval" / "golden.json"


@dataclass(slots=True)
class EvalFact:
    text: str
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5


@dataclass(slots=True)
class EvalQuery:
    query: str
    expected_substrings: list[str]
    require_tags: tuple[str, ...] = ()
    description: str = ""


@dataclass(slots=True)
class QueryReport:
    query: str
    description: str
    rank: int | None
    hit_at_k: dict[int, bool]
    retrieved: list[str]


@dataclass(slots=True)
class EvalReport:
    queries: list[QueryReport]
    recall_at_k: dict[int, float]
    mrr: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "queries": len(self.queries),
                "mrr": round(self.mrr, 4),
                "recall_at_k": {str(k): round(v, 4) for k, v in self.recall_at_k.items()},
            },
            "queries": [
                {
                    "query": item.query,
                    "description": item.description,
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
    facts = [
        EvalFact(
            text=str(item["text"]).strip(),
            tags=[str(tag) for tag in item.get("tags", [])],
            importance=float(item.get("importance", 0.5)),
        )
        for item in data.get("facts", [])
        if str(item.get("text", "")).strip()
    ]
    queries = [
        EvalQuery(
            query=str(item["query"]).strip(),
            expected_substrings=[str(value) for value in item.get("expected_substrings", [])],
            require_tags=tuple(str(tag) for tag in item.get("require_tags", [])),
            description=str(item.get("description", "")),
        )
        for item in data.get("queries", [])
        if str(item.get("query", "")).strip()
    ]
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
    for fact in facts:
        service.remember(
            fact.text,
            tags=fact.tags,
            importance=fact.importance,
        )
    return store, service


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
    return EvalReport(queries=reports, recall_at_k=recall_at_k, mrr=mrr)


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
