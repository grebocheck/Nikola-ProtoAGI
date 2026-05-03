from __future__ import annotations

import hmac
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .memory import MemoryItem, MemoryStore, utc_now


FEDERATION_FORMAT = "protoagi.memory_federation.v1"


class MemoryFederationError(RuntimeError):
    pass


@dataclass(slots=True)
class FederationExportResult:
    path: Path
    exported: int
    signature: str


@dataclass(slots=True)
class FederationImportResult:
    imported: int
    skipped: int
    source: str


def export_memory_bundle(
    store: MemoryStore,
    path: Path,
    *,
    secret: str,
    source: str = "protoagi",
    scope: str | None = None,
    require_tags: Iterable[str] = (),
    limit: int = 1000,
) -> FederationExportResult:
    if not secret:
        raise MemoryFederationError("federation secret is required")
    tag_filter = tuple(str(tag).strip() for tag in require_tags if str(tag).strip())
    items = [
        item
        for item in store.list_memories(scope=scope, include_superseded=False, limit=limit)
        if not tag_filter or all(tag in item.tags for tag in tag_filter)
    ]
    payload: dict[str, Any] = {
        "format": FEDERATION_FORMAT,
        "created_at": utc_now(),
        "source": source,
        "items": [_export_item(item, source=source) for item in items],
    }
    signature = _signature(payload, secret)
    payload["signature"] = signature
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return FederationExportResult(path=path, exported=len(items), signature=signature)


def import_memory_bundle(
    store: MemoryStore,
    path: Path,
    *,
    secret: str,
) -> FederationImportResult:
    if not secret:
        raise MemoryFederationError("federation secret is required")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MemoryFederationError(f"could not read federation bundle: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != FEDERATION_FORMAT:
        raise MemoryFederationError("unsupported federation bundle")
    expected = str(payload.get("signature") or "")
    unsigned = dict(payload)
    unsigned.pop("signature", None)
    actual = _signature(unsigned, secret)
    if not hmac.compare_digest(expected, actual):
        raise MemoryFederationError("federation signature mismatch")
    imported = 0
    skipped = 0
    source = str(payload.get("source") or "unknown")
    for raw in payload.get("items") or []:
        if not isinstance(raw, dict):
            skipped += 1
            continue
        federation_id = str(raw.get("federation_id") or "").strip()
        if federation_id and store.recent_tagged_all([f"federated_id:{federation_id}"], limit=1):
            skipped += 1
            continue
        text = str(raw.get("text") or "").strip()
        if not text:
            skipped += 1
            continue
        tags = [str(tag) for tag in raw.get("tags") or [] if str(tag).strip()]
        tags.extend(["federated", f"federated_source:{_safe_tag(source)}"])
        if federation_id:
            tags.append(f"federated_id:{federation_id}")
        store.store_memory(
            text,
            kind=str(raw.get("kind") or "fact"),
            scope=str(raw.get("scope") or "global"),
            tags=tags,
            user_id=raw.get("user_id"),
            chat_id=raw.get("chat_id"),
            persona_key=raw.get("persona_key"),
            importance=float(raw.get("importance", 0.5)),
            confidence=float(raw.get("confidence", 0.7)),
            source=f"federated:{source}",
            metadata={
                "federation": {
                    "source": source,
                    "source_id": raw.get("id"),
                    "federation_id": federation_id,
                    "created_at": raw.get("created_at"),
                }
            },
        )
        imported += 1
    return FederationImportResult(imported=imported, skipped=skipped, source=source)


def _export_item(item: MemoryItem, *, source: str) -> dict[str, Any]:
    federation_id = hashlib.sha256(
        f"{source}:{item.id}:{item.text}".encode("utf-8")
    ).hexdigest()[:24]
    return {
        "id": item.id,
        "federation_id": federation_id,
        "kind": item.kind,
        "scope": item.scope,
        "text": item.text,
        "tags": list(item.tags),
        "user_id": item.user_id,
        "chat_id": item.chat_id,
        "persona_key": item.persona_key,
        "importance": item.importance,
        "confidence": item.confidence,
        "created_at": item.created_at,
    }


def _signature(payload: dict[str, Any], secret: str) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value)[:64] or "unknown"


__all__ = [
    "FEDERATION_FORMAT",
    "FederationExportResult",
    "FederationImportResult",
    "MemoryFederationError",
    "export_memory_bundle",
    "import_memory_bundle",
]
