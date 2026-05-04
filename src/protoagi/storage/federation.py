from __future__ import annotations

import hmac
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .memory import MemoryItem, MemoryStore, utc_now


FEDERATION_FORMAT = "protoagi.storage.federation.v1"


class MemoryFederationError(RuntimeError):
    pass


@dataclass(slots=True)
class FederationExportResult:
    path: Path
    exported: int
    signature: str
    deleted: int = 0
    since: str | None = None
    created_at: str = ""


@dataclass(slots=True)
class FederationImportResult:
    imported: int
    skipped: int
    source: str
    deleted: int = 0


def export_memory_bundle(
    store: MemoryStore,
    path: Path,
    *,
    secret: str,
    source: str = "protoagi",
    scope: str | None = None,
    require_tags: Iterable[str] = (),
    limit: int = 1000,
    since: str | None = None,
) -> FederationExportResult:
    if not secret:
        raise MemoryFederationError("federation secret is required")
    tag_filter = tuple(str(tag).strip() for tag in require_tags if str(tag).strip())
    all_items = [
        item
        for item in store.list_memories(scope=scope, include_superseded=False, limit=limit)
        if not tag_filter or all(tag in item.tags for tag in tag_filter)
    ]
    cursor_key = _export_state_key(source=source, scope=scope, tags=tag_filter)
    previous_state = _load_export_state(store, cursor_key)
    previous_items = {
        str(key): str(value)
        for key, value in (previous_state.get("items") or {}).items()
    }
    cutoff = _parse_iso(since) if since else None
    if cutoff is None:
        items = all_items
        deletions: list[dict[str, Any]] = []
    else:
        previous_ids = set(previous_items)
        current_manifest = {str(item.id): _federation_id(item, source=source) for item in all_items}
        items = [
            item
            for item in all_items
            if (
                str(item.id) not in previous_ids
                or current_manifest.get(str(item.id)) != previous_items.get(str(item.id))
                or _is_at_or_after(item.updated_at or item.created_at, cutoff)
            )
        ]
        deletions = [
            {
                "federation_id": federation_id,
                "source_id": source_id,
            }
            for source_id, federation_id in previous_items.items()
            if source_id not in current_manifest or current_manifest[source_id] != federation_id
        ]
    created_at = utc_now()
    payload: dict[str, Any] = {
        "format": FEDERATION_FORMAT,
        "created_at": created_at,
        "since": since,
        "source": source,
        "items": [_export_item(item, source=source) for item in items],
        "deletions": deletions,
    }
    signature = _signature(payload, secret)
    payload["signature"] = signature
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "source": source,
        "scope": scope,
        "tags": list(tag_filter),
        "last_export_at": created_at,
        "items": {str(item.id): _federation_id(item, source=source) for item in all_items},
    }
    store.set_kv(cursor_key, json.dumps(manifest, ensure_ascii=False))
    store.set_kv("memory_federation:last_export_at", created_at)
    store.set_kv(f"memory_federation:last_export_at:{_safe_tag(source)}", created_at)
    return FederationExportResult(
        path=path,
        exported=len(items),
        signature=signature,
        deleted=len(deletions),
        since=since,
        created_at=created_at,
    )


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
    deleted = 0
    source = str(payload.get("source") or "unknown")
    for raw in payload.get("deletions") or []:
        if not isinstance(raw, dict):
            continue
        federation_id = str(raw.get("federation_id") or "").strip()
        if not federation_id:
            continue
        for fact in store.recent_tagged_all([f"federated_id:{federation_id}"], limit=100):
            store.delete_memory(fact.id)
            deleted += 1
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
                    "updated_at": raw.get("updated_at"),
                }
            },
        )
        imported += 1
    return FederationImportResult(imported=imported, skipped=skipped, source=source, deleted=deleted)


def _export_item(item: MemoryItem, *, source: str) -> dict[str, Any]:
    federation_id = _federation_id(item, source=source)
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
        "updated_at": item.updated_at,
    }


def _federation_id(item: MemoryItem, *, source: str) -> str:
    return hashlib.sha256(
        f"{source}:{item.id}:{item.text}".encode("utf-8")
    ).hexdigest()[:24]


def _signature(payload: dict[str, Any], secret: str) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value)[:64] or "unknown"


def _export_state_key(*, source: str, scope: str | None, tags: tuple[str, ...]) -> str:
    raw = json.dumps(
        {"source": source, "scope": scope or "", "tags": list(tags)},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"memory_federation:export_state:{_safe_tag(source)}:{digest}"


def _load_export_state(store: MemoryStore, key: str) -> dict[str, Any]:
    raw = store.get_kv(key)
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise MemoryFederationError(f"invalid --since timestamp: {value!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _is_at_or_after(value: str, cutoff: datetime) -> bool:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed >= cutoff


__all__ = [
    "FEDERATION_FORMAT",
    "FederationExportResult",
    "FederationImportResult",
    "MemoryFederationError",
    "export_memory_bundle",
    "import_memory_bundle",
]
