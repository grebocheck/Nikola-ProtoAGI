from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from ..storage.memory import TelegramChat
from ..storage.service import MemoryService, RecallQuery
from .constants import TELEGRAM_GLOBAL_MEMORY_TAG


TELEGRAM_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search Telegram memory for facts relevant to this chat turn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remind_me",
            "description": "Create a reminder for this Telegram chat.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "in_minutes": {"type": "integer", "minimum": 1, "maximum": 525600},
                    "trigger_at": {"type": "string"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass(slots=True)
class TelegramToolEvent:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


class TelegramToolRunner:
    def __init__(
        self,
        *,
        memory_service: MemoryService,
        chat: TelegramChat,
        persona_key: str,
        user_id: str | None = None,
        global_memory: bool = True,
        max_steps: int = 4,
    ) -> None:
        self.memory_service = memory_service
        self.chat = chat
        self.persona_key = persona_key
        self.user_id = user_id
        self.global_memory = global_memory
        self.max_steps = max(1, max_steps)

    @staticmethod
    def schemas() -> list[dict[str, Any]]:
        return TELEGRAM_TOOL_SCHEMAS

    def run(
        self,
        *,
        tool_request: dict[str, Any] | None = None,
        tool_calls: Iterable[dict[str, Any]] | None = None,
    ) -> list[TelegramToolEvent]:
        events: list[TelegramToolEvent] = []
        calls = list(tool_calls or [])
        if tool_request is not None:
            calls.insert(
                0,
                {
                    "type": "function",
                    "function": {
                        "name": tool_request.get("name"),
                        "arguments": tool_request.get("arguments", {}),
                    },
                },
            )
        for call in calls:
            if len(events) >= self.max_steps:
                break
            function = call.get("function", {}) if isinstance(call, dict) else {}
            name = str(function.get("name") or "").strip()
            arguments = self._parse_arguments(function.get("arguments", {}))
            result = self.execute(name, arguments)
            events.append(TelegramToolEvent(name=name, arguments=arguments, result=result))
        return events

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "recall":
            return self._recall(arguments)
        if name == "remind_me":
            return self._remind_me(arguments)
        return {"ok": False, "error": f"unknown telegram tool: {name}"}

    def _recall(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "query is required"}
        try:
            limit = int(arguments.get("limit", 5))
        except (TypeError, ValueError):
            limit = 5
        limit = max(1, min(limit, 10))
        private_mode = not self.global_memory
        results = self.memory_service.recall(
            RecallQuery(
                text=query,
                user_id=self.user_id if private_mode else None,
                chat_id=self.chat.chat_id if private_mode else None,
                require_tags=(TELEGRAM_GLOBAL_MEMORY_TAG,),
                limit=limit,
                include_global=not private_mode,
            )
        )
        return {
            "ok": True,
            "items": [
                {
                    "id": result.item.id,
                    "text": result.item.text,
                    "kind": result.item.kind,
                    "scope": result.item.scope,
                    "tags": list(result.item.tags),
                    "created_at": result.item.created_at,
                    "score": result.score,
                }
                for result in results
            ],
        }

    def _remind_me(self, arguments: dict[str, Any]) -> dict[str, Any]:
        text = str(arguments.get("text") or "").strip()
        if not text:
            return {"ok": False, "error": "text is required"}
        trigger_at = str(arguments.get("trigger_at") or "").strip()
        if trigger_at:
            try:
                parsed = datetime.fromisoformat(trigger_at)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                trigger_at = parsed.isoformat(timespec="seconds")
            except ValueError:
                trigger_at = ""
        if not trigger_at:
            try:
                minutes = int(arguments.get("in_minutes", 60))
            except (TypeError, ValueError):
                minutes = 60
            trigger = datetime.now(timezone.utc) + timedelta(minutes=max(1, minutes))
            trigger_at = trigger.isoformat(timespec="seconds")
        reminder_id = self.memory_service.store.add_reminder(
            text=text,
            trigger_at=trigger_at,
            chat_id=self.chat.chat_id,
            persona_key=self.persona_key,
            user_id=self.user_id,
            metadata={"source": "telegram_tool"},
        )
        return {"ok": True, "id": reminder_id, "trigger_at": trigger_at, "text": text}

    @staticmethod
    def _parse_arguments(arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return dict(arguments)
        if not arguments:
            return {}
        try:
            loaded = json.loads(str(arguments))
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            return {}


__all__ = [
    "TELEGRAM_TOOL_SCHEMAS",
    "TelegramToolEvent",
    "TelegramToolRunner",
]
