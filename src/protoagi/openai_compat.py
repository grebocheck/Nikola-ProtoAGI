from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .harmony import sanitize_model_input


class OpenAICompatError(RuntimeError):
    pass


class OpenAICompatibleClient:
    """Tiny OpenAI-compatible HTTP client with no external dependencies."""

    def __init__(self, base_url: str, model: str, *, timeout_seconds: int = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        return self._request_url(method, url, payload)

    def _request_url(self, method: str, url: str, payload: dict[str, Any] | None = None) -> Any:
        body = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                data = response.read().decode("utf-8")
                if not data:
                    return {}
                return json.loads(data)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OpenAICompatError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except URLError as exc:
            raise OpenAICompatError(f"Cannot reach {url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise OpenAICompatError(f"Non-JSON response from {url}") from exc

    def models(self) -> Any:
        return self._request("GET", "/models")

    def server_props(self) -> Any:
        base_url = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        return self._request_url("GET", f"{base_url}/props")

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.6,
        top_p: float = 1.0,
        max_tokens: int = 1536,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _sanitize_messages(messages),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return self._request("POST", "/chat/completions", payload)


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        if "content" in item:
            item["content"] = _sanitize_content(item["content"])
        sanitized.append(item)
    return sanitized


def _sanitize_content(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_model_input(value)
    if isinstance(value, list):
        return [_sanitize_content(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_content(item) for key, item in value.items()}
    return value
