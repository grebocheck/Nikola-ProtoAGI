"""Thin transport layer for the Telegram Bot API.

The class wraps a handful of methods we actually use (long-polling
``getUpdates``, ``sendMessage``, ``sendSticker``, ``getFile``, etc.) so the
rest of the bot can stay HTTP-agnostic. ``urllib`` is used directly to keep
the package dependency-free.
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TELEGRAM_API_ROOT = "https://api.telegram.org"
TELEGRAM_MAX_MESSAGE_CHARS = 4096


class TelegramApiError(RuntimeError):
    pass


def is_telegram_polling_conflict(exc: BaseException | str) -> bool:
    message = str(exc)
    return "409" in message and "getUpdates" in message


class TelegramApi:
    def __init__(self, token: str, *, api_root: str = TELEGRAM_API_ROOT) -> None:
        self.token = token
        self.api_root = api_root.rstrip("/")

    def call(self, method: str, payload: dict[str, Any] | None = None, *, timeout: int = 60) -> Any:
        if not self.token:
            raise TelegramApiError("TELEGRAM_BOT_TOKEN is not set")
        url = f"{self.api_root}/bot{self.token}/{method}"
        body = json.dumps(payload or {}).encode("utf-8")
        request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise TelegramApiError(f"Telegram HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise TelegramApiError(f"Telegram network error: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise TelegramApiError("Telegram returned non-JSON response") from exc
        if not data.get("ok"):
            raise TelegramApiError(str(data.get("description", data)))
        return data.get("result")

    def get_me(self) -> dict[str, Any]:
        return dict(self.call("getMe"))

    def delete_webhook(self, *, drop_pending_updates: bool = False) -> bool:
        return bool(self.call("deleteWebhook", {"drop_pending_updates": drop_pending_updates}))

    def get_updates(
        self,
        *,
        offset: int | None,
        timeout_seconds: int,
        allowed_updates: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"timeout": timeout_seconds, "limit": 50}
        if offset is not None:
            payload["offset"] = offset
        if allowed_updates is not None:
            payload["allowed_updates"] = allowed_updates
        return list(self.call("getUpdates", payload, timeout=timeout_seconds + 10))

    def send_message(
        self,
        chat_id: str | int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        disable_notification: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text[:TELEGRAM_MAX_MESSAGE_CHARS],
            "disable_notification": disable_notification,
        }
        if reply_to_message_id is not None:
            payload["reply_parameters"] = {"message_id": reply_to_message_id}
        return dict(self.call("sendMessage", payload))

    def send_chat_action(self, chat_id: str | int, action: str = "typing") -> bool:
        return bool(self.call("sendChatAction", {"chat_id": chat_id, "action": action}, timeout=20))

    def set_message_reaction(
        self,
        chat_id: str | int,
        message_id: int,
        emoji: str | None,
        *,
        is_big: bool = False,
    ) -> bool:
        reaction = [{"type": "emoji", "emoji": emoji}] if emoji else []
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": int(message_id),
            "reaction": reaction,
            "is_big": bool(is_big),
        }
        return bool(self.call("setMessageReaction", payload, timeout=20))

    def get_sticker_set(self, name: str) -> dict[str, Any]:
        return dict(self.call("getStickerSet", {"name": name}))

    def get_file(self, file_id: str) -> dict[str, Any]:
        return dict(self.call("getFile", {"file_id": file_id}))

    def download_file(self, file_path: str, *, max_bytes: int) -> bytes:
        if not self.token:
            raise TelegramApiError("TELEGRAM_BOT_TOKEN is not set")
        url = f"{self.api_root}/file/bot{self.token}/{file_path.lstrip('/')}"
        request = Request(url, method="GET")
        try:
            with urlopen(request, timeout=60) as response:
                data = response.read(max_bytes + 1)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise TelegramApiError(f"Telegram file HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise TelegramApiError(f"Telegram file network error: {exc}") from exc
        if len(data) > max_bytes:
            raise TelegramApiError(f"Telegram file is larger than configured max_bytes={max_bytes}")
        return data

    def send_sticker(
        self,
        chat_id: str | int,
        sticker: str,
        *,
        reply_to_message_id: int | None = None,
        disable_notification: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "sticker": sticker,
            "disable_notification": disable_notification,
        }
        if reply_to_message_id is not None:
            payload["reply_parameters"] = {"message_id": reply_to_message_id}
        return dict(self.call("sendSticker", payload))

    def send_voice_bytes(
        self,
        chat_id: str | int,
        data: bytes,
        *,
        filename: str = "reply.ogg",
        reply_to_message_id: int | None = None,
        disable_notification: bool = False,
        mime_type: str = "audio/ogg",
    ) -> dict[str, Any]:
        fields: dict[str, str] = {
            "chat_id": str(chat_id),
            "disable_notification": "true" if disable_notification else "false",
        }
        if reply_to_message_id is not None:
            fields["reply_parameters"] = json.dumps({"message_id": reply_to_message_id})
        return dict(
            self._call_multipart(
                "sendVoice",
                fields,
                {"voice": (filename, data, mime_type)},
            )
        )

    def send_audio_bytes(
        self,
        chat_id: str | int,
        data: bytes,
        *,
        filename: str = "reply.mp3",
        mime_type: str = "audio/mpeg",
        reply_to_message_id: int | None = None,
        disable_notification: bool = False,
    ) -> dict[str, Any]:
        fields: dict[str, str] = {
            "chat_id": str(chat_id),
            "disable_notification": "true" if disable_notification else "false",
        }
        if reply_to_message_id is not None:
            fields["reply_parameters"] = json.dumps({"message_id": reply_to_message_id})
        return dict(
            self._call_multipart(
                "sendAudio",
                fields,
                {"audio": (filename, data, mime_type)},
            )
        )

    def _call_multipart(
        self,
        method: str,
        fields: dict[str, str],
        files: dict[str, tuple[str, bytes, str]],
        *,
        timeout: int = 60,
    ) -> Any:
        if not self.token:
            raise TelegramApiError("TELEGRAM_BOT_TOKEN is not set")
        boundary = f"----protoagi-{uuid.uuid4().hex}"
        chunks: list[bytes] = []
        for name, value in fields.items():
            chunks.append(f"--{boundary}\r\n".encode("ascii"))
            chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii"))
            chunks.append(str(value).encode("utf-8"))
            chunks.append(b"\r\n")
        for name, (filename, data, content_type) in files.items():
            chunks.append(f"--{boundary}\r\n".encode("ascii"))
            chunks.append(
                (
                    f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                    f"Content-Type: {content_type}\r\n\r\n"
                ).encode("ascii")
            )
            chunks.append(data)
            chunks.append(b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode("ascii"))
        url = f"{self.api_root}/bot{self.token}/{method}"
        request = Request(
            url,
            data=b"".join(chunks),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise TelegramApiError(f"Telegram HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise TelegramApiError(f"Telegram network error: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise TelegramApiError("Telegram returned non-JSON response") from exc
        if not payload.get("ok"):
            raise TelegramApiError(str(payload.get("description", payload)))
        return payload.get("result")


__all__ = [
    "TELEGRAM_API_ROOT",
    "TELEGRAM_MAX_MESSAGE_CHARS",
    "TelegramApi",
    "TelegramApiError",
    "is_telegram_polling_conflict",
]
