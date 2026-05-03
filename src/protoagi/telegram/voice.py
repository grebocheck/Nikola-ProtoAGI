from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .api import TelegramApi, TelegramApiError


@dataclass(slots=True)
class VoiceAttachment:
    file_id: str
    mime_type: str
    duration: int
    label: str = "voice"


@dataclass(slots=True)
class VoiceTranscriptionConfig:
    base_url: str = ""
    model: str = ""
    timeout_seconds: int = 120
    max_bytes: int = 16 * 1024 * 1024

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)


@dataclass(slots=True)
class VoiceSynthesisConfig:
    base_url: str = ""
    model: str = ""
    voice: str = "alloy"
    timeout_seconds: int = 120
    max_chars: int = 600
    enabled: bool = False


class VoiceTranscriber:
    def __init__(
        self,
        telegram: TelegramApi,
        config: VoiceTranscriptionConfig,
    ) -> None:
        self.telegram = telegram
        self.config = config

    def transcribe(self, attachment: VoiceAttachment | None) -> str:
        if attachment is None or not attachment.file_id or not self.config.enabled:
            return ""
        try:
            file_info = self.telegram.get_file(attachment.file_id)
            file_path = str(file_info.get("file_path") or "")
            if not file_path:
                return ""
            data = self.telegram.download_file(file_path, max_bytes=self.config.max_bytes)
            return self._transcribe_bytes(data, filename=f"{attachment.file_id}.ogg").strip()
        except (TelegramApiError, OSError, ValueError):
            return ""

    def _transcribe_bytes(self, data: bytes, *, filename: str) -> str:
        if not data:
            return ""
        fields = {"model": self.config.model}
        files = {"file": (filename, data, "audio/ogg")}
        body, content_type = _multipart_body(fields, files)
        request = Request(
            self.config.base_url.rstrip("/") + "/audio/transcriptions",
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, json.JSONDecodeError):
            return ""
        return str(payload.get("text") or "")


class VoiceSynthesizer:
    def __init__(self, config: VoiceSynthesisConfig) -> None:
        self.config = config

    def synthesize(self, text: str) -> bytes | None:
        text = text.strip()
        if not self.config.enabled or not self.config.base_url or not self.config.model or not text:
            return None
        payload = {
            "model": self.config.model,
            "voice": self.config.voice,
            "input": text[: self.config.max_chars],
        }
        request = Request(
            self.config.base_url.rstrip("/") + "/audio/speech",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                data = response.read()
        except (HTTPError, URLError):
            return None
        return data or None


def _multipart_body(
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> tuple[bytes, str]:
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
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


__all__ = [
    "VoiceAttachment",
    "VoiceSynthesisConfig",
    "VoiceSynthesizer",
    "VoiceTranscriptionConfig",
    "VoiceTranscriber",
    "_multipart_body",
]
