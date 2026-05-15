from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .api import TelegramApi, TelegramApiError


_RESPONSE_FORMAT_MIME: dict[str, str] = {
    "opus": "audio/ogg",
    "ogg": "audio/ogg",
    "mp3": "audio/mpeg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/wav",
}


@dataclass(slots=True)
class VoiceAttachment:
    file_id: str
    mime_type: str
    duration: int
    label: str = "voice"


@dataclass(slots=True)
class VoiceTranscriptionResult:
    text: str = ""
    data: bytes = b""
    mime_type: str = "audio/ogg"
    file_id: str = ""


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
    # Telegram ``sendVoice`` requires OGG/Opus. ``opus`` is what most
    # OpenAI-compatible TTS servers (openedai-speech, kokoro,
    # piper-server) emit when asked for it. Set ``mp3`` only if you
    # plan to route through ``sendAudio`` instead.
    response_format: str = "opus"
    speed: float = 1.0


class VoiceTranscriber:
    def __init__(
        self,
        telegram: TelegramApi,
        config: VoiceTranscriptionConfig,
    ) -> None:
        self.telegram = telegram
        self.config = config

    def transcribe(self, attachment: VoiceAttachment | None) -> str:
        return self.transcribe_with_bytes(attachment).text

    def transcribe_with_bytes(
        self,
        attachment: VoiceAttachment | None,
    ) -> VoiceTranscriptionResult:
        if attachment is None or not attachment.file_id or not self.config.enabled:
            return VoiceTranscriptionResult()
        try:
            file_info = self.telegram.get_file(attachment.file_id)
            file_path = str(file_info.get("file_path") or "")
            if not file_path:
                return VoiceTranscriptionResult(
                    mime_type=attachment.mime_type or "audio/ogg",
                    file_id=attachment.file_id,
                )
            data = self.telegram.download_file(file_path, max_bytes=self.config.max_bytes)
            return VoiceTranscriptionResult(
                text=self._transcribe_bytes(data, filename=f"{attachment.file_id}.ogg").strip(),
                data=data,
                mime_type=attachment.mime_type or "audio/ogg",
                file_id=attachment.file_id,
            )
        except (TelegramApiError, OSError, ValueError):
            return VoiceTranscriptionResult(
                mime_type=attachment.mime_type or "audio/ogg",
                file_id=attachment.file_id if attachment is not None else "",
            )

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
        self.last_error: str | None = None

    @property
    def response_format(self) -> str:
        return (self.config.response_format or "opus").strip().lower() or "opus"

    @property
    def expected_mime(self) -> str:
        fmt = self.response_format
        return _RESPONSE_FORMAT_MIME.get(fmt, "application/octet-stream")

    def synthesize(self, text: str, *, voice: str | None = None) -> bytes | None:
        text = text.strip()
        if not self.config.enabled or not self.config.base_url or not self.config.model or not text:
            return None
        active_voice = (voice or "").strip() or self.config.voice
        payload: dict[str, Any] = {
            "model": self.config.model,
            "voice": active_voice,
            "input": text[: self.config.max_chars],
            "response_format": self.response_format,
        }
        if self.config.speed and self.config.speed != 1.0:
            payload["speed"] = float(self.config.speed)
        request = Request(
            self.config.base_url.rstrip("/") + "/audio/speech",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                data = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            self.last_error = f"HTTP {exc.code}: {detail}"
            return None
        except URLError as exc:
            self.last_error = f"network: {exc.reason}"
            return None
        if not data:
            self.last_error = "empty response from TTS server"
            return None
        # Sanity-check: a JSON error blob is NOT audio. openedai-speech and
        # similar return application/json on bad params.
        if data.lstrip().startswith(b"{") and b'"error"' in data[:200]:
            self.last_error = (
                f"TTS server returned a JSON error instead of audio: "
                f"{data[:200].decode('utf-8', errors='replace')}"
            )
            return None
        self.last_error = None
        return data


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
    "VoiceTranscriptionResult",
    "VoiceSynthesisConfig",
    "VoiceSynthesizer",
    "VoiceTranscriptionConfig",
    "VoiceTranscriber",
    "_multipart_body",
]
