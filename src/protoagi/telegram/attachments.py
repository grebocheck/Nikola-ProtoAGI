"""Telegram incoming attachment extraction helpers."""

from __future__ import annotations

from typing import Any

from .json_io import ImageAttachment, StickerAttachment
from .voice import VoiceAttachment


class TelegramAttachmentMixin:
    def _extract_image_attachment(self, message: dict[str, Any]) -> ImageAttachment | None:
        photos = message.get("photo")
        if isinstance(photos, list) and photos:
            photo = max(
                (item for item in photos if isinstance(item, dict) and item.get("file_id")),
                key=lambda item: int(item.get("file_size") or item.get("width", 0) * item.get("height", 0) or 0),
                default=None,
            )
            if photo:
                return ImageAttachment(
                    file_id=str(photo["file_id"]),
                    mime_type="image/jpeg",
                    label="photo",
                )

        document = message.get("document")
        if isinstance(document, dict):
            mime_type = str(document.get("mime_type") or "")
            file_id = str(document.get("file_id") or "")
            if file_id and mime_type.startswith("image/"):
                return ImageAttachment(
                    file_id=file_id,
                    mime_type=mime_type,
                    label="image document",
                    file_name=str(document.get("file_name") or ""),
                )
        return None

    @staticmethod
    def _extract_voice_attachment(message: dict[str, Any]) -> VoiceAttachment | None:
        voice = message.get("voice")
        label = "voice"
        if not isinstance(voice, dict):
            voice = message.get("audio")
            label = "audio"
        if not isinstance(voice, dict):
            return None
        file_id = str(voice.get("file_id") or "")
        if not file_id:
            return None
        return VoiceAttachment(
            file_id=file_id,
            mime_type=str(voice.get("mime_type") or "audio/ogg"),
            duration=int(voice.get("duration") or 0),
            label=label,
        )

    @staticmethod
    def _extract_sticker_attachment(message: dict[str, Any]) -> StickerAttachment | None:
        sticker = message.get("sticker")
        if not isinstance(sticker, dict):
            return None
        file_id = str(sticker.get("file_id") or "")
        if not file_id:
            return None
        if sticker.get("is_video"):
            kind = "video sticker"
        elif sticker.get("is_animated"):
            kind = "animated sticker"
        else:
            kind = "sticker"
        return StickerAttachment(
            file_id=file_id,
            emoji=str(sticker.get("emoji") or ""),
            set_name=str(sticker.get("set_name") or ""),
            kind=kind,
        )

    @staticmethod
    def _voice_to_payload(voice: VoiceAttachment | None) -> dict[str, Any] | None:
        if voice is None:
            return None
        return {
            "file_id": voice.file_id,
            "mime_type": voice.mime_type,
            "duration": voice.duration,
            "label": voice.label,
        }


__all__ = ["TelegramAttachmentMixin"]
