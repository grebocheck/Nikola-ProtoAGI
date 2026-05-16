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
                # Animated GIFs uploaded as documents (mime=image/gif) are
                # still GIFs — most vision models choke on them, so prefer
                # the server-side thumbnail when available and fall back
                # to the raw file otherwise.
                thumbnail = _thumbnail_attachment(document, label="GIF (still frame)")
                if thumbnail is not None and mime_type == "image/gif":
                    return thumbnail
                return ImageAttachment(
                    file_id=file_id,
                    mime_type=mime_type,
                    label="image document",
                    file_name=str(document.get("file_name") or ""),
                )

        # Animations (Telegram GIFs are delivered as video/mp4 with a
        # ``animation`` field) and ordinary videos come with a thumbnail
        # the server already extracted. We surface that single frame to
        # the vision model instead of trying to decode mp4 locally.
        for key, label in (
            ("animation", "GIF (still frame)"),
            ("video", "video (still frame)"),
            ("video_note", "video note (still frame)"),
        ):
            payload = message.get(key)
            if not isinstance(payload, dict):
                continue
            thumbnail = _thumbnail_attachment(payload, label=label)
            if thumbnail is not None:
                return thumbnail
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
        thumb = sticker.get("thumbnail")
        if not isinstance(thumb, dict):
            thumb = sticker.get("thumb")
        thumbnail_file_id = str(thumb.get("file_id") or "") if isinstance(thumb, dict) else ""
        return StickerAttachment(
            file_id=file_id,
            emoji=str(sticker.get("emoji") or ""),
            set_name=str(sticker.get("set_name") or ""),
            kind=kind,
            thumbnail_file_id=thumbnail_file_id,
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


def _thumbnail_attachment(
    payload: dict[str, Any], *, label: str
) -> ImageAttachment | None:
    """Return the JPEG thumbnail Telegram bundles with media payloads.

    ``thumbnail`` is the canonical key in current API versions, ``thumb``
    is the legacy alias older clients sometimes still emit. The
    thumbnail is always a small JPEG, which is exactly what our vision
    pipeline already knows how to handle.
    """

    thumb = payload.get("thumbnail")
    if not isinstance(thumb, dict):
        thumb = payload.get("thumb")
    if not isinstance(thumb, dict):
        return None
    file_id = str(thumb.get("file_id") or "")
    if not file_id:
        return None
    file_name = str(payload.get("file_name") or "")
    return ImageAttachment(
        file_id=file_id,
        mime_type="image/jpeg",
        label=label,
        file_name=file_name,
    )


__all__ = ["TelegramAttachmentMixin"]
