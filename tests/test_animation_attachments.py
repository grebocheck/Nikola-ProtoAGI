"""GIF / animation / video attachment extraction.

Telegram doesn't send GIFs as ``photo`` — they come as ``animation``
(an mp4 with a thumbnail). The bot's vision model can't read mp4, so
the orchestrator falls back to the server-provided still-frame
thumbnail. Same logic covers ordinary videos and video notes.
"""

from __future__ import annotations

import unittest

from protoagi.telegram.attachments import TelegramAttachmentMixin


class _Stub(TelegramAttachmentMixin):
    """Just exposes the mixin's protected methods for direct testing."""


class AnimationAttachmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bot = _Stub()

    def test_animation_returns_thumbnail_as_image(self) -> None:
        msg = {
            "animation": {
                "file_id": "ANIM_FILE",
                "mime_type": "video/mp4",
                "duration": 3,
                "thumbnail": {"file_id": "THUMB_FILE", "width": 320, "height": 180},
            }
        }
        result = self.bot._extract_image_attachment(msg)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.file_id, "THUMB_FILE")
        self.assertEqual(result.mime_type, "image/jpeg")
        self.assertIn("GIF", result.label)

    def test_video_returns_thumbnail(self) -> None:
        msg = {
            "video": {
                "file_id": "VID",
                "thumbnail": {"file_id": "VID_THUMB"},
                "file_name": "clip.mp4",
            }
        }
        result = self.bot._extract_image_attachment(msg)
        assert result is not None
        self.assertEqual(result.file_id, "VID_THUMB")
        self.assertEqual(result.file_name, "clip.mp4")
        self.assertIn("video", result.label)

    def test_video_note_returns_thumbnail(self) -> None:
        msg = {
            "video_note": {
                "file_id": "VN",
                "thumbnail": {"file_id": "VN_THUMB"},
            }
        }
        result = self.bot._extract_image_attachment(msg)
        assert result is not None
        self.assertEqual(result.file_id, "VN_THUMB")

    def test_legacy_thumb_key_still_works(self) -> None:
        # Older Telegram libs may still emit ``thumb`` instead of
        # ``thumbnail``. We accept either.
        msg = {
            "animation": {
                "file_id": "ANIM",
                "thumb": {"file_id": "LEGACY_THUMB"},
            }
        }
        result = self.bot._extract_image_attachment(msg)
        assert result is not None
        self.assertEqual(result.file_id, "LEGACY_THUMB")

    def test_animation_without_thumbnail_returns_none(self) -> None:
        # Rare but possible — encoder didn't bundle a thumbnail.
        # We prefer "no image" over "vision model chokes on mp4".
        msg = {"animation": {"file_id": "ANIM_NO_THUMB"}}
        self.assertIsNone(self.bot._extract_image_attachment(msg))

    def test_gif_document_prefers_thumbnail_over_raw_gif(self) -> None:
        # When a GIF is sent as a document (mime=image/gif) it has the
        # same thumbnail. Vision models usually can't decode GIF either,
        # so we route through the thumbnail in that case too.
        msg = {
            "document": {
                "file_id": "GIF_DOC",
                "mime_type": "image/gif",
                "file_name": "meme.gif",
                "thumbnail": {"file_id": "GIF_DOC_THUMB"},
            }
        }
        result = self.bot._extract_image_attachment(msg)
        assert result is not None
        self.assertEqual(result.file_id, "GIF_DOC_THUMB")

    def test_gif_document_without_thumbnail_falls_back_to_raw(self) -> None:
        msg = {
            "document": {
                "file_id": "GIF_DOC_RAW",
                "mime_type": "image/gif",
                "file_name": "meme.gif",
            }
        }
        result = self.bot._extract_image_attachment(msg)
        assert result is not None
        self.assertEqual(result.file_id, "GIF_DOC_RAW")
        self.assertEqual(result.mime_type, "image/gif")

    def test_regular_jpeg_document_unaffected(self) -> None:
        # Sanity: regular image documents still hit the old branch with
        # their original mime, not forced through thumbnail.
        msg = {
            "document": {
                "file_id": "JPEG_DOC",
                "mime_type": "image/jpeg",
                "file_name": "photo.jpg",
                "thumbnail": {"file_id": "JPEG_THUMB"},
            }
        }
        result = self.bot._extract_image_attachment(msg)
        assert result is not None
        self.assertEqual(result.file_id, "JPEG_DOC")
        self.assertEqual(result.mime_type, "image/jpeg")

    def test_photo_still_wins_over_animation(self) -> None:
        # Photos come first in the priority list — they're better
        # quality than thumbnails when both are present (shouldn't
        # actually happen in one Telegram message, but be safe).
        msg = {
            "photo": [{"file_id": "PHOTO", "file_size": 9999}],
            "animation": {"thumbnail": {"file_id": "WRONG"}},
        }
        result = self.bot._extract_image_attachment(msg)
        assert result is not None
        self.assertEqual(result.file_id, "PHOTO")


if __name__ == "__main__":
    unittest.main()
