"""Behavioural tests for the tightened sticker policy.

The previous policy fired on single-word triggers ("чай", "кава", "дякую",
"грати") and frequently sent two stickers in a row. After the rebalance the
bot defaults to text-only and only stickerizes on:

- explicit laughter ("ахах", "lol", "🤣"),
- explicit warmth ("обійми", "❤", "дякую тобі"),
- explicit gameplay context (controller emoji or specific phrasing).

Long replies, recent stickers, initiative-time messages, and a
``concise``-leaning style tuner all suppress stickers as well.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from protoagi.config import AgentConfig
from protoagi.storage.memory import MemoryStore
from protoagi.telegram import (
    NikolaBot,
    TelegramConfig,
    TELEGRAM_GLOBAL_MEMORY_TAG,
)
from protoagi.telegram.stickers import auto_sticker_choice


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.stickers: list[dict] = []

    def get_me(self) -> dict:
        return {"id": 99, "username": "PolicyTestBot"}

    def delete_webhook(self, *, drop_pending_updates: bool = False) -> bool:
        return True

    def get_updates(self, **kwargs):
        return []

    def send_chat_action(self, *args, **kwargs):
        return True

    def send_message(self, chat_id, text, *, reply_to_message_id=None, disable_notification=False):
        self.sent.append({"text": text})
        return {"message_id": len(self.sent)}

    def send_sticker(self, chat_id, sticker, *, reply_to_message_id=None, disable_notification=False):
        self.stickers.append({"sticker": sticker})
        return {"message_id": 100 + len(self.stickers)}

    def get_sticker_set(self, name: str) -> dict:
        return {"stickers": [{"file_id": f"{name}:smile", "emoji": "🙂"}]}

    def get_file(self, file_id):
        return {"file_path": ""}

    def download_file(self, file_path, *, max_bytes):
        return b""


class FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content

    def chat_completion(self, messages, **kwargs):
        return {"choices": [{"message": {"content": self.content}}]}


class AutoStickerChoiceNarrowingTests(unittest.TestCase):
    def test_bare_word_no_longer_fires(self) -> None:
        """The old policy fired on 'чай' / 'кава' / 'дякую' alone."""

        self.assertIsNone(auto_sticker_choice("я налив собі кави", "ну ок"))
        self.assertIsNone(auto_sticker_choice("дякую за пораду", "будь ласка"))
        self.assertIsNone(auto_sticker_choice("граю в Steam деку зараз", "ага"))

    def test_explicit_laughter_still_fires(self) -> None:
        choice = auto_sticker_choice("ахах ну погнали", "ахах, звучить як план")
        self.assertIsNotNone(choice)
        assert choice is not None
        self.assertEqual(choice["pack"], "Bocchi_the_Rock_sticker_pack2")

    def test_explicit_warmth_still_fires(self) -> None:
        choice = auto_sticker_choice("ой, дякую тобі ❤️", "тримайся, обнімаю")
        self.assertIsNotNone(choice)
        assert choice is not None
        self.assertEqual(choice["pack"], "SenkoSan")

    def test_long_reply_blocks_auto_sticker(self) -> None:
        long_reply = "ахах " + ("слухай це окрема довга думка яку я зараз пояснюю " * 4)
        self.assertIsNone(auto_sticker_choice("ахах щось смішне", long_reply))

    def test_emoji_only_laughter_works(self) -> None:
        choice = auto_sticker_choice("🤣🤣🤣", "ну і реакція")
        self.assertIsNotNone(choice)


class StickerPolicyEndToEndTests(unittest.TestCase):
    def _make_bot(
        self,
        tmp: Path,
        *,
        llm_content: str,
        sticker_frequency: str = "always",
    ) -> NikolaBot:
        memory = MemoryStore(tmp / "memory.sqlite3")
        return NikolaBot(
            telegram=FakeTelegram(),
            llm=FakeLLM(llm_content),
            memory=memory,
            telegram_config=TelegramConfig(
                token="t",
                persona_key="solomiya",
                sticker_frequency=sticker_frequency,
                sticker_cooldown_messages=3,
            ),
            agent_config=AgentConfig(database_path=tmp / "memory.sqlite3"),
        )

    def test_long_reply_drops_llm_emitted_sticker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            long_reply = "це довга людська думка яку я зараз пояснюю " * 6
            payload = {
                "should_reply": True,
                "reply": long_reply,
                "stickers": [{"pack": "SenkoSan", "emoji": "🙂", "reason": "warmth"}],
                "memories": [],
                "next_check_minutes": 60,
            }
            bot = self._make_bot(Path(tmp), llm_content=json.dumps(payload))
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 1,
                    "message": {
                        "message_id": 10,
                        "chat": {"id": 321, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 321, "first_name": "Vadim"},
                        "text": "розкажи історію",
                    },
                }
            )
            assert isinstance(bot.telegram, FakeTelegram)
            self.assertEqual(bot.telegram.stickers, [])
            self.assertGreater(len(bot.telegram.sent), 0)

    def test_recent_sticker_suppresses_llm_followup_sticker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            memory.log_telegram_message(
                chat_id=321,
                message_id=99,
                persona_key="solomiya",
                role="assistant",
                sender_id=None,
                sender_name="Соломія",
                text="[sticker:SenkoSan]",
            )
            payload = {
                "should_reply": True,
                "reply": "ахах окей",
                "stickers": [{"pack": "Bocchi_the_Rock_sticker_pack2", "emoji": "🙂", "reason": "laughter"}],
                "memories": [],
                "next_check_minutes": 60,
            }
            bot = NikolaBot(
                telegram=FakeTelegram(),
                llm=FakeLLM(json.dumps(payload)),
                memory=memory,
                telegram_config=TelegramConfig(
                    token="t",
                    persona_key="solomiya",
                    sticker_frequency="always",
                    sticker_cooldown_messages=6,
                ),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 2,
                    "message": {
                        "message_id": 100,
                        "chat": {"id": 321, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 321, "first_name": "Vadim"},
                        "text": "ахах ну і",
                    },
                }
            )
            assert isinstance(bot.telegram, FakeTelegram)
            self.assertEqual(bot.telegram.stickers, [])

    def test_concise_bandit_arm_silences_auto_sticker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = {
                "should_reply": True,
                "reply": "ахах ну і",
                "stickers": [],
                "memories": [],
                "next_check_minutes": 60,
            }
            bot = self._make_bot(Path(tmp), llm_content=json.dumps(payload))
            # Force the tuner into the ``concise`` arm without mutating
            # internal API: drive the kv state directly.
            bot.memory.set_kv(
                "telegram:style:321",
                json.dumps(
                    {
                        "arms": {
                            "balanced": {"trials": 0, "successes": 0.0},
                            "concise": {"trials": 5, "successes": 5.0},
                            "expressive": {"trials": 0, "successes": 0.0},
                        },
                        "signals": {},
                        "last_choice": "concise",
                    }
                ),
            )
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 3,
                    "message": {
                        "message_id": 200,
                        "chat": {"id": 321, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 321, "first_name": "Vadim"},
                        "text": "ахах ну і",
                    },
                }
            )
            assert isinstance(bot.telegram, FakeTelegram)
            self.assertEqual(bot.telegram.stickers, [])

    def test_initiative_messages_drop_stickers_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            chat = memory.upsert_telegram_chat(
                {"id": 321, "type": "private", "first_name": "Vadim"},
                {"id": 321, "first_name": "Vadim"},
            )
            memory.mark_telegram_user_message(chat.chat_id)
            payload = {
                "send": True,
                "message": "тихий привіт як настрій?",
                "stickers": [{"pack": "SenkoSan", "emoji": "🙂", "reason": "soft hello"}],
                "memories": [],
                "next_check_minutes": 360,
            }
            bot = NikolaBot(
                telegram=FakeTelegram(),
                llm=FakeLLM(json.dumps(payload)),
                memory=memory,
                telegram_config=TelegramConfig(
                    token="t",
                    persona_key="solomiya",
                    sticker_initiative_enabled=False,
                ),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            sent = bot.run_initiative_once()
            self.assertEqual(sent, 1)
            assert isinstance(bot.telegram, FakeTelegram)
            self.assertEqual(bot.telegram.stickers, [])
            self.assertEqual(bot.telegram.sent[0]["text"], "тихий привіт як настрій?")


if __name__ == "__main__":
    unittest.main()
