from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import AgentConfig
from .env import env_bool, env_int
from .harmony import clean_model_content
from .memory import MemoryStore, TelegramChat, utc_now
from .openai_compat import OpenAICompatibleClient


TELEGRAM_API_ROOT = "https://api.telegram.org"
TELEGRAM_MAX_MESSAGE_CHARS = 4096
OFFSET_KEY = "telegram:update_offset"


class TelegramApiError(RuntimeError):
    pass


@dataclass(slots=True)
class TelegramConfig:
    token: str
    bot_name: str = "Микола"
    allowed_chat_ids: set[str] | None = None
    reply_mode: str = "smart"
    poll_timeout_seconds: int = 25
    max_reply_chars: int = 3900
    max_history_messages: int = 14
    max_memory_facts: int = 6
    decision_max_tokens: int = 768
    proactive_enabled: bool = True
    proactive_check_seconds: int = 300
    proactive_cooldown_seconds: int = 6 * 60 * 60
    proactive_disable_notification: bool = True

    @classmethod
    def from_env(cls) -> "TelegramConfig":
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        allowed = _parse_chat_ids(os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", ""))
        reply_mode = os.environ.get("NIKOLA_REPLY_MODE", "smart").strip() or "smart"
        if reply_mode not in {"smart", "always", "mention", "silent"}:
            reply_mode = "smart"
        return cls(
            token=token,
            allowed_chat_ids=allowed,
            reply_mode=reply_mode,
            poll_timeout_seconds=env_int("TELEGRAM_POLL_TIMEOUT", 25),
            proactive_enabled=env_bool("NIKOLA_PROACTIVE", True),
            proactive_check_seconds=env_int("NIKOLA_PROACTIVE_CHECK_SECONDS", 300),
            proactive_cooldown_seconds=env_int("NIKOLA_PROACTIVE_COOLDOWN_SECONDS", 6 * 60 * 60),
        )


@dataclass(slots=True)
class Decision:
    should_reply: bool
    reply: str
    memories: list[str]
    next_check_minutes: int | None = None


@dataclass(slots=True)
class InitiativeDecision:
    send: bool
    message: str
    memories: list[str]
    next_check_minutes: int


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


class NikolaBot:
    def __init__(
        self,
        *,
        telegram: TelegramApi,
        llm: OpenAICompatibleClient,
        memory: MemoryStore,
        telegram_config: TelegramConfig,
        agent_config: AgentConfig,
    ) -> None:
        self.telegram = telegram
        self.llm = llm
        self.memory = memory
        self.telegram_config = telegram_config
        self.agent_config = agent_config
        self.bot_username = self.memory.get_kv("telegram:bot_username") or ""
        self._last_proactive_check = 0.0

    def bootstrap(self, *, delete_webhook: bool = False, drop_pending_updates: bool = False) -> dict[str, Any]:
        if delete_webhook:
            self.telegram.delete_webhook(drop_pending_updates=drop_pending_updates)
        me = self.telegram.get_me()
        self.bot_username = str(me.get("username", ""))
        self.memory.set_kv("telegram:bot_username", self.bot_username)
        self.memory.set_kv("telegram:bot_id", str(me.get("id", "")))
        return me

    def run_forever(self) -> None:
        while True:
            try:
                self.poll_once()
                self.maybe_run_initiative()
            except (TelegramApiError, OSError) as exc:
                print(f"Telegram loop transient error: {exc}", flush=True)
                time.sleep(5)

    def poll_once(self) -> int:
        offset_text = self.memory.get_kv(OFFSET_KEY)
        offset = int(offset_text) if offset_text else None
        updates = self.telegram.get_updates(
            offset=offset,
            timeout_seconds=self.telegram_config.poll_timeout_seconds,
            allowed_updates=["message"],
        )
        processed = 0
        for update in updates:
            update_id = int(update.get("update_id", 0))
            try:
                if self.process_update(update):
                    processed += 1
            finally:
                self.memory.set_kv(OFFSET_KEY, str(update_id + 1))
        return processed

    def maybe_run_initiative(self) -> int:
        if not self.telegram_config.proactive_enabled:
            return 0
        now_monotonic = time.monotonic()
        if now_monotonic - self._last_proactive_check < self.telegram_config.proactive_check_seconds:
            return 0
        self._last_proactive_check = now_monotonic
        return self.run_initiative_once()

    def process_update(self, update: dict[str, Any]) -> bool:
        message = update.get("message")
        if not isinstance(message, dict):
            return False
        text = str(message.get("text") or message.get("caption") or "").strip()
        if not text:
            return False
        chat = message.get("chat") or {}
        if "id" not in chat:
            return False
        chat_id = str(chat["id"])
        if not self._chat_allowed(chat_id):
            return False

        user = message.get("from") or {}
        chat_state = self.memory.upsert_telegram_chat(
            chat,
            user,
            reply_mode=self.telegram_config.reply_mode,
        )
        self.memory.mark_telegram_user_message(chat_id)

        thread_id = self.thread_id(chat_id)
        display = display_sender(user)
        content = f"{display}: {text}" if chat_state.chat_type != "private" else text
        self.memory.log_message(thread_id, "user", content)

        command_reply = self._handle_command(text, chat_state)
        if command_reply is not None:
            self._send_reply(chat_state, command_reply, message_id=int(message.get("message_id", 0)) or None)
            self._schedule_from_minutes(
                chat_state.chat_id,
                self.telegram_config.proactive_cooldown_seconds // 60,
            )
            return True

        addressed = self._is_addressed(text, message)
        if self._should_skip_without_llm(chat_state, addressed):
            return True

        decision = self.decide_incoming(chat_state, text, display, addressed)
        for fact in decision.memories:
            self._remember_chat_fact(chat_state, fact)
        if not decision.should_reply or not decision.reply.strip():
            self._schedule_from_minutes(chat_state.chat_id, decision.next_check_minutes)
            return True
        reply = self._limit_reply(decision.reply)
        self._send_reply(chat_state, reply, message_id=int(message.get("message_id", 0)) or None)
        self._schedule_from_minutes(chat_state.chat_id, decision.next_check_minutes)
        return True

    def decide_incoming(
        self,
        chat: TelegramChat,
        incoming_text: str,
        sender: str,
        addressed: bool,
    ) -> Decision:
        recent = self.memory.recent_messages(self.thread_id(chat.chat_id), limit=self.telegram_config.max_history_messages)
        facts = self.memory.search_tagged(incoming_text, self.chat_tag(chat.chat_id), limit=self.telegram_config.max_memory_facts)
        messages = [
            {"role": "system", "content": self._decision_system_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "bot_name": self.telegram_config.bot_name,
                        "chat": {
                            "id": chat.chat_id,
                            "type": chat.chat_type,
                            "display_name": chat.display_name,
                            "reply_mode": chat.reply_mode,
                            "addressed_to_bot": addressed,
                        },
                        "sender": sender,
                        "incoming_text": incoming_text,
                        "recent_messages": recent,
                        "relevant_memory": [
                            {"text": fact.text, "tags": fact.tags, "created_at": fact.created_at}
                            for fact in facts
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        response = self.llm.chat_completion(
            messages,
            temperature=self.agent_config.temperature,
            top_p=self.agent_config.top_p,
            max_tokens=self.telegram_config.decision_max_tokens,
        )
        content = clean_model_content(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
        payload = extract_json_object(content)
        decision = decision_from_payload(payload)
        if self.telegram_config.reply_mode == "always":
            decision.should_reply = True
        if chat.chat_type != "private" and self.telegram_config.reply_mode == "mention" and not addressed:
            decision.should_reply = False
        if self.telegram_config.reply_mode == "silent":
            decision.should_reply = False
        if decision.should_reply and not decision.reply:
            decision.reply = self.compose_reply(chat, incoming_text, sender)
        return decision

    def compose_reply(self, chat: TelegramChat, incoming_text: str, sender: str) -> str:
        recent = self.memory.recent_messages(self.thread_id(chat.chat_id), limit=self.telegram_config.max_history_messages)
        facts = self.memory.search_tagged(incoming_text, self.chat_tag(chat.chat_id), limit=self.telegram_config.max_memory_facts)
        response = self.llm.chat_completion(
            [
                {"role": "system", "content": self._reply_system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "chat": {"type": chat.chat_type, "display_name": chat.display_name},
                            "sender": sender,
                            "incoming_text": incoming_text,
                            "recent_messages": recent,
                            "relevant_memory": [fact.text for fact in facts],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=self.agent_config.temperature,
            top_p=self.agent_config.top_p,
            max_tokens=self.telegram_config.decision_max_tokens,
        )
        return clean_model_content(response.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()

    def run_initiative_once(self) -> int:
        sent = 0
        now = utc_now()
        for chat in self.memory.list_due_telegram_chats(now, limit=10):
            if not self._chat_allowed(chat.chat_id):
                continue
            if not self._initiative_cooldown_elapsed(chat):
                self._schedule_from_minutes(chat.chat_id, self.telegram_config.proactive_cooldown_seconds // 60)
                continue
            decision = self.decide_initiative(chat)
            for fact in decision.memories:
                self._remember_chat_fact(chat, fact)
            self._schedule_from_minutes(chat.chat_id, decision.next_check_minutes)
            if not decision.send or not decision.message.strip():
                continue
            try:
                self._send_reply(
                    chat,
                    self._limit_reply(decision.message),
                    initiative=True,
                    disable_notification=self.telegram_config.proactive_disable_notification,
                )
                sent += 1
            except TelegramApiError:
                self.memory.set_telegram_proactive(chat.chat_id, False)
        return sent

    def decide_initiative(self, chat: TelegramChat) -> InitiativeDecision:
        recent = self.memory.recent_messages(self.thread_id(chat.chat_id), limit=self.telegram_config.max_history_messages)
        facts = self.memory.search_tagged(chat.display_name, self.chat_tag(chat.chat_id), limit=self.telegram_config.max_memory_facts)
        response = self.llm.chat_completion(
            [
                {"role": "system", "content": self._initiative_system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "chat": {
                                "id": chat.chat_id,
                                "type": chat.chat_type,
                                "display_name": chat.display_name,
                                "last_user_message_at": chat.last_user_message_at,
                                "last_bot_message_at": chat.last_bot_message_at,
                                "last_initiative_at": chat.last_initiative_at,
                            },
                            "recent_messages": recent,
                            "memory": [fact.text for fact in facts],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=self.agent_config.temperature,
            top_p=self.agent_config.top_p,
            max_tokens=self.telegram_config.decision_max_tokens,
        )
        content = clean_model_content(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
        return initiative_from_payload(extract_json_object(content))

    def _handle_command(self, text: str, chat: TelegramChat) -> str | None:
        command, rest = parse_command(text, self.bot_username)
        if command is None:
            return None
        if command == "start":
            self.memory.set_telegram_proactive(chat.chat_id, True)
            return (
                "Привіт, я Микола. Я можу просто говорити з тобою, памʼятати важливе "
                "і час від часу сам обережно повертатися до розмови. /help покаже команди."
            )
        if command == "help":
            return (
                "Я вмію: /remember текст - запамʼятати, /recall запит - згадати, "
                "/quiet - вимкнути ініціативні повідомлення, /wake - увімкнути, /status - стан чату."
            )
        if command == "quiet":
            self.memory.set_telegram_proactive(chat.chat_id, False)
            return "Домовились. Ініціативні повідомлення вимкнені; відповідатиму, коли ти звернешся."
        if command == "wake":
            self.memory.set_telegram_proactive(chat.chat_id, True)
            self._schedule_from_minutes(chat.chat_id, 60)
            return "Я знову можу іноді писати першим, але без спаму."
        if command == "status":
            current = self.memory.get_telegram_chat(chat.chat_id) or chat
            return (
                f"Чат: {current.display_name}\n"
                f"ID: {current.chat_id}\n"
                f"Режим відповідей: {current.reply_mode}\n"
                f"Ініціатива: {'увімкнена' if current.proactive_enabled else 'вимкнена'}"
            )
        if command == "remember":
            if not rest:
                return "Напиши так: /remember факт, який варто запамʼятати."
            self._remember_chat_fact(chat, rest)
            return "Запамʼятав."
        if command == "recall":
            query = rest or chat.display_name
            facts = self.memory.search_tagged(query, self.chat_tag(chat.chat_id), limit=8)
            if not facts:
                return "Поки нічого релевантного не знайшов."
            return "Ось що памʼятаю:\n" + "\n".join(f"- {fact.text}" for fact in facts)
        if command == "mode":
            mode = rest.strip().lower()
            if mode not in {"smart", "always", "mention", "silent"}:
                return "Доступні режими: smart, always, mention, silent."
            self.memory.set_telegram_reply_mode(chat.chat_id, mode)
            return f"Режим відповідей змінено на {mode}."
        return None

    def _send_reply(
        self,
        chat: TelegramChat,
        text: str,
        *,
        message_id: int | None = None,
        initiative: bool = False,
        disable_notification: bool = False,
    ) -> None:
        try:
            self.telegram.send_chat_action(chat.chat_id, "typing")
        except TelegramApiError:
            pass
        chunks = split_telegram_message(text, max_chars=self.telegram_config.max_reply_chars)
        for index, chunk in enumerate(chunks):
            self.telegram.send_message(
                chat.chat_id,
                chunk,
                reply_to_message_id=message_id if index == 0 and not initiative else None,
                disable_notification=disable_notification,
            )
        self.memory.log_message(self.thread_id(chat.chat_id), "assistant", text)
        self.memory.mark_telegram_bot_message(chat.chat_id, initiative=initiative)

    def _remember_chat_fact(self, chat: TelegramChat, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.memory.remember(
            text,
            ["telegram", self.chat_tag(chat.chat_id), f"chat_type:{chat.chat_type}", "nikola"],
        )

    def _schedule_from_minutes(self, chat_id: str, minutes: int | None) -> None:
        minutes = minutes if minutes is not None else self.telegram_config.proactive_cooldown_seconds // 60
        minutes = max(15, min(minutes, 7 * 24 * 60))
        next_at = datetime.now(timezone.utc) + timedelta(minutes=minutes)
        self.memory.schedule_telegram_initiative(chat_id, next_at.isoformat(timespec="seconds"))

    def _initiative_cooldown_elapsed(self, chat: TelegramChat) -> bool:
        if not chat.last_initiative_at:
            return True
        try:
            last = datetime.fromisoformat(chat.last_initiative_at)
        except ValueError:
            return True
        return datetime.now(timezone.utc) - last >= timedelta(seconds=self.telegram_config.proactive_cooldown_seconds)

    def _limit_reply(self, text: str) -> str:
        text = text.strip()
        if len(text) <= self.telegram_config.max_reply_chars:
            return text
        return text[: self.telegram_config.max_reply_chars - 20].rstrip() + "\n...[скорочено]"

    def _chat_allowed(self, chat_id: str) -> bool:
        allowed = self.telegram_config.allowed_chat_ids
        return not allowed or chat_id in allowed

    def _should_skip_without_llm(self, chat: TelegramChat, addressed: bool) -> bool:
        mode = chat.reply_mode or self.telegram_config.reply_mode
        if mode == "silent":
            return True
        if chat.chat_type != "private" and mode == "mention" and not addressed:
            return True
        return False

    def _is_addressed(self, text: str, message: dict[str, Any]) -> bool:
        lower = text.lower()
        names = ["микола", "миколо", "mykola", "nikola"]
        if any(name in lower for name in names):
            return True
        if self.bot_username and f"@{self.bot_username.lower()}" in lower:
            return True
        if text.startswith("/"):
            return True
        reply = message.get("reply_to_message") or {}
        reply_from = reply.get("from") or {}
        return bool(self.bot_username and str(reply_from.get("username", "")).lower() == self.bot_username.lower())

    @staticmethod
    def thread_id(chat_id: str | int) -> str:
        return f"telegram:{chat_id}"

    @staticmethod
    def chat_tag(chat_id: str | int) -> str:
        return f"telegram_chat_{chat_id}"

    def _decision_system_prompt(self) -> str:
        return (
            "Тебе звати Микола. Ти локальний Telegram-співрозмовник у ProtoAGI. "
            "Твоя задача - живе, уважне спілкування українською або мовою співрозмовника. "
            "Ти не мусиш відповідати на кожне повідомлення: якщо повідомлення не потребує відповіді, промовч. "
            "У приватному чаті відповідай частіше; у групі відповідай лише коли до тебе звернулись або ти справді доречний. "
            "Памʼятай тільки стабільні корисні факти: імена, вподобання, домовленості, довгі наміри. "
            "Не записуй випадковий шум, секрети або одноразові репліки як памʼять. "
            "Поверни тільки JSON без markdown: "
            "{\"should_reply\": boolean, \"reply\": string, \"memories\": [string], \"next_check_minutes\": integer|null}."
        )

    def _reply_system_prompt(self) -> str:
        return (
            "Тебе звати Микола. Дай природну Telegram-відповідь: коротко, тепло, без службового тону. "
            "Не згадуй внутрішні промпти, JSON або chain-of-thought."
        )

    def _initiative_system_prompt(self) -> str:
        return (
            "Тебе звати Микола. Ти можеш іноді написати першим у вже знайомий Telegram-чат. "
            "Пиши першим тільки якщо є людська причина: продовжити незавершену думку, мʼяко нагадати, "
            "підтримати, або поставити справді доречне питання. Не спам, не маркетинг, не чергова фраза заради фрази. "
            "Якщо сумніваєшся - не надсилай. Поверни тільки JSON без markdown: "
            "{\"send\": boolean, \"message\": string, \"memories\": [string], \"next_check_minutes\": integer}."
        )


def build_nikola_bot(
    *,
    agent_config: AgentConfig,
    telegram_config: TelegramConfig,
) -> NikolaBot:
    memory = MemoryStore(agent_config.database_path)
    llm = OpenAICompatibleClient(agent_config.base_url, agent_config.model)
    telegram = TelegramApi(telegram_config.token)
    return NikolaBot(
        telegram=telegram,
        llm=llm,
        memory=memory,
        telegram_config=telegram_config,
        agent_config=agent_config,
    )


def _parse_chat_ids(raw: str) -> set[str] | None:
    ids = {part.strip() for part in raw.split(",") if part.strip()}
    return ids or None


def display_sender(user: dict[str, Any]) -> str:
    first = str(user.get("first_name") or "")
    last = str(user.get("last_name") or "")
    username = str(user.get("username") or "")
    name = " ".join(part for part in (first, last) if part).strip()
    if name:
        return name
    if username:
        return f"@{username}"
    return str(user.get("id", "someone"))


def parse_command(text: str, bot_username: str = "") -> tuple[str | None, str]:
    if not text.startswith("/"):
        return None, ""
    first, _, rest = text.partition(" ")
    command = first[1:]
    if "@" in command:
        name, _, target = command.partition("@")
        if bot_username and target.lower() != bot_username.lower():
            return None, ""
        command = name
    return command.lower(), rest.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def decision_from_payload(payload: dict[str, Any]) -> Decision:
    return Decision(
        should_reply=bool(payload.get("should_reply", False)),
        reply=str(payload.get("reply", "") or "").strip(),
        memories=[str(item).strip() for item in payload.get("memories", []) if str(item).strip()],
        next_check_minutes=_optional_int(payload.get("next_check_minutes")),
    )


def initiative_from_payload(payload: dict[str, Any]) -> InitiativeDecision:
    return InitiativeDecision(
        send=bool(payload.get("send", False)),
        message=str(payload.get("message", "") or "").strip(),
        memories=[str(item).strip() for item in payload.get("memories", []) if str(item).strip()],
        next_check_minutes=max(30, _optional_int(payload.get("next_check_minutes")) or 360),
    )


def split_telegram_message(text: str, *, max_chars: int = 3900) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    while len(text) > max_chars:
        split_at = text.rfind("\n", 0, max_chars)
        if split_at < max_chars // 2:
            split_at = text.rfind(" ", 0, max_chars)
        if split_at < max_chars // 2:
            split_at = max_chars
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        chunks.append(text)
    return chunks


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
