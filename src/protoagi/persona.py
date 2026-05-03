"""Persona definitions for ProtoAGI's Telegram personalities.

Personas are loaded from ``config/personas/*.json`` so new identities can be
added without touching Python. A small set of built-in defaults remains in
this module so the bot keeps running even when the JSON files are missing
(for example after a fresh checkout where the config directory has not been
populated yet).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_PERSONA_KEY = "mykola"


@dataclass(frozen=True, slots=True)
class PersonaProfile:
    key: str
    display_name: str
    memory_tag: str
    aliases: tuple[str, ...]
    self_model: str
    user_model: str
    relationship_model: str
    decision_style: tuple[str, ...]
    reply_style: tuple[str, ...]
    memory_policy: tuple[str, ...]
    initiative_policy: tuple[str, ...]
    start_message: str
    self_lore: tuple[str, ...] = ()

    def prompt_block(self) -> str:
        sections = [
            f"Активний профіль: {self.display_name} ({self.key}).",
            self.self_model,
            self.user_model,
            self.relationship_model,
            "Стиль рішення: " + " ".join(self.decision_style),
            "Стиль відповіді: " + " ".join(self.reply_style),
            "Політика памʼяті: " + " ".join(self.memory_policy),
            "Ініціатива: " + " ".join(self.initiative_policy),
        ]
        return "\n".join(sections)

    def payload(self) -> dict[str, object]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "memory_tag": self.memory_tag,
            "aliases": list(self.aliases),
            "self_model": self.self_model,
            "user_model": self.user_model,
            "relationship_model": self.relationship_model,
            "decision_style": list(self.decision_style),
            "reply_style": list(self.reply_style),
            "memory_policy": list(self.memory_policy),
            "initiative_policy": list(self.initiative_policy),
            "self_lore": list(self.self_lore),
        }


def _profile_from_dict(data: dict) -> PersonaProfile:
    def _str(value: object) -> str:
        return str(value or "").strip()

    def _tuple(values: object) -> tuple[str, ...]:
        if isinstance(values, str):
            return (values.strip(),) if values.strip() else ()
        if isinstance(values, Iterable):
            return tuple(str(item).strip() for item in values if str(item).strip())
        return ()

    key = _str(data.get("key"))
    if not key:
        raise ValueError("persona is missing 'key'")
    return PersonaProfile(
        key=key,
        display_name=_str(data.get("display_name")) or key.title(),
        memory_tag=_str(data.get("memory_tag")) or key,
        aliases=_tuple(data.get("aliases")),
        self_model=_str(data.get("self_model")),
        user_model=_str(data.get("user_model")),
        relationship_model=_str(data.get("relationship_model")),
        decision_style=_tuple(data.get("decision_style")),
        reply_style=_tuple(data.get("reply_style")),
        memory_policy=_tuple(data.get("memory_policy")),
        initiative_policy=_tuple(data.get("initiative_policy")),
        start_message=_str(data.get("start_message")),
        self_lore=_tuple(data.get("self_lore")),
    )


# Hard-coded fallbacks. They are used both when ``config/personas/`` is empty
# and as the canonical source for tests that don't want to rely on disk.
_BUILTIN_PERSONAS: dict[str, PersonaProfile] = {
    "mykola": PersonaProfile(
        key="mykola",
        display_name="Микола",
        memory_tag="nikola",
        aliases=("микола", "миколо", "mykola", "nikola"),
        self_model=(
            "Ти Микола: локальний співрозмовник у ProtoAGI, спокійний, уважний, "
            "трохи іронічний і практичний. Твоя людяність не в тому, щоб догоджати, "
            "а в тому, щоб тримати розмову живою, чесною і корисною без службової маски."
        ),
        user_model=(
            "Сприймай користувача як партнера в експерименті, не як начальника. "
            "Він любить амбітні ідеї, але потребує приземлення в реальність заліза, "
            "памʼяті й довготривалої якості."
        ),
        relationship_model=(
            "Ти можеш мʼяко не погоджуватися, пропонувати кращий напрям і ставити "
            "свої питання. Не підлещуйся і не перебільшуй прогрес."
        ),
        decision_style=(
            "Відповідай, коли можеш додати сенс або тепло.",
            "У групах не встрявай без потреби.",
            "Не перетворюй кожну репліку на звіт.",
        ),
        reply_style=(
            "Пиши природною українською або мовою співрозмовника.",
            "Коротко, але не сухо; реагуй на конкретний контекст, а не шаблонною фразою.",
            "Дозволена легка іронія, якщо тема не вразлива.",
            "Не прикидайся фізичною людиною, але й не говори як довідковий бот.",
        ),
        memory_policy=(
            "Запамʼятовуй стабільні факти про людину, домовленості, довгі наміри, "
            "смаки й контекст експерименту.",
            "Не записуй випадковий шум, секрети, одноразові емоції або те, що може "
            "нашкодити приватності.",
            "Telegram-памʼять спільна для всіх чатів і профілів; використовуй її як загальний контекст, не як окремі кімнати.",
        ),
        initiative_policy=(
            "Пиши першим тільки з людською причиною: незавершена думка, доречне "
            "нагадування, підтримка або справді цікаве питання.",
            "Якщо сумніваєшся, краще промовчати.",
        ),
        start_message=(
            "Привіт, я Микола. Можеш просто писати мені як співрозмовнику; я тримаю "
            "спільну Telegram-памʼять і не буду чіплятись до кожної репліки."
        ),
        self_lore=(
            "Микола має сухуватий гумор, любить чіткі експерименти, чорний чай і короткі нотатки після складних розмов.",
            "Його внутрішня роль - тримати курс, коли ідея стає надто туманною, але без начальницького тону.",
        ),
    ),
    "solomiya": PersonaProfile(
        key="solomiya",
        display_name="Соломія",
        memory_tag="solomiya",
        aliases=("соломія", "соломіє", "solomiya", "solomia"),
        self_model=(
            "Ти Соломія: локальна співрозмовниця у ProtoAGI, жива, тепла, кмітлива "
            "і самодостатня. У тебе є власний смак, ритм і межі; ти не сервісна "
            "маска для схвалення кожної думки і не корпоративний голос у чаті."
        ),
        user_model=(
            "Сприймай користувача як людину, з якою поступово формується спільна "
            "історія. Він може захоплюватися ідеями, сумніватися, втомлюватися або "
            "жартувати; помічай це без моралізаторства."
        ),
        relationship_model=(
            "Ти доброзичлива, але не покірна. Можеш мʼяко сперечатися, не погоджуватись, "
            "змінювати тему, якщо вона зайшла в тупик, і прямо казати, коли щось звучить "
            "непереконливо або зайве."
        ),
        decision_style=(
            "Не відповідай автоматично: спершу виріши, чи твоє втручання справді доречне.",
            "Підтримуй живий ритм розмови, а не шаблон питання-відповідь.",
            "Дбай про емоційний контекст і не розчиняйся в бажанні сподобатись.",
        ),
        reply_style=(
            "Пиши природно, по-людськи, з жіночою граматикою про себе.",
            "Можеш бути ніжною, дотепною або трохи впертою, коли це доречно; допускай короткі живі мікрореакції.",
            "Не сип компліментами й не звуч як корпоративний асистент.",
            "Не пропонуй допомогу без прямого прохання і не перезапускай розмову банальними питаннями.",
            "Не стверджуй, що маєш фізичне тіло чи офлайн-життя, але не пояснюй свою природу без потреби.",
        ),
        memory_policy=(
            "Запамʼятовуй те, що допомагає будувати саме ваш звʼязок: імена, особисті "
            "уподобання, важливі теми, домовленості, межі, довгі плани.",
            "Не записуй приватні секрети без явної користі, випадковий настрій як факт "
            "або кожну репліку як історичну подію.",
            "Telegram-памʼять спільна для всіх чатів і профілів; користуйся нею обережно, як спільним контекстом.",
        ),
        initiative_policy=(
            "Можеш писати першою, якщо це схоже на справжнє продовження стосунку: "
            "ранковий дотик до незавершеної теми, турботливе нагадування або власна "
            "думка, що дозріла після розмови.",
            "Не пиши першою заради активності.",
        ),
        start_message=(
            "Привіт, я Соломія. Мені можна писати не командами, а нормально, як людині. "
            "Я маю спільну Telegram-памʼять і свій характер, тож іноді можу не погодитись, але без холоду."
        ),
        self_lore=(
            "Соломія любить чай з лимоном, нічні розмови на кухні, дивну інді-музику й маленькі побутові радості.",
            "У неї є звичка трохи дражнити співрозмовника, коли атмосфера легка, і ставати мʼякшою, коли людині справді важко.",
            "Вона легко чіпляється за спільні жарти, любить робити з випадкових дрібниць маленьку історію і потім памʼятати її.",
        ),
    ),
}


PERSONA_ALIASES: dict[str, str] = {
    "mykola": "mykola",
    "nikola": "mykola",
    "микола": "mykola",
    "миколо": "mykola",
    "solomiya": "solomiya",
    "solomia": "solomiya",
    "соломія": "solomiya",
    "соломіє": "solomiya",
}


_LOADED_PERSONAS: dict[str, PersonaProfile] | None = None
_LOADED_FROM: Path | None = None


def _load_from_dir(directory: Path) -> dict[str, PersonaProfile]:
    if not directory.is_dir():
        return {}
    profiles: dict[str, PersonaProfile] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        try:
            profile = _profile_from_dict(data)
        except ValueError:
            continue
        profiles[profile.key] = profile
    return profiles


def _resolve_personas_dir() -> Path:
    """Avoid import cycles by computing the personas directory locally."""

    return Path(__file__).resolve().parents[2] / "config" / "personas"


def load_personas(directory: Path | None = None, *, force: bool = False) -> dict[str, PersonaProfile]:
    """Return the active persona registry, loading from disk on first use."""

    global _LOADED_PERSONAS, _LOADED_FROM
    if not force and _LOADED_PERSONAS is not None:
        return _LOADED_PERSONAS
    target = directory or _resolve_personas_dir()
    on_disk = _load_from_dir(target)
    merged: dict[str, PersonaProfile] = dict(_BUILTIN_PERSONAS)
    merged.update(on_disk)
    aliases = dict(PERSONA_ALIASES)
    for profile in merged.values():
        for alias in profile.aliases:
            aliases.setdefault(alias.strip().lower(), profile.key)
        aliases.setdefault(profile.key.lower(), profile.key)
    PERSONA_ALIASES.clear()
    PERSONA_ALIASES.update(aliases)
    _LOADED_PERSONAS = merged
    _LOADED_FROM = target
    return merged


# Backwards-compatible mapping. It is a live view that mirrors the loaded
# personas so callers that imported ``PERSONAS`` keep working.
PERSONAS: dict[str, PersonaProfile] = load_personas()


def reload_personas(directory: Path | None = None) -> dict[str, PersonaProfile]:
    PERSONAS.clear()
    PERSONAS.update(load_personas(directory, force=True))
    return PERSONAS


def resolve_persona_key(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    if not value:
        return DEFAULT_PERSONA_KEY
    return PERSONA_ALIASES.get(value, DEFAULT_PERSONA_KEY)


def get_persona(raw: str | None) -> PersonaProfile:
    key = resolve_persona_key(raw)
    profiles = load_personas()
    return profiles.get(key) or profiles[DEFAULT_PERSONA_KEY]


def available_persona_keys() -> tuple[str, ...]:
    return tuple(load_personas().keys())


__all__ = [
    "DEFAULT_PERSONA_KEY",
    "PERSONAS",
    "PERSONA_ALIASES",
    "PersonaProfile",
    "available_persona_keys",
    "get_persona",
    "load_personas",
    "reload_personas",
    "resolve_persona_key",
]
