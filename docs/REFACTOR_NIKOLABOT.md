# Design note — taming the `NikolaBot` god-object (Phase R)

Status: **proposed** (awaiting go-ahead). Author pass: 2026-06-09.

## Problem

[`telegram/orchestrator.py`](../src/protoagi/telegram/orchestrator.py) is
~3400 lines. `NikolaBot` alone has **91 methods**. Every change to the
Telegram logic — a decision tweak, a new tool, a sending change — edits
the same enormous class, and no one can hold it in their head. This is
the top maintainability risk flagged in the 2026-06-09 audit.

[`storage/memory.py`](../src/protoagi/storage/memory.py) (2717 lines) is
the second god-object; it is **out of scope for this note** and handled
as a later sub-phase (R5) because its split is along table/feature lines,
not behavioural seams.

## Constraint that shapes the approach

`NikolaBot` is one object whose methods freely share instance state
(`self.memory`, `self.telegram`, `self.persona`, `self.telegram_config`,
per-chat locks, cooldown caches, …). Two ways to split it:

1. **Mixins** — move a cohesive method cluster into a `*Mixin` class in
   its own module; `NikolaBot` keeps inheriting it. No call-site changes,
   no state re-threading, behaviour provably identical (same `self`).
2. **Collaborators (composition)** — extract a real object that holds a
   back-reference to the bot. Cleaner long-term, but every `self.x`
   becomes `self._bot.x`, which is large churn and real regression risk.

The codebase **already uses option 1**: `NikolaBot` is
`class NikolaBot(TelegramAttachmentMixin, TelegramStickerMixin)`. We
extend that established, low-risk pattern. Composition can come later for
any cluster that proves to have a clean, narrow state interface — but it
is not the goal of Phase R.

### Why mixins are safe here

- The mixin sees the same `self`, so no behaviour can change.
- `mypy --strict` now runs for real (Phase 14), so a mistyped or missing
  attribute on a mixin is caught. We declare shared attributes once via a
  `typing.Protocol` (or a thin typed base) so each mixin type-checks in
  isolation instead of relying on `Any`.
- 385 unit tests exercise the orchestrator; each extraction step is
  green-to-green.

## Proposed clusters (the seams)

Grouping the 91 methods by responsibility. Each becomes one module +
mixin under `src/protoagi/telegram/orchestrator/` (the file becomes a
package; the public `NikolaBot` / `build_nikola_bot` API is unchanged and
re-exported from `orchestrator/__init__.py`).

| Mixin / module | Responsibility | Representative methods |
| --- | --- | --- |
| `lifecycle.py` | supervisor loop, bootstrap, endpoint warnings, error log | `bootstrap`, `run_forever`, `poll_once`, `maybe_run_initiative`, `maybe_dispatch_reminders`, `maybe_run_reflection`, `_warn_about_unreachable_endpoints`, `_log_loop_exception`, `_rotate_error_log` |
| `maintenance.py` | reflection, consolidation, conflict resolution, user-state refresh, reminders dispatch | `run_reflection_pass`, `_write_reflection_memory`, `try_resolve_conflict`, `_run_conflict_resolution`, `refresh_user_state`, `_maybe_bootstrap_user_state`, `_run_user_state_refresh`, `dispatch_due_reminders` |
| `decisions.py` | incoming pipeline + initiative decisions | `process_update`, `decide_incoming`, `_merge_decision_tool_results`, `_inline_tool_decision`, `_record_decision_metrics`, `_tool_result_reply`, `compose_reply`, `run_initiative_once`, `decide_initiative`, `_handle_command`, `_handle_edited_message`, `_should_skip_without_llm`, `_is_addressed` |
| `reactions_mixin.py` | emoji reaction policy | `_handle_message_reaction`, `_apply_reactions`, `_reaction_denylist`, `_reaction_cooldown_active`, `_mark_reaction_sent`, `_record_reaction_denylist` |
| `transport.py` | outbound: send, chat actions, TTS/voice delivery, reply shaping | `_send_reply`, `_send_chat_action_safely`, `_chat_action_loop`, `_log_sent_telegram_message`, `_try_send_tts_reply`, `_should_send_auto_voice_reply`, `_mark_auto_voice_reply_sent`, `_resolve_reply_target`, `_send_tts_audio`, `_limit_reply`, `_clean_reply_text` |
| `memory_write.py` | persisting facts/media/voice/goals/reminders/notes | `_remember_chat_fact`, `_remember_media_fact`, `_remember_voice_fact`, `_transcribe_voice`, `_persist_reminder_requests`, `_remember_persona_self_fact`, `_persist_temporary_notes`, `_persist_goal_actions` |
| `context.py` | building the prompt context payloads (recall, stickers, goals, history, reasoning capture) | `_relevant_memory_payload`, `_tensions_for_facts`, `_available_stickers_payload`, `_rank_stickers`, `_user_state_payload`, `_open_goals_payload`, `_due_goals_payload`, `_search_chat_memory`, `_persona_self_context`, `_fact_view`, `_style_payload`, `_persona_context_payload`, `_recent_*`, `_history_*`, `_incoming_text_with_media`, `_capture_reasoning` |
| `orchestrator/__init__.py` (core) | the `NikolaBot` class body: `__init__`, attribute wiring, the small glue helpers (`_chat_lock`, `thread_id`, `chat_tag`, `_chat_allowed`, prompt-builder helpers), and the mixin composition | — |

Module-level free functions (`_decision_to_payload`, `_goal_summary`,
`_compact_prompt_payload`, …) move to a `orchestrator/payloads.py`
helper module; they are already stateless and trivially relocatable.

## Sequencing (one PR-sized step each, green-to-green)

Ordered from lowest-risk / least-entangled to most:

- **R0** Turn `orchestrator.py` into the `orchestrator/` package with the
  whole class still in `orchestrator/__init__.py`; add the shared-state
  `Protocol`. No methods move yet. Verifies the package split + imports
  in isolation. *(XS)*
- **R1** Extract the stateless module-level functions → `payloads.py`. *(S)*
- **R2** `reactions_mixin.py` — small, self-contained, few cross-calls. *(S)*
- **R3** `transport.py` — outbound side, clear boundary. *(M)*
- **R4** `maintenance.py` and `memory_write.py` — periodic + persistence. *(M)*
- **R5** `context.py` and `decisions.py` last — the most interconnected;
  by now everything they call already lives in a typed mixin. *(M–L)*
- **R6** *(separate, later)* begin the `memory.py` split along feature
  lines (items / reminders / goals / conflicts / media / kv). *(L)*

Each step: move methods verbatim, run `ruff` + `mypy --strict` +
`unittest` (385), commit. No behaviour edits ride along with a move; any
behaviour fix is a separate commit so review stays honest.

## Acceptance per step

- `git diff` for a move step is pure relocation (reviewable as such).
- `ruff check src/`, `mypy --strict src/protoagi/`, and the full unit
  suite are all green before commit.
- `from protoagi.telegram import NikolaBot, build_nikola_bot` and the
  existing test imports keep working unchanged.

## Risks & mitigations

- **Hidden cross-cluster calls.** A method in one mixin calling a private
  of another is fine (same `self`), but the `Protocol` must list it or
  mypy fails — that failure is the safety net, not a problem.
- **Import cycles** between mixin modules: avoided because mixins never
  import each other — they only import shared types and the `Protocol`.
- **Churn vs. value.** R0–R2 are cheap and already de-risk the worst of
  the file size; if appetite runs out, stopping after R3 still leaves the
  core class roughly half its current size.

## Open question for the reviewer

Mixins (recommended, lowest-risk, matches existing pattern) vs.
composition for one or two clusters where a clean state interface exists
(e.g. `transport` arguably only needs `telegram`, `telegram_config`, and
the error log). Default to mixins unless you want a composition pilot on
`transport`.
