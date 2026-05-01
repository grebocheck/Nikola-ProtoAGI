from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import uuid
from typing import Any

from .config import AgentConfig
from .harmony import clean_model_content
from .memory import MemoryStore
from .openai_compat import OpenAICompatibleClient
from .tools import ToolRegistry, result_to_tool_content


SYSTEM_PROMPT = """You are ProtoAGI, a local experimental agent running on the user's machine.

Mission:
- Solve user tasks with a disciplined plan-act-observe-reflect loop.
- Use tools when they materially improve correctness.
- Store durable, useful facts with remember; retrieve them with recall.
- Inspect files before changing them.
- Keep final answers concise and in the user's language.

Constraints:
- Do not claim to be true AGI. You are an experimental local agent scaffold.
- Do not expose hidden chain-of-thought. Summarize reasoning briefly when useful.
- Prefer small reversible workspace edits.
- If shell access is denied by policy, use non-shell tools or explain the blocker.
- For code changes, verify with tests or a concrete command whenever possible.
"""


@dataclass(slots=True)
class AgentRun:
    thread_id: str
    final: str
    steps: int
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    raw_response: dict[str, Any] | None = None


class ProtoAgent:
    def __init__(
        self,
        *,
        config: AgentConfig,
        client: OpenAICompatibleClient,
        memory: MemoryStore,
        tools: ToolRegistry,
    ) -> None:
        self.config = config
        self.client = client
        self.memory = memory
        self.tools = tools

    def run(self, user_prompt: str, *, thread_id: str | None = None, max_steps: int = 8) -> AgentRun:
        thread_id = thread_id or f"thread-{uuid.uuid4().hex[:12]}"
        recalled = self.memory.search(user_prompt, limit=5)
        memory_context = "\n".join(
            f"- [{fact.id}] {fact.text} (tags: {', '.join(fact.tags)})" for fact in recalled
        )
        system = SYSTEM_PROMPT
        if memory_context:
            system += "\nRelevant durable memory:\n" + memory_context + "\n"

        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        messages.extend(self.memory.recent_messages(thread_id, limit=10))
        messages.append({"role": "user", "content": user_prompt})
        self.memory.log_message(thread_id, "user", user_prompt)

        tool_events: list[dict[str, Any]] = []
        raw_response: dict[str, Any] | None = None

        for step in range(1, max_steps + 1):
            response = self.client.chat_completion(
                messages,
                tools=self.tools.schemas(),
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
            )
            raw_response = response
            message = response.get("choices", [{}])[0].get("message", {})
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []

            fallback_call = self._extract_fallback_tool_call(content)
            if not tool_calls and fallback_call is not None:
                tool_calls = [fallback_call]
                content = ""

            if not tool_calls:
                final = clean_model_content(content)
                self.memory.log_message(thread_id, "assistant", final)
                return AgentRun(
                    thread_id=thread_id,
                    final=final,
                    steps=step,
                    tool_events=tool_events,
                    raw_response=raw_response,
                )

            assistant_message: dict[str, Any] = {"role": "assistant", "content": content, "tool_calls": tool_calls}
            messages.append(assistant_message)

            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                name = str(function.get("name", ""))
                arguments = self._parse_arguments(function.get("arguments", {}))
                result = self.tools.execute(name, arguments)
                event = {"name": name, "arguments": arguments, "result": result}
                tool_events.append(event)
                self.memory.log_tool_event(thread_id, name, arguments, result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "name": name,
                        "content": result_to_tool_content(result),
                    }
                )

        final = self._finalize_after_step_limit(messages)
        self.memory.log_message(thread_id, "assistant", final)
        return AgentRun(
            thread_id=thread_id,
            final=final,
            steps=max_steps,
            tool_events=tool_events,
            raw_response=raw_response,
        )

    def _finalize_after_step_limit(self, messages: list[dict[str, Any]]) -> str:
        messages = [
            *messages,
            {
                "role": "user",
                "content": (
                    "The tool-use step budget is exhausted. Produce the best final answer now "
                    "using the observations already available. Do not request more tools."
                ),
            },
        ]
        try:
            response = self.client.chat_completion(
                messages,
                tools=None,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
            )
        except Exception:  # noqa: BLE001 - final fallback should not mask the run.
            return (
                "I reached the step limit before a final answer. "
                "The latest tool results were captured in memory."
            )
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return clean_model_content(content) or (
            "I reached the step limit before a final answer. "
            "The latest tool results were captured in memory."
        )

    @staticmethod
    def _parse_arguments(arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if not arguments:
            return {}
        try:
            loaded = json.loads(str(arguments))
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _extract_fallback_tool_call(content: str) -> dict[str, Any] | None:
        """Accept a simple JSON fallback if a server does not emit tool_calls."""
        match = re.search(r"\{[\s\S]*\"tool\"[\s\S]*\"arguments\"[\s\S]*\}", content.strip())
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict) or "tool" not in payload:
            return None
        return {
            "id": f"fallback_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": str(payload.get("tool")),
                "arguments": json.dumps(payload.get("arguments", {}), ensure_ascii=False),
            },
        }
