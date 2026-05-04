from pathlib import Path
import tempfile
import unittest

from protoagi.agent import ProtoAgent
from protoagi.config import AgentConfig, ToolPolicy
from protoagi.agent_tools.core import default_registry
from protoagi.storage.memory import MemoryStore


class FakeAgentClient:
    def __init__(self) -> None:
        self.calls = []

    def chat_completion(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if kwargs.get("response_format"):
            if len([call for call in self.calls if call["kwargs"].get("response_format")]) == 1:
                content = '{"plan": ["read the README", "summarize useful changes"], "step": 1}'
            else:
                content = '{"plan": ["summarize useful changes"], "step": 2}'
            return {"choices": [{"message": {"content": content}}]}
        if kwargs.get("tools"):
            tool_calls = [
                {
                    "id": "call_read",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "README.md"}',
                    },
                }
            ]
            return {"choices": [{"message": {"content": "", "tool_calls": tool_calls}}]}
        return {"choices": [{"message": {"content": "done"}}]}


class FinalAfterToolClient(FakeAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.tool_rounds = 0

    def chat_completion(self, messages, **kwargs):
        if kwargs.get("tools"):
            self.tool_rounds += 1
            if self.tool_rounds == 1:
                return super().chat_completion(messages, **kwargs)
            self.calls.append({"messages": messages, "kwargs": kwargs})
            return {"choices": [{"message": {"content": "README inspected."}}]}
        return super().chat_completion(messages, **kwargs)


class AgentPlanReflectTests(unittest.TestCase):
    def test_agent_runs_initial_plan_and_one_reflection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("hello from readme", encoding="utf-8")
            memory = MemoryStore(root / "memory.sqlite3")
            config = AgentConfig(
                database_path=root / "memory.sqlite3",
                tool_policy=ToolPolicy(allow_write=True),
                plan_reflect=True,
                plan_call_limit=2,
            )
            client = FinalAfterToolClient()
            agent = ProtoAgent(
                config=config,
                client=client,
                memory=memory,
                tools=default_registry(memory, config.tool_policy, root=root),
            )
            run = agent.run("read README and summarize", max_steps=4)
            self.assertEqual(run.final, "README inspected.")
            self.assertEqual(run.plan, ["summarize useful changes"])
            self.assertEqual(len(run.plan_updates), 1)
            self.assertEqual(len(run.tool_events), 1)
            planning_calls = [call for call in client.calls if call["kwargs"].get("response_format")]
            self.assertEqual(len(planning_calls), 2)


if __name__ == "__main__":
    unittest.main()
