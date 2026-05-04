from pathlib import Path
import tempfile
import unittest

from protoagi.config import ToolPolicy
from protoagi.agent_tools.core import ToolContext, ToolRegistry
from protoagi.storage.memory import MemoryStore


class ToolTests(unittest.TestCase):
    def make_registry(self, *, allow_write: bool = True, allow_shell: bool = False) -> ToolRegistry:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        memory = MemoryStore(root / "memory.sqlite3")
        return ToolRegistry(
            ToolContext(
                root=root,
                memory=memory,
                policy=ToolPolicy(allow_write=allow_write, allow_shell=allow_shell),
            )
        )

    def tearDown(self) -> None:
        tmp = getattr(self, "tmp", None)
        if tmp is not None:
            tmp.cleanup()

    def test_write_and_read_file(self) -> None:
        registry = self.make_registry(allow_write=True)
        written = registry.execute("write_file", {"path": "notes/a.txt", "content": "hello"})
        self.assertTrue(written["ok"])
        read = registry.execute("read_file", {"path": "notes/a.txt"})
        self.assertTrue(read["ok"])
        self.assertEqual(read["data"]["content"], "hello")

    def test_path_escape_is_blocked(self) -> None:
        registry = self.make_registry(allow_write=True)
        result = registry.execute("read_file", {"path": "../outside.txt"})
        self.assertFalse(result["ok"])
        self.assertIn("escapes workspace", result["error"])

    def test_shell_denied_by_default(self) -> None:
        registry = self.make_registry(allow_shell=False)
        result = registry.execute("run_powershell", {"command": "Get-ChildItem"})
        self.assertFalse(result["ok"])
        self.assertIn("denied", result["error"])


if __name__ == "__main__":
    unittest.main()
