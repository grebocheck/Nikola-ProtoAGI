import tempfile
import unittest
from pathlib import Path

from protoagi.config import ToolPolicy
from protoagi.memory import MemoryStore
from protoagi.tools import ToolContext, ToolRegistry, _validate_public_url


class SsrfTests(unittest.TestCase):
    def test_blocks_loopback_hostname(self) -> None:
        self.assertIsNotNone(_validate_public_url("http://localhost/secret"))
        self.assertIsNotNone(_validate_public_url("http://127.0.0.1:8080/v1/models"))

    def test_blocks_link_local(self) -> None:
        self.assertIsNotNone(_validate_public_url("http://169.254.169.254/latest/meta-data/"))

    def test_blocks_credentialed_url(self) -> None:
        self.assertIsNotNone(_validate_public_url("http://user:pass@example.com/"))

    def test_rejects_non_http_schemes(self) -> None:
        self.assertIsNotNone(_validate_public_url("file:///etc/passwd"))
        self.assertIsNotNone(_validate_public_url("ftp://example.com/file"))


class ShellBlocklistTests(unittest.TestCase):
    def _make_registry(self, *, allow_unsafe: bool = False) -> ToolRegistry:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        memory = MemoryStore(root / "memory.sqlite3")
        policy = ToolPolicy(allow_shell=True, allow_unsafe_shell=allow_unsafe)
        return ToolRegistry(ToolContext(root=root, memory=memory, policy=policy))

    def tearDown(self) -> None:
        tmp = getattr(self, "tmp", None)
        if tmp is not None:
            tmp.cleanup()

    def test_blocks_remove_item_with_pipe(self) -> None:
        registry = self._make_registry()
        result = registry.execute(
            "run_powershell",
            {"command": "Get-ChildItem | Remove-Item"},
        )
        self.assertFalse(result["ok"])
        self.assertIn("blocked", result["error"].lower())

    def test_blocks_reg_delete(self) -> None:
        registry = self._make_registry()
        result = registry.execute(
            "run_powershell",
            {"command": "reg delete HKLM\\Software\\Foo"},
        )
        self.assertFalse(result["ok"])
        self.assertIn("blocked", result["error"].lower())

    def test_unsafe_shell_skips_blocklist(self) -> None:
        registry = self._make_registry(allow_unsafe=True)
        # We do not actually run a destructive command here, but the
        # blocklist must not return an error for a normal command in unsafe
        # mode — exit code is independent of the blocklist.
        result = registry.execute(
            "run_powershell",
            {"command": "Write-Output ok"},
        )
        # Either succeeds (powershell available) or fails for OS reasons —
        # but the "blocked" path must not trigger.
        if not result["ok"] and result.get("error"):
            self.assertNotIn("blocked", str(result["error"]).lower())


if __name__ == "__main__":
    unittest.main()
