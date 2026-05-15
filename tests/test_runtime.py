import unittest
from pathlib import Path

from protoagi.config import LlamaServerProfile


class RuntimeTests(unittest.TestCase):
    def test_server_command_contains_gpt_oss_critical_flags(self) -> None:
        profile = LlamaServerProfile(ctx_size=8192, n_cpu_moe=4)
        cmd = profile.server_command()
        joined = " ".join(cmd)
        self.assertIn("--ctx-size 8192", joined)
        self.assertIn("--jinja", cmd)
        self.assertIn("--skip-chat-parsing", cmd)
        self.assertIn("-fa on", joined)
        self.assertIn("--n-cpu-moe 4", joined)

if __name__ == "__main__":
    unittest.main()
