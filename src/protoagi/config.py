from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

from .env import env_bool, env_int, load_project_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "gpt-oss-20b-MXFP4.gguf"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "protoagi.sqlite3"
DEFAULT_LLAMA_DIR = PROJECT_ROOT / "tools" / "llama.cpp"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "protoagi.json"
EXAMPLE_CONFIG_PATH = PROJECT_ROOT / "config" / "protoagi.example.json"
DEFAULT_PERSONAS_DIR = PROJECT_ROOT / "config" / "personas"


def _path_from_config(value: str | os.PathLike[str], *, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


@dataclass(slots=True)
class ToolPolicy:
    allow_write: bool = True
    allow_shell: bool = False
    allow_unsafe_shell: bool = False
    command_timeout_seconds: int = 30
    max_tool_output_chars: int = 12000

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ToolPolicy":
        if not data:
            return cls()
        return cls(
            allow_write=bool(data.get("allow_write", True)),
            allow_shell=bool(data.get("allow_shell", False)),
            allow_unsafe_shell=bool(data.get("allow_unsafe_shell", False)),
            command_timeout_seconds=int(data.get("command_timeout_seconds", 30)),
            max_tool_output_chars=int(data.get("max_tool_output_chars", 12000)),
        )


@dataclass(slots=True)
class EmbeddingSettings:
    base_url: str = ""
    model: str = ""
    timeout_seconds: int = 30
    request_dimensions: int | None = None
    backend: str = "flat"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EmbeddingSettings":
        if not data:
            return cls()
        dims = data.get("request_dimensions")
        return cls(
            base_url=str(data.get("base_url", "")),
            model=str(data.get("model", "")),
            timeout_seconds=int(data.get("timeout_seconds", 30)),
            request_dimensions=int(dims) if dims else None,
            backend=str(data.get("backend", "flat") or "flat"),
        )


@dataclass(slots=True)
class AgentConfig:
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "gpt-oss-20b-MXFP4"
    database_path: Path = DEFAULT_DB_PATH
    temperature: float = 0.6
    top_p: float = 1.0
    max_tokens: int = 1536
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    llm_importance: bool = False
    plan_reflect: bool = True
    plan_call_limit: int = 2

    @classmethod
    def load(cls, path: Path | None = None) -> "AgentConfig":
        load_project_env(PROJECT_ROOT)
        config_path = path or DEFAULT_CONFIG_PATH
        data: dict[str, Any] = {}
        if config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))
        elif EXAMPLE_CONFIG_PATH.exists():
            data = json.loads(EXAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))

        embedding_data = dict(data.get("embedding") or {})
        if env_value := os.environ.get("PROTOAGI_EMBED_BASE_URL"):
            embedding_data["base_url"] = env_value
        if env_value := os.environ.get("PROTOAGI_EMBED_MODEL"):
            embedding_data["model"] = env_value
        if env_value := os.environ.get("PROTOAGI_EMBED_DIMENSIONS"):
            embedding_data["request_dimensions"] = env_value
        if env_value := os.environ.get("PROTOAGI_EMBED_TIMEOUT_SECONDS"):
            embedding_data["timeout_seconds"] = env_value
        if env_value := os.environ.get("PROTOAGI_EMBED_BACKEND"):
            embedding_data["backend"] = env_value

        return cls(
            base_url=os.environ.get("PROTOAGI_BASE_URL", data.get("base_url", cls.base_url)),
            model=os.environ.get("PROTOAGI_MODEL", data.get("model", cls.model)),
            database_path=_path_from_config(
                os.environ.get("PROTOAGI_DB", data.get("database_path", str(DEFAULT_DB_PATH)))
            ),
            temperature=float(os.environ.get("PROTOAGI_TEMPERATURE", data.get("temperature", 0.6))),
            top_p=float(os.environ.get("PROTOAGI_TOP_P", data.get("top_p", 1.0))),
            max_tokens=env_int("PROTOAGI_MAX_TOKENS", int(data.get("max_tokens", 1536))),
            tool_policy=ToolPolicy.from_dict(data.get("tool_policy")),
            embedding=EmbeddingSettings.from_dict(embedding_data),
            llm_importance=env_bool(
                "PROTOAGI_LLM_IMPORTANCE",
                bool(data.get("llm_importance", False)),
            ),
            plan_reflect=env_bool(
                "PROTOAGI_PLAN_REFLECT",
                bool(data.get("plan_reflect", True)),
            ),
            plan_call_limit=env_int(
                "PROTOAGI_PLAN_CALL_LIMIT",
                int(data.get("plan_call_limit", 2)),
            ),
        )

    def with_cli_overrides(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        allow_write: bool | None = None,
        allow_shell: bool | None = None,
        allow_unsafe_shell: bool | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        database_path: str | Path | None = None,
    ) -> "AgentConfig":
        policy = ToolPolicy(
            allow_write=self.tool_policy.allow_write if allow_write is None else allow_write,
            allow_shell=self.tool_policy.allow_shell if allow_shell is None else allow_shell,
            allow_unsafe_shell=(
                self.tool_policy.allow_unsafe_shell
                if allow_unsafe_shell is None
                else allow_unsafe_shell
            ),
            command_timeout_seconds=self.tool_policy.command_timeout_seconds,
            max_tool_output_chars=self.tool_policy.max_tool_output_chars,
        )
        return AgentConfig(
            base_url=base_url or self.base_url,
            model=model or self.model,
            database_path=(
                _path_from_config(database_path)
                if database_path is not None
                else self.database_path
            ),
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p if top_p is None else top_p,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            tool_policy=policy,
            embedding=self.embedding,
            llm_importance=self.llm_importance,
            plan_reflect=self.plan_reflect,
            plan_call_limit=self.plan_call_limit,
        )


@dataclass(slots=True)
class LlamaServerProfile:
    model_path: Path = DEFAULT_MODEL_PATH
    llama_dir: Path = DEFAULT_LLAMA_DIR
    host: str = "127.0.0.1"
    port: int = 8080
    ctx_size: int = 8192
    batch_size: int = 1024
    ubatch_size: int = 1024
    n_cpu_moe: int | None = 4
    flash_attn: str = "on"
    jinja: bool = True
    skip_chat_parsing: bool = True
    reasoning: str = "auto"
    reasoning_format: str = "deepseek"
    temperature: float = 1.0
    top_p: float = 1.0

    @property
    def server_exe(self) -> Path:
        return self.llama_dir / "llama-server.exe"

    @property
    def bench_exe(self) -> Path:
        return self.llama_dir / "llama-bench.exe"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def server_command(self) -> list[str]:
        cmd = [
            str(self.server_exe),
            "-m",
            str(self.model_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.ctx_size),
            "-fa",
            self.flash_attn,
            "-b",
            str(self.batch_size),
            "-ub",
            str(self.ubatch_size),
            "--temp",
            str(self.temperature),
            "--top-p",
            str(self.top_p),
            "--reasoning",
            self.reasoning,
            "--reasoning-format",
            self.reasoning_format,
        ]
        if self.jinja:
            cmd.append("--jinja")
        else:
            cmd.append("--no-jinja")
        if self.skip_chat_parsing:
            cmd.append("--skip-chat-parsing")
        else:
            cmd.append("--no-skip-chat-parsing")
        if self.n_cpu_moe is not None:
            cmd.extend(["--n-cpu-moe", str(self.n_cpu_moe)])
        return cmd
