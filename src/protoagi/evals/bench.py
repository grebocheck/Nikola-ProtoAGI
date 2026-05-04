from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time

from ..config import LlamaServerProfile
from ..harmony import clean_model_content
from ..openai_compat import OpenAICompatibleClient


@dataclass(slots=True)
class EndpointBenchResult:
    round_index: int
    seconds: float
    chars: int
    preview: str


def bench_endpoint(
    client: OpenAICompatibleClient,
    *,
    prompt: str,
    rounds: int = 3,
    max_tokens: int = 256,
) -> list[EndpointBenchResult]:
    results: list[EndpointBenchResult] = []
    messages = [
        {
            "role": "system",
            "content": "You are a concise benchmark assistant. Answer directly.",
        },
        {"role": "user", "content": prompt},
    ]
    for index in range(1, rounds + 1):
        start = time.perf_counter()
        response = client.chat_completion(messages, temperature=0.2, top_p=1.0, max_tokens=max_tokens)
        elapsed = time.perf_counter() - start
        content = clean_model_content(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
        results.append(
            EndpointBenchResult(
                round_index=index,
                seconds=elapsed,
                chars=len(content),
                preview=content.replace("\n", " ")[:160],
            )
        )
    return results


def run_llama_bench(
    profile: LlamaServerProfile,
    *,
    output_path: Path,
    n_cpu_moe_values: list[int],
    prompt_tokens: int = 512,
    gen_tokens: int = 128,
    repetitions: int = 2,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(profile.bench_exe),
        "-m",
        str(profile.model_path),
        "-ngl",
        "99",
        "-fa",
        "1",
        "-b",
        str(profile.batch_size),
        "-ub",
        str(profile.ubatch_size),
        "-p",
        str(prompt_tokens),
        "-n",
        str(gen_tokens),
        "-r",
        str(repetitions),
        "-o",
        "jsonl",
        "-ncmoe",
        ",".join(str(value) for value in n_cpu_moe_values),
    ]
    completed = subprocess.run(cmd, cwd=profile.llama_dir, capture_output=True, text=True)
    output_path.write_text(completed.stdout, encoding="utf-8")
    err_path = output_path.with_suffix(output_path.suffix + ".stderr.txt")
    err_path.write_text(completed.stderr, encoding="utf-8")
    return completed.returncode


def endpoint_results_to_json(results: list[EndpointBenchResult]) -> str:
    return json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False)
