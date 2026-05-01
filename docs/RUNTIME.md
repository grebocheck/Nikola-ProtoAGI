# Runtime Notes

## Recommended first launch

```powershell
.\scripts\start-server.ps1 -CtxSize 8192 -CpuMoE 4
```

This profile keeps several early MoE layers on CPU. It is slower than full GPU
but safer on 16 GB VRAM under Windows.

The launch script uses `--reasoning-format deepseek` so internal reasoning is
returned separately by llama.cpp instead of leaking into `message.content`.

## If VRAM is comfortable

Try full GPU:

```powershell
.\scripts\start-server.ps1 -CtxSize 8192 -FullGpu
```

Then compare latency and GPU memory.

## If VRAM is tight

Increase CPU MoE offload:

```powershell
.\scripts\start-server.ps1 -CtxSize 8192 -CpuMoE 8
```

If the system still pages VRAM, close GPU-heavy apps and retry.

## Context scaling

Start with:

- 8192 for stable tool work
- 16384 after a successful 8k benchmark
- 32768 only if memory use remains stable

Large context is useful, but a slow unstable agent is worse than a smaller
stable one.

## Benchmark matrix

```powershell
.\scripts\bench-llama.ps1 -CpuMoE "0,4,8,12"
```

The output is written to `runs/llama-bench.jsonl`.

## Agent endpoint benchmark

Start the server first, then:

```powershell
$env:PYTHONPATH="src"
python -m protoagi bench --rounds 3
```
