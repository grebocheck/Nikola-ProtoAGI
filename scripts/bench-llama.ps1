param(
    [string]$Output = "runs\llama-bench.jsonl",
    [string]$CpuMoE = "0,4,8,12",
    [int]$BatchSize = 1024,
    [int]$UBatchSize = 1024,
    [int]$PromptTokens = 512,
    [int]$GenTokens = 128,
    [int]$Repetitions = 2
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $Root "src"
python -m protoagi bench-llama `
    --output $Output `
    --n-cpu-moe $CpuMoE `
    --batch-size $BatchSize `
    --ubatch-size $UBatchSize `
    --prompt-tokens $PromptTokens `
    --gen-tokens $GenTokens `
    --repetitions $Repetitions
