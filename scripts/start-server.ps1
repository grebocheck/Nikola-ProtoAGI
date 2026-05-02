param(
    [string]$ModelPath = "",
    [int]$Port = 8080,
    [int]$CtxSize = 8192,
    [int]$BatchSize = 1024,
    [int]$UBatchSize = 1024,
    [int]$CpuMoE = 4,
    [switch]$FullGpu,
    [ValidateSet("on", "off", "auto")]
    [string]$FlashAttn = "on"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Server = Join-Path $Root "tools\llama.cpp\llama-server.exe"
if ($ModelPath -eq "") {
    $ModelPath = Join-Path $Root "gpt-oss-20b-MXFP4.gguf"
}

if (!(Test-Path $Server)) {
    throw "llama-server.exe not found at $Server"
}
if (!(Test-Path $ModelPath)) {
    throw "Model not found at $ModelPath"
}

$Args = @(
    "-m", $ModelPath,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--ctx-size", "$CtxSize",
    "--jinja",
    "-fa", $FlashAttn,
    "-b", "$BatchSize",
    "-ub", "$UBatchSize",
    "--temp", "1.0",
    "--top-p", "1.0",
    "--reasoning", "auto",
    "--reasoning-format", "deepseek",
    "--skip-chat-parsing"
)

if (-not $FullGpu) {
    $Args += @("--n-cpu-moe", "$CpuMoE")
}

Write-Host "Starting llama-server on http://127.0.0.1:$Port"
Write-Host "$Server $($Args -join ' ')"
& $Server @Args
