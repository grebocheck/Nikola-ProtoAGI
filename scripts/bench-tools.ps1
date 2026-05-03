param(
    [int]$Rounds = 5,
    [int]$MaxTokens = 512,
    [string]$Prompt = "Use the recall tool to find what you remember about coffee.",
    [switch]$UpdateBaseline,
    [string]$BaselinePath = "",
    [string]$OutputPath = "",
    [double]$AllowNeitherPct = 20.0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($BaselinePath)) {
    $BaselinePath = Join-Path $Root "runs\bench-tools-baseline.json"
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $Stamp = (Get-Date -Format "yyyy-MM-dd-HHmm")
    $OutputPath = Join-Path $Root "runs\bench-tools-$Stamp.json"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputPath) | Out-Null

$env:PYTHONPATH = Join-Path $Root "src"
Write-Host "Running protoagi bench-tools (rounds=$Rounds)..."
& python -m protoagi bench-tools `
    --rounds $Rounds `
    --max-tokens $MaxTokens `
    --prompt $Prompt `
    --output $OutputPath `
    --summary
if ($LASTEXITCODE -ne 0) {
    throw "bench-tools exited with code $LASTEXITCODE"
}

if ($UpdateBaseline) {
    Copy-Item -Force $OutputPath $BaselinePath
    Write-Host "Updated baseline at $BaselinePath"
    exit 0
}

Write-Host "Comparing against $BaselinePath"
& python (Join-Path $PSScriptRoot "check_baseline.py") `
    bench-tools `
    --report $OutputPath `
    --baseline $BaselinePath `
    --allow-neither-pct $AllowNeitherPct
$Code = $LASTEXITCODE
if ($Code -ne 0) {
    Write-Error "bench-tools regression: see report at $OutputPath"
    exit $Code
}
Write-Host "bench-tools within tolerance"
