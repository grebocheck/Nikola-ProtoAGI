param(
    [switch]$WithEmbeddings,
    [switch]$UpdateBaseline,
    [string]$BaselinePath = "",
    [string]$OutputPath = "",
    [double]$MaxDropPp = 5.0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($BaselinePath)) {
    $BaselinePath = Join-Path $Root "runs\memory-eval-baseline.json"
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $Stamp = (Get-Date -Format "yyyy-MM-dd-HHmm")
    $OutputPath = Join-Path $Root "runs\memory-eval-$Stamp.json"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputPath) | Out-Null

$env:PYTHONPATH = Join-Path $Root "src"
$EvalArgs = @("-m", "protoagi", "memory-eval", "--json")
if ($WithEmbeddings) {
    $EvalArgs += "--with-embeddings"
}

$EvalSuffix = ""
if ($WithEmbeddings) { $EvalSuffix = " --with-embeddings" }
Write-Host "Running protoagi memory-eval$EvalSuffix..."
$Json = & python @EvalArgs
if ($LASTEXITCODE -ne 0) {
    throw "memory-eval exited with code $LASTEXITCODE"
}
Set-Content -Path $OutputPath -Value $Json -Encoding utf8
Write-Host "wrote $OutputPath"

$Result = $Json | ConvertFrom-Json
$Recall = $Result.summary.recall_at_k
$Mrr = [double]$Result.summary.mrr
Write-Host ("recall@1: {0:N3} | recall@3: {1:N3} | recall@5: {2:N3} | MRR: {3:N3}" -f `
    [double]$Recall.'1', [double]$Recall.'3', [double]$Recall.'5', $Mrr)

if ($UpdateBaseline) {
    $Mode = if ($WithEmbeddings) { "with-embeddings" } else { "fts-only" }
    $NewBaseline = [PSCustomObject]@{
        captured_at = (Get-Date -Format "yyyy-MM-dd")
        mode        = $Mode
        summary     = $Result.summary
    }
    $NewBaseline | ConvertTo-Json -Depth 6 | Set-Content -Path $BaselinePath -Encoding utf8
    Write-Host "Updated baseline at $BaselinePath"
    exit 0
}

if (-not (Test-Path $BaselinePath)) {
    Write-Warning "no baseline at $BaselinePath; skipping regression check (use -UpdateBaseline to capture one)"
    exit 0
}

$Baseline = Get-Content $BaselinePath -Raw | ConvertFrom-Json
$BaselineRecall = $Baseline.summary.recall_at_k
$Threshold = $MaxDropPp / 100.0
$Failed = $false
foreach ($k in @("1", "3", "5")) {
    $current = [double]$Recall.$k
    $base = [double]$BaselineRecall.$k
    $delta = $current - $base
    $sign = if ($delta -ge 0) { "+" } else { "" }
    Write-Host ("recall@{0}: current {1:N3} vs baseline {2:N3} ({3}{4:N3})" -f $k, $current, $base, $sign, $delta)
    if ($delta -lt -$Threshold) {
        Write-Warning ("recall@{0} dropped by more than {1}pp" -f $k, $MaxDropPp)
        $Failed = $true
    }
}

if ($Failed) {
    Write-Error "memory eval regression: recall dropped past tolerance"
    exit 1
}

Write-Host "memory eval within tolerance"
