param(
    [string]$HfRepo = "ggml-org/SmolVLM2-2.2B-Instruct-GGUF:Q4_K_M",
    [string]$Alias = "smolvlm2-2.2b-instruct",
    [int]$Port = 8081,
    [int]$CtxSize = 4096,
    [int]$BatchSize = 256,
    [int]$UBatchSize = 256,
    [string]$GpuLayers = "0",
    [switch]$MmprojOffload,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$ModelCacheRoot = Join-Path $Root "models\hf-cache"
$env:HF_HOME = $ModelCacheRoot
$env:HF_HUB_CACHE = Join-Path $ModelCacheRoot "hub"
$env:HUGGINGFACE_HUB_CACHE = $env:HF_HUB_CACHE
$env:LLAMA_CACHE = $env:HF_HUB_CACHE
$Server = Join-Path $Root "tools\llama.cpp\llama-server.exe"
$StdOut = Join-Path $Root "runs\llama-vision.stdout.log"
$StdErr = Join-Path $Root "runs\llama-vision.stderr.log"
$ServerUrl = "http://127.0.0.1:$Port/v1/models"

function Test-VisionServer {
    try {
        $null = Invoke-WebRequest -Uri $ServerUrl -UseBasicParsing -TimeoutSec 2
        return $true
    } catch {
        return $false
    }
}

function Get-VisionServerProcesses {
    @(Get-CimInstance Win32_Process -Filter "name = 'llama-server.exe'" |
        Where-Object { $_.CommandLine -match "--port\s+$Port\b" })
}

if (!(Test-Path $Server)) {
    throw "llama-server.exe not found at $Server"
}

if (Test-VisionServer) {
    Write-Host "Vision llama-server already running on http://127.0.0.1:$Port"
    return
}

$Existing = @(Get-VisionServerProcesses)
if ($Existing.Count -gt 0) {
    throw "A llama-server process is already bound to port $Port but is not ready. Check runs\llama-vision.stderr.log or stop it first."
}

New-Item -ItemType Directory -Force -Path `
    (Join-Path $Root "runs"), `
    $env:HF_HUB_CACHE | Out-Null

$Args = @(
    "-hf", $HfRepo,
    "--alias", $Alias,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--ctx-size", "$CtxSize",
    "--jinja",
    "-fa", "auto",
    "-b", "$BatchSize",
    "-ub", "$UBatchSize",
    "-ngl", "$GpuLayers",
    "--temp", "0.2",
    "--top-p", "1.0"
)

if (-not $MmprojOffload) {
    $Args += "--no-mmproj-offload"
}

Write-Host "Starting vision llama-server on http://127.0.0.1:$Port"
Write-Host "$Server $($Args -join ' ')"

if ($Foreground) {
    & $Server @Args
    exit $LASTEXITCODE
}

Start-Process `
    -FilePath $Server `
    -ArgumentList $Args `
    -WorkingDirectory (Join-Path $Root "tools\llama.cpp") `
    -WindowStyle Hidden `
    -RedirectStandardOutput $StdOut `
    -RedirectStandardError $StdErr | Out-Null

$Ready = $false
foreach ($i in 1..600) {
    Start-Sleep -Seconds 1
    if (Test-VisionServer) {
        $Ready = $true
        break
    }
}

if (-not $Ready) {
    throw "Vision llama-server did not become ready. Check runs\llama-vision.stderr.log"
}

Write-Host "Vision llama-server ready."
