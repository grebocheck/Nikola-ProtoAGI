param(
    [string]$HfRepo = "CompendiumLabs/bge-m3-gguf:Q4_K_M",
    [string]$Alias = "bge-m3",
    [int]$Port = 8082,
    [int]$CtxSize = 8192,
    [int]$BatchSize = 512,
    [int]$UBatchSize = 512,
    [string]$GpuLayers = "0",
    [int]$Threads = 4,
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
$StdOut = Join-Path $Root "runs\llama-embed.stdout.log"
$StdErr = Join-Path $Root "runs\llama-embed.stderr.log"
$ServerUrl = "http://127.0.0.1:$Port/v1/models"

function Test-EmbedServer {
    try {
        $null = Invoke-WebRequest -Uri $ServerUrl -UseBasicParsing -TimeoutSec 2
        return $true
    } catch {
        return $false
    }
}

function Get-EmbedServerProcesses {
    @(Get-CimInstance Win32_Process -Filter "name = 'llama-server.exe'" |
        Where-Object { $_.CommandLine -match "--port\s+$Port\b" })
}

if (!(Test-Path $Server)) {
    throw "llama-server.exe not found at $Server"
}

if (Test-EmbedServer) {
    Write-Host "Embedding llama-server already running on http://127.0.0.1:$Port"
    return
}

$Existing = @(Get-EmbedServerProcesses)
if ($Existing.Count -gt 0) {
    throw "A llama-server process is already bound to port $Port but is not ready. Check runs\llama-embed.stderr.log or stop it first."
}

New-Item -ItemType Directory -Force -Path `
    (Join-Path $Root "runs"), `
    $env:HF_HUB_CACHE | Out-Null

# llama-server in embedding mode: --embedding switches the inference path,
# --pooling cls is the standard for BGE-style models.
$Args = @(
    "-hf", $HfRepo,
    "--alias", $Alias,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--ctx-size", "$CtxSize",
    "--embedding",
    "--pooling", "cls",
    "-b", "$BatchSize",
    "-ub", "$UBatchSize",
    "-ngl", "$GpuLayers",
    "-t", "$Threads"
)

Write-Host "Starting embedding llama-server on http://127.0.0.1:$Port"
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
    if (Test-EmbedServer) {
        $Ready = $true
        break
    }
}

if (-not $Ready) {
    throw "Embedding llama-server did not become ready. Check runs\llama-embed.stderr.log"
}

Write-Host "Embedding llama-server ready."
