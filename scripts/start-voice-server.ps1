param(
    [int]$Port = 8083,
    [string]$Model = "large-v3",
    [ValidateSet("cpu", "cuda", "auto")]
    [string]$Device = "cpu",
    [string]$ComputeType = "",
    [string]$Language = "uk",
    [switch]$Foreground,
    [switch]$Stop,
    [switch]$Logs,
    [switch]$Reinstall
)

# faster-whisper bridge for Telegram voice messages.
#
# - Bootstraps a local venv under runs\voice-venv on first run
# - Installs faster-whisper + fastapi + uvicorn + python-multipart
# - Spins up scripts\voice-server.py on http://127.0.0.1:$Port
# - First request downloads the CTranslate2 weights (~1.5 GB for large-v3)
#   into runs\voice-cache\, then everything is local.
#
# Defaults to CPU/int8 so the GPU stays free for gpt-oss-20b. Pass
# ``-Device cuda -ComputeType float16`` if you have VRAM to spare.

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$VenvDir   = Join-Path $Root "runs\voice-venv"
$VenvPy    = Join-Path $VenvDir "Scripts\python.exe"
$ServerScript = Join-Path $Root "scripts\voice-server.py"
$CacheDir = Join-Path $Root "runs\voice-cache"
$StdOut = Join-Path $Root "runs\voice.stdout.log"
$StdErr = Join-Path $Root "runs\voice.stderr.log"
$PidFile = Join-Path $Root "runs\voice.pid"
$HealthUrl = "http://127.0.0.1:$Port/v1/models"

if ([string]::IsNullOrWhiteSpace($ComputeType)) {
    # int8 on CPU is the standard faster-whisper sweet spot; on GPU
    # we want float16 which is dramatically faster than int8_float16.
    if ($Device -eq "cuda") {
        $ComputeType = "float16"
    } else {
        $ComputeType = "int8"
    }
}

function Test-VoiceServer {
    try {
        $null = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2
        return $true
    } catch {
        return $false
    }
}

function Get-VoiceProcesses {
    @(Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
        Where-Object { ($_.CommandLine -as [string]) -match "voice-server\.py.*--port\s+$Port\b" })
}

function Stop-Voice {
    $Procs = @(Get-VoiceProcesses)
    if ($Procs.Count -eq 0) {
        Write-Host "Voice server is not running on port $Port."
        return
    }
    foreach ($Proc in $Procs) {
        Stop-Process -Id $Proc.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped voice server PID $($Proc.ProcessId)."
    }
    if (Test-Path $PidFile) { Remove-Item $PidFile -Force }
}

if ($Stop) { Stop-Voice; return }

if ($Logs) {
    if (-not (Test-Path $StdErr)) {
        Write-Host "No voice log yet at $StdErr"
        return
    }
    Get-Content -Path $StdErr -Wait -Tail 100
    return
}

if (Test-VoiceServer) {
    Write-Host "Voice server already running on http://127.0.0.1:$Port"
    return
}

# Stale process on the port? Knock it down before re-binding.
$Existing = @(Get-VoiceProcesses)
if ($Existing.Count -gt 0) {
    Write-Warning "Stale voice process on port $Port; killing before restart."
    foreach ($Proc in $Existing) {
        Stop-Process -Id $Proc.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
}

New-Item -ItemType Directory -Force -Path `
    (Join-Path $Root "runs"), `
    $CacheDir | Out-Null

if ($Reinstall -and (Test-Path $VenvDir)) {
    Write-Host "Removing existing voice venv (-Reinstall)..."
    Remove-Item -Recurse -Force $VenvDir
}

if (-not (Test-Path $VenvPy)) {
    $SystemPy = Get-Command python -ErrorAction SilentlyContinue
    if (-not $SystemPy) { throw "python 3.11+ not found on PATH" }
    Write-Host "Creating voice venv at $VenvDir ..."
    & $SystemPy.Source -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
    & $VenvPy -m pip install --upgrade pip wheel 2>&1 | Out-Host
}

# Self-heal: import-probe so a half-installed venv from an older run
# still gets repaired transparently. PS 5.1 routes native stderr through
# the error stream, so we locally relax ``$ErrorActionPreference`` to
# avoid the expected ImportError being treated as fatal.
$PrevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $VenvPy -c "import fastapi, faster_whisper, multipart" *> $null
    $DepProbeExit = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $PrevEAP
}
if ($DepProbeExit -ne 0) {
    Write-Host "Voice venv is missing required packages; installing ..."
    & $VenvPy -m pip install `
        "faster-whisper>=1.0" `
        "fastapi>=0.110" `
        "uvicorn[standard]>=0.27" `
        "python-multipart>=0.0.9" | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
}

$Args = @(
    $ServerScript,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--model", $Model,
    "--device", $Device,
    "--compute-type", $ComputeType,
    "--language", $Language,
    "--download-root", $CacheDir
)

Write-Host "Starting voice server (model=$Model device=$Device compute=$ComputeType) ..."
Write-Host "$VenvPy $($Args -join ' ')"

if ($Foreground) {
    & $VenvPy @Args
    exit $LASTEXITCODE
}

$Proc = Start-Process `
    -FilePath $VenvPy `
    -ArgumentList $Args `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $StdOut `
    -RedirectStandardError $StdErr `
    -PassThru
$Proc.Id | Out-File -FilePath $PidFile -Encoding ascii

# First-time model download can take several minutes (~1.5 GB on
# large-v3 with int8). Give it generous headroom on startup before
# declaring failure.
$Ready = $false
foreach ($i in 1..600) {
    Start-Sleep -Seconds 1
    if (Test-VoiceServer) { $Ready = $true; break }
    if ($Proc.HasExited) {
        throw "Voice server exited early (code $($Proc.ExitCode)). Check $StdErr"
    }
}

if (-not $Ready) {
    throw "Voice server did not respond within 10 minutes. Tail with: .\scripts\start-voice-server.ps1 -Logs"
}

Write-Host "Voice server ready: http://127.0.0.1:$Port"
