param(
    [int]$Port = 8084,
    [string]$Model = "uk_UA-ukrainian_tts-medium",
    [switch]$Foreground,
    [switch]$Stop,
    [switch]$Logs,
    [switch]$Reinstall
)

# Ukrainian Piper TTS server for ProtoAGI.
#
# Replaces the previous Docker openedai-speech / XTTS-v2 setup, which
# mapped Ukrainian onto Russian phonemes and produced a heavy Russian
# accent. Piper's uk_UA-ukrainian_tts-medium ships with the
# robinhad/ukrainian-tts dataset (proper Ukrainian phonetics, speakers
# mykyta / lada / dmytro / tetiana / oleksa).
#
# - Bootstraps a local venv under runs\tts-venv
# - Installs piper-tts + fastapi + uvicorn on first run
# - Downloads the Piper model to config\tts\models\ (~63 MB)
# - Exposes OpenAI-compatible /v1/audio/speech on http://127.0.0.1:$Port
# - Requires ffmpeg in PATH for opus/mp3/aac transcoding (Telegram voice)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$VenvDir   = Join-Path $Root "runs\tts-venv"
$VenvPy    = Join-Path $VenvDir "Scripts\python.exe"
$ModelDir  = Join-Path $Root "config\tts\models"
$ModelOnnx = Join-Path $ModelDir "$Model.onnx"
$ModelJson = "$ModelOnnx.json"
$VoiceMap  = Join-Path $Root "config\tts\voice_map.json"
$ServerScript = Join-Path $Root "scripts\tts-server-uk.py"
$StdOut = Join-Path $Root "runs\tts.stdout.log"
$StdErr = Join-Path $Root "runs\tts.stderr.log"
$PidFile = Join-Path $Root "runs\tts.pid"
$HealthUrl = "http://127.0.0.1:$Port/v1/voices"

function Test-TtsServer {
    try {
        $null = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2
        return $true
    } catch {
        return $false
    }
}

function Get-TtsProcesses {
    @(Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
        Where-Object { ($_.CommandLine -as [string]) -match "tts-server-uk\.py.*--port\s+$Port\b" })
}

function Stop-Tts {
    $Procs = @(Get-TtsProcesses)
    if ($Procs.Count -eq 0) {
        Write-Host "TTS server is not running on port $Port."
        return
    }
    foreach ($Proc in $Procs) {
        Stop-Process -Id $Proc.ProcessId -Force
        Write-Host "Stopped TTS server PID $($Proc.ProcessId)."
    }
    if (Test-Path $PidFile) { Remove-Item $PidFile -Force }
}

if ($Stop) { Stop-Tts; return }

if ($Logs) {
    if (-not (Test-Path $StdErr)) {
        Write-Host "No log file yet at $StdErr"
        return
    }
    Get-Content -Path $StdErr -Wait -Tail 100
    return
}

if (Test-TtsServer) {
    Write-Host "TTS server already running on http://127.0.0.1:$Port"
    return
}

$Existing = @(Get-TtsProcesses)
if ($Existing.Count -gt 0) {
    throw "A python.exe is bound to port $Port but the server is not healthy. Check $StdErr or stop it: .\scripts\start-tts-server.ps1 -Stop"
}

New-Item -ItemType Directory -Force -Path `
    (Join-Path $Root "runs"), `
    $ModelDir | Out-Null

if ($Reinstall -and (Test-Path $VenvDir)) {
    Write-Host "Removing existing TTS venv (--Reinstall)..."
    Remove-Item -Recurse -Force $VenvDir
}

if (-not (Test-Path $VenvPy)) {
    $SystemPy = (Get-Command python -ErrorAction SilentlyContinue)
    if (-not $SystemPy) { throw "python 3.11+ not found on PATH" }
    Write-Host "Creating TTS venv at $VenvDir ..."
    & $SystemPy.Source -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }

    Write-Host "Installing piper-tts + fastapi + uvicorn ..."
    & $VenvPy -m pip install --upgrade pip wheel 2>&1 | Out-Host
    & $VenvPy -m pip install "piper-tts>=1.2" "fastapi>=0.110" "uvicorn[standard]>=0.27" 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
}

if (-not (Test-Path $ModelOnnx) -or -not (Test-Path $ModelJson)) {
    $BaseUrl = "https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/ukrainian_tts/medium"
    Write-Host "Downloading Piper UA model to $ModelDir ..."
    try {
        Invoke-WebRequest -Uri "$BaseUrl/$Model.onnx"      -OutFile $ModelOnnx -UseBasicParsing
        Invoke-WebRequest -Uri "$BaseUrl/$Model.onnx.json" -OutFile $ModelJson -UseBasicParsing
    } catch {
        throw "Model download failed: $_"
    }
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg not found in PATH. Opus/mp3 transcoding will fail. Install ffmpeg or set PROTOAGI_TTS_RESPONSE_FORMAT=wav."
}

$Args = @(
    $ServerScript,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--model", $ModelOnnx,
    "--voice-map", $VoiceMap
)

Write-Host "Starting Ukrainian TTS server on http://127.0.0.1:$Port"
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

$Ready = $false
foreach ($i in 1..120) {
    Start-Sleep -Seconds 1
    if (Test-TtsServer) { $Ready = $true; break }
    if ($Proc.HasExited) {
        throw "TTS server exited early (code $($Proc.ExitCode)). Check $StdErr"
    }
}

if (-not $Ready) {
    throw "TTS server did not respond within 120s. Tail with: .\scripts\start-tts-server.ps1 -Logs"
}

Write-Host "TTS ready. Speakers: lada, mykyta, tetiana."
Write-Host "Persona voices configured in config\tts\voice_map.json (solomiya->lada, mykola->mykyta)."
