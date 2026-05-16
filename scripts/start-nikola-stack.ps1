param(
    [string]$Token = "",
    [string]$AllowedChatId = "",
    [ValidateSet("smart", "always", "mention", "silent")]
    [string]$ReplyMode = "smart",
    [int]$Port = 8080,
    [int]$CtxSize = 8192,
    [int]$CpuMoE = 4,
    [switch]$FullGpu,
    [switch]$NoVision,
    [string]$VisionRepo = "",
    [string]$VisionGpuLayers = "0",
    [switch]$NoEmbed,
    [string]$EmbedRepo = "",
    [string]$EmbedGpuLayers = "0",
    [switch]$NoVoice,
    [string]$VoiceModel = "large-v3",
    [ValidateSet("cpu", "cuda", "auto")]
    [string]$VoiceDevice = "cpu",
    [switch]$NoTts,
    [switch]$TtsCpu,
    [switch]$KeepServers,
    [switch]$KeepExistingTelegram,
    [switch]$NoProactive,
    [switch]$Once,
    [switch]$DeleteWebhook,
    [switch]$DropPendingUpdates,
    [switch]$NoAdmin,
    [int]$AdminPort = 8765,
    [switch]$ForceAdminRebuild
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$ServerUrl = "http://127.0.0.1:$Port/v1/models"
$env:PROTOAGI_CONTEXT_SIZE = [string]$CtxSize

function Read-ProtoAgiDotEnv {
    param([string]$Path)
    $Values = @{}
    if (!(Test-Path $Path)) {
        return $Values
    }
    foreach ($RawLine in Get-Content $Path) {
        $Line = $RawLine.Trim()
        if ($Line -eq "" -or $Line.StartsWith("#") -or $Line -notmatch "=") {
            continue
        }
        if ($Line.StartsWith("export ")) {
            $Line = $Line.Substring(7).Trim()
        }
        $Parts = $Line.Split("=", 2)
        $Key = $Parts[0].Trim()
        $Value = $Parts[1].Trim()
        if ($Key -notmatch "^[A-Za-z_][A-Za-z0-9_]*$") {
            continue
        }
        if ($Value.Length -ge 2 -and (($Value.StartsWith('"') -and $Value.EndsWith('"')) -or ($Value.StartsWith("'") -and $Value.EndsWith("'")))) {
            $Value = $Value.Substring(1, $Value.Length - 2)
        }
        $Values[$Key] = $Value
    }
    return $Values
}

$DotEnv = Read-ProtoAgiDotEnv (Join-Path $Root ".env")
$VisionPort = $null
$EmbedPort = $null

function Test-ProtoAgiServer {
    try {
        $null = Invoke-WebRequest -Uri $ServerUrl -UseBasicParsing -TimeoutSec 2
        return $true
    } catch {
        return $false
    }
}

function Get-ProtoAgiServerProcesses {
    @(Get-CimInstance Win32_Process -Filter "name = 'llama-server.exe'" |
        Where-Object { $_.CommandLine -match "--port\s+$Port\b" })
}

try {

$NeedsServerStart = -not (Test-ProtoAgiServer)
if (-not $NeedsServerStart) {
    $StaleServers = @(Get-ProtoAgiServerProcesses |
        Where-Object { $_.CommandLine -notmatch "--skip-chat-parsing" })
    if ($StaleServers.Count -gt 0) {
        Write-Host "Restarting llama-server with --skip-chat-parsing..."
        foreach ($Process in $StaleServers) {
            Stop-Process -Id $Process.ProcessId -Force -ErrorAction SilentlyContinue
        }
        Start-Sleep -Seconds 2
        $NeedsServerStart = $true
    }
}

if ($NeedsServerStart) {
    $Server = Join-Path $Root "tools\llama.cpp\llama-server.exe"
    $Model = Join-Path $Root "models\gpt-oss-20b-MXFP4.gguf"
    if (!(Test-Path $Model)) {
        $LegacyModel = Join-Path $Root "gpt-oss-20b-MXFP4.gguf"
        if (Test-Path $LegacyModel) {
            $Model = $LegacyModel
        }
    }
    $StdOut = Join-Path $Root "runs\llama-server.stdout.log"
    $StdErr = Join-Path $Root "runs\llama-server.stderr.log"
    $ServerArgs = @(
        "-m", $Model,
        "--host", "127.0.0.1",
        "--port", "$Port",
        "--ctx-size", "$CtxSize",
        "--jinja",
        "-fa", "on",
        "-b", "1024",
        "-ub", "1024",
        "--temp", "1.0",
        "--top-p", "1.0",
        "--reasoning", "auto",
        "--reasoning-format", "deepseek",
        "--skip-chat-parsing"
    )
    if (-not $FullGpu) {
        $ServerArgs += @("--n-cpu-moe", "$CpuMoE")
    }
    Write-Host "Starting llama-server in background..."
    Start-Process `
        -FilePath $Server `
        -ArgumentList $ServerArgs `
        -WorkingDirectory (Join-Path $Root "tools\llama.cpp") `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdOut `
        -RedirectStandardError $StdErr | Out-Null

    $Ready = $false
    foreach ($i in 1..180) {
        Start-Sleep -Seconds 1
        if (Test-ProtoAgiServer) {
            $Ready = $true
            break
        }
    }
    if (-not $Ready) {
        throw "llama-server did not become ready. Check runs\llama-server.stderr.log"
    }
}

if (-not $NoVision) {
    $VisionModel = [string]($DotEnv["PROTOAGI_VISION_MODEL"])
    $VisionBaseUrl = [string]($DotEnv["PROTOAGI_VISION_BASE_URL"])
    if (-not [string]::IsNullOrWhiteSpace($VisionModel) -and $VisionBaseUrl -match "^https?://(127\.0\.0\.1|localhost):(?<port>\d+)(/|$)") {
        $VisionPort = [int]$Matches["port"]
        # Raw image/gif documents sometimes arrive without Telegram's
        # server-side thumbnail. Vision can still describe them if we have
        # ffmpeg to extract one still frame locally.
        & (Join-Path $PSScriptRoot "ensure-ffmpeg.ps1") -Root $Root
        $ResolvedVisionRepo = $VisionRepo
        if ([string]::IsNullOrWhiteSpace($ResolvedVisionRepo)) {
            $ResolvedVisionRepo = [string]($DotEnv["PROTOAGI_VISION_HF_REPO"])
        }
        if ([string]::IsNullOrWhiteSpace($ResolvedVisionRepo)) {
            $ResolvedVisionRepo = "ggml-org/SmolVLM2-2.2B-Instruct-GGUF:Q4_K_M"
        }
        & (Join-Path $PSScriptRoot "start-vision-server.ps1") `
            -HfRepo $ResolvedVisionRepo `
            -Alias $VisionModel `
            -Port $VisionPort `
            -GpuLayers $VisionGpuLayers
    }
}

if (-not $NoEmbed) {
    $EmbedModel = [string]($DotEnv["PROTOAGI_EMBED_MODEL"])
    $EmbedBaseUrl = [string]($DotEnv["PROTOAGI_EMBED_BASE_URL"])
    if (-not [string]::IsNullOrWhiteSpace($EmbedModel) -and $EmbedBaseUrl -match "^https?://(127\.0\.0\.1|localhost):(?<port>\d+)(/|$)") {
        $EmbedPort = [int]$Matches["port"]
        $ResolvedEmbedRepo = $EmbedRepo
        if ([string]::IsNullOrWhiteSpace($ResolvedEmbedRepo)) {
            $ResolvedEmbedRepo = [string]($DotEnv["PROTOAGI_EMBED_HF_REPO"])
        }
        if ([string]::IsNullOrWhiteSpace($ResolvedEmbedRepo)) {
            $ResolvedEmbedRepo = "CompendiumLabs/bge-m3-gguf:Q4_K_M"
        }
        & (Join-Path $PSScriptRoot "start-embed-server.ps1") `
            -HfRepo $ResolvedEmbedRepo `
            -Alias $EmbedModel `
            -Port $EmbedPort `
            -GpuLayers $EmbedGpuLayers
    }
}

if (-not $NoVoice -and -not $Once) {
    # Voice transcription is auto-spun-up when the operator configured a
    # PROTOAGI_VOICE_MODEL alias. First run downloads the CTranslate2
    # weights (~1.5 GB for large-v3) into runs\voice-cache. Failure
    # here is non-fatal - the bot keeps working without transcription.
    $VoiceModelEnv = [string]($DotEnv["PROTOAGI_VOICE_MODEL"])
    $VoiceBaseUrl = [string]($DotEnv["PROTOAGI_VOICE_BASE_URL"])
    if (-not [string]::IsNullOrWhiteSpace($VoiceModelEnv) -and `
        $VoiceBaseUrl -match "^https?://(127\.0\.0\.1|localhost):(?<port>\d+)(/|$)") {
        $VoicePort = [int]$Matches["port"]
        try {
            & (Join-Path $PSScriptRoot "start-voice-server.ps1") `
                -Port $VoicePort `
                -Model $VoiceModel `
                -Device $VoiceDevice
        } catch {
            Write-Warning "Voice server failed to start: $($_.Exception.Message)"
            Write-Warning "Telegram bot will still run; voice messages will be ignored."
        }
    }
}

if (-not $NoTts -and -not $Once) {
    # TTS is auto-spun-up when the operator set PROTOAGI_TTS_ENABLED=1.
    # First run installs piper-tts in runs\tts-venv and downloads the
    # Piper UA model. Failure is non-fatal - the bot still ships text.
    $TtsEnabled = [string]($DotEnv["PROTOAGI_TTS_ENABLED"])
    $TtsBaseUrl = [string]($DotEnv["PROTOAGI_TTS_BASE_URL"])
    $TtsTruthy = $TtsEnabled -match "^(1|true|yes|on)$"
    if ($TtsTruthy -and `
        $TtsBaseUrl -match "^https?://(127\.0\.0\.1|localhost):(?<port>\d+)(/|$)") {
        $TtsPort = [int]$Matches["port"]
        # Opus/mp3/aac need ffmpeg. Prefer a local self-contained
        # bootstrap under runs\ffmpeg, then fall back to wav so the bot
        # still starts even if the download host is unavailable.
        $TtsFormat = [string]($env:PROTOAGI_TTS_RESPONSE_FORMAT)
        if ([string]::IsNullOrWhiteSpace($TtsFormat)) {
            $TtsFormat = [string]($DotEnv["PROTOAGI_TTS_RESPONSE_FORMAT"])
        }
        if ([string]::IsNullOrWhiteSpace($TtsFormat)) {
            $TtsFormat = "opus"
        }
        $TtsFormat = $TtsFormat.Trim().ToLowerInvariant()
        if ($TtsFormat -notin @("wav", "pcm")) {
            . (Join-Path $PSScriptRoot "ensure-ffmpeg.ps1") -Root $Root
        }
        $HasFfmpeg = [bool](Get-Command ffmpeg -ErrorAction SilentlyContinue)
        if (-not $HasFfmpeg -and $TtsFormat -notin @("wav", "pcm")) {
            Write-Host "ffmpeg is unavailable after bootstrap - TTS will use wav (audio bubble)." `
                "Set PROTOAGI_FFMPEG_URL to a reachable ffmpeg zip if you want opus voice waveforms."
            $env:PROTOAGI_TTS_RESPONSE_FORMAT = "wav"
        }
        try {
            $TtsArgs = @{ Port = $TtsPort }
            if ($TtsCpu) { $TtsArgs.Cpu = $true }
            & (Join-Path $PSScriptRoot "start-tts-server.ps1") @TtsArgs
        } catch {
            Write-Warning "TTS server failed to start: $($_.Exception.Message)"
            Write-Warning "Telegram bot will still run; voice replies will be skipped."
        }
    }
}

if (-not $NoAdmin -and -not $Once) {
    # Admin panel is auto-built/started alongside the bot so the operator
    # always has a UI without having to remember a separate npm/python
    # incantation. Failures here are non-fatal: we still want the bot up
    # even if node is missing or the build fails.
    try {
        $AdminArgs = @{ Port = $AdminPort }
        if ($ForceAdminRebuild) { $AdminArgs.ForceRebuild = $true }
        & (Join-Path $PSScriptRoot "start-admin-server.ps1") @AdminArgs
    } catch {
        Write-Warning "Admin server failed to start: $($_.Exception.Message)"
        Write-Warning "Telegram bot will still run; restart admin manually with .\scripts\start-admin-server.ps1"
    }
}

if (-not $KeepExistingTelegram -and -not $Once) {
    & (Join-Path $PSScriptRoot "stop-nikola.ps1") -Quiet
}

$TelegramParams = @{}
if ($PSBoundParameters.ContainsKey("ReplyMode")) {
    $TelegramParams.ReplyMode = $ReplyMode
}
if (-not [string]::IsNullOrWhiteSpace($Token)) {
    $TelegramParams.Token = $Token
}
if (-not [string]::IsNullOrWhiteSpace($AllowedChatId)) {
    $TelegramParams.AllowedChatId = $AllowedChatId
}
if ($NoProactive) {
    $TelegramParams.NoProactive = $true
}
if ($Once) {
    $TelegramParams.Once = $true
}
if ($DeleteWebhook) {
    $TelegramParams.DeleteWebhook = $true
}
if ($DropPendingUpdates) {
    $TelegramParams.DropPendingUpdates = $true
}

& (Join-Path $PSScriptRoot "start-telegram.ps1") @TelegramParams

} finally {
    & (Join-Path $PSScriptRoot "stop-nikola.ps1") -Quiet
    if (-not $NoAdmin) {
        try {
            & (Join-Path $PSScriptRoot "start-admin-server.ps1") -Stop -Port $AdminPort | Out-Null
        } catch {
            # Stop is best-effort; user can kill the process manually.
        }
    }
    if (-not $NoVoice -and -not $KeepServers) {
        # Voice server is a downstream of the bot, so we stop it when
        # the bot exits unless the operator asked to keep servers.
        $VoiceBaseUrl = [string]($DotEnv["PROTOAGI_VOICE_BASE_URL"])
        if ($VoiceBaseUrl -match ":(?<port>\d+)(/|$)") {
            try {
                & (Join-Path $PSScriptRoot "start-voice-server.ps1") -Stop -Port ([int]$Matches["port"]) | Out-Null
            } catch {
                # Best-effort.
            }
        }
    }
    if (-not $NoTts -and -not $KeepServers) {
        $TtsBaseUrl = [string]($DotEnv["PROTOAGI_TTS_BASE_URL"])
        if ($TtsBaseUrl -match ":(?<port>\d+)(/|$)") {
            try {
                & (Join-Path $PSScriptRoot "start-tts-server.ps1") -Stop -Port ([int]$Matches["port"]) | Out-Null
            } catch {
                # Best-effort.
            }
        }
    }
    if (-not $KeepServers) {
        $PortsToStop = @($Port)
        if ($VisionPort) {
            $PortsToStop += $VisionPort
        }
        if ($EmbedPort) {
            $PortsToStop += $EmbedPort
        }
        Write-Host "Stopping local llama-server processes..."
        & (Join-Path $PSScriptRoot "stop-server.ps1") -Port $PortsToStop -Quiet
    }
}
