param(
    [string]$ModelPath = "",
    [string]$Model = "",
    [int]$Port = 8090,
    [int]$CtxSize = 2048,
    [int]$CpuMoE = 8,
    [string]$Prompt = "Reply with exactly one short sentence containing the word pong.",
    [switch]$NoServer,
    [switch]$KeepServer,
    [switch]$TelegramOnce,
    [string]$Token = "",
    [string]$AllowedChatId = "",
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $Root "src"
$BaseUrl = "http://127.0.0.1:$Port/v1"
$ServerProcess = $null

function Test-SmokeServer {
    try {
        $null = Invoke-WebRequest -Uri "$BaseUrl/models" -UseBasicParsing -TimeoutSec 2
        return $true
    } catch {
        return $false
    }
}

function Resolve-SmokeModelPath {
    if (-not [string]::IsNullOrWhiteSpace($ModelPath)) {
        return (Resolve-Path $ModelPath).Path
    }
    if (-not [string]::IsNullOrWhiteSpace($env:PROTOAGI_SMOKE_MODEL_PATH)) {
        return (Resolve-Path $env:PROTOAGI_SMOKE_MODEL_PATH).Path
    }
    $Default = Join-Path $Root "models\gpt-oss-20b-MXFP4.gguf"
    if (!(Test-Path $Default)) {
        $Default = Join-Path $Root "gpt-oss-20b-MXFP4.gguf"
    }
    if (Test-Path $Default) {
        return $Default
    }
    throw "Pass -ModelPath or set PROTOAGI_SMOKE_MODEL_PATH to a small GGUF model."
}

try {
    if (-not $NoServer -and -not (Test-SmokeServer)) {
        $Server = Join-Path $Root "tools\llama.cpp\llama-server.exe"
        if (!(Test-Path $Server)) {
            throw "llama-server.exe not found at $Server"
        }
        $ResolvedModelPath = Resolve-SmokeModelPath
        $StdOut = Join-Path $Root "runs\smoke-llama.stdout.log"
        $StdErr = Join-Path $Root "runs\smoke-llama.stderr.log"
        $Args = @(
            "-m", $ResolvedModelPath,
            "--host", "127.0.0.1",
            "--port", "$Port",
            "--ctx-size", "$CtxSize",
            "--jinja",
            "-fa", "off",
            "-b", "256",
            "-ub", "256",
            "--n-cpu-moe", "$CpuMoE",
            "--skip-chat-parsing"
        )
        Write-Host "Starting smoke llama-server on $BaseUrl..."
        $ServerProcess = Start-Process `
            -FilePath $Server `
            -ArgumentList $Args `
            -WorkingDirectory (Join-Path $Root "tools\llama.cpp") `
            -WindowStyle Hidden `
            -RedirectStandardOutput $StdOut `
            -RedirectStandardError $StdErr `
            -PassThru

        $Ready = $false
        foreach ($i in 1..$TimeoutSeconds) {
            Start-Sleep -Seconds 1
            if (Test-SmokeServer) {
                $Ready = $true
                break
            }
        }
        if (-not $Ready) {
            throw "smoke llama-server did not become ready. Check runs\smoke-llama.stderr.log"
        }
    }

    if ([string]::IsNullOrWhiteSpace($Model)) {
        $Model = if ([string]::IsNullOrWhiteSpace($env:PROTOAGI_MODEL)) { "smoke" } else { $env:PROTOAGI_MODEL }
    }

    Write-Host "Running protoagi chat smoke..."
    $ChatOutput = python -m protoagi chat `
        --base-url $BaseUrl `
        --model $Model `
        --prompt $Prompt `
        --max-tokens 64
    if ($LASTEXITCODE -ne 0) {
        throw "protoagi chat smoke exited with code $LASTEXITCODE"
    }
    $Text = ($ChatOutput -join "`n").Trim()
    if ([string]::IsNullOrWhiteSpace($Text)) {
        throw "protoagi chat smoke returned an empty response"
    }
    Write-Host "Chat smoke response: $Text"

    if ($TelegramOnce) {
        if ([string]::IsNullOrWhiteSpace($Token)) {
            $Token = [string]$env:TELEGRAM_BOT_TOKEN
        }
        if ([string]::IsNullOrWhiteSpace($Token)) {
            throw "TelegramOnce requires -Token or TELEGRAM_BOT_TOKEN."
        }
        $TelegramArgs = @(
            "telegram",
            "--once",
            "--token", $Token,
            "--base-url", $BaseUrl,
            "--model", $Model,
            "--no-proactive",
            "--poll-timeout", "1"
        )
        if (-not [string]::IsNullOrWhiteSpace($AllowedChatId)) {
            $TelegramArgs += @("--allowed-chat-id", $AllowedChatId)
        }
        Write-Host "Running Telegram --once smoke. Queue a fresh user ping before this step for live inbound coverage."
        python -m protoagi @TelegramArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Telegram --once smoke exited with code $LASTEXITCODE"
        }
    }

    Write-Host "Smoke test passed."
} finally {
    if ($ServerProcess -and -not $KeepServer) {
        Stop-Process -Id $ServerProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
