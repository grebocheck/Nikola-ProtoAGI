param(
    [string]$Token = "",
    [string]$AllowedChatId = "",
    [ValidateSet("smart", "always", "mention", "silent")]
    [string]$ReplyMode = "smart",
    [int]$Port = 8080,
    [int]$CtxSize = 8192,
    [int]$CpuMoE = 4,
    [switch]$FullGpu,
    [switch]$NoProactive,
    [switch]$Once,
    [switch]$DeleteWebhook,
    [switch]$DropPendingUpdates
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$ServerUrl = "http://127.0.0.1:$Port/v1/models"

function Test-ProtoAgiServer {
    try {
        $null = Invoke-WebRequest -Uri $ServerUrl -UseBasicParsing -TimeoutSec 2
        return $true
    } catch {
        return $false
    }
}

if (-not (Test-ProtoAgiServer)) {
    $Server = Join-Path $Root "tools\llama.cpp\llama-server.exe"
    $Model = Join-Path $Root "gpt-oss-20b-MXFP4.gguf"
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
        "--reasoning-format", "deepseek"
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
