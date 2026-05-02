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
    [switch]$KeepServers,
    [switch]$KeepExistingTelegram,
    [switch]$NoProactive,
    [switch]$Once,
    [switch]$DeleteWebhook,
    [switch]$DropPendingUpdates
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$ServerUrl = "http://127.0.0.1:$Port/v1/models"

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
    if (-not $KeepServers) {
        $PortsToStop = @($Port)
        if ($VisionPort) {
            $PortsToStop += $VisionPort
        }
        Write-Host "Stopping local llama-server processes..."
        & (Join-Path $PSScriptRoot "stop-server.ps1") -Port $PortsToStop -Quiet
    }
}
