param(
    [int]$Port = 8765,
    [switch]$Foreground,
    [switch]$Stop,
    [switch]$NoBuild,
    [switch]$ForceRebuild,
    [switch]$Logs
)

# Local admin panel launcher.
#
# Auto-flow when the SPA isn't ready yet:
#   1. Verify node + npm are on PATH (gracefully no-op if missing).
#   2. ``npm install`` if ``web/node_modules`` is missing.
#   3. ``npm run build`` if ``web/dist/index.html`` is missing
#      (or always, with -ForceRebuild).
#   4. Health check: skip if another admin is already serving on the port.
#   5. Start ``python -m protoagi admin`` in the background, redirect logs
#      to ``runs/admin.{stdout,stderr}.log``, poll ``/api/health`` until ready.
#
# Use ``-Stop`` to terminate the running admin. ``-Logs`` tails stderr.

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$WebDir = Join-Path $Root "src\protoagi\admin_panel\web"
$NodeModules = Join-Path $WebDir "node_modules"
$DistIndex = Join-Path $WebDir "dist\index.html"
$StdOut = Join-Path $Root "runs\admin.stdout.log"
$StdErr = Join-Path $Root "runs\admin.stderr.log"
$PidFile = Join-Path $Root "runs\admin.pid"
$HealthUrl = "http://127.0.0.1:$Port/api/health"

function Test-AdminServer {
    try {
        $null = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2
        return $true
    } catch {
        return $false
    }
}

function Get-AdminProcesses {
    @(Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
        Where-Object { ($_.CommandLine -as [string]) -match "protoagi\s+admin" -and `
                       ($_.CommandLine -as [string]) -match "--port\s+$Port\b" })
}

function Stop-Admin {
    $Procs = @(Get-AdminProcesses)
    if ($Procs.Count -eq 0) {
        Write-Host "Admin server is not running on port $Port."
        return
    }
    foreach ($Proc in $Procs) {
        Stop-Process -Id $Proc.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped admin server PID $($Proc.ProcessId)."
    }
    if (Test-Path $PidFile) { Remove-Item $PidFile -Force }
}

if ($Stop) { Stop-Admin; return }

if ($Logs) {
    if (-not (Test-Path $StdErr)) {
        Write-Host "No admin log yet at $StdErr"
        return
    }
    Get-Content -Path $StdErr -Wait -Tail 100
    return
}

# Already healthy? Reuse the existing instance instead of churning.
if (Test-AdminServer) {
    Write-Host "Admin server already running on http://127.0.0.1:$Port"
    return
}

# Defensive: maybe a stale python.exe holds the port but isn't healthy.
$Existing = @(Get-AdminProcesses)
if ($Existing.Count -gt 0) {
    Write-Warning "Stale admin process found on port $Port; killing it before restart."
    foreach ($Proc in $Existing) {
        Stop-Process -Id $Proc.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
}

# --- Build phase ------------------------------------------------------------
# We treat node/npm as optional: if the user only wants the Telegram bot,
# missing tooling shouldn't block the stack. Just warn and bail out of
# the admin step.
if (-not $NoBuild) {
    $Npm = Get-Command npm -ErrorAction SilentlyContinue
    $Node = Get-Command node -ErrorAction SilentlyContinue
    if (-not $Npm -or -not $Node) {
        Write-Warning "node/npm not found on PATH. Skipping admin build."
        Write-Warning "Install Node 20+ then re-run, or pass -NoBuild to silence this."
        return
    }
    if (!(Test-Path $WebDir)) {
        throw "Admin SPA source missing at $WebDir. Has it been removed?"
    }
    $NeedInstall = $ForceRebuild -or -not (Test-Path $NodeModules)
    if ($NeedInstall) {
        Write-Host "Installing admin SPA dependencies (one-time)..."
        Push-Location $WebDir
        try {
            & $Npm.Source install --no-audit --no-fund --silent
            if ($LASTEXITCODE -ne 0) { throw "npm install failed (exit $LASTEXITCODE)" }
        } finally {
            Pop-Location
        }
    }
    $NeedBuild = $ForceRebuild -or -not (Test-Path $DistIndex)
    if ($NeedBuild) {
        Write-Host "Building admin SPA..."
        Push-Location $WebDir
        try {
            & $Npm.Source run build
            if ($LASTEXITCODE -ne 0) { throw "npm run build failed (exit $LASTEXITCODE)" }
        } finally {
            Pop-Location
        }
    }
} elseif (-not (Test-Path $DistIndex)) {
    Write-Warning "Admin SPA build is missing at $DistIndex and -NoBuild was set."
    Write-Warning "The admin server will start but '/' will return 404 until you build."
}

# --- Launch phase -----------------------------------------------------------
New-Item -ItemType Directory -Force -Path (Join-Path $Root "runs") | Out-Null

$Python = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $Python) { throw "python not found on PATH" }

$AdminArgs = @("-m", "protoagi", "admin", "--port", "$Port", "--host", "127.0.0.1")

Write-Host "Starting admin server on http://127.0.0.1:$Port"

$env:PYTHONPATH = (Join-Path $Root "src")

if ($Foreground) {
    & $Python.Source @AdminArgs
    exit $LASTEXITCODE
}

$Proc = Start-Process `
    -FilePath $Python.Source `
    -ArgumentList $AdminArgs `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $StdOut `
    -RedirectStandardError $StdErr `
    -PassThru
$Proc.Id | Out-File -FilePath $PidFile -Encoding ascii

$Ready = $false
foreach ($i in 1..60) {
    Start-Sleep -Seconds 1
    if (Test-AdminServer) { $Ready = $true; break }
    if ($Proc.HasExited) {
        $Proc.Refresh()
        $ExitCode = if ($null -ne $Proc.ExitCode) { $Proc.ExitCode } else { "unknown" }
        throw "Admin server exited early (code $ExitCode). Check $StdErr"
    }
}
if (-not $Ready) {
    throw "Admin server did not respond within 60s. Tail logs: .\scripts\start-admin-server.ps1 -Logs"
}
Write-Host "Admin ready: http://127.0.0.1:$Port"
