param(
    [switch]$Quiet,
    [switch]$All
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$AllProcesses = @(Get-CimInstance Win32_Process)
$ByPid = @{}
foreach ($Process in $AllProcesses) {
    $ByPid[[int]$Process.ProcessId] = $Process
}

function Test-WorkspaceAncestor {
    param($Process)

    $Current = $Process
    while ($Current -and $Current.ParentProcessId) {
        if (($Current.CommandLine -as [string]) -match [regex]::Escape($Root)) {
            return $true
        }
        $Parent = $ByPid[[int]$Current.ParentProcessId]
        if (-not $Parent) {
            break
        }
        $Current = $Parent
    }
    return $false
}

$Processes = $AllProcesses |
    Where-Object {
        $_.Name -eq "python.exe" -and
        ($_.CommandLine -as [string]) -match "(^|\s)-m\s+protoagi\s+telegram(\s|$)" -and
        ($All -or (Test-WorkspaceAncestor $_))
    }

if (-not $Processes) {
    if (-not $Quiet) {
        Write-Host "Telegram bot process is not running for this workspace."
    }
    exit 0
}

foreach ($Process in $Processes) {
    Stop-Process -Id $Process.ProcessId -Force
    if (-not $Quiet) {
        Write-Host "Stopped Telegram bot process $($Process.ProcessId)."
    }
}
