$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Processes = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -eq "python.exe" -and
        $_.CommandLine -match "protoagi telegram" -and
        $_.CommandLine -match [regex]::Escape($Root)
    }

if (-not $Processes) {
    Write-Host "Telegram bot process is not running for this workspace."
    exit 0
}

foreach ($Process in $Processes) {
    Stop-Process -Id $Process.ProcessId -Force
    Write-Host "Stopped Telegram bot process $($Process.ProcessId)."
}
