$ErrorActionPreference = "Stop"
$Processes = Get-Process llama-server -ErrorAction SilentlyContinue
if (-not $Processes) {
    Write-Host "llama-server is not running."
    exit 0
}
$Processes | Stop-Process -Force
Write-Host "Stopped llama-server."
