param(
    [int[]]$Port = @(8080, 8081),
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"

& (Join-Path $PSScriptRoot "stop-nikola.ps1") -Quiet:$Quiet
& (Join-Path $PSScriptRoot "stop-server.ps1") -Port $Port -Quiet:$Quiet
