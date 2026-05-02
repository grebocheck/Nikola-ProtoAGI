param(
    [int[]]$Port = @(),
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$Processes = @(Get-CimInstance Win32_Process -Filter "name = 'llama-server.exe'")
if ($Port.Count -gt 0) {
    $Processes = @(
        $Processes | Where-Object {
            $CommandLine = $_.CommandLine -as [string]
            $MatchesPort = $false
            foreach ($Item in $Port) {
                if ($CommandLine -match "--port\s+$Item\b") {
                    $MatchesPort = $true
                    break
                }
            }
            $MatchesPort
        }
    )
}

if (-not $Processes) {
    if (-not $Quiet) {
        Write-Host "llama-server is not running."
    }
    exit 0
}

foreach ($Process in $Processes) {
    Stop-Process -Id $Process.ProcessId -Force
    if (-not $Quiet) {
        Write-Host "Stopped llama-server process $($Process.ProcessId)."
    }
}
