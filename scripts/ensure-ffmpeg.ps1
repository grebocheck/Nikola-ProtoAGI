param(
    [string]$Root = "",
    [switch]$NoDownload,
    [switch]$Quiet,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Root)) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
} else {
    $Root = (Resolve-Path $Root).Path
}

$RunsDir = Join-Path $Root "runs"
$LocalBin = Join-Path $RunsDir "ffmpeg\bin"
$LocalExe = Join-Path $LocalBin "ffmpeg.exe"

function Write-FfmpegStatus {
    param([string]$Message)
    if (-not $Quiet) {
        Write-Host $Message
    }
}

function Add-ProtoAgiPath {
    param([string]$Path)
    $Resolved = (Resolve-Path $Path).Path
    $Parts = @($env:Path -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($Parts -notcontains $Resolved) {
        $env:Path = "$Resolved;$env:Path"
    }
}

function Assert-UnderRoot {
    param([string]$Path)
    $RootFull = [System.IO.Path]::GetFullPath($Root).TrimEnd('\')
    $Full = [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
    if (-not ($Full -eq $RootFull -or $Full.StartsWith("$RootFull\"))) {
        throw "Refusing to touch path outside repository root: $Full"
    }
}

if (-not $Force) {
    $Existing = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($Existing) {
        Write-FfmpegStatus "ffmpeg found on PATH: $($Existing.Source)"
        return
    }
    if (Test-Path $LocalExe) {
        Add-ProtoAgiPath $LocalBin
        Write-FfmpegStatus "ffmpeg ready from local cache: $LocalExe"
        return
    }
}

if ($NoDownload) {
    Write-Warning "ffmpeg not found and download is disabled."
    return
}

if ($env:OS -ne "Windows_NT") {
    Write-Warning "Automatic ffmpeg bootstrap is only implemented for Windows. Install ffmpeg with your OS package manager."
    return
}

$Url = $env:PROTOAGI_FFMPEG_URL
if ([string]::IsNullOrWhiteSpace($Url)) {
    $Url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
}

New-Item -ItemType Directory -Force -Path $RunsDir, $LocalBin | Out-Null

$Archive = Join-Path $RunsDir "ffmpeg-release-essentials.zip"
$Stage = Join-Path $RunsDir "ffmpeg-extract"
Assert-UnderRoot $Archive
Assert-UnderRoot $Stage
Assert-UnderRoot $LocalBin

if (Test-Path $Archive) {
    Remove-Item -LiteralPath $Archive -Force
}
if (Test-Path $Stage) {
    Remove-Item -LiteralPath $Stage -Recurse -Force
}

Write-FfmpegStatus "Downloading ffmpeg for local ProtoAGI runtime..."
try {
    Invoke-WebRequest -Uri $Url -OutFile $Archive -UseBasicParsing
    Expand-Archive -LiteralPath $Archive -DestinationPath $Stage -Force

    $Ffmpeg = Get-ChildItem -Path $Stage -Recurse -Filter "ffmpeg.exe" |
        Where-Object { $_.FullName -match "\\bin\\ffmpeg\.exe$" } |
        Select-Object -First 1
    if (-not $Ffmpeg) {
        $Ffmpeg = Get-ChildItem -Path $Stage -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
    }
    if (-not $Ffmpeg) {
        throw "Downloaded archive did not contain ffmpeg.exe"
    }

    foreach ($Name in @("ffmpeg.exe", "ffprobe.exe", "ffplay.exe")) {
        $Candidate = Join-Path $Ffmpeg.DirectoryName $Name
        if (Test-Path $Candidate) {
            Copy-Item -LiteralPath $Candidate -Destination (Join-Path $LocalBin $Name) -Force
        }
    }
    Add-ProtoAgiPath $LocalBin
    Write-FfmpegStatus "ffmpeg ready: $LocalExe"
} catch {
    Write-Warning "ffmpeg bootstrap failed: $($_.Exception.Message)"
} finally {
    if (Test-Path $Archive) {
        Remove-Item -LiteralPath $Archive -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path $Stage) {
        Remove-Item -LiteralPath $Stage -Recurse -Force -ErrorAction SilentlyContinue
    }
}
