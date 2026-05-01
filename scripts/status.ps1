$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $Root "src"
python -m protoagi status
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version --format=csv
