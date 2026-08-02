# DirectML free-gen sniff for a Hub epoch tag (does not touch NAS train).
# Usage:
#   .\sniff_hub_epoch.ps1 -Revision epoch-1
#   .\sniff_hub_epoch.ps1 -Revision epoch-2 -N 1500
param(
  [string]$ModelId = "thesimonharms/trocr-javanese-synthetic-v4",
  [Parameter(Mandatory = $true)][string]$Revision,
  [int]$N = 1500,
  [string]$Dataset = "thesimonharms/javanese-dataset"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $root ".venv-dml\Scripts\python.exe"
if (-not (Test-Path $py)) {
  throw "Missing DirectML venv at $py - create .venv-dml with torch-directml first."
}

$env:FORCE_DEVICE = "dml"
$env:HUB_MODEL_ID = $ModelId
$env:HUB_REVISION = $Revision
$env:DATASET_NAME = $Dataset
$env:VERIFY_SPLIT = "validation"
$env:N_SAMPLES = "$N"
$env:LOG_EVERY = "50"

$safeRev = $Revision -replace '[^\w\-]+', '_'
$log = Join-Path $root "local_verify_${N}_v4_${safeRev}_dml.log"
Write-Host "Sniff $ModelId@$Revision n=$N -> $log (DirectML)"
& $py -u (Join-Path $root "local_verify_large.py") 2>&1 | Tee-Object -FilePath $log
Write-Host "EXIT=$LASTEXITCODE log=$log"
