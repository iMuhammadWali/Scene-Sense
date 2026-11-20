# PowerShell equivalent of run_all.sh. Idempotent/resumable: a step is skipped if results\<dir>\DONE exists.
# Usage: scripts\run_all.ps1 [-AllowPartial] [-Out DIR] [-Limit N] [-E6Videos N] [-E6Queries Q] [-Force]
#   (also accepts the bash-style spellings --allow-partial --out --limit --e6-videos --e6-queries --force)
# Env: PY (default: .venv\Scripts\python.exe if present, else python), SSEVAL_EMB (default data\emb)
param([switch]$AllowPartial, [string]$Out = "", [int]$Limit = 0, [int]$E6Videos = 0, [int]$E6Queries = 0, [switch]$Force)
$ErrorActionPreference = "Continue"
$a = $args
for ($i = 0; $i -lt $a.Count; $i++) {
    switch ($a[$i]) {
        "--allow-partial" { $AllowPartial = $true }
        "--out"           { $Out = $a[++$i] }
        "--limit"         { $Limit = [int]$a[++$i] }
        "--e6-videos"     { $E6Videos = [int]$a[++$i] }
        "--e6-queries"    { $E6Queries = [int]$a[++$i] }
        "--force"         { $Force = $true }
        default           { Write-Host "unknown arg $($a[$i])"; exit 2 }
    }
}
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Repo
$venvPy = Join-Path $Repo ".venv\Scripts\python.exe"
$PY = if ($env:PY) { $env:PY } elseif (Test-Path $venvPy) { $venvPy } else { "python" }
$Emb = if ($env:SSEVAL_EMB) { $env:SSEVAL_EMB } else { Join-Path $Repo "data\emb" }
if (-not $env:SCENESENSE_DEVICE) { $env:SCENESENSE_DEVICE = "cpu" }
$env:PYTHONIOENCODING = "utf-8"
if ($Out) { $env:SSEVAL_RESULTS_DIR = [System.IO.Path]::GetFullPath($Out) }
$Res = if ($env:SSEVAL_RESULTS_DIR) { $env:SSEVAL_RESULTS_DIR } else { Join-Path $Repo "results" }
New-Item -ItemType Directory -Force -Path $Res, "logs" | Out-Null
function Log($m) { Write-Host "[$(Get-Date -Format HH:mm:ss)] $m" }

$n = @(Get-ChildItem -Path $Emb -Filter *.npz -ErrorAction SilentlyContinue | Where-Object { $_.Name -notlike "*.tmp.npz" }).Count
Log "step 1: $n / 1334 video embeddings present (encode with: python scripts\encode_all.py)"
if ($n -lt 1334 -and -not $AllowPartial) { Log "ERROR: embeddings incomplete (use -AllowPartial to run anyway)"; exit 1 }

function Step($name, $dir, $cmdArgs) {
    if ((Test-Path (Join-Path $Res "$dir\DONE")) -and -not $Force) { Log "${name}: DONE exists, skipping"; return }
    Log "${name}: running"; $t0 = Get-Date
    & $PY -W ignore @cmdArgs *> "logs\$name.log"
    Log "${name}: exit $LASTEXITCODE after $([int]((Get-Date) - $t0).TotalSeconds)s (log: logs\$name.log)"
}
$lim = @(); if ($Limit -gt 0) { $lim = @("--limit", "$Limit") }
$frc = @(); if ($Force) { $frc = @("--force") }
$e6 = @(); if ($E6Videos -gt 0) { $e6 += @("--limit", "$E6Videos") }; if ($E6Queries -gt 0) { $e6 += @("--max-queries", "$E6Queries") }
Step "e1"  "e1_pruning" (@("experiments\e1_pruning\run.py") + $lim + $frc)
Step "e1b" "e1b_long"   (@("experiments\e1b_long\run.py") + $lim + $frc)
Step "e3"  "e3_absent"  (@("experiments\e3_absent\run.py") + $lim + $frc)
Step "e6"  "e6_latency" (@("experiments\e6_latency\run.py") + $e6 + $frc)
Step "figures" "figures" @("scripts\make_figures.py")
Step "summary" "summary" @("scripts\make_summary.py")
Log "all steps finished; results in $Res (read $Res\SUMMARY.md)"
