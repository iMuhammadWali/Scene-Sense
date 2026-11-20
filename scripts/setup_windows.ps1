# Create .\.venv with the pinned environment.  Usage (repo root):  powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Repo
$req = (Select-String -Path requirements.txt -Pattern '^# Python:\s*(\d+\.\d+)' | Select-Object -First 1).Matches[0].Groups[1].Value
$ver = "import sys; print('%d.%d' % sys.version_info[:2])"
$py = $null
$v = $null; try { $v = (& py "-$req" -c $ver 2>$null) } catch {}
if ($LASTEXITCODE -eq 0 -and $v -eq $req) { $py = @("py", "-$req") }
else {
    $v = $null; try { $v = (& python -c $ver 2>$null) } catch {}
    if ($LASTEXITCODE -eq 0 -and $v -eq $req) { $py = @("python") }
}
if (-not $py) { Write-Host "ERROR: Python $req not found (tried 'py -$req' and 'python'). Install Python $req (64-bit) from python.org and re-run." -ForegroundColor Red; exit 1 }
Write-Host "Using: $($py -join ' ')"
$rest = @(); if ($py.Length -gt 1) { $rest = $py[1..($py.Length - 1)] }
& $py[0] @($rest + @("-m", "venv", ".venv"))
$vpy = Join-Path $Repo ".venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip
# CPU torch from the PyTorch index (no CUDA), then the rest pinned by requirements.txt
& $vpy -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
& $vpy scripts\check_env.py
if ($LASTEXITCODE -ne 0) { Write-Host "Environment check FAILED (see table above)." -ForegroundColor Red; exit 1 }
Write-Host "Done. Activate with:  .\.venv\Scripts\Activate.ps1"
