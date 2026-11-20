# SceneSense - start backend + frontend with one command
# Usage (from repo root):
#   .\start.ps1
#   powershell -ExecutionPolicy Bypass -File .\start.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = Join-Path $Root "backend"

Write-Host "Starting SceneSense..."
Write-Host "  Backend:  http://127.0.0.1:8000"
Write-Host "  Frontend: http://localhost:5173"
Write-Host "Press Ctrl+C in this window to stop both."
Write-Host ""

$backend = Start-Process -PassThru -NoNewWindow -WorkingDirectory (Join-Path $Root "backend") `
  -FilePath "python" -ArgumentList "-m","app.main"

$frontend = Start-Process -PassThru -NoNewWindow -WorkingDirectory (Join-Path $Root "frontend") `
  -FilePath "cmd.exe" -ArgumentList "/c","npm run dev"

function Stop-All {
  foreach ($p in @($frontend, $backend)) {
    if ($p -and -not $p.HasExited) {
      Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
  }
  foreach ($port in 8000, 5173) {
    Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
      Select-Object -ExpandProperty OwningProcess -Unique |
      ForEach-Object {
        if ($_ -and $_ -ne 0) {
          Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
        }
      }
  }
}

try {
  while ($true) {
    if ($backend.HasExited -or $frontend.HasExited) {
      Write-Host "A process exited - shutting down."
      break
    }
    Start-Sleep -Seconds 2
  }
}
finally {
  Stop-All
  Write-Host "SceneSense stopped."
}
