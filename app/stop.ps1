# SceneSense - stop backend + frontend
# Usage (from repo root):
#   .\stop.ps1
#   powershell -ExecutionPolicy Bypass -File .\stop.ps1

foreach ($port in 8000, 5173) {
  Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object {
      if ($_ -and $_ -ne 0) {
        Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped PID $_ on port $port"
      }
    }
}

Start-Sleep -Seconds 1
foreach ($port in 8000, 5173) {
  $left = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
    Where-Object { $_.State -eq "Listen" }
  if ($left) { Write-Host "Still listening on $port" }
  else { Write-Host "Port $port free" }
}
