# Install CUDA-enabled PyTorch for SceneSense (Windows NVIDIA GPUs).
# Driver must already support CUDA (check with: nvidia-smi).
#
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup_gpu.ps1

$ErrorActionPreference = "Continue"
Write-Host "Installing PyTorch with CUDA (cu126)..."
Write-Host "This replaces the CPU-only torch build."
Write-Host ""

python -m pip uninstall -y torch torchvision torchaudio
if ($LASTEXITCODE -ne 0) { Write-Host "(uninstall warnings ignored)" }

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw "pip install torch torchvision failed" }

Write-Host ""
Write-Host "Verifying CUDA..."
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

Write-Host ""
Write-Host "Done. Restart SceneSense with: .\start.ps1"
Write-Host "Set SCENESENSE_DEVICE=auto (default) or cuda in .env"
