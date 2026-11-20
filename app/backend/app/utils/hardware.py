"""Hardware detection for local inference."""

from __future__ import annotations

import platform
from typing import Any


def detect_hardware() -> dict[str, Any]:
    """Detect CPU/RAM/CUDA availability without crashing if CUDA is absent."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "cuda_available": False,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "cpu_count": None,
        "ram_gb": None,
        "recommended_device": "cpu",
    }

    try:
        import psutil

        info["cpu_count"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception as exc:  # pragma: no cover
        info["psutil_error"] = str(exc)

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(props.total_memory / (1024**3), 2)
            info["recommended_device"] = "cuda"
        else:
            info["recommended_device"] = "cpu"
    except Exception as exc:  # pragma: no cover
        info["torch_error"] = str(exc)

    return info


def resolve_device(preference: str = "auto") -> str:
    """Resolve runtime device from preference and availability."""
    pref = (preference or "auto").lower().strip()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    return detect_hardware()["recommended_device"]
