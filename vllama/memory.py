"""VRAM / unified memory tracking."""

from __future__ import annotations
import psutil

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


def total_vram_gb() -> float:
    """Total VRAM across all GPUs (or system RAM if no GPU detected)."""
    if _NVML_OK:
        total = 0
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            total += info.total
        return total / 1024**3
    # Fallback: unified memory systems (DGX Spark) — use system RAM
    return psutil.virtual_memory().total / 1024**3


def free_vram_gb() -> float:
    """Free VRAM across all GPUs (or system RAM)."""
    if _NVML_OK:
        free = 0
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            free += info.free
        return free / 1024**3
    return psutil.virtual_memory().available / 1024**3


def used_vram_gb() -> float:
    return total_vram_gb() - free_vram_gb()


def gpu_info() -> list[dict]:
    """Per-GPU stats for `status` display."""
    if not _NVML_OK:
        mem = psutil.virtual_memory()
        return [
            {
                "name": "System RAM (no GPU detected)",
                "total_gb": mem.total / 1024**3,
                "used_gb": (mem.total - mem.available) / 1024**3,
                "free_gb": mem.available / 1024**3,
            }
        ]
    gpus = []
    count = pynvml.nvmlDeviceGetCount()
    for i in range(count):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        gpus.append(
            {
                "name": name,
                "total_gb": info.total / 1024**3,
                "used_gb": info.used / 1024**3,
                "free_gb": info.free / 1024**3,
            }
        )
    return gpus
