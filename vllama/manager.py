"""vllm process lifecycle manager with LRU eviction."""

from __future__ import annotations
import asyncio
import subprocess
import time
import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import VLLM_PORT_START, VLLM_DTYPE, VLLM_MAX_MODEL_LEN, MEMORY_WARN_THRESHOLD, MEMORY_BUDGET_GB
from .memory import free_vram_gb, total_vram_gb


@dataclass
class ModelProcess:
    hf_id: str
    port: int
    proc: subprocess.Popen
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def touch(self):
        self.last_used = time.time()

    def alive(self) -> bool:
        return self.proc.poll() is None


class ModelManager:
    def __init__(self):
        self._models: dict[str, ModelProcess] = {}  # hf_id → process
        self._lock = asyncio.Lock()
        self._next_port = VLLM_PORT_START

    def _alloc_port(self) -> int:
        port = self._next_port
        self._next_port += 1
        return port

    def _budget_gb(self) -> float:
        if MEMORY_BUDGET_GB > 0:
            return MEMORY_BUDGET_GB
        return total_vram_gb() * 0.90

    async def ensure_loaded(self, hf_id: str) -> ModelProcess:
        async with self._lock:
            if hf_id in self._models and self._models[hf_id].alive():
                self._models[hf_id].touch()
                return self._models[hf_id]

            # Estimate model size from disk; if unavailable, trust vllm to OOM-warn
            model_size_gb = _estimate_size_gb(hf_id)
            free = free_vram_gb()
            budget = self._budget_gb()

            if model_size_gb and model_size_gb > budget:
                raise MemoryError(
                    f"Model {hf_id} (~{model_size_gb:.1f}GB) exceeds memory budget ({budget:.1f}GB)."
                )

            if model_size_gb and (model_size_gb / budget) > MEMORY_WARN_THRESHOLD:
                # Caller will surface this warning
                pass

            # Evict LRU models until enough free memory
            if model_size_gb:
                await self._evict_until(needed_gb=model_size_gb)

            proc = await self._spawn(hf_id)
            self._models[hf_id] = proc
            return proc

    async def _evict_until(self, needed_gb: float):
        while free_vram_gb() < needed_gb * 1.1:
            if not self._models:
                break
            lru = min(self._models.values(), key=lambda m: m.last_used)
            await self._kill(lru.hf_id)

    async def _spawn(self, hf_id: str) -> ModelProcess:
        port = self._alloc_port()
        env = {**os.environ}

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", hf_id,
            "--port", str(port),
            "--host", "127.0.0.1",
            "--dtype", VLLM_DTYPE,
            "--max-model-len", str(VLLM_MAX_MODEL_LEN),
            "--trust-remote-code",
        ]

        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        mp = ModelProcess(hf_id=hf_id, port=port, proc=proc)

        # Wait for vllm to become ready (up to 120s)
        deadline = time.time() + 120
        async with httpx.AsyncClient() as client:
            while time.time() < deadline:
                if not mp.alive():
                    stderr = proc.stderr.read().decode() if proc.stderr else ""
                    raise RuntimeError(f"vllm process for {hf_id} exited early.\n{stderr}")
                try:
                    r = await client.get(f"{mp.base_url}/health", timeout=2)
                    if r.status_code == 200:
                        return mp
                except Exception:
                    pass
                await asyncio.sleep(2)

        proc.kill()
        raise TimeoutError(f"vllm for {hf_id} did not become ready within 120s.")

    async def _kill(self, hf_id: str):
        mp = self._models.pop(hf_id, None)
        if mp and mp.alive():
            mp.proc.terminate()
            try:
                mp.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                mp.proc.kill()

    async def stop(self, hf_id: str):
        async with self._lock:
            await self._kill(hf_id)

    async def stop_all(self):
        async with self._lock:
            for hf_id in list(self._models):
                await self._kill(hf_id)

    def running(self) -> list[dict]:
        return [
            {
                "model": mp.hf_id,
                "port": mp.port,
                "loaded_at": mp.loaded_at,
                "last_used": mp.last_used,
                "pid": mp.proc.pid,
            }
            for mp in self._models.values()
            if mp.alive()
        ]


def _estimate_size_gb(hf_id: str) -> float | None:
    """Rough size estimate from HF cache metadata. Returns None if unknown."""
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == hf_id:
                return repo.size_on_disk / 1024**3
    except Exception:
        pass
    return None
