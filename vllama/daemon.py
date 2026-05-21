"""
FastAPI daemon on :11434.
- /api/*  management API (ollama-compatible where possible)
- /v1/*   OpenAI-compatible API proxied to correct vllm instance
"""

from __future__ import annotations
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

from .manager import ModelManager
from .registry import resolve, short_name
from .memory import gpu_info, free_vram_gb, total_vram_gb
from .config import MODELS_DIR

manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await manager.stop_all()


app = FastAPI(title="vllama", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Management API  (/api/*)
# ---------------------------------------------------------------------------


@app.get("/api/tags")
async def list_models():
    """ollama GET /api/tags — list downloaded models."""
    models = []
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_type == "model":
                models.append(
                    {
                        "name": short_name(repo.repo_id),
                        "model": repo.repo_id,
                        "size": repo.size_on_disk,
                        "modified_at": repo.last_modified,
                    }
                )
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"models": models}


@app.post("/api/pull")
async def pull_model(req: Request):
    """ollama POST /api/pull — download model from HuggingFace."""
    body = await req.json()
    name = body.get("name", "")
    try:
        hf_id = resolve(name)
    except ValueError as e:
        raise HTTPException(400, str(e))

    async def stream():
        from huggingface_hub import snapshot_download
        import json

        yield json.dumps({"status": f"pulling {hf_id}"}) + "\n"
        try:
            snapshot_download(repo_id=hf_id, local_files_only=False)
            yield json.dumps({"status": "success"}) + "\n"
        except Exception as e:
            yield json.dumps({"status": "error", "error": str(e)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.delete("/api/delete")
async def delete_model(req: Request):
    """ollama DELETE /api/delete — remove model from disk."""
    body = await req.json()
    name = body.get("name", "")
    try:
        hf_id = resolve(name)
    except ValueError as e:
        raise HTTPException(400, str(e))

    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        delete_strategy = cache.delete_revisions(
            *[
                rev.commit_hash
                for repo in cache.repos
                if repo.repo_id == hf_id
                for rev in repo.revisions
            ]
        )
        delete_strategy.execute()
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"status": "success"}


@app.post("/api/show")
async def show_model(req: Request):
    """ollama POST /api/show — model info."""
    body = await req.json()
    name = body.get("name", "")
    try:
        hf_id = resolve(name)
    except ValueError as e:
        raise HTTPException(400, str(e))

    try:
        from huggingface_hub import model_info

        info = model_info(hf_id)
        return {
            "model": hf_id,
            "alias": short_name(hf_id),
            "tags": info.tags or [],
            "pipeline_tag": info.pipeline_tag,
            "downloads": info.downloads,
            "likes": info.likes,
            "created_at": str(info.created_at) if info.created_at else None,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/ps")
async def running_models():
    """ollama GET /api/ps — list running model processes."""
    running = manager.running()
    now = time.time()
    return {
        "models": [
            {
                "name": short_name(m["model"]),
                "model": m["model"],
                "pid": m["pid"],
                "port": m["port"],
                "size_vram": None,  # filled in by client from /api/status
                "until": None,
                "loaded_at": m["loaded_at"],
                "expires_at": None,
            }
            for m in running
        ]
    }


@app.post("/api/stop")
async def stop_model(req: Request):
    """ollama POST /api/stop equivalent — kill vllm process."""
    body = await req.json()
    name = body.get("name", "")
    try:
        hf_id = resolve(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    await manager.stop(hf_id)
    return {"status": "stopped"}


@app.get("/api/status")
async def status():
    """vllama-specific: memory budget, GPU info, loaded models."""
    return {
        "gpus": gpu_info(),
        "total_gb": total_vram_gb(),
        "free_gb": free_vram_gb(),
        "used_gb": total_vram_gb() - free_vram_gb(),
        "models": manager.running(),
    }


@app.get("/")
async def health():
    return {"status": "ok", "version": "0.1.0"}


# ---------------------------------------------------------------------------
# OpenAI-compatible proxy  (/v1/*)
# ---------------------------------------------------------------------------


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "DELETE", "PUT", "OPTIONS"])
async def openai_proxy(path: str, request: Request):
    """Route to the correct vllm instance based on model name in request body."""
    body = b""
    model = None

    if request.method in ("POST", "PUT"):
        body = await request.body()
        try:
            import json
            data = json.loads(body)
            model = data.get("model")
        except Exception:
            pass

    if model:
        try:
            hf_id = resolve(model)
        except ValueError:
            hf_id = model  # pass through unknown names — let vllm reject

        try:
            mp = await manager.ensure_loaded(hf_id)
        except MemoryError as e:
            return JSONResponse(status_code=507, content={"error": str(e)})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

        target = f"{mp.base_url}/v1/{path}"
    else:
        # No model specified — proxy to first running instance or 400
        running = manager.running()
        if not running:
            return JSONResponse(status_code=400, content={"error": "No model loaded. Run: vllama run <model>"})
        target = f"http://127.0.0.1:{running[0]['port']}/v1/{path}"

    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    stream = "stream" in (model or "") or path.endswith("stream")

    async with httpx.AsyncClient(timeout=300) as client:
        if body:
            resp = await client.request(request.method, target, content=body, headers=headers)
        else:
            resp = await client.request(request.method, target, headers=headers)

        if resp.headers.get("content-type", "").startswith("text/event-stream"):
            async def stream_gen():
                async for chunk in resp.aiter_bytes():
                    yield chunk
            return StreamingResponse(stream_gen(), media_type="text/event-stream", status_code=resp.status_code)

        return JSONResponse(content=resp.json(), status_code=resp.status_code)
