# vllama

ollama-compatible CLI/daemon wrapping vllm for high-performance local inference. Targets NVIDIA DGX Spark (Grace Blackwell, 128GB unified memory) but works on any CUDA GPU.

## What it is

- **CLI**: identical UX to ollama (`pull`, `run`, `list`, `ps`, `show`, `rm`, `stop`, `serve`)
- **Daemon**: FastAPI on `:11434` (same as ollama — drop-in compatible)
- **Backend**: vllm processes, one per loaded model, managed by daemon
- **Proxy**: LiteLLM routes requests to correct vllm process by model name
- **Models**: HuggingFace (not ollama registry) with short aliases (`llama3:8b` → HF ID)

## Project layout

```
vllama/
├── pyproject.toml
├── CLAUDE.md
├── vllama/
│   ├── __init__.py
│   ├── cli.py          # typer CLI — thin HTTP client to daemon
│   ├── daemon.py       # FastAPI app — process manager + OpenAI API proxy
│   ├── manager.py      # vllm subprocess lifecycle, LRU eviction, VRAM tracking
│   ├── registry.py     # alias → HuggingFace model ID resolution
│   ├── memory.py       # VRAM / unified memory budget tracking
│   └── config.py       # paths, ports, defaults (XDG-compliant)
└── models/
    └── aliases.yaml    # short name → HF model ID mapping
```

## Key design decisions

- **One vllm process per model** — vllm is single-model by design; manager orchestrates multiple
- **LRU eviction** — when new model load would exceed memory budget, least-recently-used model process is killed first
- **Warn don't fail** — if model is close to budget limit, warn user but attempt load
- **`:11434`** — same port as ollama so existing tools (Open WebUI, etc.) work without reconfiguration
- **HF token** — set `HF_TOKEN` env var or `~/.cache/huggingface/token` for gated models
- **Models stored** in `~/.vllama/models/` (symlinks to HF cache to avoid duplication)

## CLI commands

```
vllama serve              # start daemon (foreground; use systemd/launchd for background)
vllama pull <model>       # download model from HuggingFace
vllama run <model>        # interactive chat (pulls + loads if needed)
vllama list               # downloaded models + disk size
vllama ps                 # running vllm processes + VRAM usage
vllama show <model>       # model info, params, quantization
vllama rm <model>         # delete from disk
vllama stop <model>       # kill vllm process, free VRAM
vllama status             # memory budget, headroom, all loaded models
```

## Model naming

```yaml
# models/aliases.yaml
llama3:8b: meta-llama/Meta-Llama-3.1-8B-Instruct
llama3:70b: meta-llama/Meta-Llama-3.1-70B-Instruct
qwen2.5:7b: Qwen/Qwen2.5-7B-Instruct
qwen2.5:72b: Qwen/Qwen2.5-72B-Instruct
mistral:7b: mistralai/Mistral-7B-Instruct-v0.3
```

Full HF IDs also work directly: `vllama pull Qwen/Qwen2.5-72B-Instruct`

## Dependency stack

- `typer` + `rich` — CLI and pretty output
- `fastapi` + `uvicorn` — daemon
- `huggingface_hub` — model download
- `vllm` — inference engine
- `litellm` — OpenAI-compatible proxy with cost tracking
- `psutil` + `pynvml` — memory/VRAM tracking

## Dev setup

```bash
pip install -e ".[dev]"
vllama serve   # start daemon
vllama run llama3:8b
```

## Daemon internals

Daemon exposes two things on `:11434`:
1. **Management API** (`/api/*`) — mirrors ollama REST API for CLI communication
2. **OpenAI-compatible API** (`/v1/*`) — proxied through LiteLLM to correct vllm instance

vllm instances start on ports `11500+` (11500, 11501, ...), allocated by manager.

## What's intentionally omitted

- `ollama create` / Modelfile — use HF model configs instead
- `ollama push` — use `huggingface-cli push`
- `ollama cp` — not applicable
- Multi-node / Ray cluster — future work; vllm supports it but adds complexity
