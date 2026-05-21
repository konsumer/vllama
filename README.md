# vllama

ollama's UX. vllm's engine.

```
vllama pull llama3:70b
vllama run llama3:70b
```

## Why not just use ollama?

ollama is great for laptops and casual use. It runs on llama.cpp, which is flexible but not optimized for serious GPU hardware.

If you have a **DGX Spark, DGX Station, or any multi-GPU NVIDIA server**, you're leaving most of your hardware on the table with ollama:

| | ollama | vllama |
|--|--|--|
| Engine | llama.cpp | vllm (PagedAttention) |
| Throughput | Moderate | 2–5× higher on NVIDIA |
| Multi-GPU | Limited | Full tensor parallelism via NVLink |
| DGX Spark (128GB unified) | Works, underutilized | Fully utilized |
| Multi-model | Native LRU | Multiple vllm processes + LRU eviction |
| API | OpenAI-compatible | OpenAI-compatible (same port: 11434) |
| HuggingFace models | Via conversion | Direct — no conversion needed |

vllama is a drop-in replacement for ollama's CLI and API. Tools that already work with ollama (Open WebUI, Continue, Cursor, etc.) work with vllama without reconfiguration.

## Requirements

**You need a CUDA GPU.** vllm does not support CPU inference or Apple Silicon.

- NVIDIA GPU with CUDA 12.1+ (Ampere/Ada/Hopper/Blackwell — RTX 3000+, A100, H100, DGX Spark)
- Python 3.10+ **or** Docker + NVIDIA Container Toolkit (see below)

Check your GPU:
```bash
nvidia-smi   # should show your GPU and driver version
```

## Installation

### Option A: Local (from source)

**1. Clone the repo:**
```bash
git clone https://github.com/youruser/vllama
cd vllama
```

**2. Install vllm** (GPU-specific wheel — do this first, before vllama):
```bash
pip install vllm
```
> vllm installation varies by CUDA version. If this fails, see the [vllm install docs](https://docs.vllm.ai/en/latest/getting_started/installation.html) for your specific CUDA version.

**3. Install vllama:**
```bash
pip install -e .
```

**4. (Optional) HuggingFace token** for gated models (Llama 3, Gemma, etc.):
```bash
huggingface-cli login
```

---

### Option B: Docker

Docker handles the CUDA/vllm setup for you. You need:
- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Verify the toolkit is working:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

**Build and start:**
```bash
git clone https://github.com/youruser/vllama
cd vllama
docker compose up -d
```

The daemon starts on `http://localhost:11434`. Model downloads and data persist in Docker volumes across restarts.

**Use the CLI against the running container:**
```bash
# Run vllama commands inside the container
docker compose exec vllama vllama pull llama3:8b
docker compose exec vllama vllama run llama3:8b

# Or install just the CLI locally (no GPU needed for the client)
pip install -e .
# Then talk to the container's daemon:
vllama pull llama3:8b   # daemon is on localhost:11434
```

**HuggingFace token for gated models:**
```bash
# Pass at startup
HF_TOKEN=hf_yourtoken docker compose up -d

# Or add to a .env file in the project root:
echo "HF_TOKEN=hf_yourtoken" >> .env
docker compose up -d
```

**Useful Docker commands:**
```bash
docker compose logs -f        # watch daemon logs
docker compose down           # stop
docker compose pull && docker compose up -d --build   # update
```

## Quick start

```bash
# Start the daemon (keep this running)
vllama serve

# In another terminal:
vllama pull llama3:8b
vllama run llama3:8b
```

That's it. The API is live on `http://localhost:11434` — same as ollama.

## Commands

```
vllama serve              Start the daemon
vllama pull <model>       Download a model
vllama run <model>        Chat interactively (pulls if needed)
vllama list               Show downloaded models
vllama ps                 Show running models + ports + PIDs
vllama show <model>       Model info (size, tags, downloads)
vllama stop <model>       Kill model process, free VRAM
vllama rm <model>         Delete model from disk
vllama status             Memory budget, GPU stats, loaded models
```

## Model names

Short aliases work like ollama tags:

```bash
vllama pull llama3:8b
vllama pull qwen2.5:72b
vllama pull deepseek-r1:32b
vllama pull mistral:7b
```

Full HuggingFace IDs also work:
```bash
vllama pull meta-llama/Meta-Llama-3.1-8B-Instruct
vllama pull Qwen/Qwen2.5-72B-Instruct
```

<details>
<summary>All built-in aliases</summary>

| Alias | HuggingFace ID |
|-------|---------------|
| llama3:8b | meta-llama/Meta-Llama-3.1-8B-Instruct |
| llama3:70b | meta-llama/Meta-Llama-3.1-70B-Instruct |
| llama3:405b | meta-llama/Meta-Llama-3.1-405B-Instruct |
| qwen2.5:7b | Qwen/Qwen2.5-7B-Instruct |
| qwen2.5:72b | Qwen/Qwen2.5-72B-Instruct |
| mistral:7b | mistralai/Mistral-7B-Instruct-v0.3 |
| mistral:nemo | mistralai/Mistral-Nemo-Instruct-2407 |
| gemma3:4b | google/gemma-3-4b-it |
| gemma3:27b | google/gemma-3-27b-it |
| phi4:14b | microsoft/phi-4 |
| deepseek-r1:7b | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B |
| deepseek-r1:32b | deepseek-ai/DeepSeek-R1-Distill-Qwen-32B |
| qwen2.5-coder:32b | Qwen/Qwen2.5-Coder-32B-Instruct |

</details>

## Multi-model

vllama loads multiple models simultaneously if your hardware can fit them. On a DGX Spark (128GB unified memory) you can comfortably run a 70B and several smaller models at the same time:

```bash
vllama run llama3:70b    # loads in background
vllama run qwen2.5:7b    # loads alongside it
vllama ps                # both show up
```

When memory is tight, the least-recently-used model is automatically evicted. You'll get a warning before this happens.

## Using with existing tools

Since vllama runs on port `11434` with an OpenAI-compatible API, no reconfiguration needed for:

- **Open WebUI**: point to `http://localhost:11434`
- **Continue** (VS Code): set `ollama` provider, default URL
- **Cursor / Copilot alternatives**: use `http://localhost:11434/v1` as base URL
- **LangChain / LlamaIndex**: use `ChatOllama` or `OpenAI(base_url="http://localhost:11434/v1")`
- Any OpenAI SDK client: set `base_url="http://localhost:11434/v1"`, `api_key="ignored"`

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLAMA_HOME` | `~/.vllama` | Data directory |
| `VLLAMA_PORT` | `11434` | Daemon port |
| `VLLAMA_HOST` | `127.0.0.1` | Bind host (use `0.0.0.0` for network access) |
| `VLLAMA_MEMORY_GB` | auto (90% of VRAM) | Memory budget for loaded models |
| `VLLAMA_ALIASES` | built-in | Path to custom `aliases.yaml` |
| `VLLM_DTYPE` | `auto` | Model dtype (`float16`, `bfloat16`, `auto`) |
| `VLLM_MAX_MODEL_LEN` | `8192` | Max context length |
| `HF_TOKEN` | — | HuggingFace token for gated models |

## Known limitations

- **vllm startup is slow** (~30–60s per model). First `run` after pulling takes time. Subsequent calls to a loaded model are instant.
- **CUDA only.** No CPU, no Metal, no ROCm (yet — vllm has ROCm support, untested here).
- **No Modelfile** equivalent. Model configuration lives in HuggingFace model cards.
- Multi-node (Ray cluster) support is on the roadmap but not yet wired up.
