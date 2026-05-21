import os
from pathlib import Path

# XDG-compliant paths
HOME = Path.home()
DATA_DIR = Path(os.environ.get("VLLAMA_HOME", HOME / ".vllama"))
MODELS_DIR = DATA_DIR / "models"
CONFIG_FILE = DATA_DIR / "config.yaml"

# Ports
DAEMON_PORT = int(os.environ.get("VLLAMA_PORT", 11434))
DAEMON_HOST = os.environ.get("VLLAMA_HOST", "127.0.0.1")
DAEMON_URL = f"http://{DAEMON_HOST}:{DAEMON_PORT}"
VLLM_PORT_START = 11500  # vllm instances get 11500, 11501, ...

# Alias registry (shipped with package, user can override)
PACKAGE_DIR = Path(__file__).parent.parent
ALIASES_FILE = Path(os.environ.get("VLLAMA_ALIASES", PACKAGE_DIR / "models" / "aliases.yaml"))

# Memory
# 0 = auto-detect (use 90% of available VRAM/unified memory)
MEMORY_BUDGET_GB = float(os.environ.get("VLLAMA_MEMORY_GB", 0))
MEMORY_WARN_THRESHOLD = 0.85  # warn when new load would exceed 85% of budget

# vllm defaults (can be overridden per-model in future)
VLLM_DTYPE = os.environ.get("VLLM_DTYPE", "auto")
VLLM_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", 8192))

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
