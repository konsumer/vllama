"""Resolve model names: alias → HuggingFace model ID."""

from pathlib import Path
import yaml
from .config import ALIASES_FILE


def _load_aliases() -> dict[str, str]:
    if not ALIASES_FILE.exists():
        return {}
    with open(ALIASES_FILE) as f:
        return yaml.safe_load(f) or {}


def resolve(name: str) -> str:
    """Return HF model ID for name. Passes through if already an HF ID (contains '/')."""
    if "/" in name:
        return name
    aliases = _load_aliases()
    if name in aliases:
        return aliases[name]
    raise ValueError(f"Unknown model '{name}'. Use full HF ID (org/model) or add to aliases.yaml.")


def list_aliases() -> dict[str, str]:
    return _load_aliases()


def short_name(hf_id: str) -> str:
    """Return alias for HF ID if one exists, else return the HF ID."""
    aliases = _load_aliases()
    reverse = {v: k for k, v in aliases.items()}
    return reverse.get(hf_id, hf_id)
