"""Gemini API key storage without a key bundled in the source tree."""

import json
import os
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "user_config.json"


def _load_config() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_config(value: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def get_api_key() -> str:
    """Use a locally saved key, then GEMINI_API_KEY; never fall back to a bundled key."""
    saved = _load_config().get("gemini_api_key", "").strip()
    return saved or os.environ.get("GEMINI_API_KEY", "").strip()


def save_api_key(new_key: str) -> None:
    config = _load_config()
    config["gemini_api_key"] = new_key.strip()
    _save_config(config)


def clear_saved_api_key() -> None:
    config = _load_config()
    config.pop("gemini_api_key", None)
    if config:
        _save_config(config)
    else:
        _CONFIG_PATH.unlink(missing_ok=True)


def is_using_custom_key() -> bool:
    return bool(_load_config().get("gemini_api_key", "").strip())
