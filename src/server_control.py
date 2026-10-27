"""Request shutdown from the single local server supervisor."""

import json
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent.parent / ".local"


def get_server_state() -> dict:
    try:
        return json.loads((STATE_DIR / "server.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def request_shutdown() -> bool:
    state = get_server_state()
    if not state.get("token"):
        return False
    temporary = STATE_DIR / "stop.tmp"
    temporary.write_text(json.dumps({"token": state["token"]}), encoding="utf-8")
    temporary.replace(STATE_DIR / "stop.json")
    return True
