"""Tracks the last-processed timestamp per agent so log_experiment.py doesn't
double-count calls across runs. Stored as a small local JSON file — simple
and sufficient for a single-instance scheduled job."""

import json
import os
from datetime import datetime, timezone

from .config import Config

DEFAULT_LOOKBACK_ISO = "1970-01-01T00:00:00+00:00"


def _load() -> dict[str, str]:
    if not os.path.exists(Config.CHECKPOINT_PATH):
        return {}
    with open(Config.CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_last_run(agent_id: str) -> datetime:
    data = _load()
    raw = data.get(agent_id, DEFAULT_LOOKBACK_ISO)
    return datetime.fromisoformat(raw)


def set_last_run(agent_id: str, when: datetime) -> None:
    data = _load()
    data[agent_id] = when.astimezone(timezone.utc).isoformat()
    with open(Config.CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
