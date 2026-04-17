from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or "run"


class ReplayLogger:
    def __init__(self, enabled: bool, log_dir: Path) -> None:
        self.enabled = enabled
        self.log_dir = log_dir

    def write(self, payload: dict[str, Any]) -> Path | None:
        if not self.enabled:
            return None

        self.log_dir.mkdir(parents=True, exist_ok=True)

        dataset_key = _slug(str(payload.get("dataset_key", "dataset")))
        session_id = _slug(str(payload.get("session_id", "session")))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        file_path = self.log_dir / f"{timestamp}_{dataset_key}_{session_id}.json"
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)

        return file_path
