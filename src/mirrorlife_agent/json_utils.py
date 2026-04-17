from __future__ import annotations

import json
from typing import Any


def _extract_balanced_json(text: str) -> str | None:
    start = None
    opening = None

    for idx, ch in enumerate(text):
        if ch in "[{":
            start = idx
            opening = ch
            break

    if start is None or opening is None:
        return None

    closing = "]" if opening == "[" else "}"
    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None


def parse_json_like(text: str, default: Any) -> Any:
    candidates = [text.strip()]
    extracted = _extract_balanced_json(text)
    if extracted:
        candidates.append(extracted)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return default


def coerce_id_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        item = value.strip()
        return [item] if item else []

    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if "id" in item:
                    text = str(item["id"]).strip()
                    if text:
                        output.append(text)
                continue
            text = str(item).strip()
            if text:
                output.append(text)
        return output

    return []
