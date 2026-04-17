from __future__ import annotations

import re
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def _coerce_float(value: object) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _header_value(headers: object, key: str) -> str | None:
    if headers is None:
        return None

    normalized = key.lower()

    if isinstance(headers, dict):
        for raw_key, raw_value in headers.items():
            if str(raw_key).lower() == normalized:
                return str(raw_value)

    getter = getattr(headers, "get", None)
    if callable(getter):
        value = getter(key)
        if value is None:
            value = getter(normalized)
        if value is not None:
            return str(value)

    return None


def _extract_reset_from_text(text: str) -> float | None:
    # OpenRouter errors may include headers inside payload metadata text.
    match = re.search(r"X-RateLimit-Reset['\"]?\s*:\s*['\"]?(\d{10,13})", text, flags=re.IGNORECASE)
    if not match:
        return None
    return _coerce_float(match.group(1))


def _rate_limit_delay_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None

    retry_after_raw = _header_value(headers, "retry-after")
    retry_after = _coerce_float(retry_after_raw) if retry_after_raw is not None else None
    if retry_after is not None and retry_after > 0:
        return retry_after

    reset_raw = _header_value(headers, "x-ratelimit-reset")
    reset_value = _coerce_float(reset_raw) if reset_raw is not None else None
    if reset_value is None:
        reset_value = _extract_reset_from_text(str(error))

    if reset_value is not None and reset_value > 0:
        now = time.time()
        # 13 digits -> epoch milliseconds, 10 digits -> epoch seconds.
        if reset_value > 1_000_000_000_000:
            return max(0.0, (reset_value / 1000.0) - now)
        if reset_value > 1_000_000_000:
            return max(0.0, reset_value - now)
        # Otherwise treat as a relative number of seconds.
        return reset_value

    message = str(error).lower()
    if error.__class__.__name__.lower() == "ratelimiterror" or "rate limit" in message:
        return 10.0

    return None


def run_with_retry(operation: Callable[[], T], max_retries: int, base_delay_seconds: float) -> T:
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    attempt = 0
    while True:
        try:
            return operation()
        except Exception as error:
            if attempt >= max_retries:
                raise

            backoff_delay = base_delay_seconds * (2 ** attempt)
            rate_limit_delay = _rate_limit_delay_seconds(error)
            delay = rate_limit_delay if rate_limit_delay is not None else backoff_delay

            # Keep retries bounded while adding a small safety buffer.
            delay = min(max(delay + 0.5, 0.0), 180.0)
            if delay > 0:
                time.sleep(delay)
            attempt += 1
