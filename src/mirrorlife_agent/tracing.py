from __future__ import annotations

from typing import Any

import ulid

from .config import Settings

try:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler
except Exception:  # pragma: no cover
    Langfuse = None
    CallbackHandler = None


class TracingManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None

        has_keys = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
        if settings.enforce_langfuse and not has_keys:
            raise ValueError(
                "Langfuse is enforced but LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are missing."
            )

        if has_keys:
            if Langfuse is None:
                raise RuntimeError("Langfuse package is not available.")
            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def generate_session_id(self) -> str:
        team = (self.settings.team_name or "team").replace(" ", "-")
        return f"{team}-{ulid.new().str}"

    def build_callback_handler(self) -> Any | None:
        if not self.enabled:
            return None
        if CallbackHandler is None:
            raise RuntimeError("Langfuse CallbackHandler is not available.")
        return CallbackHandler()

    def flush(self) -> None:
        if self._client is not None:
            self._client.flush()
