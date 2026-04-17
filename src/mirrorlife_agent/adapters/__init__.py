from __future__ import annotations

from .base import DomainAdapter
from .challenge import ChallengeAdapter
from .sandbox import SandboxAdapter


def build_adapter(mode: str, max_candidate_pool: int) -> DomainAdapter:
    normalized = mode.strip().lower()
    if normalized == "sandbox":
        return SandboxAdapter(max_candidate_pool=max_candidate_pool)
    if normalized == "challenge":
        return ChallengeAdapter(max_candidate_pool=max_candidate_pool)
    raise ValueError(f"Unsupported mode: {mode}")
