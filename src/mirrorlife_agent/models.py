from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class UsageRecord:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class DatasetContext:
    dataset_key: str
    mode: str
    id_label: str
    entity_label: str
    summary_text: str
    tool_features_text: str
    candidate_pool: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    mode: str
    phase: str
    dataset_key: str
    session_id: str
    output_path: Path
    final_ids: list[str]
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    replay_path: Path | None = None
