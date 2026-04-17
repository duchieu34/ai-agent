from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _as_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    value = _as_float(name, default)
    return min(max(value, minimum), maximum)


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    openrouter_api_key: str
    openrouter_base_url: str
    openrouter_model: str
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    llm_model_planner: str
    llm_model_decider: str
    llm_model_extractor: str
    llm_model_scorer: str
    llm_model_critic: str
    openrouter_temperature: float
    openrouter_max_tokens: int

    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str
    team_name: str
    enforce_langfuse: bool

    max_retries: int
    retry_base_delay_seconds: float

    budget_max_usd: float
    budget_max_tokens: int
    token_cost_per_1k_input: float
    token_cost_per_1k_output: float

    max_output_ids: int
    max_candidate_pool: int
    budget_profile: str
    candidate_top_k_low: int
    candidate_top_k_high: int
    adaptive_chain_enabled: bool
    force_full_chain: bool
    fast_path_min_confidence: float
    critic_min_confidence: float
    decision_profile: str
    risk_elbow_cap_enabled: bool
    risk_elbow_min_drop: float
    risk_elbow_min_ratio: float
    risk_elbow_drop_multiple: float
    submission_max_flagged_ratio: float
    challenge_min_flagged_ratio: float
    challenge_fallback_min_flagged_ratio: float
    replay_log_enabled: bool
    replay_log_dir: Path

    submission_state_file: Path

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "openrouter").strip().lower(),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip(),
            openrouter_model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip(),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
            llm_model_planner=os.getenv("LLM_MODEL_PLANNER", "").strip(),
            llm_model_decider=os.getenv("LLM_MODEL_DECIDER", "").strip(),
            llm_model_extractor=os.getenv("LLM_MODEL_EXTRACTOR", "").strip(),
            llm_model_scorer=os.getenv("LLM_MODEL_SCORER", "").strip(),
            llm_model_critic=os.getenv("LLM_MODEL_CRITIC", "").strip(),
            openrouter_temperature=_as_float("OPENROUTER_TEMPERATURE", 0.0),
            openrouter_max_tokens=_as_int("OPENROUTER_MAX_TOKENS", 1200),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "").strip(),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", "").strip(),
            langfuse_host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse").strip(),
            team_name=os.getenv("TEAM_NAME", "team").strip(),
            enforce_langfuse=_as_bool("ENFORCE_LANGFUSE", True),
            max_retries=max(0, _as_int("MAX_RETRIES", 3)),
            retry_base_delay_seconds=max(0.0, _as_float("RETRY_BASE_DELAY_SECONDS", 1.0)),
            budget_max_usd=max(0.0, _as_float("BUDGET_MAX_USD", 40.0)),
            budget_max_tokens=max(0, _as_int("BUDGET_MAX_TOKENS", 0)),
            token_cost_per_1k_input=max(0.0, _as_float("TOKEN_COST_PER_1K_INPUT", 0.0)),
            token_cost_per_1k_output=max(0.0, _as_float("TOKEN_COST_PER_1K_OUTPUT", 0.0)),
            max_output_ids=max(1, _as_int("MAX_OUTPUT_IDS", 200)),
            max_candidate_pool=max(10, _as_int("MAX_CANDIDATE_POOL", 300)),
            budget_profile=os.getenv("BUDGET_PROFILE", "auto").strip().lower(),
            candidate_top_k_low=max(5, _as_int("CANDIDATE_TOP_K_LOW", 40)),
            candidate_top_k_high=max(5, _as_int("CANDIDATE_TOP_K_HIGH", 120)),
            adaptive_chain_enabled=_as_bool("ADAPTIVE_CHAIN_ENABLED", True),
            force_full_chain=_as_bool("FORCE_FULL_CHAIN", False),
            fast_path_min_confidence=_bounded_float("FAST_PATH_MIN_CONFIDENCE", 0.8, 0.0, 1.0),
            critic_min_confidence=_bounded_float("CRITIC_MIN_CONFIDENCE", 0.65, 0.0, 1.0),
            decision_profile=os.getenv("DECISION_PROFILE", "precision_first").strip().lower(),
            risk_elbow_cap_enabled=_as_bool("RISK_ELBOW_CAP_ENABLED", True),
            risk_elbow_min_drop=max(0.0, _as_float("RISK_ELBOW_MIN_DROP", 4.0)),
            risk_elbow_min_ratio=max(1.0, _as_float("RISK_ELBOW_MIN_RATIO", 2.5)),
            risk_elbow_drop_multiple=max(1.0, _as_float("RISK_ELBOW_DROP_MULTIPLE", 2.0)),
            submission_max_flagged_ratio=_bounded_float("SUBMISSION_MAX_FLAGGED_RATIO", 0.6, 0.01, 1.0),
            challenge_min_flagged_ratio=_bounded_float("CHALLENGE_MIN_FLAGGED_RATIO", 0.22, 0.0, 0.95),
            challenge_fallback_min_flagged_ratio=_bounded_float("CHALLENGE_FALLBACK_MIN_FLAGGED_RATIO", 0.10, 0.0, 0.95),
            replay_log_enabled=_as_bool("REPLAY_LOG_ENABLED", True),
            replay_log_dir=Path(os.getenv("REPLAY_LOG_DIR", "replays")),
            submission_state_file=Path(os.getenv("SUBMISSION_STATE_FILE", ".submission_guard_state.json")),
        )
