from __future__ import annotations

import json

from ..json_utils import coerce_id_list, parse_json_like
from ..models import DatasetContext
from ..openrouter_client import OpenRouterClient


SANDBOX_SYSTEM_PROMPT = """You are the Decider Agent.
Use a compact decision path for clear cases only.
Select final suspicious IDs directly from the candidate pool.

Key signals for suboptimal trajectories:
- Negative activity_slope or sleep_slope (declining trend over time)
- High consecutive decline counts
- Multiple indicators declining simultaneously
- Low minimums combined with negative deltas

Scoring is F1-based: be precise but don't miss obvious cases.
Always return strict JSON.
"""


CHALLENGE_SYSTEM_PROMPT = """You are the Decider Agent.
Use a compact decision path for clear fraud cases only.
Select final suspicious Transaction IDs directly from the candidate pool.

Key signals:
- Very high risk_hint
- Extreme amount_max/amount_sum outliers
- Repeated suspicious patterns (row_count) plus contextual inconsistency

Be conservative but avoid empty output when clear anomalies exist.
Return strict JSON only.
"""


def run_decider(
    client: OpenRouterClient,
    session_id: str,
    context: DatasetContext,
    planner_result: dict,
    max_output_ids: int,
    model_name: str | None = None,
) -> dict:
    user_prompt = f"""
Planner output:
{json.dumps(planner_result, indent=2, ensure_ascii=False)}

Dataset summary:
{context.summary_text}

Supporting features:
{context.tool_features_text}

Candidate pool IDs:
{", ".join(context.candidate_pool)}

Task:
- For clear cases, return a conservative final suspicious list.
- Return at most {max_output_ids} IDs.
- IDs must come only from candidate pool.

Return strict JSON with fields:
- final_ids: array of strings
- confidence_0_to_1: number in [0,1]
- abstain: boolean
- reason: string
"""

    system_prompt = CHALLENGE_SYSTEM_PROMPT if context.mode == "challenge" else SANDBOX_SYSTEM_PROMPT

    raw = client.invoke(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        session_id=session_id,
        model_override=model_name,
    )
    parsed = parse_json_like(raw, default={})

    final_ids = coerce_id_list(parsed.get("final_ids") if isinstance(parsed, dict) else None)

    try:
        confidence = float(parsed.get("confidence_0_to_1", 0.5)) if isinstance(parsed, dict) else 0.5
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = min(max(confidence, 0.0), 1.0)

    abstain = bool(parsed.get("abstain", False)) if isinstance(parsed, dict) else False
    reason = str(parsed.get("reason", "")) if isinstance(parsed, dict) else ""

    return {
        "final_ids": final_ids,
        "confidence_0_to_1": confidence,
        "abstain": abstain,
        "reason": reason,
    }
