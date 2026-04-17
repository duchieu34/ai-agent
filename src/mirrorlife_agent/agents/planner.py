from __future__ import annotations

from ..json_utils import parse_json_like
from ..models import DatasetContext
from ..openrouter_client import OpenRouterClient


SANDBOX_SYSTEM_PROMPT = """You are the Planner Agent in a multi-agent detection system.
Your job is to build a high-level strategy for identifying entities with suboptimal trajectories.

Key domain knowledge:
- "Suboptimal well-being trajectory" means a DECLINING TREND over time (not just low absolute values).
- Look for: negative slope in activity/sleep indices, consecutive declines, rising environmental exposure.
- Cross-indicator patterns (multiple metrics declining together) are stronger signals.
- Scoring is F1-based: balance precision (avoid false positives) and recall (don't miss true positives).

LLM must remain the central decision-maker.
Deterministic feature lines (risk_hint, slope, deltas, consecutive declines) are supporting tools only.
Always return strict JSON.
"""


CHALLENGE_SYSTEM_PROMPT = """You are the Planner Agent in a fraud-detection multi-agent system.
Your job is to build a high-level strategy for selecting suspicious Transaction IDs.

Key domain knowledge:
- Fraud evolves over time, so combine amount anomalies with behavioral/context signals.
- Prioritize: high risk_hint, unusual amount_max/amount_sum, repeated suspicious patterns (row_count),
  unusual timing/location/device/merchant patterns when available.
- Costs are asymmetric: avoid missing likely fraud while keeping false positives controlled.

LLM must remain the central decision-maker.
Deterministic feature lines are supporting tools only.
Return strict JSON only.
"""


def run_planner(
    client: OpenRouterClient,
    session_id: str,
    context: DatasetContext,
    model_name: str | None = None,
) -> dict:
    user_prompt = f"""
Dataset summary:
{context.summary_text}

Supporting feature lines:
{context.tool_features_text}

Candidate pool ({len(context.candidate_pool)} IDs):
{", ".join(context.candidate_pool)}

Return strict JSON with fields:
- strategy: string
- priority_signals: array of strings
- failure_modes: array of strings
- guardrails: array of strings
- route_recommendation: "fast" or "full"
- planner_confidence_0_to_1: number in [0,1]
"""

    system_prompt = CHALLENGE_SYSTEM_PROMPT if context.mode == "challenge" else SANDBOX_SYSTEM_PROMPT

    raw = client.invoke(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        session_id=session_id,
        model_override=model_name,
    )
    parsed = parse_json_like(raw, default={})
    if not isinstance(parsed, dict):
        return {
            "strategy": "Prioritize strongest anomaly trajectories while minimizing false positives.",
            "priority_signals": ["trend deterioration", "high anomaly score"],
            "failure_modes": ["hallucinated IDs", "overfitting to one feature"],
            "guardrails": ["only return IDs from candidate pool"],
            "route_recommendation": "full",
            "planner_confidence_0_to_1": 0.5,
        }

    route = str(parsed.get("route_recommendation", "full")).strip().lower()
    parsed["route_recommendation"] = "fast" if route == "fast" else "full"

    try:
        confidence = float(parsed.get("planner_confidence_0_to_1", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    parsed["planner_confidence_0_to_1"] = min(max(confidence, 0.0), 1.0)

    return parsed
