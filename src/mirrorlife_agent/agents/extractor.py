from __future__ import annotations

import json

from ..json_utils import coerce_id_list, parse_json_like
from ..models import DatasetContext
from ..openrouter_client import OpenRouterClient


SANDBOX_SYSTEM_PROMPT = """You are the Extractor Agent.
Select candidate IDs likely to be true positives (suboptimal well-being trajectories).
You must only use IDs from the candidate pool.

Prioritize candidates showing:
- Negative slope in activity or sleep indices (declining trend over time)
- Multiple consecutive declines in key metrics
- Rising environmental exposure combined with declining activity/sleep
- Cross-indicator deterioration (2+ metrics declining together)

Avoid selecting IDs that only have low absolute values without a declining trend.
Scoring is F1-based: include all likely positives but avoid weak picks.
Always return strict JSON.
"""


CHALLENGE_SYSTEM_PROMPT = """You are the Extractor Agent.
Select Transaction IDs likely to be fraudulent.
You must only use IDs from the candidate pool.

Prioritize candidates showing:
- Highest risk_hint and unusual amount_max/amount_sum patterns
- Repeated suspicious behavior (row_count) or abrupt balance-related shifts
- Inconsistencies across transaction context (type, location, payment method, description) when available
- Patterns suggesting adaptive fraud rather than isolated normal behavior

If evidence is mixed, still return a focused shortlist of strongest suspects.
Scoring is F1-oriented: avoid very weak picks, but do not return an empty list when strong signals exist.
Return strict JSON only.
"""


def run_extractor(
    client: OpenRouterClient,
    session_id: str,
    context: DatasetContext,
    planner_result: dict,
    max_output_ids: int,
    model_name: str | None = None,
) -> dict:
    extraction_cap = max_output_ids if context.mode == "challenge" else (max_output_ids * 2)
    used_fallback = False

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
- Select up to {extraction_cap} IDs that are likely true positives.
- Explain each selected ID briefly.
- Do not invent IDs.

Return strict JSON with fields:
- selected_ids: array of strings
- rationale: object map of id -> short reason
"""

    system_prompt = CHALLENGE_SYSTEM_PROMPT if context.mode == "challenge" else SANDBOX_SYSTEM_PROMPT

    raw = client.invoke(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        session_id=session_id,
        model_override=model_name,
    )
    parsed = parse_json_like(raw, default={})

    selected = coerce_id_list(parsed.get("selected_ids") if isinstance(parsed, dict) else None)
    rationale = parsed.get("rationale", {}) if isinstance(parsed, dict) else {}

    if not selected and context.mode == "challenge" and context.candidate_pool:
        fallback_n = min(
            len(context.candidate_pool),
            extraction_cap,
            max_output_ids,
            max(6, int(max_output_ids * 0.35)),
        )
        selected = context.candidate_pool[:fallback_n]
        used_fallback = True
        if not isinstance(rationale, dict):
            rationale = {}
        for item in selected:
            rationale.setdefault(item, "Fallback: selected conservative subset from highest risk_hint candidates.")

    return {
        "selected_ids": selected,
        "rationale": rationale if isinstance(rationale, dict) else {},
        "used_fallback": used_fallback,
    }
