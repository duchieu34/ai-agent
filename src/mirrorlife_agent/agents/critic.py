from __future__ import annotations

import json

from ..json_utils import coerce_id_list, parse_json_like
from ..models import DatasetContext
from ..openrouter_client import OpenRouterClient


SANDBOX_SYSTEM_PROMPT = """You are the Critic Agent.
Your job is to challenge risky assumptions, remove weak picks, and return a clean final ID list.

Verification checklist:
- Does each ID show a genuine DECLINING TREND (negative slope), not just noise?
- Is the evidence consistent across multiple time steps (not just start vs end)?
- Are there at least 2 supporting signals (slope + consecutive declines, or cross-indicator)?
- Would removing this ID hurt recall more than it helps precision?

Always return strict JSON.
"""


CHALLENGE_SYSTEM_PROMPT = """You are the Critic Agent.
Challenge risky assumptions, remove weak fraud picks, and return a clean final Transaction ID list.

Verification checklist:
- Is there meaningful anomaly evidence (amount/value/context), not just random noise?
- Is evidence supported by more than one signal when possible?
- Are selected IDs still plausible under precision constraints?
- Would removing this ID likely hurt recall more than it helps precision?

Return strict JSON only.
"""


def run_critic(
    client: OpenRouterClient,
    session_id: str,
    context: DatasetContext,
    planner_result: dict,
    extractor_result: dict,
    scorer_result: dict,
    max_output_ids: int,
    model_name: str | None = None,
) -> dict:
    used_fallback = False

    user_prompt = f"""
Planner output:
{json.dumps(planner_result, indent=2, ensure_ascii=False)}

Extractor output:
{json.dumps(extractor_result, indent=2, ensure_ascii=False)}

Scorer output:
{json.dumps(scorer_result, indent=2, ensure_ascii=False)}

Dataset summary:
{context.summary_text}

Supporting features:
{context.tool_features_text}

Task:
- Cross-check scorer/extractor IDs against the raw feature data above.
- Keep only IDs with consistent evidence across multiple indicators.
- Return up to {max_output_ids} final IDs.
- IDs must be present in scorer recommended list or extractor selected list.

Return strict JSON with fields:
- final_ids: array of strings
- rejected_ids: array of strings
- critic_notes: string
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
    rejected_ids = coerce_id_list(parsed.get("rejected_ids") if isinstance(parsed, dict) else None)
    notes = parsed.get("critic_notes", "") if isinstance(parsed, dict) else ""

    if not final_ids and context.mode == "challenge":
        scorer_ids = coerce_id_list(scorer_result.get("recommended_ids") if isinstance(scorer_result, dict) else None)
        extractor_ids = coerce_id_list(extractor_result.get("selected_ids") if isinstance(extractor_result, dict) else None)
        source_ids = scorer_ids or extractor_ids
        if source_ids:
            fallback_n = min(
                max_output_ids,
                len(source_ids),
                max(4, int(len(source_ids) * 0.70)),
            )
            final_ids = source_ids[:fallback_n]
            used_fallback = True
            if not notes:
                notes = "Fallback: critic adopted conservative subset from scorer/extractor IDs due to empty critic output."

    return {
        "final_ids": final_ids,
        "rejected_ids": rejected_ids,
        "critic_notes": str(notes),
        "used_fallback": used_fallback,
    }
