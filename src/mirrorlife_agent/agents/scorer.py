from __future__ import annotations

import json

from ..json_utils import coerce_id_list, parse_json_like
from ..models import DatasetContext
from ..openrouter_client import OpenRouterClient


SANDBOX_SYSTEM_PROMPT = """You are the Scorer Agent.
Rank extracted IDs by strength of evidence for suboptimal well-being trajectory.

Scoring guidance:
- Strong evidence: negative slope + high consecutive declines + multiple indicators declining
- Moderate evidence: clear negative delta but fewer supporting signals
- Weak evidence: only one indicator declining or only low absolute values without trend

F1 scoring: recommend IDs with strong or moderate evidence. Drop weak ones to protect precision.
Always return strict JSON.
"""


CHALLENGE_SYSTEM_PROMPT = """You are the Scorer Agent.
Rank extracted Transaction IDs by fraud likelihood.

Scoring guidance:
- Strong evidence: high risk_hint plus contextual inconsistencies or repeated suspicious behavior
- Moderate evidence: clear anomaly in amount/value patterns with partial support
- Weak evidence: isolated noisy signal with no supporting pattern

Keep precision under control, but avoid over-abstaining when strong signals are present.
Return strict JSON only.
"""


def run_scorer(
    client: OpenRouterClient,
    session_id: str,
    context: DatasetContext,
    planner_result: dict,
    extractor_result: dict,
    max_output_ids: int,
    model_name: str | None = None,
) -> dict:
    used_fallback = False

    user_prompt = f"""
Planner output:
{json.dumps(planner_result, indent=2, ensure_ascii=False)}

Extractor output:
{json.dumps(extractor_result, indent=2, ensure_ascii=False)}

Dataset summary:
{context.summary_text}

Supporting features:
{context.tool_features_text}

Task:
- Rank extracted IDs.
- Return up to {max_output_ids} recommended IDs.
- Keep IDs strictly from extracted IDs.

Return strict JSON with fields:
- ranked: array of objects {{id, score_0_to_100, reason}}
- recommended_ids: array of strings
- confidence_0_to_1: number in [0,1]
- abstain: boolean
- contradiction_signals: array of strings
"""

    system_prompt = CHALLENGE_SYSTEM_PROMPT if context.mode == "challenge" else SANDBOX_SYSTEM_PROMPT

    raw = client.invoke(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        session_id=session_id,
        model_override=model_name,
    )
    parsed = parse_json_like(raw, default={})

    ranked = parsed.get("ranked", []) if isinstance(parsed, dict) else []
    if not isinstance(ranked, list):
        ranked = []

    recommended = coerce_id_list(parsed.get("recommended_ids") if isinstance(parsed, dict) else None)

    try:
        confidence = float(parsed.get("confidence_0_to_1", 0.5)) if isinstance(parsed, dict) else 0.5
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = min(max(confidence, 0.0), 1.0)

    abstain = bool(parsed.get("abstain", False)) if isinstance(parsed, dict) else False

    contradiction_signals_raw = parsed.get("contradiction_signals", []) if isinstance(parsed, dict) else []
    contradiction_signals = (
        [str(item) for item in contradiction_signals_raw if str(item).strip()]
        if isinstance(contradiction_signals_raw, list)
        else []
    )

    if not recommended and context.mode == "challenge":
        extracted_ids = coerce_id_list(extractor_result.get("selected_ids") if isinstance(extractor_result, dict) else None)
        if extracted_ids:
            fallback_n = min(
                max_output_ids,
                len(extracted_ids),
                max(4, int(len(extracted_ids) * 0.45)),
            )
            recommended = extracted_ids[:fallback_n]
            if not ranked:
                score = 90
                for item in recommended:
                    ranked.append(
                        {
                            "id": item,
                            "score_0_to_100": max(50, score),
                            "reason": "Fallback: promoted conservative subset from extractor shortlist due to empty scorer output.",
                        }
                    )
                    score -= 3
            abstain = False
            confidence = min(confidence, 0.45)
            used_fallback = True

    return {
        "ranked": ranked,
        "recommended_ids": recommended,
        "confidence_0_to_1": confidence,
        "abstain": abstain,
        "contradiction_signals": contradiction_signals,
        "used_fallback": used_fallback,
    }
