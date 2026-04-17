from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import re
from pathlib import Path
from typing import Any

from .adapters import build_adapter
from .agents import run_critic, run_decider, run_extractor, run_planner, run_scorer
from .budget_guard import BudgetGuard
from .config import Settings
from .models import RunResult
from .openrouter_client import OpenRouterClient
from .replay_logger import ReplayLogger
from .submission_guard import SubmissionGuard
from .tracing import TracingManager


def _bounded_float(value: object, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return min(max(numeric, 0.0), 1.0)


def _extract_level_index(dataset_key: str) -> int | None:
    numbers = re.findall(r"\d+", dataset_key)
    if not numbers:
        return None
    try:
        return int(numbers[-1])
    except ValueError:
        return None


def _extract_public_level(value: str) -> int | None:
    match = re.search(r"public_lev_(\d+)", value, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _extract_risk_hints(tool_features_text: str) -> dict[str, float]:
    hints: dict[str, float] = {}
    for line in tool_features_text.splitlines():
        id_match = re.search(r"\bid=([A-Za-z0-9_-]+)\b", line)
        risk_match = re.search(r"\brisk_hint=([-+]?\d+(?:\.\d+)?)\b", line)
        if not id_match or not risk_match:
            continue
        try:
            hints[id_match.group(1)] = float(risk_match.group(1))
        except ValueError:
            continue
    return hints


def _extract_risk_component_summary(tool_features_text: str, *, max_lines: int = 300) -> dict[str, Any]:
    component_sums: dict[str, float] = defaultdict(float)
    component_abs_sums: dict[str, float] = defaultdict(float)
    component_counts: dict[str, int] = defaultdict(int)
    tx_component_rows: list[tuple[float, float, str, float]] = []

    lines = [line for line in tool_features_text.splitlines() if line.strip()]
    for line in lines[:max_lines]:
        fields: dict[str, str] = {}
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key:
                fields[key] = value

        tx_id = fields.get("id", "")
        if not tx_id:
            continue

        component_total = 0.0
        component_abs_total = 0.0
        for key, raw_value in fields.items():
            if not key.startswith("c_"):
                continue
            try:
                value = float(raw_value)
            except ValueError:
                continue
            component_sums[key] += value
            component_abs_sums[key] += abs(value)
            component_counts[key] += 1
            component_total += value
            component_abs_total += abs(value)

        if component_counts:
            try:
                risk_hint = float(fields.get("risk_hint", "0"))
            except ValueError:
                risk_hint = 0.0
            tx_component_rows.append((component_total, component_abs_total, tx_id, risk_hint))

    component_mean = {
        key: (component_sums[key] / component_counts[key])
        for key in sorted(component_counts.keys())
        if component_counts[key] > 0
    }
    component_abs_mean = {
        key: (component_abs_sums[key] / component_counts[key])
        for key in sorted(component_counts.keys())
        if component_counts[key] > 0
    }

    top_positive = sorted(tx_component_rows, key=lambda item: (-item[0], -item[1], item[2]))[:10]
    top_negative = sorted(tx_component_rows, key=lambda item: (item[0], -item[1], item[2]))[:10]

    return {
        "rows_scanned": min(len(lines), max_lines),
        "component_mean": component_mean,
        "component_abs_mean": component_abs_mean,
        "top_positive_component_ids": [
            {
                "id": item[2],
                "component_sum": item[0],
                "component_abs_sum": item[1],
                "risk_hint": item[3],
            }
            for item in top_positive
        ],
        "top_negative_component_ids": [
            {
                "id": item[2],
                "component_sum": item[0],
                "component_abs_sum": item[1],
                "risk_hint": item[3],
            }
            for item in top_negative
        ],
    }


def _starts_with_fallback(value: object) -> bool:
    return str(value or "").strip().lower().startswith("fallback:")


class MultiAgentOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.budget_guard = BudgetGuard(
            max_usd=settings.budget_max_usd,
            max_tokens=settings.budget_max_tokens,
        )
        self.submission_guard = SubmissionGuard(settings.submission_state_file)
        self.tracing = TracingManager(settings)
        self.llm_client = OpenRouterClient(
            settings=settings,
            budget_guard=self.budget_guard,
            tracing=self.tracing,
        )
        self.replay_logger = ReplayLogger(
            enabled=settings.replay_log_enabled,
            log_dir=settings.replay_log_dir,
        )

    def _resolve_budget_profile(self, dataset_key: str) -> str:
        profile = self.settings.budget_profile
        if profile in {"low", "high"}:
            return profile

        level_idx = _extract_level_index(dataset_key)
        if level_idx is None:
            return "low"

        return "low" if level_idx <= 3 else "high"

    def _decision_policy(self) -> dict[str, Any]:
        requested = (self.settings.decision_profile or "precision_first").strip().lower()

        if requested == "balanced":
            return {
                "profile": "balanced",
                "use_risk_elbow_cap": False,
                "critic_is_veto": False,
                "scorer_override_confidence": 0.82,
                "decider_anchor_confidence": 0.86,
                "min_votes_for_non_anchor": 1,
                "fast_path_max_candidate_pool": 12,
                "fast_path_confidence_boost": 0.0,
                "require_no_contradictions_for_scorer_anchor": False,
                "fallback_ratio": 0.18,
                "source_priority_full": ["scorer", "critic", "decider", "extractor"],
                "source_priority_fast": ["decider", "scorer", "critic", "extractor"],
            }

        return {
            "profile": "precision_first",
            "use_risk_elbow_cap": True,
            "critic_is_veto": True,
            "scorer_override_confidence": 0.90,
            "decider_anchor_confidence": 0.92,
            "min_votes_for_non_anchor": 2,
            "fast_path_max_candidate_pool": 8,
            "fast_path_confidence_boost": 0.08,
            "require_no_contradictions_for_scorer_anchor": True,
            "fallback_ratio": 0.08,
            "source_priority_full": ["critic", "scorer", "decider", "extractor"],
            "source_priority_fast": ["decider", "scorer", "critic", "extractor"],
        }

    def _llm_selection_cap(
        self,
        *,
        mode: str,
        candidate_pool_size: int,
        effective_max_output_ids: int,
    ) -> int:
        if mode != "challenge" or candidate_pool_size <= 0:
            return effective_max_output_ids

        # Keep challenge prompts compact enough to preserve strict JSON reliability.
        dynamic_cap = max(12, int(candidate_pool_size * 0.18))
        return max(1, min(effective_max_output_ids, 80, dynamic_cap))

    def _extractor_used_fallback(self, extractor_result: dict) -> bool:
        if not isinstance(extractor_result, dict):
            return False

        if bool(extractor_result.get("used_fallback", False)):
            return True

        selected_ids = extractor_result.get("selected_ids", [])
        rationale = extractor_result.get("rationale", {})
        if not isinstance(selected_ids, list) or not selected_ids:
            return False
        if not isinstance(rationale, dict) or not rationale:
            return False

        fallback_hits = 0
        for item in selected_ids:
            reason = rationale.get(str(item), "")
            if _starts_with_fallback(reason):
                fallback_hits += 1

        return fallback_hits >= max(1, int(len(selected_ids) * 0.8))

    def _scorer_used_fallback(self, scorer_result: dict) -> bool:
        if not isinstance(scorer_result, dict):
            return False

        if bool(scorer_result.get("used_fallback", False)):
            return True

        ranked = scorer_result.get("ranked", [])
        recommended_ids = scorer_result.get("recommended_ids", [])
        if not isinstance(recommended_ids, list) or not recommended_ids:
            return False
        if not isinstance(ranked, list) or not ranked:
            return False

        fallback_ranked = 0
        for item in ranked:
            if not isinstance(item, dict):
                continue
            if _starts_with_fallback(item.get("reason", "")):
                fallback_ranked += 1

        return fallback_ranked >= max(1, int(len(recommended_ids) * 0.8))

    def _risk_elbow_cap(self, *, candidate_pool: list[str], tool_features_text: str) -> tuple[int | None, dict[str, Any]]:
        debug: dict[str, Any] = {
            "eligible": False,
            "applied": False,
            "reason": "",
            "cap": None,
        }

        if not self.settings.risk_elbow_cap_enabled:
            debug["reason"] = "disabled_by_config"
            return None, debug

        if len(candidate_pool) < 3:
            debug["reason"] = "candidate_pool_too_small"
            return None, debug

        hints = _extract_risk_hints(tool_features_text)
        scores: list[float] = []
        for item in candidate_pool:
            if item in hints:
                scores.append(hints[item])

        if len(scores) < 3:
            debug["reason"] = "insufficient_risk_hints"
            return None, debug

        drops = [scores[idx] - scores[idx + 1] for idx in range(len(scores) - 1)]
        max_drop = max(drops)
        max_idx = drops.index(max_drop)
        sorted_drops = sorted(drops, reverse=True)
        second_drop = sorted_drops[1] if len(sorted_drops) > 1 else 0.0

        upper = scores[max_idx]
        lower = scores[max_idx + 1]
        boundary_ratio = float("inf") if lower <= 0 else (upper / lower)
        mean_drop = sum(drops) / len(drops)

        debug.update(
            {
                "eligible": True,
                "scores": scores,
                "drops": drops,
                "max_drop": max_drop,
                "max_drop_index": max_idx,
                "second_drop": second_drop,
                "mean_drop": mean_drop,
                "boundary_ratio": boundary_ratio,
                "thresholds": {
                    "min_drop": self.settings.risk_elbow_min_drop,
                    "min_ratio": self.settings.risk_elbow_min_ratio,
                    "drop_multiple": self.settings.risk_elbow_drop_multiple,
                },
            }
        )

        if max_drop < self.settings.risk_elbow_min_drop:
            debug["reason"] = "max_drop_below_min_drop"
            return None, debug

        if boundary_ratio < self.settings.risk_elbow_min_ratio:
            debug["reason"] = "boundary_ratio_below_min_ratio"
            return None, debug

        if second_drop > 0 and (max_drop / second_drop) < self.settings.risk_elbow_drop_multiple:
            debug["reason"] = "max_drop_not_dominant_vs_second_drop"
            return None, debug

        if mean_drop > 0 and (max_drop / mean_drop) < self.settings.risk_elbow_drop_multiple:
            debug["reason"] = "max_drop_not_dominant_vs_mean_drop"
            return None, debug

        cap = max_idx + 1
        debug.update({"applied": True, "cap": cap, "reason": "elbow_detected"})
        return cap, debug

    def _apply_candidate_pruning(self, dataset_context, budget_profile: str) -> tuple[Any, int, int]:
        original_pool = list(dataset_context.candidate_pool)
        original_pool_size = len(original_pool)

        if not original_pool:
            return dataset_context, 0, 0

        top_k = (
            self.settings.candidate_top_k_low
            if budget_profile == "low"
            else self.settings.candidate_top_k_high
        )
        top_k = max(1, min(top_k, original_pool_size))

        pruned_pool = original_pool[:top_k]

        feature_lines = [line for line in dataset_context.tool_features_text.splitlines() if line.strip()]
        pruned_features_text = "\n".join(feature_lines[:top_k]) if feature_lines else dataset_context.tool_features_text
        pruned_summary = (
            f"{dataset_context.summary_text} BudgetProfile={budget_profile} CandidateTopK={top_k}."
        )

        pruned_context = replace(
            dataset_context,
            candidate_pool=pruned_pool,
            tool_features_text=pruned_features_text,
            summary_text=pruned_summary,
        )

        return pruned_context, original_pool_size, len(pruned_pool)

    def _should_use_fast_path(self, planner_result: dict, *, candidate_pool_size: int, policy: dict[str, Any]) -> bool:
        if self.settings.force_full_chain:
            return False
        if not self.settings.adaptive_chain_enabled:
            return False
        if candidate_pool_size > int(policy["fast_path_max_candidate_pool"]):
            return False

        route = str(planner_result.get("route_recommendation", "full")).strip().lower()
        confidence = _bounded_float(planner_result.get("planner_confidence_0_to_1"), default=0.5)
        required_confidence = min(
            1.0,
            self.settings.fast_path_min_confidence + float(policy["fast_path_confidence_boost"]),
        )

        return route == "fast" and confidence >= required_confidence

    def _should_run_critic(self, scorer_result: dict) -> bool:
        if self.settings.force_full_chain:
            return True
        if not self.settings.adaptive_chain_enabled:
            return True

        confidence = _bounded_float(scorer_result.get("confidence_0_to_1"), default=0.5)
        abstain = bool(scorer_result.get("abstain", False))
        contradiction_signals = scorer_result.get("contradiction_signals", [])
        has_contradictions = bool(contradiction_signals) if isinstance(contradiction_signals, list) else False

        return abstain or has_contradictions or confidence < self.settings.critic_min_confidence

    def _sanitize_ids(
        self,
        values: object,
        *,
        pool_set: set[str],
        id_validator,
    ) -> list[str]:
        if not isinstance(values, list):
            return []

        output: list[str] = []
        seen: set[str] = set()
        for item in values:
            text = str(item).strip()
            if not text:
                continue
            if pool_set and text not in pool_set:
                continue
            if not id_validator(text):
                continue
            if text in seen:
                continue
            seen.add(text)
            output.append(text)
        return output

    def _finalize_ids(
        self,
        candidate_pool: list[str],
        decider_result: dict | None,
        extractor_result: dict,
        scorer_result: dict,
        critic_result: dict,
        chain_path: str,
        policy: dict[str, Any],
        max_output_ids: int,
        id_validator,
    ) -> tuple[list[str], dict[str, Any]]:
        pool_set = set(candidate_pool)

        source_ids: dict[str, list[str]] = {
            "decider": self._sanitize_ids(
                decider_result.get("final_ids", []) if isinstance(decider_result, dict) else [],
                pool_set=pool_set,
                id_validator=id_validator,
            ),
            "critic": self._sanitize_ids(
                critic_result.get("final_ids", []) if isinstance(critic_result, dict) else [],
                pool_set=pool_set,
                id_validator=id_validator,
            ),
            "scorer": self._sanitize_ids(
                scorer_result.get("recommended_ids", []) if isinstance(scorer_result, dict) else [],
                pool_set=pool_set,
                id_validator=id_validator,
            ),
            "extractor": self._sanitize_ids(
                extractor_result.get("selected_ids", []) if isinstance(extractor_result, dict) else [],
                pool_set=pool_set,
                id_validator=id_validator,
            ),
        }

        rejected_ids = self._sanitize_ids(
            critic_result.get("rejected_ids", []) if isinstance(critic_result, dict) else [],
            pool_set=pool_set,
            id_validator=id_validator,
        )
        rejected_set = set(rejected_ids)

        if bool(policy["critic_is_veto"]) and rejected_set:
            for source_name, ids in source_ids.items():
                source_ids[source_name] = [item for item in ids if item not in rejected_set]

        votes: dict[str, set[str]] = {}
        for source_name, ids in source_ids.items():
            for item in ids:
                votes.setdefault(item, set()).add(source_name)

        decider_confidence = _bounded_float(
            decider_result.get("confidence_0_to_1") if isinstance(decider_result, dict) else None,
            default=0.5,
        )
        scorer_confidence = _bounded_float(
            scorer_result.get("confidence_0_to_1") if isinstance(scorer_result, dict) else None,
            default=0.5,
        )
        decider_abstain = bool(decider_result.get("abstain", False)) if isinstance(decider_result, dict) else False
        scorer_abstain = bool(scorer_result.get("abstain", False)) if isinstance(scorer_result, dict) else False
        contradiction_signals = (
            scorer_result.get("contradiction_signals", []) if isinstance(scorer_result, dict) else []
        )
        has_contradictions = bool(contradiction_signals) if isinstance(contradiction_signals, list) else False

        anchor_source = ""
        if (
            source_ids["scorer"]
            and not scorer_abstain
            and scorer_confidence >= float(policy["scorer_override_confidence"])
            and (
                not bool(policy["require_no_contradictions_for_scorer_anchor"])
                or not has_contradictions
            )
        ):
            anchor_source = "scorer"
        elif (
            source_ids["decider"]
            and not decider_abstain
            and decider_confidence >= float(policy["decider_anchor_confidence"])
        ):
            anchor_source = "decider"
        elif source_ids["critic"]:
            anchor_source = "critic"
        elif source_ids["extractor"]:
            anchor_source = "extractor"

        source_priority = (
            list(policy["source_priority_fast"])
            if chain_path == "fast"
            else list(policy["source_priority_full"])
        )
        if anchor_source and anchor_source in source_priority:
            source_priority.remove(anchor_source)
            source_priority.insert(0, anchor_source)

        pre_finalize_ids: list[str] = []
        pre_seen: set[str] = set()
        for source_name in source_priority:
            for item in source_ids.get(source_name, []):
                if item in pre_seen:
                    continue
                pre_seen.add(item)
                pre_finalize_ids.append(item)

        final_ids: list[str] = []
        filtered_by_policy: list[str] = []
        seen: set[str] = set()
        anchor_ids = set(source_ids.get(anchor_source, [])) if anchor_source else set()
        min_votes_for_non_anchor = int(policy["min_votes_for_non_anchor"])

        for item in pre_finalize_ids:
            if item in rejected_set and bool(policy["critic_is_veto"]):
                filtered_by_policy.append(item)
                continue

            vote_count = len(votes.get(item, set()))
            if item not in anchor_ids and vote_count < min_votes_for_non_anchor:
                filtered_by_policy.append(item)
                continue

            if item in seen:
                continue
            seen.add(item)
            final_ids.append(item)
            if len(final_ids) >= max_output_ids:
                break

        fallback_applied = False
        if not final_ids and candidate_pool:
            # Conservative fallback: use a strict subset of highest-priority candidates.
            fallback_limit = min(
                max_output_ids,
                max(1, int(len(candidate_pool) * float(policy["fallback_ratio"]))),
            )
            final_ids = candidate_pool[:fallback_limit]
            fallback_applied = True

        source_order_lookup = {name: idx for idx, name in enumerate(source_priority)}
        id_kept_by_primary_source: dict[str, str] = {}
        id_kept_supporting_sources: dict[str, list[str]] = {}
        for item in final_ids:
            item_sources = sorted(votes.get(item, set()), key=lambda name: source_order_lookup.get(name, 999))
            id_kept_by_primary_source[item] = item_sources[0] if item_sources else "fallback"
            id_kept_supporting_sources[item] = item_sources[1:] if len(item_sources) > 1 else []

        debug = {
            "policy_profile": str(policy["profile"]),
            "policy": {
                "critic_is_veto": bool(policy["critic_is_veto"]),
                "scorer_override_confidence": float(policy["scorer_override_confidence"]),
                "decider_anchor_confidence": float(policy["decider_anchor_confidence"]),
                "min_votes_for_non_anchor": int(policy["min_votes_for_non_anchor"]),
                "source_priority": source_priority,
                "fallback_ratio": float(policy["fallback_ratio"]),
            },
            "anchor_source": anchor_source,
            "confidence_snapshot": {
                "decider_confidence_0_to_1": decider_confidence,
                "scorer_confidence_0_to_1": scorer_confidence,
                "decider_abstain": decider_abstain,
                "scorer_abstain": scorer_abstain,
            },
            "contradiction_signals": contradiction_signals if isinstance(contradiction_signals, list) else [],
            "source_ids": source_ids,
            "source_votes": {item: sorted(list(names)) for item, names in votes.items()},
            "critic_rejected_ids": rejected_ids,
            "filtered_out_ids": filtered_by_policy,
            "pre_finalize_ids": pre_finalize_ids,
            "post_finalize_ids": final_ids,
            "cap_applied": len(final_ids) >= max_output_ids,
            "cap_limit": max_output_ids,
            "cap_is_hard_final": True,
            "fallback_applied": fallback_applied,
            "id_kept_by_primary_source": id_kept_by_primary_source,
            "id_kept_supporting_sources": id_kept_supporting_sources,
        }

        return final_ids, debug

    def run(
        self,
        *,
        mode: str,
        phase: str,
        dataset_key: str,
        dataset_path: str,
        output_path: str,
        max_output_ids: int | None = None,
    ) -> RunResult:
        self.budget_guard.reset()

        normalized_mode = mode.strip().lower()
        normalized_phase = phase.strip().lower()
        if normalized_phase not in {"training", "evaluation"}:
            raise ValueError("phase must be one of: training, evaluation")

        dataset_path_obj = Path(dataset_path)
        if normalized_mode == "sandbox":
            key_level = _extract_public_level(dataset_key)
            path_level = _extract_public_level(str(dataset_path_obj))
            if key_level is not None and path_level is not None and key_level != path_level:
                raise ValueError(
                    "Sandbox level mismatch: "
                    f"dataset_key={dataset_key} but dataset={dataset_path_obj}. "
                    "Use matching pairs such as public_lev_1 with public_lev_1.zip."
                )

        adapter = build_adapter(normalized_mode, self.settings.max_candidate_pool)
        dataset_context_loaded = adapter.load(dataset_path_obj, dataset_key=dataset_key)

        budget_profile = self._resolve_budget_profile(dataset_key)
        dataset_context, candidate_pool_size_before_prune, candidate_pool_size_after_prune = (
            self._apply_candidate_pruning(dataset_context_loaded, budget_profile)
        )
        decision_policy = self._decision_policy()

        self.submission_guard.ensure_can_submit(dataset_key=dataset_key, phase=normalized_phase)

        session_id = self.tracing.generate_session_id()
        effective_max_output_ids = max_output_ids or self.settings.max_output_ids

        if dataset_context.candidate_pool:
            effective_max_output_ids = min(effective_max_output_ids, len(dataset_context.candidate_pool))

        if dataset_context.candidate_pool:
            # Guardrail: never flag the entire population as suspicious.
            anti_all_flag_cap = max(
                1,
                int(len(dataset_context.candidate_pool) * self.settings.submission_max_flagged_ratio),
            )
            if len(dataset_context.candidate_pool) > 1:
                anti_all_flag_cap = min(anti_all_flag_cap, len(dataset_context.candidate_pool) - 1)
            effective_max_output_ids = min(effective_max_output_ids, anti_all_flag_cap)

        risk_elbow_debug: dict[str, Any] = {
            "eligible": False,
            "applied": False,
            "reason": "not_checked",
            "cap": None,
        }
        if normalized_mode == "sandbox" and bool(decision_policy.get("use_risk_elbow_cap", False)):
            elbow_cap, risk_elbow_debug = self._risk_elbow_cap(
                candidate_pool=dataset_context.candidate_pool,
                tool_features_text=dataset_context.tool_features_text,
            )
            if elbow_cap is not None:
                effective_max_output_ids = min(effective_max_output_ids, elbow_cap)

        planner_result: dict = {}
        decider_result: dict = {}
        extractor_result: dict = {}
        scorer_result: dict = {}
        critic_result: dict = {}
        chain_path = "full"
        critic_ran = False
        llm_stage_max_output_ids = effective_max_output_ids

        try:
            planner_result = run_planner(
                client=self.llm_client,
                session_id=session_id,
                context=dataset_context,
                model_name=self.settings.llm_model_planner or None,
            )

            if self._should_use_fast_path(
                planner_result,
                candidate_pool_size=len(dataset_context.candidate_pool),
                policy=decision_policy,
            ):
                decider_result = run_decider(
                    client=self.llm_client,
                    session_id=session_id,
                    context=dataset_context,
                    planner_result=planner_result,
                    max_output_ids=effective_max_output_ids,
                    model_name=self.settings.llm_model_decider or None,
                )

                decider_confidence = _bounded_float(decider_result.get("confidence_0_to_1"), default=0.5)
                decider_abstain = bool(decider_result.get("abstain", False))
                if decider_result.get("final_ids") and not decider_abstain and (
                    decider_confidence >= self.settings.fast_path_min_confidence
                ):
                    chain_path = "fast"
                else:
                    chain_path = "full-after-fast"

            if chain_path != "fast":
                llm_stage_max_output_ids = self._llm_selection_cap(
                    mode=normalized_mode,
                    candidate_pool_size=len(dataset_context.candidate_pool),
                    effective_max_output_ids=effective_max_output_ids,
                )

                extractor_result = run_extractor(
                    client=self.llm_client,
                    session_id=session_id,
                    context=dataset_context,
                    planner_result=planner_result,
                    max_output_ids=llm_stage_max_output_ids,
                    model_name=self.settings.llm_model_extractor or None,
                )
                scorer_result = run_scorer(
                    client=self.llm_client,
                    session_id=session_id,
                    context=dataset_context,
                    planner_result=planner_result,
                    extractor_result=extractor_result,
                    max_output_ids=llm_stage_max_output_ids,
                    model_name=self.settings.llm_model_scorer or None,
                )

                should_run_critic = self._should_run_critic(scorer_result)
                if normalized_mode == "challenge":
                    # Challenge scoring is sensitive to false positives; always run critic.
                    should_run_critic = True

                if should_run_critic:
                    critic_result = run_critic(
                        client=self.llm_client,
                        session_id=session_id,
                        context=dataset_context,
                        planner_result=planner_result,
                        extractor_result=extractor_result,
                        scorer_result=scorer_result,
                        max_output_ids=llm_stage_max_output_ids,
                        model_name=self.settings.llm_model_critic or None,
                    )
                    critic_ran = True
                    chain_path = "full-with-critic"
                else:
                    critic_result = {
                        "final_ids": scorer_result.get("recommended_ids", []),
                        "rejected_ids": [],
                        "critic_notes": "Critic skipped by confidence gate.",
                    }
                    critic_ran = False
                    if chain_path == "full":
                        chain_path = "full-no-critic"

            final_ids, finalize_debug = self._finalize_ids(
                candidate_pool=dataset_context.candidate_pool,
                decider_result=decider_result,
                extractor_result=extractor_result,
                scorer_result=scorer_result,
                critic_result=critic_result,
                chain_path=chain_path,
                policy=decision_policy,
                max_output_ids=effective_max_output_ids,
                id_validator=adapter.is_valid_id,
            )

            if normalized_mode == "challenge" and dataset_context.candidate_pool:
                source_votes = finalize_debug.get("source_votes", {})
                extractor_used_fallback = self._extractor_used_fallback(extractor_result)
                scorer_used_fallback = self._scorer_used_fallback(scorer_result)
                fallback_style_outputs = extractor_used_fallback and scorer_used_fallback
                risk_hints = _extract_risk_hints(dataset_context.tool_features_text)

                # Precision guard for challenge mode when scorer is fallback-heavy.
                if scorer_used_fallback and final_ids:
                    min_vote_count = 3
                    high_risk_cut = 2.2
                    precision_filtered: list[str] = []
                    for candidate_id in final_ids:
                        votes = source_votes.get(candidate_id, []) if isinstance(source_votes, dict) else []
                        vote_count = len(votes) if isinstance(votes, list) else 0
                        risk_hint = risk_hints.get(candidate_id, 0.0)
                        if vote_count >= min_vote_count or risk_hint >= high_risk_cut:
                            precision_filtered.append(candidate_id)

                    if precision_filtered:
                        final_ids = precision_filtered

                    finalize_debug["challenge_precision_filter"] = {
                        "enabled": True,
                        "scorer_used_fallback": True,
                        "min_vote_count": min_vote_count,
                        "high_risk_cut": high_risk_cut,
                        "before_count": len(source_votes) if isinstance(source_votes, dict) else len(final_ids),
                        "after_count": len(final_ids),
                    }
                else:
                    finalize_debug["challenge_precision_filter"] = {
                        "enabled": True,
                        "scorer_used_fallback": scorer_used_fallback,
                        "min_vote_count": 3,
                        "high_risk_cut": 2.2,
                        "before_count": len(final_ids),
                        "after_count": len(final_ids),
                    }

                backfill_ratio = float(self.settings.challenge_min_flagged_ratio)
                if fallback_style_outputs:
                    backfill_ratio = min(
                        backfill_ratio,
                        float(self.settings.challenge_fallback_min_flagged_ratio),
                    )

                min_target = int(
                    len(dataset_context.candidate_pool) * backfill_ratio
                )
                min_target = max(0, min(min_target, effective_max_output_ids))
                if min_target > 0 and len(final_ids) < min_target:
                    seen = set(final_ids)
                    # Controlled-recall backfill: only add a few medium/high-risk IDs.
                    risk_threshold = 0.6
                    max_backfill_extra = 3
                    preferred_candidates = [
                        candidate_id
                        for candidate_id in dataset_context.candidate_pool
                        if candidate_id not in seen and risk_hints.get(candidate_id, 0.0) >= risk_threshold
                    ]

                    missing_target = max(0, min_target - len(final_ids))
                    extra_budget = min(missing_target, max_backfill_extra)
                    relaxed_target = len(final_ids) + min(extra_budget, len(preferred_candidates))

                    for candidate_id in preferred_candidates:
                        if candidate_id in seen:
                            continue
                        final_ids.append(candidate_id)
                        seen.add(candidate_id)
                        if len(final_ids) >= relaxed_target:
                            break
                    finalize_debug["challenge_min_ratio_backfill"] = {
                        "enabled": True,
                        "base_ratio": float(self.settings.challenge_min_flagged_ratio),
                        "effective_ratio": backfill_ratio,
                        "fallback_style_outputs": fallback_style_outputs,
                        "risk_threshold": risk_threshold,
                        "max_backfill_extra": max_backfill_extra,
                        "target_count": min_target,
                        "target_count_relaxed": relaxed_target,
                        "quality_pool_count": len(preferred_candidates),
                        "final_count": len(final_ids),
                    }
                else:
                    finalize_debug["challenge_min_ratio_backfill"] = {
                        "enabled": True,
                        "base_ratio": float(self.settings.challenge_min_flagged_ratio),
                        "effective_ratio": backfill_ratio,
                        "fallback_style_outputs": fallback_style_outputs,
                        "risk_threshold": 0.6,
                        "max_backfill_extra": 3,
                        "target_count": min_target,
                        "target_count_relaxed": min_target,
                        "quality_pool_count": 0,
                        "final_count": len(final_ids),
                    }

            if not final_ids:
                raise RuntimeError(
                    "No final IDs were selected. Review prompts, adapter, or candidate features."
                )

            written_ids = self.submission_guard.write_ascii_output(
                ids=final_ids,
                output_path=Path(output_path),
                id_validator=adapter.is_valid_id,
            )

            firewall_report = self.submission_guard.firewall_validate(
                mode=normalized_mode,
                phase=normalized_phase,
                dataset_key=dataset_key,
                session_id=session_id,
                dataset_path=dataset_path_obj,
                output_path=Path(output_path),
                flagged_count=len(written_ids),
                population_count=len(dataset_context.candidate_pool),
                max_flagged_ratio=self.settings.submission_max_flagged_ratio,
            )

            self.submission_guard.register_submission(
                dataset_key=dataset_key,
                phase=normalized_phase,
                output_path=Path(output_path),
                session_id=session_id,
                metadata=firewall_report,
            )

            budget = self.budget_guard.snapshot()
            risk_component_summary = _extract_risk_component_summary(dataset_context.tool_features_text)
            feature_preview = [
                line for line in dataset_context.tool_features_text.splitlines()[:120] if line.strip()
            ]

            replay_payload = {
                "status": "success",
                "mode": normalized_mode,
                "phase": normalized_phase,
                "dataset_key": dataset_key,
                "session_id": session_id,
                "decision_profile": decision_policy["profile"],
                "budget_profile": budget_profile,
                "candidate_pool_size": len(dataset_context.candidate_pool),
                "candidate_pool_size_before_prune": candidate_pool_size_before_prune,
                "candidate_pool_size_after_prune": candidate_pool_size_after_prune,
                "candidate_pool_ids_after_prune": dataset_context.candidate_pool,
                "effective_max_output_ids": effective_max_output_ids,
                "llm_stage_max_output_ids": llm_stage_max_output_ids,
                "risk_elbow_cap": risk_elbow_debug,
                "chain_path": chain_path,
                "critic_ran": critic_ran,
                "planner_confidence_0_to_1": _bounded_float(
                    planner_result.get("planner_confidence_0_to_1"),
                    default=0.5,
                ),
                "decider_confidence_0_to_1": _bounded_float(
                    decider_result.get("confidence_0_to_1"),
                    default=0.5,
                )
                if isinstance(decider_result, dict)
                else None,
                "scorer_confidence_0_to_1": _bounded_float(
                    scorer_result.get("confidence_0_to_1"),
                    default=0.5,
                )
                if isinstance(scorer_result, dict)
                else None,
                "contradiction_signals": scorer_result.get("contradiction_signals", [])
                if isinstance(scorer_result, dict)
                else [],
                "pre_finalize_ids": finalize_debug["pre_finalize_ids"],
                "post_finalize_ids": finalize_debug["post_finalize_ids"],
                "cap_applied": finalize_debug["cap_applied"],
                "id_kept_by_primary_source": finalize_debug["id_kept_by_primary_source"],
                "risk_component_summary": risk_component_summary,
                "tool_features_preview": feature_preview,
                "planner_result": planner_result,
                "decider_result": decider_result,
                "extractor_result": extractor_result,
                "scorer_result": scorer_result,
                "critic_result": critic_result,
                "finalize_debug": finalize_debug,
                "final_ids": written_ids,
                "firewall_report": firewall_report,
                "budget": {
                    "input_tokens": budget.total_input_tokens,
                    "output_tokens": budget.total_output_tokens,
                    "total_tokens": budget.total_tokens,
                    "estimated_cost_usd": budget.estimated_cost_usd,
                },
            }
            replay_path = self.replay_logger.write(replay_payload)

            return RunResult(
                mode=normalized_mode,
                phase=normalized_phase,
                dataset_key=dataset_key,
                session_id=session_id,
                output_path=Path(output_path),
                final_ids=written_ids,
                total_input_tokens=budget.total_input_tokens,
                total_output_tokens=budget.total_output_tokens,
                total_tokens=budget.total_tokens,
                estimated_cost_usd=budget.estimated_cost_usd,
                replay_path=replay_path,
            )
        except Exception as error:
            error_payload = {
                "status": "error",
                "mode": normalized_mode,
                "phase": normalized_phase,
                "dataset_key": dataset_key,
                "session_id": session_id,
                "decision_profile": decision_policy["profile"],
                "budget_profile": budget_profile,
                "candidate_pool_size": len(dataset_context.candidate_pool),
                "candidate_pool_size_before_prune": candidate_pool_size_before_prune,
                "candidate_pool_size_after_prune": candidate_pool_size_after_prune,
                "effective_max_output_ids": effective_max_output_ids,
                "llm_stage_max_output_ids": llm_stage_max_output_ids,
                "risk_elbow_cap": risk_elbow_debug,
                "chain_path": chain_path,
                "planner_result": planner_result,
                "decider_result": decider_result,
                "extractor_result": extractor_result,
                "scorer_result": scorer_result,
                "critic_result": critic_result,
                "error": str(error),
            }
            self.replay_logger.write(error_payload)
            raise
        finally:
            self.tracing.flush()
