from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .base import DomainAdapter
from ..models import DatasetContext


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _avg(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5


def _trend_slope(values: list[float]) -> float:
    """Simple linear regression slope (units per step)."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = _avg(values)
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _consecutive_declines(values: list[float]) -> int:
    """Max count of consecutive step-over-step decreases."""
    if len(values) < 2:
        return 0
    max_run = 0
    current_run = 0
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class SandboxAdapter(DomainAdapter):
    mode = "sandbox"
    id_label = "CitizenID"
    entity_label = "citizen"

    def _read_member_text(self, dataset_path: Path, suffix: str) -> str:
        suffix = suffix.lower()

        if dataset_path.is_file() and dataset_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(dataset_path, "r") as zip_file:
                candidates = [n for n in zip_file.namelist() if n.lower().endswith(suffix)]
                if not candidates:
                    raise FileNotFoundError(f"Cannot find file ending with '{suffix}' in {dataset_path}.")
                selected = sorted(candidates, key=len)[0]
                return zip_file.read(selected).decode("utf-8", errors="replace")

        if dataset_path.is_dir():
            matches = list(dataset_path.rglob(f"*{suffix}"))
            if not matches:
                raise FileNotFoundError(f"Cannot find file ending with '{suffix}' in {dataset_path}.")
            selected_path = sorted(matches, key=lambda p: len(str(p)))[0]
            return selected_path.read_text(encoding="utf-8")

        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    def _load_status_rows(self, dataset_path: Path) -> list[dict[str, str]]:
        text = self._read_member_text(dataset_path, "status.csv")
        reader = csv.DictReader(io.StringIO(text))
        return [dict(row) for row in reader]

    def _load_users(self, dataset_path: Path) -> dict[str, dict[str, Any]]:
        text = self._read_member_text(dataset_path, "users.json")
        data = json.loads(text)
        users: dict[str, dict[str, Any]] = {}
        for item in data:
            user_id = str(item.get("user_id", "")).strip()
            if user_id:
                users[user_id] = item
        return users

    def _load_locations(self, dataset_path: Path) -> list[dict[str, Any]]:
        text = self._read_member_text(dataset_path, "locations.json")
        data = json.loads(text)
        return [dict(item) for item in data]

    def _risk_hint(
        self,
        activity_values: list[float],
        sleep_values: list[float],
        exposure_values: list[float],
    ) -> float:
        """Score how strongly a citizen's trajectory suggests declining well-being.
        Higher score = more likely suboptimal trajectory.
        """
        score = 0.0

        # --- Activity index (lower trend = worse) ---
        act_slope = _trend_slope(activity_values)
        act_delta = (activity_values[-1] - activity_values[0]) if len(activity_values) >= 2 else 0.0
        act_avg = _avg(activity_values)
        act_min = min(activity_values) if activity_values else 0.0
        act_declines = _consecutive_declines(activity_values)

        if act_slope <= -1.5:
            score += 2.5
        elif act_slope <= -0.5:
            score += 1.5
        elif act_slope < 0:
            score += 0.5

        if act_delta <= -8:
            score += 1.0
        elif act_delta <= -4:
            score += 0.5

        if act_avg < 35:
            score += 0.5
        if act_min < 20:
            score += 0.5
        if act_declines >= 4:
            score += 1.0
        elif act_declines >= 2:
            score += 0.5

        # --- Sleep quality (lower trend = worse) ---
        slp_slope = _trend_slope(sleep_values)
        slp_delta = (sleep_values[-1] - sleep_values[0]) if len(sleep_values) >= 2 else 0.0
        slp_avg = _avg(sleep_values)
        slp_min = min(sleep_values) if sleep_values else 0.0
        slp_declines = _consecutive_declines(sleep_values)

        if slp_slope <= -1.5:
            score += 2.5
        elif slp_slope <= -0.5:
            score += 1.5
        elif slp_slope < 0:
            score += 0.5

        if slp_delta <= -8:
            score += 1.0
        elif slp_delta <= -4:
            score += 0.5

        if slp_avg < 45:
            score += 0.5
        if slp_min < 25:
            score += 0.5
        if slp_declines >= 4:
            score += 1.0
        elif slp_declines >= 2:
            score += 0.5

        # --- Environmental exposure (higher trend = worse) ---
        env_slope = _trend_slope(exposure_values)
        env_delta = (exposure_values[-1] - exposure_values[0]) if len(exposure_values) >= 2 else 0.0

        if env_slope >= 1.5:
            score += 1.5
        elif env_slope >= 0.5:
            score += 0.8

        if env_delta >= 8:
            score += 0.5
        elif env_delta >= 4:
            score += 0.3

        # --- Cross-indicator: combined declining pattern ---
        declining_count = sum(1 for s in [act_slope, slp_slope] if s < -0.3) + (1 if env_slope > 0.3 else 0)
        if declining_count >= 3:
            score += 2.0
        elif declining_count >= 2:
            score += 1.0

        return score

    def load(self, dataset_path: Path, dataset_key: str) -> DatasetContext:
        users = self._load_users(dataset_path)
        status_rows = self._load_status_rows(dataset_path)
        location_rows = self._load_locations(dataset_path)

        status_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in status_rows:
            citizen_id = str(row.get("CitizenID", "")).strip()
            if citizen_id:
                status_by_id[citizen_id].append(row)

        locations_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in location_rows:
            citizen_id = str(row.get("user_id", "")).strip()
            if citizen_id:
                locations_by_id[citizen_id].append(row)

        all_ids = sorted(set(users.keys()) | set(status_by_id.keys()))
        if not all_ids:
            raise ValueError("Sandbox adapter found no citizen IDs.")

        feature_rows: list[tuple[float, str, str]] = []
        for citizen_id in all_ids:
            rows = sorted(status_by_id.get(citizen_id, []), key=lambda r: str(r.get("Timestamp", "")))

            activity_values = [_to_float(r.get("PhysicalActivityIndex")) for r in rows]
            sleep_values = [_to_float(r.get("SleepQualityIndex")) for r in rows]
            exposure_values = [_to_float(r.get("EnvironmentalExposureLevel")) for r in rows]

            activity_avg = _avg(activity_values)
            sleep_avg = _avg(sleep_values)
            exposure_avg = _avg(exposure_values)

            activity_delta = activity_values[-1] - activity_values[0] if len(activity_values) >= 2 else 0.0
            sleep_delta = sleep_values[-1] - sleep_values[0] if len(sleep_values) >= 2 else 0.0
            exposure_delta = exposure_values[-1] - exposure_values[0] if len(exposure_values) >= 2 else 0.0

            act_slope = _trend_slope(activity_values)
            slp_slope = _trend_slope(sleep_values)
            env_slope = _trend_slope(exposure_values)
            act_declines = _consecutive_declines(activity_values)
            slp_declines = _consecutive_declines(sleep_values)
            act_min = min(activity_values) if activity_values else 0.0
            slp_min = min(sleep_values) if sleep_values else 0.0

            loc_rows = locations_by_id.get(citizen_id, [])
            distinct_cities = len({str(r.get("city", "")).strip() for r in loc_rows if str(r.get("city", "")).strip()})

            user = users.get(citizen_id, {})
            birth_year = user.get("birth_year")
            age = 0
            try:
                age = 2026 - int(birth_year)
            except (TypeError, ValueError):
                age = 0
            city = str((user.get("residence") or {}).get("city", "n/a")).strip() or "n/a"

            risk = self._risk_hint(
                activity_values=activity_values,
                sleep_values=sleep_values,
                exposure_values=exposure_values,
            )

            line = (
                f"id={citizen_id} risk_hint={risk:.2f} age={age} city={city} "
                f"events={len(rows)} loc_points={len(loc_rows)} distinct_cities={distinct_cities} "
                f"activity_avg={activity_avg:.2f} activity_delta={activity_delta:.2f} "
                f"activity_slope={act_slope:.3f} activity_min={act_min:.2f} activity_consec_declines={act_declines} "
                f"sleep_avg={sleep_avg:.2f} sleep_delta={sleep_delta:.2f} "
                f"sleep_slope={slp_slope:.3f} sleep_min={slp_min:.2f} sleep_consec_declines={slp_declines} "
                f"exposure_avg={exposure_avg:.2f} exposure_delta={exposure_delta:.2f} "
                f"exposure_slope={env_slope:.3f}"
            )
            feature_rows.append((risk, citizen_id, line))

        feature_rows.sort(key=lambda item: (-item[0], item[1]))
        shortlisted = [item[1] for item in feature_rows[: self.max_candidate_pool]]
        features_text = "\n".join(item[2] for item in feature_rows[: self.max_candidate_pool])

        summary_text = (
            f"Mode=sandbox dataset_key={dataset_key} id_label={self.id_label} entity={self.entity_label}. "
            "Goal: identify citizens with suboptimal well-being trajectories. "
            "LLM must orchestrate decisions. Deterministic features are supporting evidence only."
        )

        return DatasetContext(
            dataset_key=dataset_key,
            mode=self.mode,
            id_label=self.id_label,
            entity_label=self.entity_label,
            summary_text=summary_text,
            tool_features_text=features_text,
            candidate_pool=shortlisted,
        )

    def is_valid_id(self, value: str) -> bool:
        return bool(re.fullmatch(r"[A-Z0-9]{6,20}", value))
