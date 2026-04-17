from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class SweepResult:
    dataset_key: str
    ratio_limit: float
    max_output_ids: int
    output_path: str
    replay_path: str
    final_count: int
    flagged_ratio: float
    scorer_confidence: float
    extractor_fallback: bool
    finalize_fallback: bool
    heuristic_score: float


def _parse_csv_floats(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("No float values parsed.")
    return values


def _parse_csv_ints(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("No int values parsed.")
    return values


def _extract_kv_lines(stdout_text: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in stdout_text.splitlines():
        text = line.strip()
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            output[key] = value
    return output


def _calc_heuristic(
    *,
    flagged_ratio: float,
    final_count: int,
    scorer_confidence: float,
    extractor_fallback: bool,
    finalize_fallback: bool,
) -> float:
    score = 100.0

    # Keep away from invalid-zone recall while avoiding over-flagging.
    if flagged_ratio < 0.18:
        score -= 80.0
    if flagged_ratio > 0.55:
        score -= 45.0

    target_ratio = 0.33
    score -= abs(flagged_ratio - target_ratio) * 120.0

    score += min(25.0, final_count / 8.0)
    score += scorer_confidence * 12.0

    if extractor_fallback:
        score -= 12.0
    if finalize_fallback:
        score -= 18.0

    return round(score, 3)


def _is_safe_candidate(result: SweepResult) -> bool:
    return (
        0.22 <= result.flagged_ratio <= 0.45
        and not result.finalize_fallback
        and result.final_count > 0
    )


def _load_replay(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_sweep(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    python_exe = Path(args.python_exe).resolve() if args.python_exe else Path(sys.executable).resolve()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (workspace / dataset_path).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (workspace / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ratios = _parse_csv_floats(args.ratios)
    max_outputs = _parse_csv_ints(args.max_outputs)

    run_stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    results: list[SweepResult] = []

    print(f"Sweep start: ratios={ratios} max_outputs={max_outputs}")
    print(f"Dataset: {dataset_path}")

    for ratio, max_out in itertools.product(ratios, max_outputs):
        ratio_tag = f"{int(round(ratio * 1000)):03d}"
        dataset_key = f"{args.dataset_key_prefix}_{run_stamp}_r{ratio_tag}_m{max_out}"
        output_path = output_dir / f"{dataset_key}.txt"

        env = os.environ.copy()
        env["SUBMISSION_MAX_FLAGGED_RATIO"] = f"{ratio:.4f}"

        min_ratio = min(0.22, max(0.15, ratio * 0.70))
        if min_ratio >= ratio:
            min_ratio = max(0.0, ratio - 0.02)
        env["CHALLENGE_MIN_FLAGGED_RATIO"] = f"{min_ratio:.4f}"

        command = [
            str(python_exe),
            "run_pipeline.py",
            "run",
            "--mode",
            "challenge",
            "--phase",
            "training",
            "--dataset-key",
            dataset_key,
            "--dataset",
            str(dataset_path),
            "--output",
            str(output_path),
            "--max-output-ids",
            str(max_out),
        ]

        print(f"\n[RUN] ratio={ratio:.3f} max_output={max_out} key={dataset_key}")
        proc = subprocess.run(
            command,
            cwd=str(workspace),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        if proc.returncode != 0:
            print("  -> failed")
            print(proc.stdout)
            print(proc.stderr)
            continue

        kv = _extract_kv_lines(proc.stdout)
        replay_value = kv.get("replay_path", "")
        if not replay_value:
            print("  -> missing replay_path, skip")
            continue

        replay_path = Path(replay_value)
        if not replay_path.is_absolute():
            replay_path = (workspace / replay_path).resolve()
        replay_data = _load_replay(replay_path)

        final_count = len(replay_data.get("final_ids", []))
        flagged_ratio = float(replay_data.get("firewall_report", {}).get("flagged_ratio", 0.0) or 0.0)
        scorer_confidence = float(replay_data.get("scorer_confidence_0_to_1", 0.0) or 0.0)
        extractor_fallback = any(
            "Fallback:" in str(v)
            for v in replay_data.get("extractor_result", {}).get("rationale", {}).values()
        )
        finalize_fallback = bool(replay_data.get("finalize_debug", {}).get("fallback_applied", False))

        heuristic = _calc_heuristic(
            flagged_ratio=flagged_ratio,
            final_count=final_count,
            scorer_confidence=scorer_confidence,
            extractor_fallback=extractor_fallback,
            finalize_fallback=finalize_fallback,
        )

        result = SweepResult(
            dataset_key=dataset_key,
            ratio_limit=ratio,
            max_output_ids=max_out,
            output_path=str(output_path),
            replay_path=str(replay_path),
            final_count=final_count,
            flagged_ratio=flagged_ratio,
            scorer_confidence=scorer_confidence,
            extractor_fallback=extractor_fallback,
            finalize_fallback=finalize_fallback,
            heuristic_score=heuristic,
        )
        results.append(result)

        print(
            "  -> final_count="
            f"{final_count} flagged_ratio={flagged_ratio:.3f} "
            f"score={heuristic:.2f} extractor_fallback={extractor_fallback}"
        )

    if not results:
        print("No successful runs produced replay outputs.")
        return 1

    ranked = sorted(results, key=lambda item: item.heuristic_score, reverse=True)
    safe_candidates = [item for item in ranked if _is_safe_candidate(item)]
    safe_pick = safe_candidates[0] if safe_candidates else ranked[0]
    aggressive_pick = sorted(results, key=lambda item: (item.final_count, item.heuristic_score), reverse=True)[0]

    summary = {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "workspace": str(workspace),
        "dataset": str(dataset_path),
        "ratios": ratios,
        "max_outputs": max_outputs,
        "safe_pick": safe_pick.__dict__,
        "aggressive_pick": aggressive_pick.__dict__,
        "ranked": [item.__dict__ for item in ranked],
    }

    summary_json = output_dir / f"lev2_sweep_summary_{run_stamp}.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Lev2 Sweep Summary",
        "",
        f"- Dataset: {dataset_path}",
        f"- Created (UTC): {summary['created_at_utc']}",
        "",
        "## Safe Pick",
        f"- dataset_key: {safe_pick.dataset_key}",
        f"- output: {safe_pick.output_path}",
        f"- replay: {safe_pick.replay_path}",
        f"- flagged_ratio: {safe_pick.flagged_ratio:.4f}",
        f"- final_count: {safe_pick.final_count}",
        f"- heuristic_score: {safe_pick.heuristic_score:.3f}",
        "",
        "## Aggressive Pick",
        f"- dataset_key: {aggressive_pick.dataset_key}",
        f"- output: {aggressive_pick.output_path}",
        f"- replay: {aggressive_pick.replay_path}",
        f"- flagged_ratio: {aggressive_pick.flagged_ratio:.4f}",
        f"- final_count: {aggressive_pick.final_count}",
        f"- heuristic_score: {aggressive_pick.heuristic_score:.3f}",
        "",
        "## Ranked Results",
        "",
        "| rank | dataset_key | ratio_limit | max_output | final_count | flagged_ratio | score | extractor_fallback | finalize_fallback |",
        "|---:|---|---:|---:|---:|---:|---:|:---:|:---:|",
    ]

    for idx, item in enumerate(ranked, 1):
        lines.append(
            f"| {idx} | {item.dataset_key} | {item.ratio_limit:.3f} | {item.max_output_ids} | "
            f"{item.final_count} | {item.flagged_ratio:.4f} | {item.heuristic_score:.3f} | "
            f"{str(item.extractor_fallback)} | {str(item.finalize_fallback)} |"
        )

    summary_md = output_dir / f"lev2_sweep_summary_{run_stamp}.md"
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nSweep completed.")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary MD:   {summary_md}")
    print(f"Safe output:  {safe_pick.output_path}")
    print(f"Aggressive output: {aggressive_pick.output_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run lev2 parameter sweep and summarize safe/aggressive picks.")
    parser.add_argument("--workspace", default=".", help="Workspace root containing run_pipeline.py")
    parser.add_argument(
        "--dataset",
        default="Brave+New+World+-+train.zip",
        help="Dataset path (absolute or relative to workspace)",
    )
    parser.add_argument("--python-exe", default="", help="Python executable to use. Defaults to current interpreter.")
    parser.add_argument("--output-dir", default="outputs/stability", help="Folder for sweep outputs and reports.")
    parser.add_argument("--dataset-key-prefix", default="lev2_sweep", help="Prefix for generated dataset keys.")
    parser.add_argument("--ratios", default="0.22,0.28,0.34,0.40", help="Comma-separated SUBMISSION_MAX_FLAGGED_RATIO values.")
    parser.add_argument("--max-outputs", default="140,180,220", help="Comma-separated --max-output-ids values.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
