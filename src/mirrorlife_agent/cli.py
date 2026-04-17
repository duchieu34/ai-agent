from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .config import Settings
from .submission_guard import SubmissionGuard


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mirrorlife-agent",
        description="Reusable multi-agent runner for Reply AI Agent Challenge 2026.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run multi-agent pipeline and write output txt.")
    run_parser.add_argument("--mode", required=True, choices=["sandbox", "challenge"])
    run_parser.add_argument("--phase", required=True, choices=["training", "evaluation"])
    run_parser.add_argument("--dataset-key", required=True, help="Stable key like public_lev_1")
    run_parser.add_argument("--dataset", required=True, help="Path to dataset folder or zip")
    run_parser.add_argument("--output", required=True, help="Path to output txt")
    run_parser.add_argument("--max-output-ids", type=int, default=None)

    subparsers.add_parser("status", help="Print local submission guard state file.")

    return parser


def _run(args: argparse.Namespace) -> int:
    from .orchestrator import MultiAgentOrchestrator

    settings = Settings.from_env()
    orchestrator = MultiAgentOrchestrator(settings)

    result = orchestrator.run(
        mode=args.mode,
        phase=args.phase,
        dataset_key=args.dataset_key,
        dataset_path=args.dataset,
        output_path=args.output,
        max_output_ids=args.max_output_ids,
    )

    print("Run completed.")
    print(f"session_id={result.session_id}")
    print(f"mode={result.mode} phase={result.phase} dataset_key={result.dataset_key}")
    print(f"output_path={result.output_path}")
    print(f"ids_written={len(result.final_ids)}")
    if result.replay_path is not None:
        print(f"replay_path={result.replay_path}")
    print(
        "budget="
        f"input_tokens={result.total_input_tokens} "
        f"output_tokens={result.total_output_tokens} "
        f"total_tokens={result.total_tokens} "
        f"estimated_usd={result.estimated_cost_usd:.6f}"
    )

    return 0


def _status() -> int:
    settings = Settings.from_env()
    guard = SubmissionGuard(settings.submission_state_file)
    print(json.dumps(guard.read_state(), indent=2, sort_keys=True))
    return 0


def main() -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run":
        return _run(args)
    if args.command == "status":
        return _status()

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
