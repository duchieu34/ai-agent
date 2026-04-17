from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable


class SubmissionGuardError(RuntimeError):
    pass


class SubmissionGuard:
    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file

    def _sha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            while True:
                chunk = file.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def fingerprint_input(self, dataset_path: Path) -> str:
        if dataset_path.is_file():
            return self._sha256_file(dataset_path)

        if dataset_path.is_dir():
            digest = hashlib.sha256()
            for child in sorted(path for path in dataset_path.rglob("*") if path.is_file()):
                rel = str(child.relative_to(dataset_path)).replace("\\", "/")
                digest.update(rel.encode("utf-8"))
                with child.open("rb") as file:
                    while True:
                        chunk = file.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
            return digest.hexdigest()

        raise SubmissionGuardError(f"Dataset path does not exist: {dataset_path}")

    def _load_state(self) -> dict:
        if not self.state_file.exists():
            return {"evaluation": {}}
        with self.state_file.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        if "evaluation" not in raw:
            raw["evaluation"] = {}
        return raw

    def _save_state(self, state: dict) -> None:
        if self.state_file.parent and not self.state_file.parent.exists():
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with self.state_file.open("w", encoding="utf-8") as file:
            json.dump(state, file, indent=2, sort_keys=True)

    def ensure_can_submit(self, dataset_key: str, phase: str) -> None:
        if phase != "evaluation":
            return
        state = self._load_state()
        if dataset_key in state.get("evaluation", {}):
            raise SubmissionGuardError(
                f"Evaluation submission already recorded for '{dataset_key}'."
            )

    def write_ascii_output(
        self,
        ids: list[str],
        output_path: Path,
        id_validator: Callable[[str], bool],
    ) -> list[str]:
        if not ids:
            raise SubmissionGuardError("No IDs to write.")

        unique_ids: list[str] = []
        seen: set[str] = set()

        for raw in ids:
            text = str(raw).strip()
            if not text:
                continue
            if "\n" in text or "\r" in text or "\t" in text:
                raise SubmissionGuardError(f"Invalid ID with control characters: {text!r}")
            if not id_validator(text):
                raise SubmissionGuardError(f"ID does not match adapter contract: {text!r}")
            try:
                text.encode("ascii")
            except UnicodeEncodeError as exc:
                raise SubmissionGuardError(f"Non-ASCII ID found: {text!r}") from exc
            if text in seen:
                continue
            seen.add(text)
            unique_ids.append(text)

        if not unique_ids:
            raise SubmissionGuardError("No valid IDs remained after normalization.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(unique_ids)
        with output_path.open("w", encoding="ascii", newline="\n") as file:
            file.write(body)
            file.write("\n")

        return unique_ids

    def register_submission(
        self,
        dataset_key: str,
        phase: str,
        output_path: Path,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        if phase != "evaluation":
            return

        state = self._load_state()
        digest = self._sha256_file(output_path)

        entry = {
            "sha256": digest,
            "output_path": str(output_path),
        }
        if session_id:
            entry["session_id"] = session_id
        if metadata:
            entry["metadata"] = metadata

        state.setdefault("evaluation", {})[dataset_key] = entry
        self._save_state(state)

    def firewall_validate(
        self,
        *,
        mode: str,
        phase: str,
        dataset_key: str,
        session_id: str,
        dataset_path: Path,
        output_path: Path,
        flagged_count: int,
        population_count: int,
        max_flagged_ratio: float,
    ) -> dict:
        normalized_mode = mode.strip().lower()
        normalized_phase = phase.strip().lower()

        if normalized_mode not in {"sandbox", "challenge"}:
            raise SubmissionGuardError(f"Unsupported mode for submission firewall: {mode}")
        if normalized_phase not in {"training", "evaluation"}:
            raise SubmissionGuardError(f"Unsupported phase for submission firewall: {phase}")

        key = dataset_key.strip()
        if not key:
            raise SubmissionGuardError("dataset_key cannot be empty.")

        sid = session_id.strip()
        if not sid or any(ch.isspace() for ch in sid):
            raise SubmissionGuardError("Langfuse session_id must be non-empty and contain no spaces.")

        if flagged_count <= 0:
            raise SubmissionGuardError("Submission must contain at least one flagged ID.")

        if population_count > 0:
            if flagged_count >= population_count:
                raise SubmissionGuardError(
                    "Invalid output: all entities are flagged as suspicious."
                )
            ratio = flagged_count / float(population_count)
            if ratio > max_flagged_ratio:
                raise SubmissionGuardError(
                    f"Flagged ratio too high: {ratio:.3f} > {max_flagged_ratio:.3f}."
                )
        else:
            ratio = 1.0

        if not output_path.exists() or not output_path.is_file():
            raise SubmissionGuardError(f"Output file not found: {output_path}")

        output_hash = self._sha256_file(output_path)
        input_hash = self.fingerprint_input(dataset_path)

        return {
            "mode": normalized_mode,
            "phase": normalized_phase,
            "dataset_key": key,
            "session_id": sid,
            "flagged_count": flagged_count,
            "population_count": population_count,
            "flagged_ratio": ratio,
            "output_sha256": output_hash,
            "input_sha256": input_hash,
        }

    def read_state(self) -> dict:
        return self._load_state()
