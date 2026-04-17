# Multi-Agent Implementation Direction (Current Baseline)

This document summarizes the direction already implemented for Reply AI Agent Challenge 2026.
It is written to be reusable when scaling the system.

## 1. Design Targets

- Reusable core for orchestration, tracing, retry, budget control, and submission safety.
- Python-first stack for lower packaging/debug risk on challenge day.
- Dual mode operation:
  - sandbox: output entity is Citizen ID
  - challenge: output entity is Transaction ID (adapter-driven)
- LLM remains the central reasoning/orchestration brain.
- Deterministic rules/features are support tools, not the main decision engine.
- Adaptive execution path to save token and latency on clear cases.

## 2. Current Implemented Architecture

### Entry and execution

- `run_pipeline.py` is the thin entrypoint.
- `src/mirrorlife_agent/cli.py` provides commands:
  - `run`: execute full multi-agent pipeline and write output file
  - `status`: show local evaluation one-shot state

### Core orchestration

- `src/mirrorlife_agent/orchestrator.py`
- Sequence:
  1. Build adapter by mode (`sandbox` or `challenge`)
  2. Resolve budget profile (`auto`/`low`/`high`) and prune candidate pool to top-K
  3. Enforce evaluation one-shot guard locally
  4. Generate Langfuse session id
  5. Run adaptive chain:
    - Fast path: Planner -> Decider
    - Full path: Planner -> Extractor -> Scorer -> Critic
  6. Apply confidence gate:
    - skip Critic when scorer confidence is high and no contradiction signals
  7. Finalize IDs with adapter validation and pool filtering
  8. Write ASCII output file (one ID per line, no header)
  9. Run submission firewall checks (session id, ratio, input/output hashes)
  10. Register evaluation submission state
  11. Write replay log for offline debugging
  12. Return run metrics (tokens/cost estimate)

### Adapter layer (domain switch without core rewrite)

- `src/mirrorlife_agent/adapters/sandbox.py`
  - Reads `status.csv`, `users.json`, `locations.json`
  - Builds feature hints for well-being trajectories
  - Advanced trend analysis: linear regression slope, consecutive declines, min values, cross-indicator correlation
  - Risk scoring considers: slope direction, decline streaks, absolute minimums, multi-indicator deterioration
  - Feature line includes: risk_hint, age, city, activity/sleep/exposure avg/delta/slope/min/consec_declines
  - Outputs candidate Citizen IDs + context text
- `src/mirrorlife_agent/adapters/challenge.py`
  - Reads csv/json tables from zip or directory
  - Infers transaction-like ID column
  - Builds risk hints from counts/amount-like columns
  - Outputs candidate Transaction IDs + context text
- `src/mirrorlife_agent/adapters/base.py`
  - Common adapter contract (`load`, `is_valid_id`)

### Agent layer

- `src/mirrorlife_agent/agents/planner.py`
- `src/mirrorlife_agent/agents/decider.py`
- `src/mirrorlife_agent/agents/extractor.py`
- `src/mirrorlife_agent/agents/scorer.py`
- `src/mirrorlife_agent/agents/critic.py`

All agents:
- consume dataset summary + tool features + previous agent outputs
- call LLM through shared client
- return strict JSON-like structures (with parser fallback)
- support role-based model override (`LLM_MODEL_PLANNER`, etc.)
- pass inter-agent data as `json.dumps()` (not Python dict repr) for reliable LLM parsing
- include domain-specific guidance in system prompts:
  - F1-based scoring awareness (balance precision and recall)
  - "suboptimal trajectory" = declining trend over time (negative slope)
  - key signals: slope, consecutive declines, cross-indicator patterns
- Critic agent also receives `tool_features_text` for independent evidence verification

### Infra and safety guards

- OpenRouter client: `src/mirrorlife_agent/openrouter_client.py`
- Langfuse manager: `src/mirrorlife_agent/tracing.py`
- Retry with exponential backoff: `src/mirrorlife_agent/retry.py`
- Budget guard (tokens/USD estimate): `src/mirrorlife_agent/budget_guard.py`
- Submission guard (ASCII format + one-shot eval): `src/mirrorlife_agent/submission_guard.py`
- Submission firewall checks (mode/session/hash/ratio): `src/mirrorlife_agent/submission_guard.py`
- Parsing helpers for unstable model JSON output: `src/mirrorlife_agent/json_utils.py`
- Offline replay logger: `src/mirrorlife_agent/replay_logger.py`
- Runtime models: `src/mirrorlife_agent/models.py`
- Env config loader: `src/mirrorlife_agent/config.py`

## 3. Hard Contracts Enforced

- Output writer contract:
  - ASCII text file
  - one ID per line
  - no header
- ID type by mode:
  - sandbox -> Citizen ID validation regex
  - challenge -> Transaction ID validation regex
- Evaluation local one-shot guard per `dataset_key`.
- Optional strict Langfuse enforcement via `ENFORCE_LANGFUSE=true`.
- Ratio guard to avoid submitting all entities as suspicious.
- Session ID no-space validation inside submission firewall.
- Input/output SHA256 fingerprints captured per run.

## 4. Why This Direction Scales

- Core pipeline is mode-agnostic; only adapters/prompts need change for new schema.
- Clear boundaries:
  - Adapter = data interpretation + candidate pool
  - Agents = LLM reasoning and decision chain
  - Guards = operational safety
- Challenge day pivot path is fast:
  - update challenge adapter rules and prompt packs
  - keep orchestration, tracing, retry, budget, and submission guards unchanged

## 5. Practical Scale-Up Plan

1. Add versioned prompt packs (sandbox_v1, challenge_v1, challenge_v2...).
2. Add batch runner for multiple datasets with summary report.
3. Add pre-submit validator for final zip checklist (deps, config, instructions).
4. Extend challenge adapter profiler once official schema is published.
5. Add automated run-to-run diff utility for replay logs.

## 6. Current Known Gaps

- Challenge adapter is intentionally generic until official challenge-day schema is final.
- Cost estimate depends on configured `TOKEN_COST_PER_1K_INPUT/OUTPUT`.
- Local one-shot guard protects your process, but platform submission rules still remain authoritative.
- BudgetGuard does not reset between multiple `run()` calls on the same orchestrator instance.
- DatasetContext is mutated in-place by `_apply_candidate_pruning()` (safe for single-run, not for reuse).

## 7. Files to Treat as Priority When Extending

- Pipeline behavior: `src/mirrorlife_agent/orchestrator.py`
- Domain adaptation: `src/mirrorlife_agent/adapters/`
- Agent reasoning policy: `src/mirrorlife_agent/agents/`
- Runtime safety: `src/mirrorlife_agent/submission_guard.py`, `src/mirrorlife_agent/budget_guard.py`
- LLM/tracing integration: `src/mirrorlife_agent/openrouter_client.py`, `src/mirrorlife_agent/tracing.py`
