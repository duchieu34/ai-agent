# Source Context Map: Files, Imports, and Call Flow

Use this file as a single context import to understand how the current codebase executes.

## 1. Full Runtime Call Chain

### A) Run command path

1. `run_pipeline.py`
   - imports `main` from `src/mirrorlife_agent/cli.py`
   - executes `main()`

2. `src/mirrorlife_agent/cli.py:main()`
   - `load_dotenv()`
   - `_build_parser()`
   - dispatch by command:
     - `run` -> `_run(args)`
     - `status` -> `_status()`

3. `src/mirrorlife_agent/cli.py:_run(args)`
   - `Settings.from_env()` from `config.py`
   - `MultiAgentOrchestrator(settings)` from `orchestrator.py`
   - `orchestrator.run(...)`

4. `src/mirrorlife_agent/orchestrator.py:MultiAgentOrchestrator.__init__`
   - creates `BudgetGuard`
   - creates `SubmissionGuard`
   - creates `TracingManager`
   - creates `OpenRouterClient`

5. `src/mirrorlife_agent/orchestrator.py:run(...)`
   - `build_adapter(mode, max_candidate_pool)` from `adapters/__init__.py`
   - `adapter.load(dataset_path, dataset_key)`
  - resolve budget profile and prune candidate pool top-K
   - `submission_guard.ensure_can_submit(...)`
   - `tracing.generate_session_id()`
  - adaptive chain:
    - fast path: `run_planner(...)` -> `run_decider(...)`
    - full path: `run_planner(...)` -> `run_extractor(...)` -> `run_scorer(...)` -> `run_critic(...)`
  - confidence gate may skip critic when confidence is high
   - `_finalize_ids(...)`
   - `submission_guard.write_ascii_output(...)`
  - `submission_guard.firewall_validate(...)`
  - `submission_guard.register_submission(...)`
  - `replay_logger.write(...)`
   - `budget_guard.snapshot()`
   - return `RunResult`
   - finally block always calls `tracing.flush()`

6. Agent functions call LLM via shared client
   - `run_planner` -> `OpenRouterClient.invoke`
  - `run_decider` -> `OpenRouterClient.invoke`
   - `run_extractor` -> `OpenRouterClient.invoke`
   - `run_scorer` -> `OpenRouterClient.invoke`
   - `run_critic` -> `OpenRouterClient.invoke`

7. `src/mirrorlife_agent/openrouter_client.py:OpenRouterClient.invoke`
   - wraps internal `_call` with `run_with_retry(...)` from `retry.py`
   - `_call`:
     - `tracing.build_callback_handler()`
     - model invoke (`ChatOpenAI.invoke`) with metadata `langfuse_session_id`
     - `_extract_usage(...)` to build `UsageRecord`
     - `budget_guard.consume(usage)`
     - `_normalize_text(response.content)`

### B) Status command path

1. `run_pipeline.py` -> `cli.main()`
2. `cli.main()` with command `status` -> `_status()`
3. `_status()`:
   - `Settings.from_env()`
   - `SubmissionGuard(settings.submission_state_file)`
   - `guard.read_state()` and print json

## 2. File-by-File Dependency Map

## Entry layer

- `run_pipeline.py`
  - imports: `src.mirrorlife_agent.cli.main`
  - called by: shell command

- `src/mirrorlife_agent/cli.py`
  - imports: `dotenv.load_dotenv`, `Settings`, `SubmissionGuard`
  - runtime import inside `_run`: `MultiAgentOrchestrator`
  - calls: `_build_parser`, `_run`, `_status`

## Core orchestration layer

- `src/mirrorlife_agent/orchestrator.py`
  - imports:
    - adapter builder
    - five agent functions
    - budget/submission/tracing/openrouter components
    - replay logger
    - `RunResult`
  - calls:
    - adapter loading and validation
    - adaptive fast/full chain
    - submission writer/register
    - submission firewall validation
    - replay write
    - budget snapshot
  - internal method:
    - `_finalize_ids` merges results from decider/critic/scorer/extractor and adapter pool

## Adapter layer

- `src/mirrorlife_agent/adapters/base.py`
  - defines `DomainAdapter` interface

- `src/mirrorlife_agent/adapters/__init__.py`
  - `build_adapter(mode, max_candidate_pool)`
  - routes to `SandboxAdapter` or `ChallengeAdapter`

- `src/mirrorlife_agent/adapters/sandbox.py`
  - calls internal helpers:
    - `_read_member_text`
    - `_load_status_rows`
    - `_load_users`
    - `_load_locations`
    - `_risk_hint` (uses slope, consecutive declines, min values, cross-indicator)
    - `_trend_slope` (linear regression)
    - `_consecutive_declines` (max streak of step-over-step drops)
    - `_std` (standard deviation)
  - `load(...)` returns `DatasetContext` with enriched feature lines
  - `is_valid_id(...)` enforces Citizen ID regex

- `src/mirrorlife_agent/adapters/challenge.py`
  - calls internal helpers:
    - `_iter_zip_files`
    - `_iter_dir_files`
    - `_load_tables`
    - `_id_column_score`
    - `_pick_table_and_id_column`
  - `load(...)` returns `DatasetContext`
  - `is_valid_id(...)` enforces transaction-like ID regex

## Agent layer

- `src/mirrorlife_agent/agents/__init__.py`
  - re-exports `run_planner`, `run_decider`, `run_extractor`, `run_scorer`, `run_critic`

- `src/mirrorlife_agent/agents/decider.py`
  - calls `client.invoke(...)`
  - calls `parse_json_like(...)`, `coerce_id_list(...)`
  - uses `json.dumps()` for inter-agent data serialization

- `src/mirrorlife_agent/agents/planner.py`
  - calls `client.invoke(...)`
  - calls `parse_json_like(...)`

- `src/mirrorlife_agent/agents/extractor.py`
  - calls `client.invoke(...)`
  - calls `parse_json_like(...)`, `coerce_id_list(...)`
  - uses `json.dumps()` for inter-agent data serialization

- `src/mirrorlife_agent/agents/scorer.py`
  - calls `client.invoke(...)`
  - calls `parse_json_like(...)`, `coerce_id_list(...)`
  - uses `json.dumps()` for inter-agent data serialization

- `src/mirrorlife_agent/agents/critic.py`
  - calls `client.invoke(...)`
  - calls `parse_json_like(...)`, `coerce_id_list(...)`
  - uses `json.dumps()` for inter-agent data serialization
  - receives `tool_features_text` for independent evidence cross-checking
- `src/mirrorlife_agent/openrouter_client.py`
  - calls `ChatOpenAI.invoke`
  - calls `run_with_retry`
  - calls `tracing.build_callback_handler`
  - calls `budget_guard.consume`
  - supports `model_override` for per-agent model routing
  - uses `UsageRecord`

- `src/mirrorlife_agent/tracing.py`
  - constructs `Langfuse` client (if keys exist)
  - creates callback handler
  - generates `session_id` (`team-ULID`)
  - flushes traces

- `src/mirrorlife_agent/retry.py`
  - `run_with_retry(operation, max_retries, base_delay_seconds)`

- `src/mirrorlife_agent/budget_guard.py`
  - `BudgetGuard.consume(UsageRecord)`
  - raises `BudgetExceededError`
  - `snapshot()` returns totals

- `src/mirrorlife_agent/submission_guard.py`
  - `ensure_can_submit(dataset_key, phase)`
  - `write_ascii_output(ids, output_path, id_validator)`
  - `firewall_validate(...)`
  - `fingerprint_input(dataset_path)`
  - `register_submission(dataset_key, phase, output_path)`
  - `read_state()`

- `src/mirrorlife_agent/replay_logger.py`
  - writes per-run replay json for offline debugging

- `src/mirrorlife_agent/json_utils.py`
  - `parse_json_like(text, default)` for robust parsing
  - `coerce_id_list(value)` for ID extraction normalization

- `src/mirrorlife_agent/config.py`
  - `Settings.from_env()` loads all runtime controls

- `src/mirrorlife_agent/models.py`
  - shared dataclasses: `UsageRecord`, `DatasetContext`, `RunResult`

## 3. Data Passed Between Key Functions

- Adapter `load(...)` -> `DatasetContext`
- Agent outputs:
  - planner: strategy object
  - decider: fast-path final IDs + confidence + abstain
  - extractor: selected IDs + rationale
  - scorer: ranked + recommended IDs + confidence + abstain + contradiction signals
  - critic: final IDs + rejected IDs + notes
- Orchestrator `_finalize_ids(...)` merges and validates IDs
- Submission guard writes final output txt and evaluation state hash
- Submission firewall validates ratio/session/hash before registration
- Budget guard accumulates usage from every LLM call

## 4. Failure Points You Should Expect

- Missing OpenRouter key -> `OpenRouterClient` init error
- Enforced Langfuse without keys -> `TracingManager` init error
- Adapter cannot infer schema/IDs -> adapter `load(...)` error
- Budget exceeded -> `BudgetExceededError`
- Invalid output IDs or duplicate evaluation submit -> `SubmissionGuardError`
- Non-JSON or malformed agent output -> parser fallback may reduce quality

## 5. Fast Modification Guide

- Change model/provider params: `config.py` + environment values
- Change mode-specific schema logic: `adapters/sandbox.py` or `adapters/challenge.py`
- Change reasoning policy: files under `agents/`
- Change final filtering policy: `orchestrator.py:_finalize_ids`
- Change output constraints: `submission_guard.py`
- Change retry/budget policy: `retry.py`, `budget_guard.py`, env config
