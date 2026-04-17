# Agent Activity Flow

This document focuses on how the multi-agent reasoning chain operates inside the orchestrator.

## 1) Agent steps in order

1. Planner analyzes dataset summary + feature lines and suggests route (fast or full).
2. Orchestrator evaluates fast-path conditions:
   - adaptive chain enabled 
   - candidate pool small enough
   - planner route is fast with enough confidence
3. If fast path is eligible:
   - Decider produces direct final IDs with confidence/abstain signal.
   - If decider is strong enough, pipeline can finish as fast path.
   - Otherwise pipeline falls back to full path.
4. Full path sequence:
   - Extractor selects likely IDs.
   - Scorer ranks and recommends IDs, emits confidence + contradiction signals.
   - Critic runs only when needed (low confidence, abstain, or contradictions).
5. Finalizer merges IDs from decider/critic/scorer/extractor with policy:
   - anchor source priority
   - vote threshold for non-anchor IDs
   - optional critic veto
   - cap by max_output_ids
   - fallback to top candidate subset if empty
6. Submission/output stage:
   - write ASCII output
   - firewall validation
   - register submission state
   - write replay log

## 2) Visual flow

```mermaid
flowchart TD
    A[DatasetContext from adapter] --> B[Planner]
    B --> C{Fast path eligible?}

    C -->|Yes| D[Decider]
    D --> E{Decider strong enough?\nnot abstain + confidence ok + has IDs}
    E -->|Yes| F[Finalize IDs]
    E -->|No| G[Extractor]

    C -->|No| G[Extractor]
    G --> H[Scorer]
    H --> I{Run Critic?\nlow confidence OR abstain OR contradictions}
    I -->|Yes| J[Critic]
    I -->|No| K[Use scorer recommendation as critic final]

    J --> F
    K --> F

    F --> L[Policy merge: anchor + votes + veto + cap + fallback]
    L --> M[Write ASCII output]
    M --> N[Firewall validate]
    N --> O[Register submission]
    O --> P[Write replay]
    P --> Q[Return RunResult]

    B -.LLM invoke.-> X[OpenRouterClient]
    D -.LLM invoke.-> X
    G -.LLM invoke.-> X
    H -.LLM invoke.-> X
    J -.LLM invoke.-> X

    X --> X1[run_with_retry]
    X1 --> X2[ChatOpenAI.invoke]
    X2 --> X3[Extract usage]
    X3 --> X4[BudgetGuard.consume]
```

## 3) Per-agent role summary

1. Planner: choose strategy and route recommendation.
2. Decider: quick direct decision for clear cases.
3. Extractor: maximize likely true positives from pool.
4. Scorer: rank evidence and output recommended IDs.
5. Critic: challenge weak picks and return cleaner final IDs.
