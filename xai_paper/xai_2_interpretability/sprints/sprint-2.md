# Sprint 2 — Gradient oracle + cross-check + first attribution pilot (jutari)

## Goal
Complete the ground-truth oracle (add the gradient companion + the intervention↔gradient
cross-check) and run the **first attribution method scored against the oracle** — all on
**jutari** (the verified fast substrate), building on E1-1's `oracle_intervene.jl` +
`common/jutari_oracle.jl` + the proven `tools/xai_si_gradient` (Zygote real-ROM gradients).

## Planning
Deps satisfied: E1-1 (jutari intervention oracle) **done**; E2-1 (T3 candidates) done.
All items jutari except E0-4 (data audit, substrate-agnostic). Run as one Workflow,
≤3 parallel, two waves; file-scopes pairwise disjoint.

- **Wave A (parallel):**
  - `P2-E1-2` Gradient oracle (jutari/Zygote, content-path; document the index/position
    vanishing per experiment_design §1) — scope `ground_truth/oracle_grad.*`.
  - `P2-E0-4` Game-set + T3-coverage audit (reads `t3/out/candidates_*.json`) — scope
    `common/game_set.{md,json}`.
- **Wave B (after E1-2):**
  - `P2-E1-3` Oracle cross-check: correlate intervention Δy (E1-1) vs gradient ∂y (E1-2);
    flag non-smooth/index points — scope `ground_truth/oracle_xcheck.*`.
  - `P2-E4-0` Phase-B pilot: Integrated Gradients vs the oracle on one Pong output; report
    faithfulness (corr + deletion/insertion AUC on the true VCS) — scope
    `phaseB_attribution/pilot_ig_vs_oracle.*`.

Deferred to Sprint 3: pilots `E3-0` (Phase A) and `E5-0` (Phase C) — both need a **jutari
state-trajectory recorder** (E0-2 is currently the jaxtari version); add that first.

ENV: `julia --project=<primary>/jutari`; ROMs at the primary abs path; never touch the
emulator core; no shared-venv pip (SCRUM §7).

## Review
_(to be filled at the Sprint-2 barrier)_
