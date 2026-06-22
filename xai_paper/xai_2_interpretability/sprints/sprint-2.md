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

## Review — closed 2026-06-22, **4/4 DONE** (all built, run on jutari, self-tested, pushed)

- **E1-2 gradient oracle** (`22881c2`) — content `∂pixel/∂COLUP1 = 1.0`, IG completeness
  err 0, **agrees exactly with the intervention oracle on the content path**; naive index
  grad `∂pixel/∂ball_x = 0` (vanishes) → E1-1 is the sole truth for position/index; bilinear
  sampler restores the position grad (documented workaround). `out/oracle_grad_pong.{json,npz}`.
- **E0-4 game-set/T3 audit** (`e327c74`) — §G frozen: pilot = Space Invaders (A) / Pong
  (oracle,B,C); **core (6)** = pong, breakout, space_invaders, seaquest, ms_pacman, qbert
  (full F/S/M); breadth = rest (T1/T2-only, Supplementary). 183 candidate labels; counts
  asserted == source files. `common/game_set.{json,md}`.
- **E1-3 oracle cross-check** (`4a4e84f`, 19/19 test) — CONTENT path **Spearman ρ = 1.0,
  Pearson r = 1.0, max|pred−exact| = 0** (gradient companion validated exact); index /
  non-smooth disagreements flagged with reasons; **intervention oracle declared the A/B/C
  reference instrument**. `out/oracle_xcheck_pong.{json,npz}`.
- **E4-0 Phase-B pilot — IG vs oracle** (`0dc6692`) — **first attribution method scored
  against ground truth.** `p0_score` (content): corr 0.73 / Spearman 0.998, **precision@3 =
  1.0**, deletion AUC 0.028 (↓), insertion 0.917 (↑), IG top-3 == oracle top-3 → **faithful**.
  `ball_pixel` (position): corr 0, precision@3 0, max|attr| 0 vs real oracle signal → the
  **"plausible ≠ faithful"** result. Harness positive control corr 1.0. Fixes the
  `compute_faithfulness`/`write_faithfulness` contract E4-1..E4-13 reuse.
  `out/phaseB_attribution/pilotB_faithfulness_ig_pong_*.{json,npz}`.

**Verification:** all four `status: done` on main; board regenerated (10 done); all 6 §R
artifact pairs present on disk. Agent self-tests passed (E1-2 selftest; E1-3 19/19; E4-0
selftest + positive control). Headline E4-0 numbers are agent-reported — to be independently
re-confirmed when the Phase-B full sweep (E4-1..13) runs the same scorer at scale.

**Retro:** clean sprint — pairwise-disjoint scopes, every item ran fast on jutari (~13–20 s),
no venv/cluster contention. Scope note: E4-0 added `pilot_ig_vs_oracle.jl` (Julia mandated;
`.py` shim retained at the original path).

**New item added:** `P2-E0-2j` — jutari state/trajectory recorder (the Phase-A tuning curves
and Phase-C activation capture need per-frame snapshots; the existing E0-2 recorder is the
jaxtari version). Goes first in Sprint 3.

**Next:** Sprint 3 — `E0-2j` recorder → `E3-0` (Phase A pilot) ∥ `E5-0` (Phase C pilot).
