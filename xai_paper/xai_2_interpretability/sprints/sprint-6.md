# Sprint 6 — Synthesis, benchmark, figures, T3 completion (experiment-tier, auto-advance)

## Goal
Turn the complete A/B/C method matrix into the paper's headline artifacts: the cross-method
**faithfulness-vs-plausibility leaderboard**, the packaged **benchmark**, the **figures**, and
the finished **T3** ground-truth (verify-by-intervention + discovery). All local jutari /
offline aggregation. After this sprint → **PAUSE for PO** before E8 (writing) + E9 (submission).

## Planning
Deps: all method results (E3/E4/E5) done; oracle + game set done; E2-1 (T3 candidates) done.
≤3 local workers/wave; pairwise-disjoint scopes.

- **Wave 1 (parallel):**
  - `P2-E6-1` Cross-method leaderboard — aggregate every §R record
    (`phase{A,B,C}/out/*.json`) into the master faithfulness-vs-plausibility table + per-phase
    rollups. Scope `compare/leaderboard.*` + `compare/out/*`.
  - `P2-E2-2` T3 verify-by-intervention + offset-correct — perturb each candidate RAM byte, confirm
    the object moves on the bit-exact framebuffer, emit per-game verified-rate. Scope `t3/verify_*`.
  - `P2-E2-3` T3 discovery — correlation + intervention sweep to find NEW causally-grounded labels
    (unlabeled cells that drive identifiable screen regions). Scope `t3/discover_*`.
- **Wave 2 (after E6-1):**
  - `P2-E6-2` Package the reusable benchmark (tasks + oracle + metrics). Scope `compare/benchmark_*`.
  - `P2-E6-3` Faithful-method demonstration (causal near-ceiling vs popular near-chance). Scope
    `compare/demo_*`.
  - `P2-E7-1` Figure 1 (platform & oracle schematic). Scope `paper/figures/fig1_*`.
- **Wave 3 (figures, after E6-1):** `P2-E7-2..E7-6` (headline axes, Phase-A battery, B-vs-C,
  representativeness map, failure taxonomy) — one figure per item, disjoint `paper/figures/figN_*`.

Figures read the committed leaderboard/§R records (matplotlib via the jaxtari venv numpy/mpl, no
new pip) → `.pdf` + the generating `.py`.

## Review — closed 2026-06-23: synthesis + benchmark + figures + T3 DONE (the experimental program is complete)

- **E6-1 leaderboard** (`bd1317f`) — 257 §R records → 31 methods on the faithfulness-vs-plausibility
  axes; headline gap causal/intervention **0.68** vs gradient/correlational **0.29** (position
  regime 0.41 vs 0.07); danger-zone topped by expected_gradients Δ0.87, linear_probing Δ0.69.
- **E6-2 benchmark** (`1d7d909`) — packaged reusable benchmark (14 tasks + oracle interface +
  metric API + runnable example; oracle_copy control F=1.0). `compare/benchmark/`.
- **E6-3 faithful-method demo** (`3ce0687`) — activation_patching **1.000** vs vanilla saliency
  **0.000** on the position regime (gap matches the leaderboard exactly).
- **E7-1..6 figures** (`935ea72`,`8b91976`,`a8a47ef`,`2df2fcd`,`a4c88f0`,`3732551`) — all six
  publication-quality vector PDFs: platform/oracle schematic, the headline scatter +
  danger-zone, the Kording battery, B-vs-C, the VCS↔NN representativeness map (the "single
  strongest objection" rebuttal), and the failure taxonomy. Every plotted number traced to a
  committed record; self-checks pass.
- **E2-2 / E2-3 T3** (`4c65045`,`02543e7`) — 53/96 candidate labels verified causal-by-intervention
  + 30 newly discovered causally-grounded labels.

**Verification:** 13/13 Sprint-6 items `status: done`; 6/6 figure PDFs render; leaderboard
self-check 5/5. **CI:** the nightly heavy suite, broken for days, is fixed — 6/7 groups green
(autodiff + screen all pass after the OOM-split + ROM-skip + JAX-cache-clear fixes); `boot`
finishing.

**Program state:** the entire **experimental program for Paper 2 is complete** — substrate
(Paper 1), oracle, T3, Phase A/B/C method matrix (~30 methods × 6 games vs the exact oracle),
cross-method leaderboard, packaged benchmark, and all figures. **55/70 backlog items done.**

**Remaining = the paper + submission, both PO-gated:** E8 (Nature draft, 10 items) and E9
(submission prep, 4 items); E3-9 (optional Visual6502) deferrable. **PAUSED for PO before E8.**
