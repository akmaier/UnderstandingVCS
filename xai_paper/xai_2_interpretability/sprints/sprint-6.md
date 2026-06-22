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

## Review
_(to be filled at the Sprint-6 barrier)_
