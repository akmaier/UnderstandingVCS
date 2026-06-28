---
id: P2-R7-EXP-sampler
title: Keystone experiment — sampler ON makes gradient methods faithful, yet no semantics
epic: R7 (universal semantic gap)
status: done
sprint: 15
owner:
where: local
depends_on: []
file_scope:
  - tools/xai_study/phaseB_attribution/sampler_on/run_sampler_faithfulness.jl
  - tools/xai_study/phaseB_attribution/sampler_on/out/
  - tools/xai_study/compare/out/sampler_faithfulness.csv
  - xai_paper/xai_2_interpretability/paper/figures/fig7_sampler_faithful_no_semantics.py
  - xai_paper/xai_2_interpretability/paper/figures/fig7_sampler_faithful_no_semantics.pdf
estimate: L
spec_ref: plan.md (storyline step 4; honesty contract claim 4)
---

## Goal
Build and RUN, on **jutari**, the keystone experiment of the sharpened thesis: turning the
Paper-1 **bilinear sampler ON** makes the gradient/correlational attribution methods faithful on
the discrete position/index regime (where their naive gradient is provably zero), and yet their
**semantic recovery stays zero**, exactly like every other method. This forecloses the objection
that the danger zone is a fixable technical artifact. Re-run vanilla saliency, grad×input,
SmoothGrad, Integrated Gradients, and Expected Gradients on the 6 core games on the position
regime in two conditions — naive gradient (sampler off, the committed baseline) and bilinear
sampler on (Paper-1's index-boundary surrogate, the same construction as the SI joystick-gradient
tool `tools/xai_si_gradient/`) — and score both against the intervention oracle. Report, per
method: faithfulness_naive (≈0), faithfulness_sampler (>0, risen off the floor), and
semantic_recovery (= 0 in both — no attribution map names a game concept; the only semantic label
is the imported T3 annotation, used only to *check* localization, never produced by the method).

## Definition of Done
- [x] runs to completion: `julia --project=<primary>/jutari tools/xai_study/phaseB_attribution/sampler_on/run_sampler_faithfulness.jl --games all` (exit 0)
- [x] per-method/per-game records in `tools/xai_study/phaseB_attribution/sampler_on/out/*.json` (30 real records, committed — NOT a placeholder)
- [x] `tools/xai_study/compare/out/sampler_faithfulness.csv` with columns: method, game, regime, faithfulness_naive, faithfulness_sampler, semantic_recovery, record_path (30 rows)
- [x] **the headline holds and is checked in the script:** for ≥1 gradient method, faithfulness_sampler > faithfulness_naive on the position regime (8 (method,game) pairs rose), AND semantic_recovery == 0 for all 30 records in both conditions; self-check `@assert`s ⇒ uncaught error exits non-zero on failure
- [x] `fig7_*` rendered (faithfulness naive→sampler rises; semantics flat at 0) + render-checked (rasterized PDF inspected)
- [x] every number traces to a committed record; nothing fabricated; Paper-1 gates untouched (no emulator core modified)
- [x] committed + pushed (rebase-before-push); primary pulled ff-only; `status: done`

## Measured result (real run, jutari, tf=120 hz=30)
- NAIVE faithfulness = 0.000 for ALL 30 (method × game) records (Prop. prop:zero floor) →
  reproduces the committed `saliency_<game>_position` baseline (sal_max=0, pearson=0).
- SAMPLER ON restores faithfulness off the floor:
  * **pong** 0.791 — vanilla_saliency / gradxinput / smoothgrad / expected_gradients
    (record e.g. `tools/xai_study/phaseB_attribution/sampler_on/out/sampler_vanilla_saliency_pong.json`)
  * **qbert** 0.529 — vanilla_saliency / gradxinput / smoothgrad / integrated_gradients
    (`sampler_vanilla_saliency_qbert.json`)
  * 8 (method,game) pairs rose; the keystone holds.
- HONEST non-rises (recorded, not hidden): IG vanishes on pong/SI (zeros-baseline path
  integral lands outside the triangular-kernel support); SI's restored gradient lands on
  RAM[49], which is NOT one of the 2 oracle-active causes at that pixel ⇒ Pearson 0;
  breakout/seaquest/ms_pacman are static at this state (no moving sprite at the tracked
  cell) ⇒ the sampler has no position to restore. The keystone needs only ≥1 method to
  rise; 8 pairs do.
- semantic_recovery = 0 for ALL 30 records in BOTH conditions (an attribution map names
  no T3 concept on its own) — faithfulness is repairable, the semantic gap is not.

## Notes / handoff
- The sampler is Paper-1's bilinear index-boundary interpolation (spatial-transformer style), the
  SAME mechanism as `tools/xai_si_gradient/` (memory: real-ROM SI joystick gradient via the
  bilinear sampler; naive index gradient vanishes, sampler recovers it). Do NOT conflate with a
  generic HARD/SOFT/STE mode sweep — the point is the position-gradient surrogate specifically.
- `semantic_recovery` is operationalized as: does the method, on its own, emit/name a T3
  game-concept (missile, collision, score, restart)? Attribution maps do not → 0 for all. The
  metric exists to make "faithful yet meaningless" a measured 0, not rhetoric.
- THIS is the item DeepSeek deleted as "redundant" (commit 4002e6e4). It is the keystone; it is
  not redundant. Do not mark done without the committed records + figure.
