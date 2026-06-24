---
id: P2-3M-SOFT
title: Run gradient XAI methods in SOFT mode with alpha/T grid sweep
epic: 3M (Three Modes)
status: todo
sprint:
owner:
where: cluster
depends_on: [P2-3M-RUNNER]
file_scope:
  - tools/xai_study/phaseB_attribution/out/multimode_soft_*
estimate: L
spec_ref: experiment_design.md
---

## Goal
Run the multi-mode runner for SOFT (fully relaxed) mode with alpha/T grid sweep across the 6 core games for vanilla saliency, grad×input, SmoothGrad, and Integrated Gradients. Alpha grid: [2, 6, 10, 20]; T grid: [0.05, 0.1, 0.14, 0.2, 0.5]. Score each against the intervention oracle (HARD mode). Results show at which alpha/T settings gradient methods become faithful on position outputs, and where the forward pass breaks.

## Definition of Done
- [ ] all 4 gradient methods × 6 games × alpha grid × T grid run
- [ ] results written to `out/multimode_soft_*`
- [ ] alpha/T faithfulness landscape available per method per game
- [ ] committed + pushed to main; status: done

## Notes / handoff
- SOFT mode = `using_relax(on=true, alpha=X, temperature=Y)` with `set_mode!(SOFT)`
- Expected: at recommended settings (T~0.14, alpha>=6), soft gradients give non-zero signal on position outputs
- At large T/small alpha, forward pass diverges — document where
- This is the experiment that shows the three-mode claim empirically
