---
id: P2-3M-STE
title: Run gradient XAI methods in SOFT-STE mode across 6 core games
epic: 3M (Three Modes)
status: todo
sprint:
owner:
where: local
depends_on: [P2-3M-RUNNER]
file_scope:
  - tools/xai_study/phaseB_attribution/out/multimode_ste_*
estimate: M
spec_ref: experiment_design.md
---

## Goal
Run the multi-mode runner for SOFT-STE mode across all 6 core games (Pong, Breakout, Space Invaders, Seaquest, Ms. Pac-Man, Q*bert) for vanilla saliency, grad×input, SmoothGrad, and Integrated Gradients. Score each against the intervention oracle (HARD mode). Results show whether SOFT-STE mode changes faithfulness scores on position outputs vs the naive-gradient baseline.

## Definition of Done
- [ ] all 4 gradient methods × 6 games run in SOFT-STE mode
- [ ] results written to `out/multimode_ste_*`
- [ ] comparison available: naive-gradient F scores vs SOFT-STE F scores
- [ ] committed + pushed to main; status: done

## Notes / handoff
- SOFT-STE = SOFT mode with relaxation OFF — forward exact, gradient routes through content path
- Expected: STE makes no difference on position outputs (STE gradient is content-only, still zero on index)
- The key result is that STE does NOT fix position outputs — only soft mode can
