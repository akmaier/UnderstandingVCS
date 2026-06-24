---
id: P2-N-BILINEAR
title: Bilinear-sampler experiment — make gradient methods faithful, show gap is universal
epic: E4 (Phase B / Semantic Gap)
status: todo
sprint:
owner:
where: local
depends_on: [P2-E4-1, P2-E4-2, P2-E4-4, P2-E4-5]
file_scope:
  - tools/xai_study/phaseB_attribution/bilinear_sampler_sweep.jl
  - tools/xai_study/phaseB_attribution/out/bilinear_sampler_*
estimate: L
spec_ref: plan.md (semantic gap, item iv)
---

## Goal
Run gradient-based attribution methods (vanilla saliency, grad×input, SmoothGrad,
Integrated Gradients) with Paper 1's bilinear sampler turned on, so that on the
position-output regime they now have a non-zero gradient and become faithful (F~1.0).
Demonstrate that even with perfect faithfulness across all methods, every method still
delivers only causal/mechanical relations — "wiggle X, Y moves" — and no semantics.
This proves faithfulness is not the dividing line: the gap is universal and fundamental,
not a methods-underperformance issue.

## Definition of Done
- [ ] bilinear sampler integrated into gradient methods for position-output attribution
- [ ] run on at least 2 games (Breakout, Pong) on the position regime
- [ ] results show all methods achieve F~1.0 but none recover semantics
- [ ] comparison: naive-gradient F values vs bilinear-sampler F values
- [ ] output written to `tools/xai_study/phaseB_attribution/out/bilinear_sampler_*`
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
- The bilinear sampler is Paper 1's construction (jaderberg2015spatial). It interpolates
  the index boundary to manufacture a usable position gradient — the escape hatch from
  Prop.~zero.
- Key claim: with sampler ON, gradient methods become faithful (F~1.0 on position regime),
  yet still deliver no semantics. The gap is universal.
- Results feed into the semantic gap subsection (item iv) and the Discussion.
