---
id: P2-R7-S03-methods
title: Methods — document the sampler-on protocol (bilinear surrogate for the position regime)
epic: R7 (universal semantic gap)
status: todo
sprint: 16
owner:
where: local
depends_on: [P2-R7-EXP-sampler]
file_scope:
  - xai_paper/xai_2_interpretability/paper/sections/03_methods.tex
estimate: S
spec_ref: plan.md (thesis; honesty contract claim 4)
---

## Goal
Add a short methods paragraph describing the sampler-on protocol: how the Paper-1 bilinear sampler
(index-boundary interpolation, spatial-transformer style) is enabled so the gradient attribution
methods receive a non-zero surrogate position gradient instead of the naive zero (Prop. prop:zero),
and how both conditions (naive / sampler-on) are scored against the same intervention oracle. Keep
the existing scoping that names the bilinear sampler + STE as the explicit exceptions to prop:zero.
Define the `semantic_recovery` measure used in the keystone experiment (does the method emit a
T3 game-concept on its own → 0 for attribution maps). Maier voice.

## Definition of Done
- [ ] methods paragraph describes the sampler-on protocol + semantic_recovery measure, consistent with P2-R7-EXP-sampler
- [ ] paper compiles: `latexmk -pdf` exit 0, 0 undefined
- [ ] nothing outside file_scope changed
- [ ] committed + pushed (rebase-before-push); primary pulled ff-only; `status: done`

## Notes / handoff
- HARD dep on the experiment (protocol must match what the runner actually does).
