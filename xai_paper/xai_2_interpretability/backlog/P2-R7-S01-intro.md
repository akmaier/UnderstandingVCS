---
id: P2-R7-S01-intro
title: Reframe intro to the universal semantic gap + behavior-as-reference
epic: R7 (universal semantic gap)
status: todo
sprint: 16
owner:
where: local
depends_on: []
file_scope:
  - xai_paper/xai_2_interpretability/paper/sections/01_intro.tex
estimate: M
spec_ref: plan.md (central question; thesis; storyline)
---

## Goal
Reframe the introduction to the sharpened thesis (plan.md): faithfulness is **not** the dividing
line. State that we put the question in its strongest form — turning the bilinear sampler on so the
gradient methods become faithful too — and that even *universal* faithfulness yields only
"wiggle $X$, $Y$ moves" causal maps, never the meaning ("missile", "collision", "game restart").
Make the positive claim explicit: the reference for meaning is the system's **behavior**, which for
software lives in its **documentation and data** (IEEE-610.12); in this study the only semantics
came from imported T3 (OCAtari/AtariARI), never discovered by a method. Keep the faithful-vs-causal
result as a way-station, not the headline. Maier voice (STYLE.md), no run-in headings, vivid via the
measured gap, not adjectives.

## Definition of Done
- [ ] intro leads with the universal-gap thesis + names the sampler-on move + behavior-as-reference
- [ ] paper still compiles: `cd paper && latexmk -pdf` exit 0, 0 undefined cites/refs
- [ ] consistent with number_audit §1 canonical numbers; ACDC=Breakout exemplar / SAE=Pong exemplar / 1.0-vs-0.0=position regime; gap not robust off position
- [ ] nothing outside file_scope changed
- [ ] committed + pushed (rebase-before-push); primary pulled ff-only; `status: done`

## Notes / handoff
- Reference the sampler-on finding qualitatively (the experiment item P2-R7-EXP-sampler supplies the
  numbers; cite its result, do not invent figures). Soft dependency, not hard.
