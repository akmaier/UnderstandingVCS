---
id: P2-R7-S08-discussion
title: Discussion — meaning is not in the mechanism; its reference is behavior (docs + data)
epic: R7 (universal semantic gap)
status: todo
sprint: 16
owner:
where: local
depends_on: [P2-R7-EXP-sampler]
file_scope:
  - xai_paper/xai_2_interpretability/paper/sections/08_discussion.tex
estimate: M
spec_ref: plan.md (honesty contract claim 4; storyline step 6)
---

## Goal
Make the discussion carry the positive thesis: the sampler-on result shows the shortfall is not a
faithfulness deficit to be engineered away (faithful methods, made universally faithful, still
recover only mechanism), so **meaning is not present in the wiring for any method to recover**. Its
reference is the system's **behavior**, which for software is fixed externally by its documentation
and data (IEEE-610.12: software = programs *and* documentation); in this study the only semantics
entered from imported T3 (OCAtari/AtariARI) and were merely verified. Frame the open problem for XAI
as: understanding requires an external behavioral reference, which no internal interpretability
method supplies — closing the gap (recovering the documentation/design) is Paper 4. Keep the
necessary-condition-screen scoping; no "no business"/"proven floor" overclaim. Maier voice.

## Definition of Done
- [ ] discussion states the behavior/documentation-as-reference thesis, anchored to the sampler-on result
- [ ] scope kept honest (necessary-condition screen, not a proven NN floor); points to SUPP S6 not-claimed box
- [ ] paper compiles: `latexmk -pdf` exit 0, 0 undefined
- [ ] no fabricated numbers; nothing outside file_scope changed
- [ ] committed + pushed (rebase-before-push); primary pulled ff-only; `status: done`

## Notes / handoff
- Soft+hard mix: cite the sampler-on outcome (hard dep on the CSV for any number); the conceptual
  argument can be written in parallel with the experiment but numbers wait for it.
