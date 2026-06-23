---
id: P2-R-S01-intro
title: Revise intro — cut 25-35%, tone down, scope NN-transfer claim, fix oracle/section refs
epic: RV (Revision — section)
status: done
sprint: 13
owner:
where: local
depends_on: []
file_scope:
  - paper/sections/01_intro.tex
estimate: M
spec_ref: ../SPEC.md#e8-nature-paper-draft-writing-sprint-6-local-one-section-per-item
---

## Goal
Revise the introduction per improvement_instructions P3#45 (and P0#12, P3#33, P3#37, P4#44).
Apply IN THIS FILE:
- **Cut 25–35% (P3#45):** state problem, substrate, what is measured, three results, limits;
  keep the semantic-gap framing but do NOT repeat all examples before they are defined.
- **Tone down NN-transfer (P0#12):** line 80 "...has no business being trusted on a network..."
  → narrower wording: "Failure on this fully observable known system is strong negative evidence
  for claims that the same method reliably recovers causal mechanisms in harder unknown systems."
- **Reduce rhetorical heat (P3#33):** soften any "remove the excuse"/"field's default tool"/
  combative phrasing present here.
- **Oracle/section refs (P4#44):** the oracle is defined in §3.2 — ensure any oracle reference in
  the intro points to \S\ref{sec:oracle} (Section 3.2), not "Section 1"/"Section 4". (line 65
  has a `Section~` ref — verify its target.)
- **Companion-paper roadmap (P3#37):** if the intro leans on Papers 4/5, add/keep only a brief
  roadmap and cite Paper 1 as arXiv:2606.22447 (key `maier2025vcs`); do not lean on unpublished
  papers in the main argument.

## Definition of Done
- [ ] grep clean in this file: no "no business", no stale "Section 1"/"Section 4" oracle ref
- [ ] intro length reduced ≥25% vs current (report before/after word count in the commit)
- [ ] NN-transfer claim uses the narrowed wording; tone softened
- [ ] `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main` clean, refs resolve
- [ ] nothing outside `file_scope` changed; Paper-1 gates untouched/green
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
Tone-down + scope-narrowing is shared with S08-discussion (which owns the discussion-side
copies of "no business"/"floor"/NN-transfer). Keep the two consistent (revision_plan maps
both). The "what this paper does not claim" box lives in SUPP (S6) — intro may point to it.
