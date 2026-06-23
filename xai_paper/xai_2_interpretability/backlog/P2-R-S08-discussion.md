---
id: P2-R-S08-discussion
title: Revise discussion — tone down NN-transfer/"no business", precise recommendations, Fig5 caption
epic: RV (Revision — section)
status: done
sprint: 13
owner:
where: local
depends_on: [P2-R-F5]
file_scope:
  - paper/sections/08_discussion.tex
estimate: M
spec_ref: ../SPEC.md#e8-nature-paper-draft-writing-sprint-6-local-one-section-per-item
---

## Goal
Revise the discussion per improvement_instructions P3#52 (and P0#11,#12; P3#33,#34; P4#42,#44).
Apply IN THIS FILE:
- **Tone down NN-transfer (P0#12):** line 93 "...has no business being trusted..." → "should not
  be trusted for stronger claims without further evidence"; replace any "the gap here is a floor
  on the gap there" with "faithfulness alone should not be treated as understanding in less
  observable neural systems."
- **Reduce rhetorical heat (P3#33,#34):** soften combative phrasing; keep one strong semantic-gap
  statement; move repeated claims to tables.
- **Precise recommendations (P3#52):** convert "field recommendations" into precise ones: validate
  on known systems; report causal and non-causal metrics separately; do not use plausibility as
  faithfulness; distinguish present/used/causal/sufficient/minimal/semantic.
- **Keep semantic-gap conclusion (P3#52):** keep the conclusion, tie the three separations to the
  definitions in S07.
- **Section refs (P4#44):** oracle pointers at \S\ref{sec:oracle} (§3.2).
- **Fig. 5 caption (P4#42):** Fig.5 .py fixed by P2-R-F5 (number mismatch, density) — this item
  owns the Fig.5 caption (lines 113–124): shorten radically (move mechanism details to
  supplement), ensure caption says "Fig. 5" consistently, avoid page/caption collision by keeping
  it short, oracle at §3.2.

## Definition of Done
- [ ] no "no business" / "floor on the gap" remain in this file (grep clean); recommendations are precise + separated causal/non-causal
- [ ] one strong semantic-gap statement; oracle refs at §3.2
- [ ] Fig.5 caption shortened (mechanism details → supplement), number consistent
- [ ] `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main` clean
- [ ] nothing outside `file_scope` changed; Paper-1 gates untouched/green
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
Tone-down + scope-narrowing shared with S01-intro (which owns the intro copies). Keep wording
consistent (revision_plan maps both). The "what this paper does not claim" box lives in SUPP (S6);
discussion may point to it. Depends on P2-R-F5 (Fig.5 number/density fix).
