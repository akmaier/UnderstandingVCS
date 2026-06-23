---
id: P2-R-S00-abstract
title: Revise abstract — reframe, exact method count, plausibility-proxy, drop rhetorical heat
epic: RV (Revision — section)
status: todo
sprint:
owner:
where: local
depends_on: [P2-R-UNC, P2-R-S07-compare]
file_scope:
  - paper/sections/00_abstract.tex
estimate: S
spec_ref: ../SPEC.md#e8-nature-paper-draft-writing-sprint-6-local-one-section-per-item
---

## Goal
Rewrite the abstract per improvement_instructions P3#32 (and P0#9, P1#16, P3#33). Apply IN THIS
FILE:
- **Exact method count (P0#4, P3#32):** replace "Scoring about~31 methods" (line 10) with the
  single authoritative count agreed across the paper (S07/SUPP). No "about".
- **Plausibility proxy (P0#9):** "human plausibility" (line 13) → "plausibility proxy" (and make
  clear it is a documented method-tradition proxy, not a human-subjects measurement).
- **Rhetorical heat (P3#33):** "We remove the excuse" (line 7) → calmer wording, e.g. "We provide
  a setting where this judgement can be measured."
- **Reframe (P3#32):** add the exact substrate, the six-core-games + 64-game T1/T2 scope, that
  code/data are available for review, that semantic labels are supplied/verified not discovered,
  and the central empirical numbers **with uncertainty** (CIs from P2-R-UNC's leaderboard_ci.csv).

## Definition of Done
- [ ] no "about", no "human plausibility", no "remove the excuse" remain in this file (grep clean)
- [ ] method count matches S07/SUPP exactly; central numbers carry CIs from leaderboard_ci.csv
- [ ] abstract states scope (6 core + 64 T1/T2), proxy nature of plausibility, review availability, supplied-not-discovered labels
- [ ] `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main` clean; abstract within length budget (document_check.md)
- [ ] nothing outside `file_scope` changed; Paper-1 gates untouched/green
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
Depends on S07 (it fixes the canonical count + the leaderboard numbers) and on P2-R-UNC (CIs).
Schedule the abstract in a LATER sprint than S07 so the count is settled first.
