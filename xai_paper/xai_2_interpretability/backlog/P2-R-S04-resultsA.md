---
id: P2-R-S04-resultsA
title: Revise Results Phase A — per-game values + CIs, justify analogues, Fig3 caption fixes
epic: RV (Revision — section)
status: todo
sprint:
owner:
where: local
depends_on: [P2-R-UNC, P2-R-F3]
file_scope:
  - paper/sections/04_results_A.tex
estimate: M
spec_ref: ../SPEC.md#e8-nature-paper-draft-writing-sprint-6-local-one-section-per-item
---

## Goal
Revise Phase-A results per improvement_instructions P3#48 (and P1#16, P4#40, P4#44). Apply IN
THIS FILE:
- **Per-game values + uncertainty (P3#48, P1#16):** report per-game values (point to the
  Supplement table from SUPP), not only averages; attach CIs from P2-R-UNC's leaderboard_ci.csv
  to the family mean ($0.49\pm0.24$ → with bootstrap CI) and the per-method numbers.
- **Justify each neuroscience analogue (P3#48):** add one line per analysis A1–A8 explaining why
  it is a reasonable analogue; do not overstate methods used outside their intended use (state
  when that is the case).
- **Section ref (P4#44):** Fig.3 caption (lines 100–118) uses \S\ref{sec:methods}; ensure the
  oracle pointer resolves to §3.2 (use \S\ref{sec:oracle}).
- **Fig. 3 caption (P4#40):** apply caption-shortening; mark oracle-like-intervention methods
  separately in the caption; the Fig.3 .py is fixed by P2-R-F3 — this item owns the caption.

## Definition of Done
- [ ] per-game Phase-A values referenced (Supplement table) + family mean carries a bootstrap CI from leaderboard_ci.csv
- [ ] each A1–A8 has a one-line analogue justification; no overstatement of out-of-intended-use methods
- [ ] Fig.3 caption shortened, oracle anchored at §3.2, oracle-like methods marked
- [ ] `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main` clean
- [ ] nothing outside `file_scope` changed; Paper-1 gates untouched/green
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
Depends on P2-R-UNC (CIs) and P2-R-F3 (figure/caption consistency). Per-game tables themselves
live in SUPP (S3 coverage); this section cites them.
