---
id: P2-R-S00-abstract
title: Revise abstract — reframe, exact method count, plausibility-proxy, drop rhetorical heat
epic: RV (Revision — section)
status: done
sprint: 14
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

### R6 done (2026-06-24)
Reframed to the semantic-gap thesis (faithful attribution recovers the causal wiring, not the
meaning; faithfulness necessary but not sufficient). Canonical numbers verified against
`tools/xai_study/compare/out/{leaderboard.json,leaderboard_ci.csv,faithful_demo.json}` +
the §R exemplar records: 30 methods + oracle control = 31 rows, 257 per-game records; family
contrast 0.68 [0.59,0.78] vs 0.29 [0.20,0.38] with the honest caveat that the robust
separation is the position regime (act.patching 1.000 vs vanilla saliency 0.000); ACDC
sufficiency 0.44 **on Breakout**, SAE matched 1.0 / F=0.04 **on Pong**. Honesty clause added
(labels imported and only causally verified; plausibility a documented proxy). number_audit §2
[00] fixes all applied; banned strings ("about~31", "human plausibility", "remove the excuse")
grep-clean. Cooled "We remove the excuse" → "The VCS makes that ground truth available".
Build: `latexmk -pdf` exit 0, 36 pages, 0 undefined cites/refs. Only this file + the abstract
touched (main.tex left to the SM).
