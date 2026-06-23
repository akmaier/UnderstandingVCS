---
id: P2-R-S05-resultsB
title: Revise Results Phase B — separate regimes, baselines/variance, CIs, drop "default tool", §1→§3.2, Fig4 caption
epic: RV (Revision — section)
status: todo
sprint:
owner:
where: local
depends_on: [P2-R-UNC, P2-R-F4]
file_scope:
  - paper/sections/05_results_B.tex
estimate: M
spec_ref: ../SPEC.md#e8-nature-paper-draft-writing-sprint-6-local-one-section-per-item
---

## Goal
Revise Phase-B (attribution) results per improvement_instructions P3#49 (and P1#16, P3#33,
P4#41, P4#44). Apply IN THIS FILE:
- **Separate result classes (P3#49):** content-output gradient results vs position/index gradient
  failure vs perturbation/intervention methods vs N/A methods, reported distinctly.
- **Baselines + variance (P3#49, P1#16):** for Integrated/Expected Gradients show baseline
  distributions and sensitivity; for LIME/SHAP/RISE report sampling budgets and variance (read
  from P2-R-UNC's leaderboard_ci.csv / seed_sd column); attach CIs to the family means
  (gradient family $0.294$, perturbation $>0.6$, etc.).
- **Stale §1 oracle refs (P4#44):** lines 10, 29, 144 use "\S1 intervention oracle" → change to
  \S\ref{sec:oracle} (Section 3.2). These are exactly the "Section 1 oracle" the reviewers
  flagged.
- **Rhetorical heat (P3#33):** line 127 "the field's default tool" → "a widely used baseline".
- **Fig. 4 caption (P4#41):** the Fig.4 .py is fixed by P2-R-F4 (count, plausibility-proxy axis,
  clipped label) — this item owns the Fig.4 caption (lines 27–37): shorten it, say "plausibility
  proxy", state the exact method count, explain jitter/danger-zone compactly, add uncertainty.

## Definition of Done
- [ ] no "\S1" oracle ref and no "default tool" remain in this file (grep clean)
- [ ] result classes separated; IG/EG baseline-sensitivity + LIME/SHAP/RISE variance reported with CIs from leaderboard_ci.csv
- [ ] Fig.4 caption shortened, exact count, "plausibility proxy", uncertainty noted, oracle at §3.2
- [ ] `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main` clean
- [ ] nothing outside `file_scope` changed; Paper-1 gates untouched/green
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
Depends on P2-R-UNC (variance/CIs) + P2-R-F4 (figure consistency: count + proxy axis). The exact
method count must match S07/S00/SUPP — use the canonical count set in S07. NOTE: fig4's pdf is
embedded here (Fig. 4) but fig2/fig6 are embedded in S07 and fig5 in S08 — figure-number ≠
section order, so coordinate captions via revision_plan.
