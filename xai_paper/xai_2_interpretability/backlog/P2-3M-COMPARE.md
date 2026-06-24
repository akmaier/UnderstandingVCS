---
id: P2-3M-COMPARE
title: Update Results Compare with three-oracle framing
epic: 3M (Three Modes)
status: todo
sprint:
owner:
where: local
depends_on: [P2-3M-RESULTS-B]
file_scope:
  - paper/sections/07_results_compare.tex
estimate: M
spec_ref: plan.md
---

## Goal
Update `07_results_compare.tex` to reflect the three-mode / three-oracle framing. The semantic gap section should acknowledge that even with three execution modes (HARD, SOFT-STE, SOFT) and three gradient constructions, every method still delivers mechanism, not semantics. The gap is universal across modes.

## Definition of Done
- [ ] Three-mode framing in leaderboard and semantic gap sections
- [ ] SOFT-STE and soft mode results referenced
- [ ] Claim: even with three gradient constructions, no method recovers semantics
- [ ] Paper compiles cleanly with no undefined refs
- [ ] committed + pushed to main; status: done
