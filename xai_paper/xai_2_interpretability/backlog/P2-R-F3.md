---
id: P2-R-F3
title: Redraw Fig 3 (attribution vs mechanistic) — facet strips, legends outside, mark oracle-like, fonts
epic: RV (Revision — figure)
status: todo
sprint: 10
owner:
where: local
depends_on: [P2-R-UNC]
file_scope:
  - paper/figures/fig3_phaseA_battery.py
estimate: M
spec_ref: ../SPEC.md#e7-figures-sprint-5-local-one-figure-per-item
---

## Goal
Fix Figure 3 per figure_detail_pass "Fig. 3" + improvement_instructions P4#40. Edit ONLY the .py.
(NOTE: filename is fig3_phaseA_battery.py and it is embedded as **Figure 3** in
sections/04_results_A.tex.) Apply:
- **Family-label overlaps (BLOCKER):** the "INTERVENTION/PERTURBATION" and "GRADIENT FAMILY"
  overlay labels cross bars/value labels → replace with proper facet strips (Gradient /
  Intervention-perturbation / Causal-intervention / Approximate-search / Present-not-used).
- **Section ref (BLOCKER):** any "Section 1 intervention oracle" in the figure → "Section 3.2
  oracle" or just "intervention oracle".
- **Declutter:** do not print every "0.00"; use one annotation ("all gradient-position scores =
  0.00") or faint zero markers; move legends outside panels; widen right margin so "oracle
  ceiling 1.00" + bar-end labels are not clipped.
- **Mark oracle-like methods** separately (P4#40); separate content-output vs position-output
  panels if possible.
- **Uncertainty:** add error bars from P2-R-UNC's leaderboard_ci.csv.
- **Contrast/fonts:** darker/black labels with colored bars; fonts ≥ 7.5 pt; remove source line;
  colorblind-safe palette; embedded fonts.

## Definition of Done
- [ ] regenerates: `python paper/figures/fig3_phaseA_battery.py` → updated pdf
- [ ] facet strips replace overlay family labels; no "Section 1 oracle"; oracle-like methods marked
- [ ] legends outside; no clipped labels; error bars from leaderboard_ci.csv; fonts ≥ 7.5 pt
- [ ] nothing outside `file_scope` changed; Paper-1 gates untouched/green
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
CAPTION RULE: caption owned by P2-R-S04-resultsA. HANDOFF to S04: shorten caption, mark
oracle-like methods, oracle at §3.2. Depends on P2-R-UNC for error bars.
