---
id: P2-R-F2
title: Redraw Fig 2 (Kording battery) — fix callout overlaps, legends outside, key-findings table, fonts
epic: RV (Revision — figure)
status: todo
sprint: 10
owner:
where: local
depends_on: [P2-R-UNC]
file_scope:
  - paper/figures/fig2_faithfulness_vs_plausibility.py
estimate: M
spec_ref: ../SPEC.md#e7-figures-sprint-5-local-one-figure-per-item
---

## Goal
Fix Figure 2 per figure_detail_pass "Fig. 2" + improvement_instructions P4#39. Edit ONLY the .py.
Apply:
- **Callout overlaps (BLOCKER):** the A3/A2/A7 right-side callout boxes collide — remove callout
  boxes; replace the right panel with a compact key-results table (row A6/A3/A2/A7; columns
  finding/number/interpretation) OR move findings to text.
- **Legends outside (MAJOR):** move the two upper-left legends out of the plotting area; enlarge
  the dashed oracle-line label.
- **Axis labels:** x-axis method names cramped/multi-line → rotate/abbreviate; one numeric label
  per bar, placed consistently.
- **Method count:** line 27/210 reference "all 31 rows (30 methods + ...)" — align the wording
  with the canonical count chosen in S07; assert uses the canonical count.
- **Uncertainty:** add error bars to bars from P2-R-UNC's leaderboard_ci.csv (this figure shows
  Phase-A faithfulness — attach CIs).
- **Remove source microtext** from the figure body; fonts ≥ 7.5–8 pt; colorblind-safe palette;
  embedded fonts.

## Definition of Done
- [ ] regenerates: `python paper/figures/fig2_faithfulness_vs_plausibility.py` → updated pdf
- [ ] no callout-box overlaps; legends outside the plot; error bars from leaderboard_ci.csv present
- [ ] no source-path microtext drawn; fonts ≥ 7.5 pt; method-count wording matches canonical count
- [ ] nothing outside `file_scope` changed; Paper-1 gates untouched/green
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
NOTE on numbering: this .py file is "fig2_*" but it is embedded as **Figure 2** in
sections/07_results_compare.tex (figure number ≠ filename-vs-section order). CAPTION RULE: caption
owned by P2-R-S07-compare. HANDOFF to S07: shorten caption, key-findings as table, "plausibility
proxy", oracle at §3.2. Depends on P2-R-UNC for error bars.
