---
id: P2-R-F4
title: Redraw Fig 4 (faithfulness vs plausibility plane) — exact count, proxy axis, fix clip+label collisions
epic: RV (Revision — figure)
status: todo
sprint: 10
owner:
where: local
depends_on: [P2-R-UNC]
file_scope:
  - paper/figures/fig4_attribution_vs_mechanistic.py
estimate: M
spec_ref: ../SPEC.md#e7-figures-sprint-5-local-one-figure-per-item
---

## Goal
Fix Figure 4 per figure_detail_pass "Fig. 4" + improvement_instructions P4#41. Edit ONLY the .py.
(NOTE: filename is fig4_attribution_vs_mechanistic.py, embedded as **Figure 4** in
sections/05_results_B.tex.) Apply:
- **Method count (BLOCKER):** title "All 30 methods" → the exact canonical count from S07
  ("N methods" / "N method rows incl. oracle"). No "30".
- **Plausibility-proxy axis (BLOCKER):** Y axis "PLAUSIBILITY — how convincing it LOOKS / not a
  measurement" → "plausibility proxy"; remove the "how convincing it LOOKS" decoration.
- **Clipped label (BLOCKER):** "activation patching" label clipped at the right edge (line 431
  area) → increase x-limit / right margin so it is not clipped.
- **Top-left label collisions (MAJOR):** expected gradients / tuning curves / Granger / vanilla
  saliency leader lines collide → use label repulsion (adjustText) or label only 5–7 anchor
  methods; move the full list to a side table/SUPP.
- **Legends + callout outside (MAJOR):** move the two in-plot legends + the central white
  callout box out of the data field; de-emphasize or remove the dashed interpolation curve.
- **Uncertainty:** add point uncertainty (CI whiskers) from P2-R-UNC's leaderboard_ci.csv; do not
  overinterpret exact point positions.
- **Remove source path**; fonts ≥ 7.5 pt; colorblind-safe; embedded fonts.

## Definition of Done
- [ ] regenerates: `python paper/figures/fig4_attribution_vs_mechanistic.py` → updated pdf
- [ ] no "All 30 methods" / "how convincing it LOOKS"; axis says "plausibility proxy"; activation-patching label not clipped
- [ ] ≤7 anchor labels (rest in side table); legends/callout outside data; uncertainty from leaderboard_ci.csv
- [ ] no source path drawn; fonts ≥ 7.5 pt
- [ ] nothing outside `file_scope` changed; Paper-1 gates untouched/green
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
CAPTION RULE: caption owned by P2-R-S05-resultsB. HANDOFF to S05: shorten caption, exact count,
"plausibility proxy", explain jitter/danger-zone, uncertainty note. Method count must equal the
canonical count from S07. Depends on P2-R-UNC.
