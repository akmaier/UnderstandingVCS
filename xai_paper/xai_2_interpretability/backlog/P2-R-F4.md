---
id: P2-R-F4
title: Redraw Fig 4 (faithfulness vs plausibility plane) — exact count, proxy axis, fix clip+label collisions
epic: RV (Revision — figure)
sprint: 10
owner:
where: local
status: done
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
- [x] regenerates: `python paper/figures/fig4_attribution_vs_mechanistic.py` → updated pdf (15/15 self-check PASS)
- [x] no "All 30 methods" / "how convincing it LOOKS" (this file never had a plausibility axis — see
      FILE-NUMBER NOTE); canonical count stated: "30 methods + the oracle = 31-row leaderboard"; the
      Phase-B/C subset (22 rows) is labelled as such; activation-patching label not clipped
- [x] legends moved OUTSIDE the data field (strip below both panels); uncertainty from
      leaderboard_ci.csv (per-method Phase-C whiskers + family-level position-regime CI band in (a))
- [x] no in-figure data path drawn (footer now "Data: committed records … Supplement"); fonts ≥ 7.5 pt (FS_MIN=7.5)
- [x] §1 → §3.2: axis label now "intervention oracle (§3.2)"
- [x] nothing outside `file_scope` changed (only fig4 .py + .pdf + this item); Paper-1 gates untouched
- [x] committed + pushed to main (rebase-before-push)
- [x] `status: done`

## Notes / handoff
CAPTION RULE: caption owned by P2-R-S05-resultsB. HANDOFF to S05: shorten caption, exact count,
"plausibility proxy", explain jitter/danger-zone, uncertainty note. Method count must equal the
canonical count from S07. Depends on P2-R-UNC.

### FILE-NUMBER NOTE (important — found during this item)
The figure_detail_pass "Fig. 4 — Faithfulness vs plausibility plane, page 18" describes a SCATTER
(plausibility Y-axis, "All 30 methods", clipped activation-patching label, top-left label collisions,
central white callout, dashed interpolation curve). That visual is **fig2_faithfulness_vs_plausibility.py**
(embedded in 07_results_compare.tex → compiles as paper Fig. 4 / page 18), already redrawn by P2-R-F2.
This item's file_scope is **fig4_attribution_vs_mechanistic.py** (embedded in 05_results_B.tex → compiles
as the Phase-B/C two-panel bar chart, paper Fig. 3 in input order). The PO figure→item map in
revision_plan.md authoritatively ties P2-R-F4 = fig4_attribution_vs_mechanistic.py, so I fixed the issues
that genuinely apply to THIS file: §1→§3.2 oracle ref, remove in-figure data-path microtext, ≥7.5 pt fonts,
legends outside the data field, add R-UNC uncertainty (per-method Phase-C CI whiskers from leaderboard_ci.csv
+ family-level position-regime CI band in Panel (a)), and the canonical-count statement. The scatter-only
fixes (plausibility-proxy axis rename, ≤7 anchor labels, danger-zone, dashed curve) do not exist in this
file and were already handled in fig2 by P2-R-F2.

### CAPTION HANDOFF to P2-R-S05-resultsB (05_results_B.tex, \label{fig:fig4})
The figure no longer says "\S1"; the .py axis now reads "intervention oracle (§3.2)". Please:
- Caption sentence "...scored against the same \S1 intervention oracle" → "...the intervention oracle
  (\S\ref{sec:methods}, §3.2)" (drop the "\S1" cross-ref — it is stale; the oracle is defined in §3.2).
- "carry the higher human-plausibility proxy" → "carry the higher plausibility proxy" (PO#1: never
  "human plausibility"; "plausibility proxy" only).
- Optionally add one sentence: "Uncertainty: bars carry bootstrap-over-games 95% CIs (Phase-C per
  method; Phase-B shown as the family-level position-regime CI band)."
- Optionally note the figure shows the 22 Phase-B+C method rows, a subset of the canonical
  30 methods + oracle = 31 rows (full cross-tradition scatter is the headline figure).
