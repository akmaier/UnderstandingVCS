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
status: done
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
- [x] regenerates: `python paper/figures/fig3_phaseA_battery.py` → updated pdf (self-check OK)
- [x] no "§1 oracle" (y-axis now "vs §3.2 intervention oracle"); oracle positive control marked
      separately (dashed reference line + hatched GT-oracle control bar + labelled annotation)
- [x] legends outside the data field (horizontal strip above panel a); no clipped labels
      (oracle annotation left-anchored in headroom — was clipped at right edge); error bars =
      asymmetric 95% bootstrap CIs read from leaderboard_ci.csv (R-UNC); all in-figure fonts ≥ 7.5 pt
- [x] nothing outside `file_scope` changed (only fig3_phaseA_battery.py + .pdf + this item)
- [x] committed + pushed to main (rebase-before-push)
- [x] `status: done`

## What I changed (R2 redraw)
Note: filename ≠ caption-number alias is real here — `fig3_phaseA_battery.py` is the **Phase-A
Kording battery** (A1–A8 scored bars + 4 findings cards), embedded as Figure 3 in 04_results_A.tex.
The fig_pass "Fig. 3" prose (page 12) literally describes the *attribution-vs-mechanistic* figure
(INTERVENTION/PERTURBATION + GRADIENT FAMILY overlay labels, "present not used" bracket, right-panel
legend) — that layout is `fig4_attribution_vs_mechanistic.py` (P2-R-F4), NOT this file. So the
family-facet-strip / "present-not-used" / Phase-B-vs-C-split items belong to F4, not F3. For THIS
figure I applied the fig_pass items that genuinely exist in it:
- §1 → §3.2: y-axis "(vs §1 oracle)" → "(vs §3.2 intervention oracle)"; docstring updated.
- Legends OUT of the data field: the two in-axes legends (which sat over the A1/A2 bars + triad
  markers) moved to a horizontal strip above panel (a).
- Right-edge clip fixed: "oracle-as-method (positive control) = 1.0" was clipped at the right
  (ha="right" near n); now left-anchored in the headroom band, fully visible, no bar-label collision.
- Min-font: removed the 5.8 pt per-bar metric microcaption row and the 6.1 pt data-path footnote;
  footnote is now 7.5 pt "Data: committed §R records …; see Supplement provenance table." (no paths).
  Finding-card body lines bumped 7.4 → 7.5 pt. All in-figure text now ≥ 7.5 pt.
- Uncertainty (R-UNC): whiskers now read asymmetric [ci_lo, ci_hi] from leaderboard_ci.csv (was the
  symmetric faithfulness_ci95 from leaderboard.json). Bars still = leaderboard.json faithfulness;
  self-check unchanged + green.
- Okabe-Ito palette, vector PDF, embedded subset TrueType fonts retained.

## Notes / handoff
CAPTION RULE: caption owned by **P2-R-S04-resultsA** (sections/04_results_A.tex L101). HANDOFF to S04:
1. **Shorten** the caption (fig_pass: captions too long) — first sentence = takeaway.
2. **Oracle ref → §3.2**: verify the caption says "§3.2 intervention oracle", not "§1".
3. **Per-metric meanings moved here**: the per-bar metric microcaptions (what each headline number
   measures) were REMOVED from the figure for the font floor. If the caption/Supplement should carry
   them, the mapping is: A1 = data-flow-graph F1 vs oracle; A2 = lesion-importance ρ vs true role;
   A3 = 1 − spurious-tuning rate; A4 = coupling corr. vs true coupling; A5 = 1 − clock-explained var.;
   A6 = 1 − false-edge rate vs true data-flow; A7 = NMF matched-component fraction; A8 = minimality M.
4. **Whiskers**: state in the caption that whiskers = 95% bootstrap CIs (leaderboard_ci.csv, R-UNC).
Depends on P2-R-UNC (done) for the CIs.
