---
id: P2-R-F3
title: Redraw Fig 3 (attribution vs mechanistic) — facet strips, legends outside, mark oracle-like, fonts
epic: RV (Revision — figure)
status: done
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

## What I changed (R2 **Revision Sprint 2** redraw — corrected disambiguation)
DISAMBIGUATION CORRECTION: this file compiles as **paper Figure 2** (the BAR-CHART Phase-A
neuroscience battery), embedded via `sections/04_results_A.tex`. The figure-number scrambling in
the backlog/fig_pass is real — the R1 note above mistakenly chased the fig_pass "Fig. 3"
(attribution) block. Revision Sprint 2 applies the correct block: fig_pass **"Fig. 2 — Kording
neuroscience battery"** (page 11; review lines 105–133), the BAR-CHART block. The scatter-plane
("Fig. 3"/attribution) fixes belong to `fig4_attribution_vs_mechanistic.py` (P2-R-F4), not here.

fig_pass "Fig. 2" items resolved:
- **Callout-box overlaps (BLOCKER):** the four key-finding callout boxes in panel (b) had head
  text overlapping body text and bleeding into the next box. Replaced the boxes with a **compact
  key-results table** (fig_pass recommended redraw: "Replace right panel with a compact table —
  Row: A6/A3/A2/A7; Columns: finding, number, interpretation"). Each row = colour chip + tag, bold
  finding head, the driving number (bold), 2-line interpretation, faint rules between rows. No boxes
  over data; verified pixel-level no-overlap via pdftoppm crops.
- **One numeric label per bar:** removed the F/S/M triad scatter overlay that sat ON the bars
  (extra symbols competing with the value labels). Each bar now carries **exactly one** numeric
  label, placed consistently above the upper CI whisker, in black for contrast on the coloured bars
  (fig_pass: "Keep only one numeric label per bar, placed consistently above").
- **Legend OUTSIDE the axes:** the tradition legend is now a single horizontal strip in a reserved
  band BELOW panel (a) (figure-level legend at fixed y), between the x-tick labels and the footnote
  — nothing over the bars, nothing clipping the figure edge (fig_pass: "Put the legends outside the
  plotting area"). The second (triad) legend was dropped along with the triad overlay.
- **Source-path microtext removed:** the figure body carries no data paths; footnote reads "Data:
  committed §R records (6 core games); whiskers = 95% bootstrap CI (every bar has a CI record). See
  Supplement provenance table." (fig_pass #6 / "Fig. 2": "Remove the tiny source path… cite a
  supplement table instead").
- **CI whiskers from leaderboard_ci.csv:** asymmetric [ci_lo, ci_hi] read for every Phase-A bar.
  All 8 Phase-A methods have a CI row → 8/8 bars get a whisker; none fabricated (self-check asserts
  every bar has a CI record). No "single record" label needed (no Phase-A bar lacks a CI).

PO decisions applied:
- Oracle referenced as **"intervention oracle (§3.2)"** (y-axis: "vs intervention oracle, §3.2";
  positive-control annotation + docstring) — never "Section 1".
- "plausibility proxy" wording rule noted (this figure has no plausibility axis, so nothing to
  rename here).
- No method count appears in this figure, so the "30 methods + the oracle = 31 rows" string is not
  added (would be out of place; the count is owned by P2-R-S07-compare).
- **8 pt minimum font everywhere** (bumped from the R1 7.5 pt floor — no fontsize literal < 8.0);
  Okabe-Ito colour-blind-safe palette; vector PDF; `pdf.fonttype=42` → all 6 fonts embedded
  (verified with pdffonts: emb=yes, subsetted). Every plotted number traced to a committed record;
  self-check green.

Self-check / render-check: PDF exists, `%PDF-` header, 192 KB (>4 KB); pdftoppm rasterized at
150 dpi, visually confirmed no overlaps / no clipping in both panels and the legend strip.

## Notes / handoff
CAPTION RULE: caption owned by **P2-R-S04-resultsA** (sections/04_results_A.tex). HANDOFF to S04
(this is **paper Figure 2**, the Phase-A bar-chart battery):
1. **Shorten** the caption (fig_pass: captions too long) — first sentence = the takeaway.
2. **Oracle ref → §3.2**: caption must say "intervention oracle (§3.2)" / "vs the §3.2 intervention
   oracle", never "§1" / "Section 1".
3. **Per-metric meanings moved to caption/Supplement**: the per-bar metric microcaptions (what each
   headline number measures) are NOT in the figure (font floor). Mapping the caption/Supplement
   should carry: A1 = data-flow-graph F1 vs oracle; A2 = lesion-importance ρ vs true role;
   A3 = 1 − spurious-tuning rate; A4 = coupling corr. vs true coupling; A5 = 1 − clock-explained var.;
   A6 = 1 − false-edge rate vs true data-flow; A7 = NMF matched-component fraction; A8 = minimality M.
4. **Whiskers**: state in the caption that whiskers = asymmetric 95% bootstrap CIs (leaderboard_ci.csv,
   R-UNC); all eight Phase-A bars have a CI.
5. **Panel (b) is now a table** (was callout boxes): if the caption refers to "callout boxes" /
   "cards", reword to "key-results table".
Depends on P2-R-UNC (done) for the CIs.
