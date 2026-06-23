---
id: P2-R-F5
title: Redraw Fig 5 (failure taxonomy) — fix "Figure 6" title mismatch, declutter to table, supplement decision
epic: RV (Revision — figure)
status: done
sprint: 10
owner:
where: local
depends_on: []
file_scope:
  - paper/figures/fig5_representativeness_map.py
estimate: M
spec_ref: ../SPEC.md#e7-figures-sprint-5-local-one-figure-per-item
---

## Goal
Fix Figure 5 per figure_detail_pass "Fig. 5" + improvement_instructions P4#42. Edit ONLY the .py.
(NOTE: filename fig5_representativeness_map.py is embedded as **Figure 5** in
sections/08_discussion.tex. Per the figure_detail_pass the rendered page-30 figure showed a
"Figure 6" in-figure title while the caption said Fig. 5 — the in-figure title/number must match
its caption number, which is Fig. 5.) Apply:
- **Number/title mismatch (BLOCKER):** ensure any in-figure title says "Figure 5" (or carries no
  hardcoded number) to match the Fig.5 caption. (line 2 docstring says "Figure 5"; check the
  drawn title text and the self-check print on line 581.)
- **Declutter (BLOCKER/density):** the figure is too dense for a main page — convert the
  tree to a compact table (Branch / Failure mechanism / Example methods / Faithfulness range /
  Interpretation); remove the right-side mini bar chart or reduce to a single numeric column;
  move the N/A block + provenance paragraph out of the figure body.
- **Legend/labels:** legend outside the data; readable left-axis label; fonts ≥ 7.5 pt.
- **Layout:** keep the figure short enough (or landscape) to avoid the page-number/caption
  collision flagged on page 30.
- **Remove source path**; colorblind-safe; embedded fonts.
- **Main-vs-supplement:** per the global decision (figure_detail_pass §"Decide which figures
  belong in the main paper"), the FULL taxonomy tree is a supplement candidate; produce a
  simplified main-paper version here and note the supplement-move option to S08.

## Definition of Done
- [ ] regenerates: `python paper/figures/fig5_representativeness_map.py` → updated pdf
- [ ] in-figure title/number matches caption (Fig. 5); tree simplified to a compact table form
- [ ] mini bar chart removed/reduced; provenance + N/A block out of figure body; fonts ≥ 7.5 pt
- [ ] nothing outside `file_scope` changed; Paper-1 gates untouched/green
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
CAPTION RULE: caption owned by P2-R-S08-discussion. HANDOFF to S08: shorten caption (mechanism
details → supplement), ensure caption number is Fig. 5, keep it short to avoid the page/caption
collision. The main-vs-supplement final call is a PO decision (revision_plan §PO).

## DONE (P2 R2) — what was actually built
IMPORTANT FILENAME↔CAPTION CLARIFICATION (verified against the compiled main.pdf):
`fig5_representativeness_map.py` is embedded in `08_discussion.tex` (`\label{fig:representativeness}`)
and renders, in document order (after `07_results_compare.tex`), as the figure whose caption is
**"The VCS↔neural-network failure-mode map"** — i.e. it is **Fig. 6 on page 31** in the reviewed
PDF, NOT the page-30 taxonomy. The "Figure 6"-title-vs-"Fig. 5"-caption mismatch the fig_pass
flagged on page 30 belongs to the OTHER file, `fig6_failure_taxonomy.py` (owned by P2-R-F6). So
the fig_pass block applied here was **"Fig. 6 — VCS to neural-network failure-mode map" (lines
257-294)** plus the global PO decisions, since that is what THIS .py actually draws.

Redraw (one script → two PDFs, PO#4 simplified-main + full-supplement):
- `fig5_representativeness_map.pdf` — SIMPLIFIED main-text version (landscape, 4-col table:
  Failure mode | VCS measured evidence + score badge | NN analogue | Status; one sentence/cell).
- `fig5_representativeness_map_full.pdf` — FULL Supplementary-Information version (fuller per-cell
  evidence + analogue-reference key + full provenance line).

fig_pass "Fig. 6" issues resolved (in this figure):
- Pink callout removed → no callout-over-subtitle overlap; single calm header (no competing
  subtitle+callout).
- "...has no business being trusted there..." → "...should not be trusted for stronger claims
  without further evidence." (P0#12).
- Repeated 6× green/orange per-row status badges → one status header + one footer legend
  ("VCS evidence: proven in this benchmark / NN analogue: documented, not proven here").
- In-cell citation NAMES removed from the table; the seven (now eleven, by mechanism) analogue
  works are listed in the supplement reference key and must be cited in the caption/text + bib.
- Large row-number circles shrunk; cell fonts increased; one sentence per cell (main).
- Source/data-path microtext removed from the figure body (provenance → caption / Supplement).
- "§1 oracle" / "Section 1 oracle" → "§3.2 oracle" / "intervention oracle" everywhere drawn.
- Landscape page (no caption/page-number collision); all drawn fonts ≥ 7.5 pt; legend outside
  the table; colour-blind-safe (Okabe-Ito); TrueType fonts embedded; no hard-coded figure number
  in the drawn header (caption number is authoritative).

PO decisions applied: "plausibility proxy" N/A (no plausibility axis in this figure);
"30 methods + oracle = 31 rows" asserted in self-check; uncertainty added — the four rows whose
badge maps to a `leaderboard_ci.csv` mean (A6_granger, A1_connectomics,
linear_probing_control_tasks, expected_gradients) show the R-UNC bootstrap-over-games 95% CI; the
position-regime / derived-margin rows correctly show "single record" (no CI fabricated); the
headline gap carries its committed CI95. Self-check: ALL CHECKS PASSED.

### HANDOFF to P2-R-S08-discussion (caption owner of Fig. <whatever LaTeX assigns this file>)
1. CAPTION: shorten to one takeaway + a data pointer (Supplement). Use calm wording: NN column is
   a "documented analogue, not proven by this benchmark" (no "no business"). Oracle = §3.2.
2. ANALOGUE CITATIONS: the in-figure citation NAMES were removed. The caption (or nearby text)
   should cite the analogue works — Balduzzi 2017, Sundararajan 2017, Adebayo 2018, Anand 2019,
   Jain & Wallace 2019, Morcos 2018, Leavitt & Morcos 2020, Hewitt & Liang 2019, Elhage 2022,
   Kindermans 2019, Sturmfels 2020 — all of which P2-R-REF must ensure exist in `references.bib`.
   (The supplement-version PDF embeds the full reference key as a fallback.)
3. SUPPLEMENT WIRING: per PO#4 the FULL version moves to Supplementary Information. The SM/SUPP
   integration item should `\includegraphics{fig5_representativeness_map_full.pdf}` in the
   supplement and keep `fig5_representativeness_map.pdf` (simplified) in the main text.
HANDOFF to P2-R-REF: confirm the eleven analogue works above are in `references.bib`.
