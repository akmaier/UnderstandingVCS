---
id: P2-R-F5
title: Redraw Fig 5 (failure taxonomy) — fix "Figure 6" title mismatch, declutter to table, supplement decision
epic: RV (Revision — figure)
status: todo
sprint:
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
