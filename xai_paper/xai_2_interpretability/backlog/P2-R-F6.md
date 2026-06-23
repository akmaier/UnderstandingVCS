---
id: P2-R-F6
title: Redraw Fig 6 (VCS->NN map) — pink-callout overlap, "no business" wording, §1 refs, analogy labels
epic: RV (Revision — figure)
status: todo
sprint:
owner:
where: local
depends_on: []
file_scope:
  - paper/figures/fig6_failure_taxonomy.py
estimate: M
spec_ref: ../SPEC.md#e7-figures-sprint-5-local-one-figure-per-item
---

## Goal
Fix Figure 6 per figure_detail_pass "Fig. 6" + improvement_instructions P4#43 (and P0#12).
Edit ONLY the .py. (NOTE: filename fig6_failure_taxonomy.py is embedded as **Figure 6** in
sections/07_results_compare.tex.) Apply:
- **Pink-callout overlap (BLOCKER):** the pink callout overlaps/obscures the subtitle → remove
  the pink callout (or move it above with vertical space) and delete the subtitle if the callout
  stays; do not use both.
- **"Section 1 oracle" refs (BLOCKER):** any "Section 1 oracle"-style reference in the
  figure/source text → "Section 3.2 oracle" or "intervention oracle".
- **Missing references (BLOCKER):** the figure names cited works (Balduzzi 2017, Jain & Wallace
  2019, Morcos 2018, Leavitt & Morcos 2020, Elhage 2022, Kindermans 2019, Sturmfels 2020). EITHER
  keep the names AND ensure P2-R-REF added them to the bib (so the caption can cite them) OR
  remove the named citations from the figure and let S07 discuss them in caption/text. (Pick the
  remove-from-figure option to keep the figure clean; record the choice in the S07 handoff.)
- **"no business" wording (P0#12):** line 12 "...has no business being trusted..." → "should not
  be trusted for stronger claims without further evidence."
- **Analogy labels (P0#12, P4#43):** label NN counterparts as "analogue/counterpart", NOT proof;
  replace the repeated per-row status badges with a single legend ("VCS evidence: proven in this
  benchmark / NN counterpart: documented analogue, not proven here").
- **Declutter / supplement:** reduce per-cell text to one sentence; remove source path; consider
  a 6-row table form (Failure mode / VCS evidence / NN analogue / Boundary of inference); fonts
  ≥ 7.5 pt; colorblind-safe; embedded fonts. Full map is a supplement candidate.

## Definition of Done
- [ ] regenerates: `python paper/figures/fig6_failure_taxonomy.py` → updated pdf
- [ ] no pink-callout/subtitle overlap; no "no business"; no "Section 1 oracle"; NN cells labeled "analogue" not proof
- [ ] in-figure citation names removed (discussed via caption/REF instead); single status legend; fonts ≥ 7.5 pt
- [ ] nothing outside `file_scope` changed; Paper-1 gates untouched/green
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
CAPTION RULE: caption owned by P2-R-S07-compare. HANDOFF to S07: shorten caption, label NN map as
analogy, discuss the seven works in caption/text (they are added to the bib by P2-R-REF), oracle
at §3.2. Main-vs-supplement final call is a PO decision (revision_plan §PO).
