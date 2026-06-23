---
id: P2-R-F6
title: Redraw Fig 6 (VCS->NN map) — pink-callout overlap, "no business" wording, §1 refs, analogy labels
epic: RV (Revision — figure)
status: done
sprint: 10
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

## Resolution (R2, 2026-06-23)
Redrew `fig6_failure_taxonomy.py`. The figure had ALREADY migrated from the old VCS→NN map (with
pink callout / repeated status badges / in-cell citation names) to a clean mechanism-taxonomy tree,
so several fig_pass items were structurally moot in the prior render; the remaining ones are now
fixed and the PO decisions applied.

Per **PO #4** the script now emits TWO variants in one run:
  * `fig6_failure_taxonomy.pdf`       — SIMPLIFIED main-text version (no per-branch mechanism
    micro-notes, no bottom provenance banner; caption owns the prose).
  * `fig6_failure_taxonomy_full.pdf`  — FULL supplementary version (keeps the per-branch
    "mechanism:" notes + a one-line Supplement provenance pointer + the canonical-count note).

fig_pass items resolved:
  - Pink callout / subtitle overlap: N/A — no pink callout in this design (tree, not map).
  - "Section 1 oracle" → all in-figure text now reads "intervention oracle" / "§3.2" (no "§1").
  - In-figure citation names: none present (no Balduzzi/Jain&Wallace/etc. names in the figure);
    S07 discusses the works in caption/text, refs added by P2-R-REF.
  - "no business" wording: docstring narrative changed to "should not be trusted for stronger
    claims … without further evidence" (P0#12).
  - Repeated per-row status badges: N/A (tree design uses a single legend, not per-row badges).
  - Source-path microtext REMOVED from the figure: no leaderboard.json / faithful_demo.json /
    plan.md / exp_design paths rendered. The full variant says only "full data provenance in the
    Supplement".
  - Min font: every plotted glyph now ≥ 8.0 pt (was 6.2–7.8 pt in places).
  - Legend kept OUTSIDE the axes (figure-coords top-right, 3 cols); no collision with title/data.
  - R-UNC: added 95% bootstrap CI whiskers on every F mini-bar, read from
    `tools/xai_study/compare/out/leaderboard_ci.csv` (ci_lo/ci_hi); self-check asserts each whisker
    brackets its leaderboard.json point estimate.
  - Canonical count (PO #2): the full variant carries "30 interpretability methods + the
    intervention oracle positive control = 31 rows; only the failure-exhibiting subset is drawn".
  - Colour-blind-safe Okabe-Ito palette retained; PDF fonts embedded (Type 42).

Self-check: ALL CHECKS PASS for both variants (20 scored leaves + 3 N/A; 20 CI whiskers; both PDFs
%PDF-/>4 KB). Verified visually via pdftoppm — no overlaps/clipping, CI whiskers and chips legible.

HANDOFF to **P2-R-S07-compare** (caption owner, `07_results_compare.tex`):
  - The `\includegraphics` for Fig. 6 should point at the SIMPLIFIED main-text PDF
    (`fig6_failure_taxonomy.pdf`) — unchanged filename, no .tex edit needed unless re-wiring.
  - Add a Supplement figure entry for the FULL version `fig6_failure_taxonomy_full.pdf` (the
    keep-in-main-vs-move-to-supplement wiring of `main.tex`/supplement is a PO/SM call per
    revision_plan §PO #4; the FULL PDF is committed and ready to `\includegraphics`).
  - Caption: keep it short (< 120–160 words); the figure no longer embeds provenance or the long
    rationale, so the caption can state oracle = §3.2 intervention oracle, "F ± 95% bootstrap CI",
    and label the NN parallels as documented analogues (not proof) in the surrounding text.
