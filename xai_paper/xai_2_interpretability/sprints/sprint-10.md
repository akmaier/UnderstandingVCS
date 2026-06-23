# Sprint 10 — Revision R2: figure redraws (fig_pass + PO decisions)

## Goal
Fix every figure per `reviews/review_figure_detail_pass.txt`, applying the resolved PO decisions:
plausibility-proxy axis labels, the canonical count ("30 methods + oracle = 31 rows"), CIs from
`compare/out/leaderboard_ci.csv` (R-UNC), and **full Fig 5 / Fig 6 → Supplementary** (simplified
main versions). Each item owns ONLY its `paper/figures/figN_*.py` (captions live in the section
items, fixed in R3–R5) — file_scopes pairwise disjoint.

## Planning (≤3 local workers; two waves)
- **Wave A:** P2-R-F1, P2-R-F2, P2-R-F3
  - **F1** (platform/oracle schematic): `jutari (JAX)`→**jutari (Julia)**; split clutter into clean
    columns; min 8 pt; drop micro source line; oracle vs method-under-test visually distinct.
  - **F2** (faithfulness-vs-plausibility = the headline scatter): y-axis → **"plausibility proxy"**;
    fix top-left label collisions (label only 5–7 anchors / repulsion); un-clip "activation patching";
    legends outside; **add CIs** from leaderboard_ci; exact count; remove the in-plot source path.
  - **F3** (Phase-A battery): callout-box overlaps → compact table/facets; legends outside; one
    numeric label per bar; **CI whiskers** from leaderboard_ci; drop micro source line.
- **Wave B:** P2-R-F4, P2-R-F5, P2-R-F6
  - **F4** (attribution vs mechanistic): "All 30 methods"→canonical count; family facet strips not
    overlay labels; legends outside; CIs; right-margin so labels aren't clipped.
  - **F5** (failure taxonomy): fix "Figure 6"/Fig 5 title-vs-number mismatch + caption/page collision;
    produce a **simplified 5-row main version** + a **full SI version** (per PO#4); remove the source path.
  - **F6** (VCS→NN map): remove pink-callout overlap; the Fig-6 cited works now exist in references.bib
    (R-REF) — keep them; replace "Section 1 oracle"→"§3.2 oracle" in any in-figure text; calmer wording
    ("should not be trusted for stronger claims …"); **simplified main + full SI** (PO#4).

Each item: regenerate the PDF, **render-check** (no overlaps/clipping/stale labels), self-check.
Captions (length, "§3.2 oracle", count) are fixed by the owning section items in R3–R5 — note any
caption change needed in the item's handoff. main.tex main-vs-SI wiring happens at R6 integration.

## Review
_(to be filled at the R2 barrier)_
