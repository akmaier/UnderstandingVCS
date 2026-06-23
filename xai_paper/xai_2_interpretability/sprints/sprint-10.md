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

## Review — R2 closed 2026-06-23 (6/6 figures redrawn; paper recompiles, 0 undefined, 31 pp)

**Truth table discovered (figure FILES are numbered in creation order, NOT paper order):**

| Paper Fig | file | item | commit |
|---|---|---|---|
| Fig 1 | fig1_platform_oracle | F1 | `ed0df46` |
| Fig 2 | fig3_phaseA_battery | F3 | `33f8857` (re-run; `eb71f04` did it first but missed the status flip) |
| Fig 3 | fig4_attribution_vs_mechanistic | F4 | done |
| Fig 4 (scatter, p18) | fig2_faithfulness_vs_plausibility | F2 | `7f49d9b` |
| Fig 5 (taxonomy, p30) | fig6_failure_taxonomy | F6 | `a353609` |
| Fig 6 (VCS→NN, p31) | fig5_representativeness_map | F5 | `0c73238` |

Every fig_pass block is covered exactly once. Four agents independently flagged the file↔number
scramble and each applied the block matching what their file actually draws (verified correct).

**Applied across figures:** plausibility-proxy axes (no "human plausibility"); "intervention oracle
(§3.2)" (no §1); R-UNC 95% bootstrap CI whiskers (F2/F3/F4/F6); canonical count asserted; legends
moved outside; 8 pt font floor; Okabe-Ito; embedded fonts; in-figure data-path microtext removed;
"no business" → "should not be trusted … without further evidence" (P0#12). **PO#4:** Fig 5 and
Fig 6 each emit a simplified main `.pdf` + a full `_full.pdf` for the SI (SI/main wiring at R6).

**Carry-forward to R3–R5 (recorded in `reviews/r2_caption_handoffs.md`):** the redraws stripped
in-figure prose/legends/keys/citations to hit the font floor, so each owning section item MUST
update its caption (per-metric keys, F-detail, CI wording, count, §3.2 oracle, cite the analogue
works, reword "callout"→"table"). Also a data reconciliation: `sparse_autoencoder` F differs between
leaderboard.json (0.1947) and leaderboard_ci.csv (mean 0.1404) — S07/SUPP to reconcile.

**Recompile:** `latexmk` exit 0, 0 undefined cites/refs, 31 pp. Minor: one Overfull \hbox (63 pt,
lines 143–177) + one "Float too large by 20 pt" (line 192) — for the owning section item / R6 to
absorb. **Next:** R3 supplement + P0 core (SUPP, S03-methods, S07-compare).
