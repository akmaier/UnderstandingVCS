# P2 revision plan — peer-review response (Sprint-0 for the revision)

Built from the three reviewer docs in `reviews/`:
- `review_final.txt` — the summary verdict (major revision before submission).
- `review_improvement_instructions.txt` — the exhaustive **P0–P7** list (point numbers cited
  below as e.g. `improvement P0#5`).
- `review_figure_detail_pass.txt` — per-figure overlap/typesetting notes (cited as `fig_pass`).

The backlog is organized by **file ownership** so every item's `file_scope` is pairwise
disjoint (SCRUM §0). At most one item owns any file. The cross-cutting string fixes are applied
by whichever section item owns each occurrence (the exact line numbers are recorded in each
item).

## PO DECISIONS — RESOLVED 2026-06-23 (agents MUST follow these)
1. **Plausibility:** use "plausibility proxy" everywhere (abstract, axes, text) + a rubric + a
   sensitivity analysis. **No human-subjects study.**
2. **Canonical method count:** "**30 interpretability methods + the oracle positive control = 31
   rows**." `P2-R-S07-compare` sets it; every other section/figure/supplement follows verbatim.
3. **Artifact & blinding** (verified against NMI policy 2026-06):
   - NMI peer review is **single-blind by default** (double-anonymized is opt-in, NOT required) →
     **do not anonymize**; a **public GitHub link works for review** (`https://github.com/akmaier/UnderstandingVCS`).
   - GitHub alone is **insufficient for the published Code Availability statement** — Nature
     requires a DOI-minting archive → archive a **Zenodo DOI snapshot** and cite it.
   - **Code Ocean** (still partnered with Springer Nature, 2025) is the recommended optional route
     for *code peer review*: a Compute Capsule that **regenerates the leaderboard + figures from the
     committed §R records (no ROMs needed)**, private during review, published on acceptance.
   - `P2-R-REPRO` + `P2-R-S09-endmatter` wording: code is "available for review at
     github.com/akmaier/UnderstandingVCS; a versioned snapshot will be archived on Zenodo (DOI) and
     an optional Code Ocean capsule reproduces the leaderboard and figures." Drop the "anonymized
     link" placeholder.
4. **Figures:** move the FULL Fig 5 (taxonomy tree) and Fig 6 (VCS→NN map) to **Supplementary
   Information**; keep simplified main-text versions (consistent with Nature SI policy).

## CANONICAL NUMBERS + audit fix-list → `reviews/number_audit.md` (2026-06-24; agents MUST apply §2)
The number-provenance + consistency audit (119 findings; **no fatal error**; `prop:zero` SOUND;
F/S/M consistent except the M4 F-formula) is authoritative for every quantitative claim. §1 is the
canonical-numbers table; §2 is the owner-tagged fix-list. R4/R5/R6 section items MUST apply their
tagged §2 fixes; the R3-done files (03_methods, 07_results_compare, supplement) are patched in a
separate R3-fix pass. Recurring rules: ACDC `F=1.0/S=0.44` is the **Breakout exemplar** (always
"on Breakout"); ACDC family mean is F=0.45/S=0.32. SAE `F=0.04` is the **Pong exemplar** ("on
Pong"); SAE family is F=0.19. The 1.000-vs-0.000 contrast is **position regime only**. The
all-regimes gap 0.387 is **not robust** (collapses off the position regime).

## Decision: figure-caption ownership (one rule, stated)
Figure captions live inside the section `.tex` that embeds the figure. To keep file_scopes
disjoint, **the section item owns the caption; the figure item (`P2-R-F*`) edits ONLY the `.py`**
and records the caption-shortening request as a Note for the owning section item. (The reverse —
figure item owning caption — is impossible because captions live in shared section files.)

### Figure → section embedding map (figure number ≠ filename ≠ section order)
| Figure (caption #) | .py file (figure item) | embedded in section (caption owner) |
|---|---|---|
| Fig. 1 | `fig1_platform_oracle.py` (P2-R-F1) | `03_methods.tex` (P2-R-S03-methods) |
| Fig. 2 | `fig2_faithfulness_vs_plausibility.py` (P2-R-F2) | `07_results_compare.tex` (P2-R-S07-compare) |
| Fig. 3 | `fig3_phaseA_battery.py` (P2-R-F3) | `04_results_A.tex` (P2-R-S04-resultsA) |
| Fig. 4 | `fig4_attribution_vs_mechanistic.py` (P2-R-F4) | `05_results_B.tex` (P2-R-S05-resultsB) |
| Fig. 5 | `fig5_representativeness_map.py` (P2-R-F5) | `08_discussion.tex` (P2-R-S08-discussion) |
| Fig. 6 | `fig6_failure_taxonomy.py` (P2-R-F6) | `07_results_compare.tex` (P2-R-S07-compare) |

Note S07-compare owns BOTH the Fig.2 and Fig.6 captions (both embedded there). That is fine —
one section file, one item.

## Items created (20)
| ID | Priority | file_scope (own only) |
|---|---|---|
| P2-R-UNC | P1 (empirical, runs first) | `tools/xai_study/compare/uncertainty.py` + `compare/out/{leaderboard_ci.json,leaderboard_ci.csv,threshold_sensitivity.csv,aggregation_robustness.csv}` |
| P2-R-REF | P0/P2 | `paper/references.bib` |
| P2-R-SUPP | P0 | `paper/supplement/{supplement,S1_per_method_protocols,S2_benchmark_schema,S3_coverage_applicability,S4_claims_evidence,S5_number_provenance,S6_not_claimed}.tex` |
| P2-R-REPRO | P0 | `xai_2_interpretability/REPRODUCIBILITY.md` + `tools/xai_study/repro/{rom_hash_table.csv,action_stream_hashes.csv,make_hash_tables.py}` |
| P2-R-S00-abstract | P0 | `paper/sections/00_abstract.tex` |
| P2-R-S01-intro | P3 | `paper/sections/01_intro.tex` |
| P2-R-S02-related | P2 | `paper/sections/02_related.tex` |
| P2-R-S03-methods | P0 | `paper/sections/03_methods.tex` |
| P2-R-S04-resultsA | P1 | `paper/sections/04_results_A.tex` |
| P2-R-S05-resultsB | P1 | `paper/sections/05_results_B.tex` |
| P2-R-S06-resultsC | P1 | `paper/sections/06_results_C.tex` |
| P2-R-S07-compare | P0 | `paper/sections/07_results_compare.tex` |
| P2-R-S08-discussion | P0/P3 | `paper/sections/08_discussion.tex` |
| P2-R-S09-endmatter | P0 | `paper/sections/09_endmatter.tex` |
| P2-R-F1 | P3 | `paper/figures/fig1_platform_oracle.py` |
| P2-R-F2 | P3 | `paper/figures/fig2_faithfulness_vs_plausibility.py` |
| P2-R-F3 | P3 | `paper/figures/fig3_phaseA_battery.py` |
| P2-R-F4 | P3 | `paper/figures/fig4_attribution_vs_mechanistic.py` |
| P2-R-F5 | P3 | `paper/figures/fig5_representativeness_map.py` |
| P2-R-F6 | P3 | `paper/figures/fig6_failure_taxonomy.py` |

## Review-point → item mapping

### P0 — fix before submission
| improvement point | item(s) |
|---|---|
| P0#1 review-time artifact / code availability | P2-R-REPRO; P2-R-S09-endmatter (in-paper wording) |
| P0#2 Supplementary Information (per-method protocols) | P2-R-SUPP (S1) |
| P0#3 cite Paper 1 arXiv:2606.22447 + code URL, self-contained theorem | P2-R-REF (bib); P2-R-S03-methods (proposition+cite); P2-R-S09-endmatter |
| P0#4 unit of evaluation / benchmark schema / one method count | P2-R-SUPP (S2); canonical count set by P2-R-S07-compare; consumed by S00/S05/F2/F4 |
| P0#5 formalize faithfulness (+ CIs) | P2-R-S03-methods; CIs from P2-R-UNC |
| P0#6 formalize sufficiency S (+ bootstrap CI; ACDC S=0.44) | P2-R-S03-methods (def); P2-R-S06-resultsC (ACDC); P2-R-UNC (CI) |
| P0#7 formalize minimality M (whole-state M=0.07) | P2-R-S03-methods |
| P0#8 separate predicates from scores (Eq. 3) | P2-R-S03-methods |
| P0#9 repair plausibility axis ("plausibility proxy") | P2-R-S03-methods, P2-R-S00-abstract, P2-R-S07-compare (prose); P2-R-F1/F4 (axis labels) |
| P0#10 "provably zero" self-contained proposition | P2-R-S03-methods |
| P0#11 tighten semantic-gap definition (three separations) | P2-R-S07-compare; P2-R-S08-discussion (ties to it) |
| P0#12 scope NN-transfer / "no business" / "floor" | P2-R-S01-intro, P2-R-S08-discussion (prose); P2-R-F6 (figure) |

### P1 — empirical design + reporting
| point | item(s) |
|---|---|
| P1#13 coverage table (64 games) | P2-R-SUPP (S3) |
| P1#14 method applicability table | P2-R-SUPP (S3) |
| P1#15 separate upper-bound controls from methods | P2-R-S06-resultsC |
| P1#16 report uncertainty (CIs, bootstrap, threshold curves, seed var) | P2-R-UNC; consumed by S04/S05/S06 + F2/F3/F4 |
| P1#17 robust aggregation (alternate weightings) | P2-R-UNC; reported by P2-R-S07-compare |
| P1#18 clarify output regimes | P2-R-S03-methods |
| P1#19 intervention validity (on/off-manifold) | P2-R-S03-methods |
| P1#20 ROM + action-stream reproducibility (SHA-256) | P2-R-REPRO |
| P1#21 artifact traceability / number provenance | P2-R-SUPP (S5) |
| P1#22 strengthen SAE analysis (+ SAEBench) | P2-R-S06-resultsC; positioning in P2-R-S02-related |
| P1#23 strengthen probing (cell-84 case study) | P2-R-S06-resultsC |
| P1#24 T3 label provenance + remove "verified:false" | P2-R-S06-resultsC (provenance); "verified:false" token removed by P2-R-S07-compare (it owns line 165) |
| P1#25 clarify validation horizon | P2-R-S06-resultsC |

### P2 — literature
| point | item(s) |
|---|---|
| P2#26 MIB 2025 | P2-R-REF (bib) + P2-R-S02-related |
| P2#27 SAEBench 2025 | P2-R-REF + P2-R-S02-related + P2-R-S06-resultsC |
| P2#28 M4 2023 + BAM/BIM + [35]/Yang–Kim title | P2-R-REF + P2-R-S02-related |
| P2#29 substrate-adjacent Atari systems | P2-R-S02-related |
| P2#30 missing Figure-6 references | P2-R-REF (bib entries); P2-R-F6 removes names from figure; P2-R-S07-compare discusses in caption |
| P2#31 tone down "first" claims | P2-R-S02-related |

### P3 — structure + writing
| point | item(s) |
|---|---|
| P3#32 reframe abstract (exact count, proxy, scope, numbers+uncertainty) | P2-R-S00-abstract |
| P3#33 reduce rhetorical heat | P2-R-S00-abstract, P2-R-S01-intro, P2-R-S05-resultsB ("default tool"), P2-R-S07-compare, P2-R-S08-discussion |
| P3#34 move repeated claims into tables | P2-R-S07-compare, P2-R-S08-discussion |
| P3#35 claims-and-evidence table | P2-R-SUPP (S4) |
| P3#36 "what this paper does not claim" box | P2-R-SUPP (S6) |
| P3#37 companion-paper roadmap | P2-R-S01-intro, P2-R-S09-endmatter |

### P4 — figures + layout
| point | item(s) |
|---|---|
| P4#38 Fig 1 (jutari Julia, proxy, declutter) | P2-R-F1 (.py) + P2-R-S03-methods (caption) |
| P4#39 Fig 2 (callout overlaps, legends, table) | P2-R-F2 (.py) + P2-R-S07-compare (caption) |
| P4#40 Fig 3 (family labels, oracle-like marks) | P2-R-F3 (.py) + P2-R-S04-resultsA (caption) |
| P4#41 Fig 4 (count, proxy axis, clip) | P2-R-F4 (.py) + P2-R-S05-resultsB (caption) |
| P4#42 Fig 5 (number mismatch, density, page collision) | P2-R-F5 (.py) + P2-R-S08-discussion (caption) |
| P4#43 Fig 6 (pink callout, refs, analogy labels) | P2-R-F6 (.py) + P2-R-S07-compare (caption) + P2-R-REF (refs) |
| P4#44 cross-references (§1→§3.2 oracle, grep) | applied per-file: S01,S03,S04,S05(\S1→§3.2),S06,S07,S08; figures F3/F6 |

### P5 — section-by-section (45–52) → the S0x items (intro→S01, related→S02, methods→S03, A→S04, B→S05, C→S06, compare→S07, discussion→S08).
### P6 — reproducibility package (53–56) → P2-R-REPRO (+ P2-R-SUPP for tables).
### P7 — minimal viable path (57) is exactly the P0 front-loaded set below.

## Cross-cutting string fixes — file ownership (the §0 mechanical-consistency pass)
| string | location(s) | owning item |
|---|---|---|
| "Section 1 oracle" / "\S1 oracle" → §3.2 | `05_results_B.tex` lines 10,29,144 | P2-R-S05-resultsB |
| oracle ref → §3.2 (verify targets) | `01_intro.tex` L65; `04_results_A.tex` L92,102; `03_methods.tex` | S01 / S04 / S03 |
| "human plausibility" → "plausibility proxy" | `03_methods.tex` L24,160 | P2-R-S03-methods |
| "human plausibility" → "plausibility proxy" | `00_abstract.tex` L13 | P2-R-S00-abstract |
| "human-plausibility" (axis) → "plausibility proxy" | `fig1_*.py` L28-29,286; `fig4_*.py` axis | P2-R-F1 / P2-R-F4 |
| "no business" → narrowed wording | `01_intro.tex` L80 | P2-R-S01-intro |
| "no business" → narrowed wording | `08_discussion.tex` L93 | P2-R-S08-discussion |
| "no business" → narrowed wording | `fig6_*.py` L12 | P2-R-F6 |
| "verified:false" token removed | `07_results_compare.tex` L165 | P2-R-S07-compare |
| method count "about 31" → exact | `00_abstract.tex` L10 | P2-R-S00-abstract |
| method count "31 method rows" → canonical | `07_results_compare.tex` L28 | P2-R-S07-compare (CANONICAL SOURCE) |
| "All 30 methods" → exact count | `fig4_*.py` title | P2-R-F4 |
| "all 31 rows (30 methods+...)" wording | `fig2_*.py` L27,210 | P2-R-F2 |
| "field's default tool" → "a widely used baseline" | `05_results_B.tex` L127 | P2-R-S05-resultsB |
| "field's default tool" | `07_results_compare.tex` L42 | P2-R-S07-compare |
| "We remove the excuse" → calmer | `00_abstract.tex` L7 | P2-R-S00-abstract |
| "released ... upon acceptance" → review-time | `09_endmatter.tex` L41-43 | P2-R-S09-endmatter |
| Paper-4 forward pointer only | `09_endmatter.tex` L9 | P2-R-S09-endmatter |

## Sprint roadmap (≤3 local workers/sprint; disjoint file_scopes; deps honored)

Front-load P0 + the mechanical-consistency pass. P2-R-UNC runs first because the figure +
results-section items read its CIs.

### Revision Sprint R1 — foundations (no cross-deps; runs first)
Workers: **P2-R-UNC**, **P2-R-REF**, **P2-R-REPRO**.
Disjoint: `tools/xai_study/compare/*` vs `paper/references.bib` vs `REPRODUCIBILITY.md +
tools/xai_study/repro/*`. All three are dependency roots. (3 workers.)

### Revision Sprint R2 — figures (all read leaderboard_ci.csv from R1)
Workers (split across two waves of ≤3): **P2-R-F1, P2-R-F2, P2-R-F3** then **P2-R-F4, P2-R-F5,
P2-R-F6**. Each owns one `figures/figN_*.py` — pairwise disjoint. F2/F3/F4 depend on P2-R-UNC
(R1 done); F1/F5/F6 have no data dep. (6 items, 2 waves.)

### Revision Sprint R3 — supplement + the P0 metric/leaderboard core
Workers: **P2-R-SUPP** (depends UNC), **P2-R-S03-methods** (depends F1), **P2-R-S07-compare**
(depends UNC,F2,F6,REF). Disjoint: `paper/supplement/*` vs `03_methods.tex` vs
`07_results_compare.tex`. S07 sets the canonical method count here. (3 workers.)

### Revision Sprint R4 — results sections + literature
Workers: **P2-R-S04-resultsA** (UNC,F3), **P2-R-S05-resultsB** (UNC,F4), **P2-R-S06-resultsC**
(UNC). Disjoint section files. (3 workers.) Then a short wave: **P2-R-S02-related** (REF) — owns
`02_related.tex`, disjoint from everything in R4, can run with any one of them if a slot is free.

### Revision Sprint R5 — framing sections (depend on settled numbers/count)
Workers: **P2-R-S01-intro**, **P2-R-S08-discussion** (depends F5), **P2-R-S09-endmatter**
(depends REPRO, REF). Disjoint section files. (3 workers.)

### Revision Sprint R6 — abstract last (depends on canonical count + numbers)
Worker: **P2-R-S00-abstract** (depends UNC, S07). Single item — the abstract is written once the
count + headline numbers + uncertainty are final. SM Review then: SM-only integration writes —
`\input{supplement/...}` into `main.tex`, full recompile, `document_check.md` length pass, final
grep for the banned strings, figure-number/caption checklist (fig_pass §"Concrete checks").

Dependency summary: `UNC, REF, REPRO` (roots) → figures + SUPP + section items → `S00-abstract`
(leaf). `S07-compare` is the count authority; `S03-methods` is the metric-formalization authority.

## What needs the PO (human / Andreas) — explicit
1. **Human-plausibility study decision (improvement P0#9 "stronger option", P3-final).** The
   backlog implements the *minimum* path: rename to "plausibility proxy" everywhere + a documented
   rubric + sensitivity analysis. Whether to instead **run a real human/expert plausibility-rating
   study** (with IRB/ethics) is a PO call. If yes, it becomes a new empirical item (cluster/local
   data collection) and the abstract/Fig.2/Fig.4 wording changes from "proxy" to "measured".
2. **Double-blind review logistics (P0#1).** The anonymized artifact link (Zenodo private /
   OpenReview supplementary / anonymized repo snapshot) is a PO/venue decision; S09-endmatter +
   REPRODUCIBILITY.md write the wording with a placeholder the PO fills.
3. **Main-vs-supplement figure call (fig_pass §"Decide which figures belong in the main paper").**
   F5 (full taxonomy tree) and F6 (full VCS→NN map) are flagged as supplement candidates unless
   radically simplified. The backlog produces simplified main-paper versions; the final
   keep-in-main-vs-move-to-supplement decision (and any `main.tex` re-wiring) is a PO/SM call.
4. **Any unsupportable claim.** If, after formalization, a headline claim cannot be backed by a
   reviewable artifact + CI (e.g. a number whose provenance P2-R-SUPP S5 cannot trace, or a
   semantic-gap separation whose diagnostic is not reproducible), the PO must decide to **drop or
   soften** it rather than ship it. Candidates to watch: the "provably zero" wording (now a scoped
   proposition — PO confirms the Paper-1 theorem pointer is exact), and the NN-transfer "floor"
   framing (now narrowed to evidence-transfer — PO confirms the softened wording is acceptable).
5. **Canonical method count (P0#4).** S07-compare picks ONE count ("N methods" vs "N method rows
   incl. oracle"). The PO should confirm which framing to standardize on, since it propagates to
   the abstract, two figures, and the supplement schema.
