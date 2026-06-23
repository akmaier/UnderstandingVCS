# Sprint 9 — Revision R1: foundations (uncertainty · references · reproducibility)

## Goal
The three dependency-root revision items, all disjoint, all P0/P1 from the reviews. These unblock
the figures (need CI numbers), the supplement (needs the artifact/repro spine), and the related-work
+ citations. No prose-section edits this sprint (those are R3–R6).

## Planning (≤3 local workers; pairwise-disjoint file_scopes — verified by the backlog build)
- **P2-R-UNC** — bootstrap CI over the 6 games + seed variance (sampling methods) + ACDC threshold
  sensitivity + aggregation robustness, from the committed §R records (re-run on jutari only where a
  record lacks the needed dispersion). Owns `tools/xai_study/compare/uncertainty.py` +
  `compare/out/{leaderboard_ci.*, threshold_sensitivity.csv, aggregation_robustness.csv}`. (improvement P1#16,#17)
- **P2-R-REF** — `paper/references.bib`: cite Paper 1 as **arXiv:2606.22447** + code URL
  https://github.com/akmaier/UnderstandingVCS; add **MIB 2025, SAEBench 2025, M4 2023**; add the
  Fig-6 missing refs (Balduzzi 2017; Jain & Wallace 2019; Morcos 2018; Leavitt & Morcos 2020;
  Elhage 2022; Kindermans 2019; Sturmfels 2020); fix the Yang–Kim/BIM entry. WebSearch-verify each
  new entry (no hallucinations). (improvement P0#3, P2#26–30)
- **P2-R-REPRO** — `REPRODUCIBILITY.md` + `tools/xai_study/repro/{rom_hash_table.csv,
  action_stream_hashes.csv, make_hash_tables.py}`: review-time artifact availability wording, real
  SHA-256 ROM hash table (from xitari/roms), action-stream hashes, artifact traceability. (P0#1, P1#20, P6)

## PO defaults applied this revision (unless the PO overrides)
- Plausibility = **"plausibility proxy"** + rubric + sensitivity (no human study) — PO may flip to measured.
- Canonical count = **"30 interpretability methods (+ the oracle positive control) = 31 rows"** (S07 sets it; all others follow).
- Review-time artifact link = a **placeholder** the PO fills (Zenodo/OpenReview/anon repo).
- Full Fig 5 / Fig 6 → simplify in main, full versions to supplement (figure items decide).

## Review — R1 closed 2026-06-23 (3/3 done; paper recompiles, 0 undefined)

- **P2-R-UNC** (`3bfa0fe`): `compare/uncertainty.py` + bootstrap CIs (n=2000 over the 6 games),
  threshold-sensitivity, aggregation-robustness. CIs bracket the committed point estimates.
  **Carry-forward finding (R4/R5/figures MUST disclose):** the causal>gradient gap is robust under
  equal-by-method / equal-by-game / position-only (CIs exclude 0) BUT **shrinks with CI crossing 0
  under `content_only` (0.045 [−0.087, 0.153]) and `excluding_oracle_like` (0.123 [−0.008, 0.256])**
  — the gap is driven by the position/index regime and the near-ceiling oracle-like methods. Report
  this honestly (it answers reviewer "separate oracle controls" + "regime-specific" cautions).
- **P2-R-REF** (`dcb3751`): Paper 1 → arXiv:2606.22447 (+ Siming author fix); **BIM→BAM** title fix
  (1907.09701 is BAM); +MIB2025/SAEBench2025/M4-2023 + all 7 Fig-6 refs; WebSearch-verified. The 10
  new keys are unused until S07/S02/S08 `\cite` them (R3–R5).
- **P2-R-REPRO** (`8e7760b`): `REPRODUCIBILITY.md` §0 + real ROM SHA-256 table + action-stream
  hashes + traceability. Open: §0 link placeholder + endmatter "upon acceptance" → trued-up to the
  GitHub→Zenodo→Code Ocean decision (revision_plan PO#3) by the **S09-endmatter** item in R5.

**Recompile:** `latexmk` exit 0, 0 undefined citations, 31 pp. **Next:** R2 figures (sprint 10).
