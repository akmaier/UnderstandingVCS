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

## Review
_(to be filled at the R1 barrier)_
