---
id: P2-R7-SENT
title: Split overly long sentences throughout the paper (copy-edit pass, LAST but before abstract)
epic: R7 (universal semantic gap)
status: todo
sprint:
owner:
where: local
depends_on: [P2-R7-S01-intro, P2-R7-S03-methods, P2-R7-S07-compare, P2-R7-S08-discussion]
file_scope:
  - xai_paper/xai_2_interpretability/paper/sections/01_intro.tex
  - xai_paper/xai_2_interpretability/paper/sections/02_related.tex
  - xai_paper/xai_2_interpretability/paper/sections/03_methods.tex
  - xai_paper/xai_2_interpretability/paper/sections/04_results_A.tex
  - xai_paper/xai_2_interpretability/paper/sections/05_results_B.tex
  - xai_paper/xai_2_interpretability/paper/sections/06_results_C.tex
  - xai_paper/xai_2_interpretability/paper/sections/07_results_compare.tex
  - xai_paper/xai_2_interpretability/paper/sections/08_discussion.tex
  - xai_paper/xai_2_interpretability/paper/sections/09_endmatter.tex
  - xai_paper/xai_2_interpretability/paper/supplement/S1_per_method_protocols.tex
  - xai_paper/xai_2_interpretability/paper/supplement/S2_benchmark_schema.tex
  - xai_paper/xai_2_interpretability/paper/supplement/S3_coverage_applicability.tex
  - xai_paper/xai_2_interpretability/paper/supplement/S4_claims_evidence.tex
  - xai_paper/xai_2_interpretability/paper/supplement/S5_number_provenance.tex
  - xai_paper/xai_2_interpretability/paper/supplement/S6_not_claimed.tex
  - xai_paper/xai_2_interpretability/paper/supplement/S7_plausibility.tex
estimate: M
spec_ref: STYLE.md (Maier voice)
---

## Goal
A single copy-edit pass that splits over-long sentences (the multi-clause, comma-and-em-dash chains)
into shorter declarative sentences, in Maier's voice (STYLE.md: flat declarative openers, one idea
per sentence). **Meaning-preserving only — change NO numbers, claims, citations, or labels.** Runs
as the LAST writing pass over all sections (owns them all, so it must run alone — after the R7
reframe items, before the abstract), to avoid file-scope collisions.

## Definition of Done
- [ ] long sentences split across all sections + supplement; readability improved; no content changed
- [ ] `git diff` shows ONLY sentence-structure edits (no number/claim/citation deltas) — spot-verify
- [ ] paper compiles: `latexmk -pdf` exit 0, 0 undefined; page count reported
- [ ] committed + pushed (rebase-before-push); primary pulled ff-only; `status: done`

## Notes / handoff
- This is the item DeepSeek marked done (P2-N-SENT) entangled with its off-thesis rewrite; redo it
  cleanly as a pure copy-edit on the corrected paper. Must run alone (whole-paper scope).
