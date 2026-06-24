---
id: P2-N-CONSISTENCY
title: Full-paper consistency scan — main text, figures, tables, captions, supplement
epic: RV (Revision)
status: in-sprint
sprint: 15
owner: agent-1
where: local
depends_on: []
file_scope:
  - paper/main.tex
  - paper/sections/00_abstract.tex
  - paper/sections/01_intro.tex
  - paper/sections/02_related.tex
  - paper/sections/03_methods.tex
  - paper/sections/04_results_A.tex
  - paper/sections/05_results_B.tex
  - paper/sections/06_results_C.tex
  - paper/sections/07_results_compare.tex
  - paper/sections/08_discussion.tex
  - paper/sections/09_endmatter.tex
  - paper/supplement/supplement.tex
  - paper/supplement/S1_per_method_protocols.tex
  - paper/supplement/S2_benchmark_schema.tex
  - paper/supplement/S3_coverage_applicability.tex
  - paper/supplement/S4_claims_evidence.tex
  - paper/supplement/S5_number_provenance.tex
  - paper/supplement/S6_not_claimed.tex
  - paper/supplement/S7_plausibility.tex
  - paper/figures/
estimate: M
spec_ref: plan.md (revision)
---

## Goal
After all other N items are done, scan the entire paper for inconsistencies with the
new framing: the sharper thesis (faithfulness is not the dividing line; the gap is
universal; IEEE-610.12; "explainable AI does not understand a thing"). Check: main text
claims vs figures vs tables vs captions vs supplementary material. Every claim, number,
label, and framing must be consistent. Also update the title and add the IEEE definition
of software (code, documentation, data) toward the end.

## Definition of Done
- [ ] all sections scanned for inconsistencies with the new framing
- [ ] title updated to reference IEEE-610.12 / the universal gap if needed
- [ ] IEEE definition of software as code + documentation + data woven into discussion
- [ ] no stale "faithfulness is the dividing line" language remains
- [ ] claims in main text match figures, tables, captions, supplement
- [ ] paper compiles cleanly with no undefined refs
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
- Key phrases to scan for: "faithfulness is necessary/sufficient", "the easy half",
  "the problem is methods underperform" — replace with the universal-gap framing.
- Add IEEE-610.12: software = code + documentation + data. Interpretability recovers
  code-level mechanism. Documentation and data (semantics) are unreachable from mechanism.
- Title update: consider adding "according to IEEE-610.12, explainable AI does not
  understand a thing" or similar.
