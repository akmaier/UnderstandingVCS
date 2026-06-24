---
id: P2-N-SENT
title: Split overly long sentences throughout the paper (style pass)
epic: RV (Revision)
status: done
sprint: 16
owner: agent-1
where: local
depends_on: [P2-N-CONSISTENCY]
file_scope:
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
estimate: M
spec_ref: plan.md (Maier voice / style)
---

## Goal
Systematically identify and split sentences that are too long (over ~35-40 words,
or containing multiple clauses joined by semicolons/commas that strain readability)
throughout all paper sections. Follow Maier's style: prefer shorter, declarative
sentences. Each sentence should make one point. Do not change scientific content,
only syntactic structure.

## Definition of Done
- [ ] every section file read and edited for sentence length
- [ ] no sentence exceeds ~40 words (exceptions for deliberate rhetorical effect)
- [ ] paper compiles cleanly with no undefined refs
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
- Work section by section. Read each file, identify run-on / multi-clause sentences,
  split them. Preserve all \ref{}, \cite{}, and LaTeX commands intact.
- Follow STYLE.md voice exactly: concise, declarative, Maier register.
- Run the STYLE.md §7 self-check before marking done.
