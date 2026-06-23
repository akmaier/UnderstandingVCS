---
id: P2-R-S02-related
title: Revise related work — add MIB/SAEBench/M4 + substrate-adjacent Atari, reorganize, tone "first"
epic: RV (Revision — section)
status: todo
sprint: 12
owner:
where: local
depends_on: [P2-R-REF]
file_scope:
  - paper/sections/02_related.tex
estimate: M
spec_ref: ../SPEC.md#e8-nature-paper-draft-writing-sprint-6-local-one-section-per-item
---

## Goal
Complete the novelty framing per improvement_instructions P3#46 (and P2#26,#27,#28,#29,#31).
Apply IN THIS FILE:
- **Reorganize (P3#46):** into XAI/faithfulness evaluation; known-answer interpretability
  benchmarks; mechanistic-interpretability benchmarks; Atari/ALE/object-state resources;
  software reverse-engineering/semantics.
- **MIB 2025 (P2#26):** add a paragraph — MIB benchmarks causal pathways/variables in neural
  LMs; our benchmark differs by using a real hand-authored emulator artifact with exact
  intervention over software state; our semantic-gap claim complements MIB. Cite `mib2025`.
- **SAEBench 2025 (P2#27):** SAE proxy metrics can fail to translate to practical
  interpretability; our oracle gives a different causal-use test on a known system. Cite
  `saebench2025`.
- **M4 2023 + BAM/BIM (P2#28):** position our exact-intervention ground truth vs pseudo/synthetic
  feature ground truth; correct the Yang/Kim ([35]) reference usage. Cite `m4_2023`.
- **Substrate-adjacent Atari (P2#29):** mention ALE/Farama, xitari, OCAtari, JAXAtari, CuLE,
  CALE; be explicit the novelty is "bit-exact differentiable/intervention oracle for
  interpretability," not "Atari environment."
- **Tone down "first" claims (P2#31):** "To our knowledge, this is the first cross-tradition
  audit of these methods on a real hand-authored software artifact with an exact intervention
  oracle." Avoid broad "first ground-truth calibration" claims (exclude Tracr, InterpBench, MIB,
  M4, BAM/BIM, synthetic known-feature benchmarks, Jonas/Kording).

## Definition of Done
- [ ] `mib2025`, `saebench2025`, `m4_2023` cited in this file and resolve (depends_on P2-R-REF)
- [ ] related work reorganized into the five subsections; substrate-adjacent Atari systems named
- [ ] "first" claims narrowed to the cross-tradition/real-artifact wording
- [ ] `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main` clean, 0 undefined cites
- [ ] nothing outside `file_scope` changed; Paper-1 gates untouched/green
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
Depends on P2-R-REF for the new bib keys — schedule REF in the same or an earlier sprint. SAE
discussion alignment with SAEBench is shared in spirit with S06-results-C (which owns the Phase-C
SAE prose); this item only adds the related-work positioning, not the results discussion.
