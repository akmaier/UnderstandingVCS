---
id: P2-R-S06-resultsC
title: Revise Results Phase C — oracle-equiv controls, ACDC/SAE/probe rigor + CIs, "verified:false", §refs
epic: RV (Revision — section)
status: done
sprint: 12
owner:
where: local
depends_on: [P2-R-UNC]
file_scope:
  - paper/sections/06_results_C.tex
estimate: M
spec_ref: ../SPEC.md#e8-nature-paper-draft-writing-sprint-6-local-one-section-per-item
---

## Goal
Revise Phase-C (mechanistic) results per improvement_instructions P3#50 (and P0#6, P1#15,#16,#22,
#23,#24,#25; P4#44). Apply IN THIS FILE:
- **Separate positive controls from recovery (P3#50, P1#15):** explicitly mark oracle-equivalent
  positive controls (activation patching == oracle operation on this substrate) as upper bounds,
  not surprising method wins; introduce the three-category framing (oracle/control vs coarse
  intervention vs non-intervention).
- **ACDC/patching sensitivity (P3#50, P0#6, P1#16):** for ACDC, path patching, attribution
  patching show threshold/approximation sensitivity (read from P2-R-UNC's
  threshold_sensitivity.csv); explain how S=0.44 is computed and that it is not an artifact of a
  wrong intervention distribution or strict tolerance; report a bootstrap CI for S.
- **SAE rigor (P1#22, P3#50):** clarify SAE architecture, dictionary size, sparsity penalty,
  training data, train/test split, reconstruction metrics, feature-matching algorithm, causal
  ablation protocol, why S can be negative; align discussion with SAEBench (cite via S02).
- **Probing rigor (P1#23):** report chance accuracy, control-task accuracy, selectivity, probe
  class, regularization, split, label balance, causal-use criterion; expand the cell-84 mini case
  study (what it encodes, probe, intervention, why unused within horizon, may matter outside).
- **T3 label provenance + remove tokens (P1#24):** make T3 source concrete; this file has no
  `verified:false` (that token lives in S07) — but ensure no implementation tokens leak here.
- **Validation horizon (P1#25):** confirm every Phase-C experiment lies within Paper-1's certified
  horizon (≈30 RAM / 60 screen frames); distinguish recorded-trajectory exactness vs intervention
  exactness vs training-data generation if SAE/probe training uses longer trajectories.
- **Section refs (P4#44):** keep oracle pointers at \S\ref{sec:oracle} (§3.2).

## Definition of Done
- [ ] oracle-equivalent controls labeled as upper bounds; ACDC/SAE/probe protocol details present; S=0.44 derivation + CI shown
- [ ] cell-84 mini case study expanded; validation-horizon confirmation present
- [ ] no implementation tokens (e.g. "verified:false") in this file; oracle refs at §3.2
- [ ] `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main` clean
- [ ] nothing outside `file_scope` changed; Paper-1 gates untouched/green
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
Depends on P2-R-UNC (threshold curves + S CIs). Full SAE/probe protocol tables go to SUPP (S1);
this section keeps the evaluable core. SAEBench/control-task citations come from P2-R-REF via
the S02 positioning.
