---
id: P2-R-S03-methods
title: Revise methods — formalize F/S/M + metrics, predicates-vs-scores, proxy, "provably zero", Fig1 caption
epic: RV (Revision — section)
status: todo
sprint: 11
owner:
where: local
depends_on: [P2-R-F1]
file_scope:
  - paper/sections/03_methods.tex
estimate: L
spec_ref: ../SPEC.md#e8-nature-paper-draft-writing-sprint-6-local-one-section-per-item
---

## Goal
Formalize the metrics the reviewers say are underdefined — the heart of the major weakness.
Apply IN THIS FILE (improvement_instructions P0#5,#6,#7,#8,#9,#10,#11; P1#18,#19; P3#47; P4#44):
- **Formalize faithfulness (P0#5):** state whether the leaderboard is single- or multi-metric;
  give the per-record metric, normalization to [0,1], handling of negative correlations/
  sufficiency, vector-output reduction, graph→attribution scale mapping, averaging across
  outputs/games, game-vs-method weighting; include formulas for Pearson/Spearman, precision@k,
  deletion/insertion AUC, graph F1, sign/magnitude agreement, and the final faithfulness score.
- **Formalize sufficiency S (P0#6):** define S as an estimator (held-out intervention set,
  train/calibration vs test split, what an explanation predicts, metric/range, tolerance, meaning
  of negative S).
- **Formalize minimality M (P0#7):** reproducible formula + target level (register/RAM cell/
  opcode/module/...); explain why whole-state recording gets M=0.07.
- **Predicates vs scores (P0#8):** Eq. 3 boolean conjunction → define F_score/S_score/M_score in
  their real ranges + thresholds tau_F/tau_S/tau_M, with right(E)=1 iff all exceed threshold (or
  drop boolean "right" entirely and report continuous only).
- **Plausibility proxy (P0#9):** lines 24 and 160 "human plausibility" → "plausibility proxy";
  add the rubric (who assigned, criteria, preregistration-vs-not, single tradition-level proxy).
- **"Provably zero" self-contained (P0#10):** add a short proposition stating the discrete-index/
  position conditions, why the naive gradient is zero a.e. under the hard forward map, which
  surrogate/bilinear-sampler gradients are exceptions; cite the exact Paper-1 theorem
  (arXiv:2606.22447) and replace broad "provably zero" with the precise wording from P0#10.
- **Output regimes + intervention validity (P1#18,#19):** define content/position-index/event/
  score/pixel outputs and per-regime gradient validity; state per-intervention-type whether it is
  surgical-off-manifold / donor-on-manifold / resampling / occlusion / counterfactual-valid, and
  why off-manifold interventions are acceptable for oracle scoring.
- **Section refs (P4#44):** keep the oracle anchored at \S\ref{sec:oracle} (3.2).
- **Fig. 1 caption:** apply the figure_detail_pass caption-shortening + "plausibility proxy" +
  jutari(Julia) corrections to the Fig.1 caption that lives in THIS file (lines 16–28). (The
  Fig.1 .py is fixed by P2-R-F1; this item owns the caption.)

## Definition of Done
- [ ] no "human plausibility" remains in this file; F/S/M each have an explicit formula + range + threshold; Eq. 3 reconciled (predicates vs scores)
- [ ] a Proposition for the discrete-index zero-gradient result with a precise Paper-1 theorem cite
- [ ] output-regime + intervention-validity definitions present; oracle anchored at §3.2
- [ ] Fig.1 caption shortened, says "plausibility proxy" and "jutari (Julia)"
- [ ] `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main` clean
- [ ] nothing outside `file_scope` changed; Paper-1 gates untouched/green
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
The detailed per-method protocol + metric formulas table go to SUPP (S1/S2); methods keeps the
self-contained core. Depends on P2-R-F1 only for caption/figure consistency (proxy + jutari
label) — schedule F1 in the same or earlier sprint. The numeric S/M values (0.44, 0.07) and
their CIs come from P2-R-UNC; keep them consistent with S06.
