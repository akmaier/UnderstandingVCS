---
id: P2-R-UNC
title: Bootstrap CIs + threshold/sampling sensitivity over the committed leaderboard
epic: RV (Revision — empirical)
status: done
sprint: 9
owner:
where: local
depends_on: []
file_scope:
  - tools/xai_study/compare/uncertainty.py
  - tools/xai_study/compare/out/leaderboard_ci.json
  - tools/xai_study/compare/out/leaderboard_ci.csv
  - tools/xai_study/compare/out/threshold_sensitivity.csv
  - tools/xai_study/compare/out/aggregation_robustness.csv
estimate: L
spec_ref: ../SPEC.md#e6-cross-tradition-comparison-benchmark-artifact-sprint-5-local-sm-integrated
---

## Goal
Add the quantitative uncertainty the reviewers require (improvement_instructions P1#16, P1#17;
also feeds P0#5 "confidence intervals and uncertainty", P0#6 "bootstrap intervals for S").
Pure re-read + resample over the already-committed per-game §R records under
`tools/xai_study/phase{A,B,C}/.../out/*.json` (do NOT re-run any emulator experiment; this
must not touch jutari/jaxtari core — SCRUM §7). Produce, on jutari/local:
1. **Bootstrap CIs** (≥1000 resamples over games AND over records) for every family mean and
   for each headline number (Phase-A mean, gradient-family faithfulness, causal/intervention
   faithfulness, the position-regime gap, ACDC S=0.44, SAE matched-feature S, whole-state
   M=0.07). Emit a tidy `leaderboard_ci.{json,csv}` adding `mean, ci_lo, ci_hi, n` columns
   alongside the existing leaderboard rows.
2. **Threshold sensitivity** for thresholded methods, primarily ACDC (and any path/attribution
   patching threshold): sweep the threshold grid and record S/F vs threshold →
   `threshold_sensitivity.csv` (improvement P1#16 "threshold sensitivity curves").
3. **Sampling variance** for sampling methods (LIME/SHAP/RISE, Expected/Integrated Gradients
   baselines) by reusing committed multi-seed records if present, else documenting seed-variance
   from the records' own repeats → folded into `leaderboard_ci.csv` as a `seed_sd` column.
4. **Aggregation robustness** (improvement P1#17): the family means under equal-weight-by-method
   vs equal-weight-by-game vs equal-weight-by-output-regime, excluding-oracle-like, content-only,
   position-only → `aggregation_robustness.csv`.

## Definition of Done
- [ ] runs to completion from this command: `python tools/xai_study/compare/uncertainty.py --index tools/xai_study/results_index.csv --boot 2000`
- [ ] writes `leaderboard_ci.{json,csv}`, `threshold_sensitivity.csv`, `aggregation_robustness.csv` in the §R-compatible schema, with CI columns for every family mean + every headline number cited in the paper
- [ ] a printed self-check asserts: every leaderboard row has a finite CI; ci_lo ≤ mean ≤ ci_hi; position-regime gap CI excludes 0
- [ ] nothing outside `file_scope` changed; Paper-1 gates untouched/green
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
RUNS FIRST in the revision: the figure items (F1–F6) and the results-section items
(S04/S05/S06/S07) DEPEND on this — they must read the CI numbers from
`leaderboard_ci.csv`, not invent error bars. Pure read/resample over committed records;
the original `leaderboard.py` (P2-E6-1) is NOT in scope (owned there) — this is a new
sibling script. Cross-ref experiment_design.md §9 master table for the columns to bootstrap.
