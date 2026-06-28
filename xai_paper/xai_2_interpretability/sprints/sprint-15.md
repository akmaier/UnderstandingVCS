# Sprint 15 — R7-1: foundations (keystone experiment + clean plane figure)

## R7 roadmap (so the heartbeat + SM know the sequencing)
- **R7-1 (this sprint):** `P2-R7-EXP-sampler` (keystone — build+run on jutari) + `P2-R7-FIG-plane`
  (restore the dashed-curve plane). NO prose written until the experiment's real numbers exist.
- **R7-2:** `P2-R7-S07-compare` + `P2-R7-S08-discussion` + `P2-R7-S03-methods` (all consume the
  experiment CSV), then `P2-R7-S01-intro`. Reframe to the universal-gap + behavior-as-reference.
- **R7-3:** `P2-R7-SENT` (runs ALONE — whole-paper sentence split), then `P2-R7-ABSTRACT` (ALONE,
  final, condense from settled findings).

## Goal (this sprint)
Produce the two verifiable artifacts the reframe rests on. Disjoint file_scopes, 2 local workers.

- **P2-R7-EXP-sampler** — THE KEYSTONE. Build `run_sampler_faithfulness.jl` and RUN it on jutari:
  re-run the gradient/correlational attribution methods (vanilla saliency, grad×input, SmoothGrad,
  Integrated Gradients, Expected Gradients) on the 6 core games on the position regime in two
  conditions — naive gradient (sampler off) and **bilinear sampler on** (Paper-1's index-boundary
  surrogate, the SAME mechanism as `tools/xai_si_gradient/`) — scored vs the intervention oracle.
  Emit per-method/per-game records + `compare/out/sampler_faithfulness.csv` (faithfulness_naive,
  faithfulness_sampler, semantic_recovery) + `fig7`. The headline MUST hold and be self-checked:
  faithfulness_sampler > faithfulness_naive for ≥1 gradient method on position, AND
  semantic_recovery == 0 for every method. **Real artifacts required — no stubs, no fake done.**
- **P2-R7-FIG-plane** — restore the clean dashed-curve faithfulness-vs-plausibility plane (Fig 4)
  from the `8b919764` composition (dashed offset curve + oracle ceiling at (1,1) unreached), drop
  the CI whiskers + in-panel table, keep the R2/R6 labeling fixes (§3.2 oracle, plausibility proxy,
  count, no source paths, 8pt, Okabe-Ito).

## DoD verification (SM — STRICT; the DeepSeek run faked done)
- EXP-sampler: the out/*.json + sampler_faithfulness.csv + fig7 PDF EXIST and are real; re-read the
  CSV to confirm the headline (faithfulness rises, semantic_recovery=0); trace numbers to records.
- FIG-plane: PDF regenerated + render-checked (dashed curve + oracle star present, no clutter); paper compiles.
- Neither item is "done" on a status flip alone — confirm the artifact.

## Review
_(to be filled at the R7-1 barrier)_
