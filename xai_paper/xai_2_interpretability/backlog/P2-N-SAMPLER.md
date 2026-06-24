---
id: P2-N-SAMPLER
title: Oracle-sampler-as-attribution experiment — perfect faithfulness ≠ understanding
epic: E4 (Phase B / Semantic Gap)
status: todo
sprint:
owner:
where: local
depends_on: [P2-E1-1]
file_scope:
  - tools/xai_study/phaseB_attribution/oracle_sampler_attribution.jl
  - tools/xai_study/phaseB_attribution/out/oracle_sampler_*
estimate: M
spec_ref: plan.md (semantic gap section, new item iv)
---

## Goal
Run the oracle's own intervention sampler as an attribution method — this is the
strongest possible faithful attribution. The oracle exhaustively tests each RAM
cell/register's causal effect on an output by intervening (setting the value and
re-running the emulator). Use this as an attribution method and demonstrate that
even with perfect faithfulness (F=1.0 by construction), the result is still an
effect table — not the algorithm, not the variable's meaning, not the semantics.
This experiment proves the gap is not a methods-underperformance issue but
fundamental: the causal substrate alone cannot deliver understanding.

## Definition of Done
- [ ] oracle-sampler attribution implemented: uses the intervention oracle to produce
      per-output attribution scores (causal effect per RAM cell/register)
- [ ] run on at least 2 games (e.g. Breakout, Pong) for the position-output regime
- [ ] results show perfect F but no S/M semantics recovered
- [ ] output written to `tools/xai_study/phaseB_attribution/out/oracle_sampler_*`
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
- Reuse the oracle from `tools/xai_study/ground_truth/oracle_intervene.*` — the
  sampler already exists; this item wraps it as an "attribution method" and scores it.
- Key claim: the oracle sampler achieves F=1.0 (by definition — it IS the ground
  truth), yet delivers no semantic understanding (no variable names, no algorithm,
  no documentation recovery).
- Results feed into the semantic gap subsection of the Results and the Discussion.
