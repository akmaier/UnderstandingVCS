# Sprint 5 — Method matrix: Phase-B attribution suite + Phase-C mechanistic + A2 lesion sweep

## Goal
Run the full method matrix over the 6 core games, each scored F/S/M vs the per-game oracle,
reusing the validated scorer (`phaseB_attribution/pilot_ig_vs_oracle.jl`'s
`compute_faithfulness`/`write_faithfulness`; the Phase-C pilot for patching/SAE). This is the
headline leaderboard of the paper.

## Strategy (substrate)
Cluster is provisioned + READY but the operator's queue holds ~17 pending `mayo-*` jobs, so
cluster jobs queue behind them. jutari is fast (~15-20 s/case). Therefore:
- **Cheap methods → run LOCALLY on jutari** (single/few gradient passes or few re-runs):
  immediate, exact, no queue wait.
- **Heavy methods → submit to the CLUSTER** (`tools/cluster/xai_array_jl.sbatch`, throttled
  `%N`): many re-runs / training / iterative search. They run as the `mayo-*` queue drains.
- Implement every runner locally (≤3 Opus workers per wave), validate on one game, then
  scale to the 6 core games (local run or cluster array).

## Waves (≤3 local workers each; disjoint `phaseB_/phaseC_/phaseA_` scopes)
**Cheap → LOCAL:**
- Wave 1 (attribution, gradient): `E4-1` vanilla saliency · `E4-2` Grad×Input/DeepLIFT · `E4-4` SmoothGrad
- Wave 2 (attribution, gradient/path): `E4-3` Guided Backprop (+sanity) · `E4-5` IG (+baseline sweep) · `E4-6` Expected Gradients
- Wave 3 (mechanistic, cheap): `E5-1` activation patching · `E5-2` interchange/DAS · `E5-3` attribution patching
- Wave 4 (mechanistic/probing, cheap): `E5-9` linear probing (+control tasks) · `E5-10` logit/tuned lens · `E4-12` counterfactual attribution

**Heavy → CLUSTER (array jobs, throttled `%N`, queue behind mayo):**
- `E3-2` A2 single-unit lesion importance map (per-cell × 6 games)
- `E4-7` Occlusion · `E4-8` meaningful/extremal perturbation · `E4-9` RISE · `E4-10` LIME · `E4-11` KernelSHAP/Shapley
- `E5-4` path patching · `E5-5` ACDC · `E5-6` SAE (full) · `E5-7` NMF/PCA dictionaries · `E5-8` causal scrubbing

Each cluster runner is implemented + smoke-tested locally on 1 game first, then submitted as a
`--array=0-5%6` job over the 6 games. Results land in `results/xai/` on the cluster + sync back.

## Review — closed 2026-06-22: **the full A/B/C method matrix is DONE** (23 method items across 9 waves, all jutari, all scored vs the exact oracle with a positive control = 1.0 everywhere)

**Phase B — attribution (12 methods, E4-0..E4-13).** A clean faithfulness divide on the §1
content-vs-position split:
- **Gradient family** (saliency, Grad×Input/DeepLIFT, SmoothGrad, IG+baseline-sweep, guided
  backprop, expected gradients): faithful on the CONTENT path (corr up to 0.999), **blind on
  position/index** (gradient vanishes → corr 0). Extra findings: guided-BP **fails the Adebayo
  model-randomization sanity check** (saliency is model-invariant on this substrate); SmoothGrad
  smoothing doesn't change faithfulness; EG's on-distribution baselines make it degenerate
  *more* than single-baseline IG.
- **Intervention/perturbation family** (occlusion, RISE, LIME, KernelSHAP, counterfactual,
  extremal perturbation): faithful on BOTH content AND **position** (every value a real-ROM
  re-run). KernelSHAP efficiency axiom holds to 4e-14; LIME's surrogate R² collapses on position
  (honest "diverges"); counterfactual gives real position attribution (pong ball-pixel 0.84).

**Phase C — mechanistic (10 methods, E5-0..E5-10).**
- **Exact, by construction:** activation patching (recovered == oracle, P=R=1.0, all 6 games),
  DAS/interchange (IIA aligned 1.0 / misaligned 0.0).
- **Precision-1 / recall-limited:** path patching, ACDC, causal scrubbing (single-value resample
  misses sign-dependent edges within the 30-frame window; positive controls F1=1).
- **Present ≠ used:** linear probing (6 decodable-but-not-causal cells), logit/tuned lens (R²≈0.58
  on spurious mass), SAE (features matched 0.83–1.0 but M low, polysemantic; qbert overfits),
  NMF/PCA dictionaries (NMF parts causal, SAE atoms inert); attribution patching ≈ exact only in
  the linear regime.

**Phase A — neuroscience battery (8 methods, Sprint 4):** classical methods score LOW faithfulness
vs the oracle while oracle-as-method = 1.0 (quantified Kording); lesion sweep ρ=0.991 but misses
interactions.

**Master finding (the paper's spine):** across all three traditions, methods that **re-run real
interventions on the true system** are faithful; the cheap proxies (gradients, correlations,
tuning curves, probes, dictionaries) are *plausible but not faithful* — and we can say so because
every method is scored against an **exact** causal oracle with a positive control that scores 1.0.

**Process:** 23/23 method items `done` on main; ~80 §R records under `phaseA_kording/out/`,
`phaseB_attribution/out/`, `phaseC_mechanistic/out/`. Two agent flakes (E4-1, E4-12: backgrounded
+ returned a placeholder) were caught at Review and re-run; a "RUN SYNCHRONOUSLY" guardrail in the
dev contract stopped further flakes. All runners are cluster-shardable (`--shard/--game` +
`tools/cluster/xai_array_jl.sbatch`); the cluster is provisioned but the fast jutari local path
carried the whole matrix, so the cluster is reserved for the P3+ breadth-set scale-out.

**Next:** Sprint 6 (auto-advance, experiment-tier) — **E6** cross-method leaderboard + benchmark
package + faithful-method demo, **E7** figures, **E2-2/E2-3** T3 verify/discover. Then **PAUSE for
PO** before **E8** (the Nature draft) and **E9** (submission) per the gates.
