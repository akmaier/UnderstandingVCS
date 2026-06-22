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

## Review
_(to be filled at the Sprint-5 barrier)_
