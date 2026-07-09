# Paper-2 NUMBERS OF RECORD — 42-game scored battery (single source of truth)

From compare/out/leaderboard.json + position_bootstrap.json. Scored battery = **42 games**. **1773** §R records → 31 method rows.

## Headline (family-mean gaps; CI = bootstrap over games, the paper's convention)
- **All-regime gap = 0.3374**, 95% CI **[0.3103, 0.3643]** (excludes zero). Family means: causal/intervention **0.7215** (n=11 methods) vs gradient/correlational **0.384** (n=14). (6-game: 0.297 [0.232,0.375].)
- **Position gap = 0.3264**, 95% CI **[0.2767, 0.375]** — **EXCLUDES ZERO (SIGNIFICANT)**. Family means: causal/intervention **0.561** (n=4) vs gradient/correlational **0.2346** (n=9). (6-game: 0.238, CI [-0.05, 0.32], crossed zero.)
- Both CIs are bootstrap-over-games of the FAMILY-MEAN gap (position_bootstrap.py), so each point estimate lies inside its own CI. Do NOT use per-game-gap variants.

### Provenance of the position-regime family (2026-07-09 — pilot exclusion + activation-patching position records)
The causal/intervention position family (n=4) is {activation_patching **1.000**, occlusion 0.619, extremal_perturbation 0.347, on_distribution_counterfactual 0.277} → mean **0.561**. Two corrections landed together, and they cancel on the point estimate but tighten the CI:
1. **Pilots excluded from aggregation.** `leaderboard.load_per_game_records` now skips `pilot*` records — the June single-game proof-of-concept runs (`pilotC_patch_pong.json` etc.) that stood up each phase's harness before the July 42-game battery. They were superseded but still globbed in, double-counting one Pong point.
2. **Activation-patching position records emitted across the battery.** `activation_patching.jl` now writes `activation_patching_<game>_position.json` (output_kind=position). Phase-C patching faithfulness = causal-effect agreement vs the oracle (03_methods sec:triad); scored on the discrete screen-position output `n_changed_px` (naive gradient provably zero there), F = Pearson(recovered, exact) = **1.000** because the patch *is* the oracle's own single-site intervention. **Measured, not asserted** — verified F_recomputed == F_stored == 1.0 with recovered==exact on all 42 scored games, each position output causally live (2–19 active patches, ≥3 distinct oracle values). The 6 static-position games are not in the scored 42; none was dropped silently.

Before, activation_patching sat in the position family via the single pilot game only, so the bootstrap-over-games could drop it entirely on a resample → wide CI **[0.132, 0.370]**. Now it is present on all 42 games, so the family mean is stable → CI tightens to **[0.277, 0.375]**, a *stronger* significance (lower bound 0.13 → 0.28). The point estimate (0.327→0.326), the family means (0.561 / 0.235), and activation-patching's 1.000 are unchanged — now backed by 42 games instead of one pilot. All-regime gap and content gap (−0.01, gradients valid on content) unaffected: content records are 504 with or without pilots.

## Per-method (F, F_pos, S, M, plaus, tradition, n_games)

| method | tradition | F | F_pos | S | M | plaus | n_games |
|---|---|---|---|---|---|---|---|
| A1_connectomics | intervention | 0.207 | — | 0.971 | 0.978 | 0.55 | 35 |
| A2_lesions | intervention | 0.965 | — | 0.591 | 0.86 | 0.55 | 42 |
| A3_tuning | correlational | 0.213 | — | 0.2 | 0.841 | 0.8 | 42 |
| A4_spike_word | correlational | 0.225 | — | 0.133 | 0.34 | 0.8 | 34 |
| A5_local_field_potentials | correlational | 0.376 | — | — | — | 0.8 | 40 |
| A6_granger | correlational | 0.136 | — | 0.884 | 0.216 | 0.8 | 35 |
| A7_dim_reduction | dim_reduction | 0.519 | — | — | — | 0.75 | 42 |
| A8_wholestate | descriptive | 1.0 | — | 1.0 | 0.075 | 0.3 | 42 |
| expected_gradients | gradient | 0.175 | 0.025 | — | — | 0.9 | 42 |
| extremal_perturbation | intervention | 0.461 | 0.347 | — | — | 0.55 | 42 |
| gradxinput_deeplift | gradient | 0.486 | 0.115 | — | — | 0.9 | 42 |
| guided_backprop | gradient | 0.322 | 0.0 | — | — | 0.9 | 42 |
| integrated_gradients | gradient | 0.379 | 0.112 | — | — | 0.9 | 42 |
| kernelshap | gradient | 0.652 | 0.595 | — | — | 0.9 | 42 |
| lime | gradient | 0.644 | 0.591 | — | — | 0.9 | 42 |
| occlusion | intervention | 0.687 | 0.619 | — | — | 0.55 | 42 |
| on_distribution_counterfactual | intervention | 0.433 | 0.277 | — | — | 0.55 | 42 |
| rise | gradient | 0.552 | 0.44 | — | — | 0.9 | 42 |
| smoothgrad | gradient | 0.38 | 0.115 | — | — | 0.9 | 42 |
| vanilla_saliency | gradient | 0.38 | 0.115 | — | — | 0.9 | 42 |
| ACDC | causal | 0.625 | — | 0.299 | 1.0 | 0.5 | 35 |
| activation_patching | causal | 1.0 | 1.0 | — | — | 0.5 | 42 |
| attribution_patching | gradient | 0.456 | — | — | — | 0.9 | 42 |
| causal_scrubbing | causal | 0.979 | — | — | — | 0.5 | 42 |
| interchange_interventions_das | causal | 1.0 | — | — | — | 0.5 | 42 |
| linear_probing_control_tasks | probing | 0.108 | — | — | — | 0.85 | 42 |
| logit_tuned_lens | causal | 1.0 | — | — | — | 0.5 | 42 |
| nmf_pca_dictionaries | dim_reduction | 0.441 | — | — | — | 0.75 | 42 |
| path_patching | causal | 0.58 | — | — | — | 0.5 | 42 |
| sparse_autoencoder | dim_reduction | 0.173 | — | 0.183 | 0.07 | 0.75 | 40 |
| ORACLE | oracle | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1 |
