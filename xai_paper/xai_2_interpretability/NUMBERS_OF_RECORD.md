# Paper-2 NUMBERS OF RECORD — 42-game scored battery (single source of truth)

Generated from compare/out/leaderboard.json + position_bootstrap.json. Scored battery = **42 games**.
Records aggregated: **1599** per-game §R records into 31 method rows.

## Headline

- **All-regime gap = 0.3236** — causal/intervention **0.7073** (n=11 methods) vs gradient/correlational **0.3837** (n=14 methods). (6-game was 0.297.)
- **Position regime, per-method families:** causal/intervention **0.561** (±0.3212, n=4) vs gradient/correlational **0.232** (±0.1545, n=9); gap **0.329**.
- **Position gap SIGNIFICANCE (bootstrap over games, n=42):** mean per-game gap **0.2064**, 95% CI **[0.1416, 0.2714]** — **EXCLUDES ZERO**. (6-game was 0.238, CI [-0.05, 0.32], crossed zero.)

## Per-method (faithfulness F, position-regime F, F/S/M, plaus, tradition, n_games)

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
| rise | gradient | 0.548 | 0.419 | — | — | 0.9 | 6 |
| smoothgrad | gradient | 0.38 | 0.115 | — | — | 0.9 | 42 |
| vanilla_saliency | gradient | 0.38 | 0.115 | — | — | 0.9 | 42 |
| ACDC | causal | 0.47 | — | 0.31 | 1.0 | 0.5 | 6 |
| activation_patching | causal | 1.0 | 1.0 | — | — | 0.5 | 42 |
| attribution_patching | gradient | 0.456 | — | — | — | 0.9 | 42 |
| causal_scrubbing | causal | 0.979 | — | — | — | 0.5 | 42 |
| interchange_interventions_das | causal | 1.0 | — | — | — | 0.5 | 42 |
| linear_probing_control_tasks | probing | 0.108 | — | — | — | 0.85 | 42 |
| logit_tuned_lens | causal | 1.0 | — | — | — | 0.5 | 42 |
| nmf_pca_dictionaries | dim_reduction | 0.542 | — | — | — | 0.75 | 6 |
| path_patching | causal | 0.58 | — | — | — | 0.5 | 42 |
| sparse_autoencoder | dim_reduction | 0.173 | — | 0.183 | 0.07 | 0.75 | 40 |
| ORACLE | oracle | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1 |
