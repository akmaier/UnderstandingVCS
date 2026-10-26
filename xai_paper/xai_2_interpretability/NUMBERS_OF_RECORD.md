# Paper-2 NUMBERS OF RECORD — 42-game scored battery (single source of truth)

From compare/out/leaderboard.json + position_bootstrap.json. Scored battery = **42 games**. **1772** §R records → 31 method rows.

## Headline (family-mean gaps; CI = bootstrap over games, the paper's convention)
- **All-regime gap = 0.321**, 95% CI **[0.293, 0.348]** (excludes zero). Family means: causal/intervention **0.705** (n=11 methods) vs gradient/correlational **0.385** (n=14). (6-game: 0.297 [0.232,0.375].)
- **Position gap = 0.3264**, 95% CI **[0.2767, 0.375]** — **EXCLUDES ZERO (SIGNIFICANT)**. Family means: causal/intervention **0.561** (n=4) vs gradient/correlational **0.2346** (n=9). (6-game: 0.238, CI [-0.05, 0.32], crossed zero.)
- Both CIs are bootstrap-over-games of the FAMILY-MEAN gap (position_bootstrap.py), so each point estimate lies inside its own CI. Do NOT use per-game-gap variants.

### Provenance of the position-regime family (2026-07-09 — pilot exclusion + activation-patching position records)
The causal/intervention position family (n=4) is {activation_patching **1.000**, occlusion 0.619, extremal_perturbation 0.347, on_distribution_counterfactual 0.277} → mean **0.561**. Two corrections landed together, and they cancel on the point estimate but tighten the CI:
1. **Pilots excluded from aggregation.** `leaderboard.load_per_game_records` now skips `pilot*` records — the June single-game proof-of-concept runs (`pilotC_patch_pong.json` etc.) that stood up each phase's harness before the July 42-game battery. They were superseded but still globbed in, double-counting one Pong point.
2. **Activation-patching position records emitted across the battery.** `activation_patching.jl` now writes `activation_patching_<game>_position.json` (output_kind=position). Phase-C patching faithfulness = causal-effect agreement vs the oracle (03_methods sec:triad); scored on the discrete screen-position output `n_changed_px` (naive gradient provably zero there), F = Pearson(recovered, exact) = **1.000** because the patch *is* the oracle's own single-site intervention. **Measured, not asserted** — verified F_recomputed == F_stored == 1.0 with recovered==exact on all 42 scored games, each position output causally live (2–19 active patches, ≥3 distinct oracle values). The 6 static-position games are not in the scored 42; none was dropped silently.

Before, activation_patching sat in the position family via the single pilot game only, so the bootstrap-over-games could drop it entirely on a resample → wide CI **[0.132, 0.370]**. Now it is present on all 42 games, so the family mean is stable → CI tightens to **[0.277, 0.375]**, a *stronger* significance (lower bound 0.13 → 0.28). The point estimate (0.327→0.326), the family means (0.561 / 0.235), and activation-patching's 1.000 are unchanged — now backed by 42 games instead of one pilot. All-regime gap and content gap (−0.01, gradients valid on content) unaffected: content records are 504 with or without pilots.

## Per-method (F, F_pos, S, M, plaus, tradition, n_games)
Point values from `leaderboard.json` (`faithfulness`, `faithfulness_position_regime`, `triad_S`, `triad_M`).
Rounded half-up to 3 dp from the UNROUNDED means (the 4-dp file values sit on rounding edges, e.g. A4 M 0.3725 = 0.372478 → 0.372; ACDC S 0.2995 = 0.299516 → 0.300).
Every per-record F, S, M is clipped to [0,1] by `leaderboard.py` before averaging (records keep the raw value).
Per-method bootstrap-over-games CIs (`ci_lo`/`ci_hi`) and seed dispersion (`seed_sd`) are in
`compare/out/leaderboard_ci.csv`, regenerated 2026-10-26 after the A3 fix and the A4 ms_pacman record removal (A3 M CI [0.8744, 0.9705];
seed_sd kernelshap 0.0189, lime 0.0051, rise 0.061, expected_gradients 0.0, smoothgrad 0.0).
Bootstrap RNG: `uncertainty.py` draws each bootstrap from its own stream keyed by (row name, leg),
`np.random.default_rng([20260623, crc32(name/leg)])`, so a CI moves only when its own data moves; switching
from the earlier single shared stream shifted every CI once in the 2nd–3rd decimal (point means unchanged).
A4 ms_pacman: the earlier record (commit 39cd1d3, own trajectory f330+30, pre-shared-testbed) was removed. Re-run
at the shared testbed (cluster job 780436) the A4 runner refuses ms_pacman — the oracle finds no coupled candidate
pair at this state (uninformative), as for the other games A4 does not score. A4 is scored on 33 of 42 games
(23 of the 33 show the weak-pairwise/strong-global signature).
Note: `linear_probing_control_tasks` has n_games 42 in `leaderboard.json` but its F is defined on 27 games
(`leaderboard_ci.csv` n_games 27).

| method | tradition | F | F_pos | S | M | plaus | n_games |
|---|---|---|---|---|---|---|---|
| A1_connectomics | intervention | 0.207 | — | 0.971 | 0.967 | 0.55 | 35 |
| A2_lesions | intervention | 0.965 | — | 0.589 | 1.0 | 0.55 | 42 |
| A3_tuning | correlational | 0.213 | — | 0.2 | 0.926 | 0.8 | 42 |
| A4_spike_word | correlational | 0.231 | — | 0.137 | 0.372 | 0.8 | 33 |
| A5_local_field_potentials | correlational | 0.376 | — | — | — | 0.8 | 40 |
| A6_granger | correlational | 0.136 | — | 0.884 | 0.647 | 0.8 | 35 |
| A7_dim_reduction | dim_reduction | 0.519 | — | — | — | 0.75 | 42 |
| A8_wholestate | descriptive | 1.0 | — | 1.0 | 0.076 | 0.3 | 42 |
| expected_gradients | gradient | 0.175 | 0.025 | 0.576 | 1.0 | 0.9 | 42 |
| extremal_perturbation | intervention | 0.46 | 0.347 | 0.63 | 0.969 | 0.55 | 42 |
| gradxinput_deeplift | gradient | 0.486 | 0.115 | 0.625 | 0.993 | 0.9 | 42 |
| guided_backprop | gradient | 0.322 | 0.0 | 0.572 | 0.964 | 0.9 | 42 |
| integrated_gradients | gradient | 0.379 | 0.115 | 0.591 | 0.984 | 0.9 | 42 |
| kernelshap | gradient | 0.652 | 0.595 | 0.657 | 0.57 | 0.9 | 42 |
| lime | gradient | 0.644 | 0.591 | 0.655 | 0.557 | 0.9 | 42 |
| occlusion | intervention | 0.687 | 0.619 | 0.667 | 0.973 | 0.55 | 42 |
| on_distribution_counterfactual | intervention | 0.433 | 0.277 | 0.639 | 0.989 | 0.55 | 42 |
| rise | gradient | 0.552 | 0.44 | 0.583 | 0.348 | 0.9 | 42 |
| smoothgrad | gradient | 0.38 | 0.115 | 0.597 | 0.981 | 0.9 | 42 |
| vanilla_saliency | gradient | 0.38 | 0.115 | 0.593 | 0.979 | 0.9 | 42 |
| ACDC | causal | 0.625 | — | 0.3 | 1.0 | 0.5 | 35 |
| activation_patching | causal | 1.0 | 1.0 | 0.996 | 0.973 | 0.5 | 42 |
| attribution_patching | gradient | 0.456 | — | 0.769 | 0.995 | 0.9 | 42 |
| causal_scrubbing | causal | 0.979 | — | 0.979 | 0.814 | 0.5 | 42 |
| interchange_interventions_das | causal | 1.0 | — | 1.0 | 0.89 | 0.5 | 42 |
| linear_probing_control_tasks | probing | 0.24 | — | — | 0.932 | 0.85 | 42 |
| logit_tuned_lens | causal | 0.82 | — | — | 0.224 | 0.5 | 42 |
| nmf_pca_dictionaries | dim_reduction | 0.102 | — | 0.751 | 0.966 | 0.75 | 42 |
| path_patching | causal | 0.58 | — | — | 1.0 | 0.5 | 42 |
| sparse_autoencoder | dim_reduction | 0.153 | — | 0.183 | 0.934 | 0.75 | 40 |
| ORACLE | oracle | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1 |

## End-to-end gradient control (6 core games, 2026-10-26)
`phaseB_attribution/e2e_gradient.jl`; records `out/e2e_{saliency,gradxinput}_<game>_{content,position}.json`,
merged in `out/e2e_core_summary.json`. The end-of-horizon output (f*+15) is differentiated back to RAM at f*
through every instruction of the horizon (65,081–99,703 per game), Zygote VJP checkpointed every 200
instructions, each chunk's soft forward asserted bit-equal to the hard registers + RAM. Backward 9–18 h per
game on one core; local (ARM) and cluster (x86) runs agree to ≤ 1 ulp. Scored like vanilla saliency
(Pearson vs the oracle, same causes).

| game | e2e saliency F content | e2e saliency F position |
|---|---|---|
| pong | 0.464 | 0.278 |
| breakout | 0.750 | 0.000 |
| space_invaders | 1.000 | 0.000 |
| seaquest | 0.017 | 0.710 |
| ms_pacman | 0.811 | −0.034 |
| qbert | 0.999 | −0.031 |

Grad×Input equals saliency except where the byte holds 0 at f* (seaquest content 0.000, qbert position 0.000).
The e2e records are **excluded from the leaderboard aggregation** (`leaderboard.py` skips `e2e_*`, as it skips
`pilot*`); the control is a 6-core-game sub-study, not a leaderboard row.
