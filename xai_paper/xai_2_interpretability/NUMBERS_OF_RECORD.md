# Paper-2 NUMBERS OF RECORD — 44-game scored battery (single source of truth)

From compare/out/leaderboard.json + position_bootstrap.json. Scored battery = **44 games**. **1862** §R records → 31 method rows.
Re-derived 2026-10-26 after every game boots with its real RomSettings (was GenericRomSettings for 37 games):
the gate and T3 labels were recomputed, which added asterix, enduro, skiing and removed beam_rider (42 → 44),
and the whole battery was re-run (cluster jobs 780453/780454). Previous 42-game values in git history (f547e960^).

## Headline (family-mean gaps; CI = bootstrap over games, the paper's convention)
- **All-regime gap = 0.324**, 95% CI **[0.297, 0.351]** (excludes zero). Family means: causal/intervention **0.709** (n=11 methods) vs gradient/correlational **0.384** (n=14). (6-game: 0.297 [0.232,0.375].)
- **Position gap = 0.338**, 95% CI **[0.293, 0.383]** — **EXCLUDES ZERO (SIGNIFICANT)**. Family means: causal/intervention **0.560** (n=4) vs gradient/correlational **0.222** (n=9). (6-game: 0.238, CI [-0.05, 0.32], crossed zero.)
- Both CIs are bootstrap-over-games of the FAMILY-MEAN gap (position_bootstrap.py), so each point estimate lies inside its own CI. Do NOT use per-game-gap variants.

### Provenance of the position-regime family (2026-07-09 — pilot exclusion + activation-patching position records)
The causal/intervention position family (n=4) is {activation_patching **1.000**, occlusion 0.619, extremal_perturbation 0.342, on_distribution_counterfactual 0.280} → mean **0.560** (44-game values; the 2026-07-09 history below quotes the 42-game numbers of that time). Two corrections landed together, and they cancel on the point estimate but tighten the CI:
1. **Pilots excluded from aggregation.** `leaderboard.load_per_game_records` now skips `pilot*` records — the June single-game proof-of-concept runs (`pilotC_patch_pong.json` etc.) that stood up each phase's harness before the July 42-game battery. They were superseded but still globbed in, double-counting one Pong point.
2. **Activation-patching position records emitted across the battery.** `activation_patching.jl` now writes `activation_patching_<game>_position.json` (output_kind=position). Phase-C patching faithfulness = causal-effect agreement vs the oracle (03_methods sec:triad); scored on the discrete screen-position output `n_changed_px` (naive gradient provably zero there), F = Pearson(recovered, exact) = **1.000** because the patch *is* the oracle's own single-site intervention. **Measured, not asserted** — verified F_recomputed == F_stored == 1.0 with recovered==exact on all 42 scored games, each position output causally live (2–19 active patches, ≥3 distinct oracle values). The 6 static-position games are not in the scored 42; none was dropped silently. Re-verified on the 44-game re-run: F == 1.0 on all 44 scored games.

Before, activation_patching sat in the position family via the single pilot game only, so the bootstrap-over-games could drop it entirely on a resample → wide CI **[0.132, 0.370]**. Now it is present on all 42 games, so the family mean is stable → CI tightens to **[0.277, 0.375]**, a *stronger* significance (lower bound 0.13 → 0.28). The point estimate (0.327→0.326), the family means (0.561 / 0.235), and activation-patching's 1.000 are unchanged — now backed by 42 games instead of one pilot. All-regime gap and content gap (−0.01, gradients valid on content) unaffected: content records are 504 with or without pilots.

## Per-method (F, F_pos, S, M, plaus, tradition, n_games)
Point values from `leaderboard.json` (`faithfulness`, `faithfulness_position_regime`, `triad_S`, `triad_M`).
Rounded half-up to 3 dp from the UNROUNDED means (the 4-dp file values can sit on rounding edges, e.g. causal_scrubbing F 0.9975 = 0.99749 → 0.997).
Every per-record F, S, M is clipped to [0,1] by `leaderboard.py` before averaging (records keep the raw value).
Per-method bootstrap-over-games CIs (`ci_lo`/`ci_hi`) and seed dispersion (`seed_sd`) are in
`compare/out/leaderboard_ci.csv`, regenerated 2026-10-26 on the 44-game re-run (A3 M CI [0.8636, 0.9650];
seed_sd kernelshap 0.0189, lime 0.0051, rise 0.061, expected_gradients 0.0, smoothgrad 0.0).
Bootstrap RNG: `uncertainty.py` draws each bootstrap from its own stream keyed by (row name, leg),
`np.random.default_rng([20260623, crc32(name/leg)])`, so a CI moves only when its own data moves; switching
from the earlier single shared stream shifted every CI once in the 2nd–3rd decimal (point means unchanged).
A4 ms_pacman: the earlier record (commit 39cd1d3, own trajectory f330+30, pre-shared-testbed) was removed. Re-run
at the shared testbed (cluster job 780436) the A4 runner refuses ms_pacman — the oracle finds no coupled candidate
pair at this state (uninformative), as for the other games A4 does not score. A4 is scored on 36 of 44 games
(24 of the 36 show the weak-pairwise/strong-global signature).
Note: `linear_probing_control_tasks` has n_games 44 in `leaderboard.json` but its F is defined on 26 games
(`leaderboard_ci.csv` n_games 26).

| method | tradition | F | F_pos | S | M | plaus | n_games |
|---|---|---|---|---|---|---|---|
| A1_connectomics | intervention | 0.213 | — | 0.957 | 0.971 | 0.55 | 38 |
| A2_lesions | intervention | 0.964 | — | 0.61 | 1.0 | 0.55 | 44 |
| A3_tuning | correlational | 0.207 | — | 0.196 | 0.918 | 0.8 | 44 |
| A4_spike_word | correlational | 0.251 | — | 0.147 | 0.393 | 0.8 | 36 |
| A5_local_field_potentials | correlational | 0.394 | — | — | — | 0.8 | 42 |
| A6_granger | correlational | 0.157 | — | 0.878 | 0.701 | 0.8 | 38 |
| A7_dim_reduction | dim_reduction | 0.523 | — | — | — | 0.75 | 44 |
| A8_wholestate | descriptive | 1.0 | — | 1.0 | 0.075 | 0.3 | 44 |
| expected_gradients | gradient | 0.186 | 0.029 | 0.548 | 1.0 | 0.9 | 44 |
| extremal_perturbation | intervention | 0.458 | 0.342 | 0.608 | 0.968 | 0.55 | 44 |
| gradxinput_deeplift | gradient | 0.471 | 0.092 | 0.604 | 0.993 | 0.9 | 44 |
| guided_backprop | gradient | 0.315 | 0.0 | 0.546 | 0.966 | 0.9 | 44 |
| integrated_gradients | gradient | 0.361 | 0.092 | 0.553 | 0.985 | 0.9 | 44 |
| kernelshap | gradient | 0.65 | 0.594 | 0.637 | 0.565 | 0.9 | 44 |
| lime | gradient | 0.64 | 0.581 | 0.646 | 0.553 | 0.9 | 44 |
| occlusion | intervention | 0.685 | 0.619 | 0.654 | 0.972 | 0.55 | 44 |
| on_distribution_counterfactual | intervention | 0.433 | 0.28 | 0.61 | 0.986 | 0.55 | 44 |
| rise | gradient | 0.544 | 0.426 | 0.563 | 0.356 | 0.9 | 44 |
| smoothgrad | gradient | 0.361 | 0.092 | 0.558 | 0.982 | 0.9 | 44 |
| vanilla_saliency | gradient | 0.361 | 0.092 | 0.555 | 0.98 | 0.9 | 44 |
| ACDC | causal | 0.648 | — | 0.314 | 1.0 | 0.5 | 38 |
| activation_patching | causal | 1.0 | 1.0 | 0.996 | 0.964 | 0.5 | 44 |
| attribution_patching | gradient | 0.482 | — | 0.798 | 0.99 | 0.9 | 44 |
| causal_scrubbing | causal | 0.997 | — | 0.997 | 0.8 | 0.5 | 44 |
| interchange_interventions_das | causal | 1.0 | — | 1.0 | 0.869 | 0.5 | 44 |
| linear_probing_control_tasks | probing | 0.211 | — | — | 0.944 | 0.85 | 44 |
| logit_tuned_lens | causal | 0.793 | — | — | 0.191 | 0.5 | 44 |
| nmf_pca_dictionaries | dim_reduction | 0.129 | — | 0.725 | 0.985 | 0.75 | 44 |
| path_patching | causal | 0.603 | — | — | 1.0 | 0.5 | 44 |
| sparse_autoencoder | dim_reduction | 0.147 | — | 0.165 | 0.924 | 0.75 | 42 |
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
