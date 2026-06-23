# P2 number-provenance + consistency audit (2026-06-24)

Adversarial audit of every quantitative claim vs the committed records (4 read-only auditors over
all sections + supplement + figures → synthesis). **119 findings: 85 ok, 21 label-needed, 13
mismatch. No fatal scientific error.** `prop:zero` SOUND; F/S/M consistent except M4.

## 1. Canonical numbers (authoritative — all sections/figures/supplement MUST match)

| Quantity | Canonical value | Record | Qualifier when cited |
|---|---|---|---|
| Method count | **31** = 30 methods + oracle control | leaderboard.json:n_methods | — |
| Per-game records aggregated | **257** | leaderboard.json:n_per_game_records_aggregated | — |
| Family F causal/intervention (all regimes) | **0.6811** [0.590,0.781], n=11 | leaderboard_ci.csv:family_causal_intervention | **cross-phase 11-method** family (NOT Phase-B's 3 intervention=0.3925) |
| Family F gradient/correlational (all regimes) | **0.2938** [0.203,0.375], n=14 | leaderboard_ci.csv:family_gradient_correlational | cross-phase 14-method (Phase-B-only gradient=0.298) |
| All-regimes gap | **0.3873** [0.321,0.452] | leaderboard_ci.csv:all_regime_gap | **NOT robust**: →0.123 [−0.008,0.256] excl. oracle-like; →0.045 [−0.087,0.153] content-only (cross zero) |
| Position-regime causal vs gradient | **0.4118 vs 0.0683** | leaderboard.json:headline_contrast.position_regime | — |
| Position-regime gap | **0.3435** [0.016,0.504] | aggregation_robustness.csv:position_only | **robust** (excludes zero) — the surviving headline |
| Activation patching vs vanilla saliency | **1.000 vs 0.000** | faithful_demo.json:pair_contrast | **position regime only**; all-regimes 1.0 vs 0.2667 |
| Plausibility proxy contrast | 0.9 (saliency) vs 0.5 (act.patch) | faithful_demo.json:*.plausibility_proxy | proxy by tradition, not measured |
| ACDC family (cross-game mean) | **F=0.4452 / S=0.3247 / M=1.0** | acdc_core_summary.json; leaderboard_ci.csv:ACDC | this is what the figures plot |
| ACDC Breakout exemplar | **F=1.0, M=1.0, S=0.4444** | acdc_breakout.json:extra.triad | **"on Breakout"** — NOT a family result |
| SAE leaderboard/family | **F=0.1947** (over-games 0.1404 [0.0,0.314]), plaus 0.75 | leaderboard.json:sparse_autoencoder | plotted/family value |
| SAE Pong exemplar | **matched=1.0, F=0.0408, S=−0.3362** | sae_pong.json:extra | **"on Pong"** — NOT the SAE family score |
| Whole-state (A8) | **F=1.0, S=1.0, M=0.0703** (denom 128 RAM cells) | A8_wholestate.json | M denom = 128 RAM (tape = 128 RAM + 64 TIA) |
| Phase means F | A=**0.4922**, B=**0.3216**, C=**0.6304** | leaderboard.json:phase_rollups | A's ±0.24 is **CI95**, not SD |
| Phase-A key | A1=0.4145; A2 ρ=0.99; A3=0.1161 (lowest); A4=0.2872/S=0.018; A5=0.3969; A6=0.1361; A7 **NMF=PCA tied 0.60**; A8 above | phaseA_kording/out/* | — |
| Phase-B key | occlusion 0.482 (top); kernelshap 0.446; on_dist_cf 0.350; extremal 0.346; EG 0.034 (lowest) | leaderboard.json | extremal/on_dist_cf 0.346/0.350 are **all-regime** (position 0.202/0.139) |
| Phase-C key | act_patch 1.0; causal_scrubbing 1.0 true/0.5234 wrong; DAS 1.0/0.0; logit_lens 1.0; attribution_patching 0.50/prec0.95/rec0.70; path_patching F1=0.4643; linear probing acc0.89/sel0.16, **6 decodable-not-causal / 25 not-causally-used** | phaseC_mechanistic/out/* | — |

## 2. Fix-list (by owner)

### R3-done → TARGETED PATCH NOW (03_methods, 07_results_compare, supplement)
- **M4 [03] LOAD-BEARING:** F-formula `0.5·(1+ρ)` (03:211-214) contradicts the committed estimators
  (Phase B = raw Pearson ρ; Phase A = per-game clip-at-0 mean). Under 0.5·(1+ρ) the position
  gradient methods could not "collapse to 0.00". Replace the definition with the two estimators
  actually used (or add an explicit "reported F uses raw/clipped ρ, not the rescaled F").
- **prop:zero [03] cosmetic:** add hypothesis "g non-constant (strictly monotone) on each piece" so
  the measure-zero set is literally correct; tighten "never positive" → "zero wherever defined".
- **U1–U4 + M7 [07]:** ACDC Breakout F=1.0 cite → acdc_breakout.json (07:131-133); SAE Pong
  F=0.041/S=−0.336 cite → sae_pong.json (07:146-151); cell-84 cite → linear_probing_breakout.json
  (07:168-170) AND reword "decodable"→"present-but-unused" (cell 84 decodable=false, it's in the 25
  not-causally-used set); Fig 6 caption (07:219-222) must say "F=1.0 **on Breakout**" not "from the
  leaderboard" (the figure plots the family mean 0.4452); ACDC recap (07:179-180) add "on Breakout".
- **M6 + labels [supplement]:** S5:30 0.6808→**0.6811**, S5:32 0.3869→**0.3873** (cite the
  leaderboard_ci family rows); S4:26/:28 add "position regime" to the 1.0-vs-0.0 / saliency-0.0
  claims; S4:24 add the robustness caveat (gap →0.123/0.045, CIs cross zero off the position regime).

### R4 (owns 02_related, 04_results_A, 05_results_B, 06_results_C + fig3_phaseA_battery.py)
- **M1 [04]:** A2 interaction blind-spot "25%"→**"40%"** (04:36-37 AND Fig.3 caption 04:113). 1044 px is correct.
- **M2 [04 + fig3.py]:** A7 "NMF edges out PCA" is FALSE — NMF=PCA tied (mean 0.60, best 0.80 on
  matched-component fraction). Reword 04:70-77/:113-114 AND fig3_phaseA_battery.py:428-433 to "NMF =
  PCA (both mediocre)"; drop "edges out"/"aligns better". (NMF only leads on the recon-error composite.)
- **M3 [05]:** Phase-B family means 0.681/0.412 (05:105-107, Table 1 :172) are cross-phase
  families shown in a Phase-B table — add "cross-phase family" qualifier OR use Phase-B-only 0.393/0.216;
  same for gradient 0.294 (:171; Phase-B-only 0.298).
- **M5 [05]:** ROM-scramble byte range "~95–127"→**"~69–127"** (05:76-79; seaquest=69.3).
- **labels [05]:** extremal/on_dist_cf "work on position outputs (0.346/0.350)" — those are
  all-regime; position values are 0.202/0.139. Swap values or change to "all-regime faithfulness".
- **labels [04]:** "0.49 ± 0.24" → label "(95% CI)" (04:19-22; it's the CI95 half-width, not SD).
- **labels [06]:** ACDC Breakout recap (06:115-116) add "on Breakout"; "25 such cells" (06:93-95)
  reword so 25 = all not-causally-used (superset), 6 = decodable-and-not-used.

### R5 (owns 01_intro, 08_discussion, 09_endmatter)
- **M7 [08]:** cell 84 "decodable"→"present-but-unused" (08:78).
- **labels [08]:** ACDC "F=M=1.0/S=0.44" add "on Breakout" (08:40-43); "1−S=0.56 on an exactly
  recovered circuit" (08:99) — Breakout gives 0.5556, family mean 0.675; say which; SAE Pong add
  "on Pong" (08:41-43, Fig.5 caption 08:122-123).
- **labels [01]:** SAE Pong add "on Pong" (01:69-70; intro already labels ACDC "for Breakout").

### R6 (owns 00_abstract + integration)
- **labels [00]:** ACDC "sufficiency 0.44" add "on Breakout" (00:16-17); SAE "matched 1.0, F=0.04"
  add "on Pong" (00:17-18); "1.000 versus 0.000" add "position regime" (00:12-13).

## 3. Verdicts
- **prop:zero: SOUND** (cosmetic hypothesis above). The empirical vanilla_saliency/IG position F=0.0
  is consistent — which is exactly why M4 must land (the 0.5·(1+ρ) floor would contradict it).
- **F/S/M: consistent** in definition; M (denom 128 RAM) uniform; S problems are labeling only; F has
  the single definitional bug M4. Resolve M4 → fully consistent.
- **No-change (defensible):** "about 31" (=31, ok); SAE 0.195 [0.0,0.314] cross-column pairing
  (matches figures); Breakout probe 0.72 (single decodable cell, fine).
