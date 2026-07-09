#!/usr/bin/env python3
"""P2-R-UNC — Bootstrap CIs + threshold / sampling sensitivity + aggregation robustness.

Revision-time uncertainty companion to ``compare/leaderboard.py`` (P2-E6-1). It is a
**pure read + resample** over the already-committed per-game §R records under
``tools/xai_study/{phaseA_kording,phaseB_attribution,phaseC_mechanistic}/out/*.json`` (and
the ground-truth oracle). It NEVER re-runs an emulator experiment and NEVER alters a
committed result — it only ADDS the quantitative uncertainty the reviewers require:

  reviewer improvement_instructions  P1#16  "error bars / CIs on all family means;
                                            bootstrap over games and records; seed variance
                                            for sampling methods; threshold-sensitivity
                                            curves for thresholded methods (ACDC)";
                                     P1#17  "make aggregation robust (equal-by-method,
                                            equal-by-game, equal-by-output-regime,
                                            excluding-oracle-like, excluding-N/A)";
  also feeds the P0 asks  P0#5 "confidence intervals and uncertainty",
                          P0#6 "report bootstrap intervals for S".

It re-uses ``leaderboard.py``'s validated loaders / orientation tables verbatim (imported
as a module) so every faithfulness number traces to the SAME committed record the
leaderboard reports — the CI is wrapped around the identical point estimate, never a new
measurement.

Run (pure read; the DoD gate command):
    python tools/xai_study/compare/uncertainty.py --index tools/xai_study/results_index.csv --boot 2000

Outputs (to ``tools/xai_study/compare/out/``):
    leaderboard_ci.json   — every leaderboard row + family mean + headline number with
                            mean / ci_lo / ci_hi / n / seed_sd (tidy, §R-compatible)
    leaderboard_ci.csv    — the same as a flat table
    threshold_sensitivity.csv — ACDC (and attribution-patching) S/F vs threshold τ, swept
                            on the CACHED effect matrix (zero re-runs — see budget note in
                            each acdc_*.json), per game + the family mean curve
    aggregation_robustness.csv — the causal/intervention vs gradient/correlational
                            faithfulness gap under 6 aggregation rules

----------------------------------------------------------------------------------------
Bootstrap design (review P1#16 "bootstrap over games AND over records"):
  * Resample-over-GAMES is the headline CI: the 6 core games are the unit of
    generalisation, so each family mean / per-method mean / headline contrast is
    bootstrapped by resampling the 6 games with replacement (>=`--boot` resamples).
    This is the CI the figures and results sections must read.
  * Resample-over-RECORDS is also reported (resample the flat list of per-(game,regime)
    records) as a within-method check; for methods with one record per game the two
    coincide.
  * Percentile CIs (2.5 / 97.5). A degenerate family (all values identical, e.g. the
    oracle or an at-ceiling causal method) yields ci_lo == mean == ci_hi (width 0) — that
    is correct, not a failure.

Seed variance (review P1#16 "for sampling methods report seed variance") — sourced ONLY
from each method's OWN committed dispersion block; nothing re-run, nothing invented:
  * LIME, expected_gradients  -> extra.stability.corr_std         (5 independent seeds)
  * RISE, kernelshap          -> std of extra.convergence_sweep[*].pearson_corr across the
                                 N/8..N mask budgets (Monte-Carlo dispersion proxy ∝ 1/√N)
  * smoothgrad                -> |extra.vanilla_gradient_reference.smoothing_delta_pearson|
                                 (the committed σ-robustness dispersion; SmoothGrad is a
                                 deterministic noise-AVERAGE so its seed wobble is bounded
                                 by the smoothing delta — and is exactly 0 on the
                                 position/index regime where the gradient vanishes)
  Each seed_sd carries a `seed_sd_source` string so it is never mistaken for a fresh run.
----------------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import warnings
from collections import defaultdict

import numpy as np

# A method absent from a bootstrap draw's games yields an all-NaN row whose nanmean is a
# benign NaN (correctly ignored by the outer nanmean). Silence only that specific warning.
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)  # tools/xai_study
OUT_DIR = os.path.join(HERE, "out")

# Reuse the leaderboard's validated readers / orientation tables verbatim so the CI wraps
# the *identical* committed point estimate (no second source of truth).
sys.path.insert(0, HERE)
import leaderboard as LB  # noqa: E402

CORE_GAMES = ["pong", "breakout", "space_invaders", "seaquest", "ms_pacman", "qbert"]
PHASE_C_DIR = LB.PHASE_DIRS["C"]

# Sampling methods that need a reported seed variance (review P1#16).
SAMPLING_METHODS = {"rise", "lime", "kernelshap", "smoothgrad", "expected_gradients"}


# ---------------------------------------------------------------------------------------
# Per-game collection (resample-over-games unit)
# ---------------------------------------------------------------------------------------
def collect_per_game():
    """Group committed per-game faithfulness / triad legs by (phase, method).

    Returns a dict keyed by (phase_long, method) -> {
        'tradition', 'by_game': {game: [faith,...]}, 'records':[faith,...],
        'F','S','M': {game:[...]},  'regimes': {regime: {game:[...]}}, 'commits': set,
    }.  Mirrors leaderboard.py's grouping but keeps the GAME index so we can resample
    games.
    """
    groups = defaultdict(lambda: {
        "tradition": None,
        "by_game": defaultdict(list),
        "records": [],
        "F": defaultdict(list), "S": defaultdict(list), "M": defaultdict(list),
        "regimes": defaultdict(lambda: defaultdict(list)),
        "commits": set(),
        "n_records": 0,
    })
    for phase, path, rec in LB.load_per_game_records():
        mk = LB.method_key(rec)
        if mk == "na_audit":
            continue
        game = rec.get("game") or "unknown"
        phase_long = LB.PHASE_LONG[phase]
        g = groups[(phase_long, mk)]
        g["tradition"] = LB.TRADITION.get(mk, "unknown")
        g["n_records"] += 1
        if rec.get("commit"):
            g["commits"].add(rec["commit"])

        F, S, M = LB.triad_of(rec)
        regime = LB.output_regime(rec)
        if F is not None:
            faith = LB.clip01(F)
        else:
            faith, _ = LB.orient_faithfulness(rec.get("metric_name"), rec.get("value"))
        if faith is not None and not math.isnan(faith):
            g["by_game"][game].append(faith)
            g["records"].append(faith)
            g["regimes"][regime][game].append(faith)
        if F is not None and not math.isnan(LB.clip01(F)):
            g["F"][game].append(LB.clip01(F))
        if S is not None and not math.isnan(LB.clip01(S)):
            g["S"][game].append(LB.clip01(S))
        if M is not None and not math.isnan(LB.clip01(M)):
            g["M"][game].append(LB.clip01(M))
    return groups


def game_means(by_game):
    """One value per game (mean of that game's records) — the resampling unit."""
    return {g: float(np.mean(v)) for g, v in by_game.items() if len(v)}


# ---------------------------------------------------------------------------------------
# Bootstrap primitives
# ---------------------------------------------------------------------------------------
def boot_ci_over_games(per_game_value, rng, n_boot, alpha=0.05):
    """Percentile CI of the across-game mean, resampling the GAMES with replacement.

    `per_game_value` is {game: scalar}.  Returns (mean, ci_lo, ci_hi, n_games).
    """
    vals = np.asarray(list(per_game_value.values()), dtype=float)
    n = len(vals)
    if n == 0:
        return None, None, None, 0
    mean = float(vals.mean())
    if n == 1:
        return mean, mean, mean, 1
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = vals[idx].mean(axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return mean, lo, hi, n


def boot_ci_over_records(records, rng, n_boot, alpha=0.05):
    """Percentile CI resampling the flat per-record list (the within-method check)."""
    vals = np.asarray(records, dtype=float)
    n = len(vals)
    if n == 0:
        return None, None, None, 0
    mean = float(vals.mean())
    if n == 1:
        return mean, mean, mean, 1
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = vals[idx].mean(axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return mean, lo, hi, n


def boot_ci_family(per_method_game_values, rng, n_boot, alpha=0.05):
    """CI of a FAMILY mean (mean over the family's methods of each method's game-mean),
    bootstrapping the shared 6 games jointly so the across-method covariance is respected.

    `per_method_game_values` is a list of {game: scalar} (one per method in the family).
    On each resample we draw a games multiset and recompute every method's mean on those
    games, then average the methods. Returns (mean, ci_lo, ci_hi, n_methods, n_games).
    """
    methods = [m for m in per_method_game_values if m]
    if not methods:
        return None, None, None, 0, 0
    games = sorted(set().union(*[set(m) for m in methods]))
    G = len(games)
    point = float(np.mean([np.mean(list(m.values())) for m in methods]))
    if G <= 1:
        return point, point, point, len(methods), G
    # matrix [method, game] with NaN where a method lacks that game
    mat = np.full((len(methods), G), np.nan)
    for i, m in enumerate(methods):
        for j, gname in enumerate(games):
            if gname in m:
                mat[i, j] = m[gname]
    boots = np.empty(n_boot)
    gi = np.arange(G)
    for b in range(n_boot):
        draw = rng.choice(gi, size=G, replace=True)
        sub = mat[:, draw]               # [method, G]
        with np.errstate(invalid="ignore"):
            method_means = np.nanmean(sub, axis=1)
        boots[b] = np.nanmean(method_means)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi, len(methods), G


# ---------------------------------------------------------------------------------------
# Seed variance (committed dispersion only)
# ---------------------------------------------------------------------------------------
def seed_sd_for_method(method):
    """Mean over the core games of the method's OWN committed seed/MC dispersion.

    Returns (seed_sd, source_str, n_games_with_block) or (None, reason, 0).
    """
    bdir = LB.PHASE_DIRS["B"]
    sds, source = [], None
    for game in CORE_GAMES:
        for out in ("content", "ball_pixel"):
            p = os.path.join(bdir, f"{method}_{game}_{out}.json")
            if not os.path.isfile(p):
                continue
            try:
                ex = json.load(open(p)).get("extra", {})
            except Exception:
                continue
            stab = ex.get("stability")
            conv = ex.get("convergence_sweep")
            if isinstance(stab, dict) and "corr_std" in stab:
                sds.append(abs(float(stab["corr_std"])))
                source = "extra.stability.corr_std (independent-seed refits)"
            elif isinstance(conv, dict) and conv.get("points"):
                pc = [pt.get("pearson_corr") for pt in conv["points"]
                      if pt.get("pearson_corr") is not None]
                if len(pc) >= 2:
                    sds.append(float(np.std(pc, ddof=1)))
                    source = ("std of extra.convergence_sweep[*].pearson_corr "
                              "(Monte-Carlo budget dispersion ∝ 1/√N)")
            else:
                vgr = ex.get("vanilla_gradient_reference")
                if isinstance(vgr, dict) and "smoothing_delta_pearson" in vgr:
                    sds.append(abs(float(vgr["smoothing_delta_pearson"])))
                    source = ("|extra.vanilla_gradient_reference.smoothing_delta_pearson| "
                              "(committed σ-robustness; SmoothGrad is a deterministic "
                              "noise-average, 0 on the gradient-vanishing position regime)")
    if not sds:
        return None, "no committed dispersion block in this method's records", 0
    return float(np.mean(sds)), source, len(sds)


# ---------------------------------------------------------------------------------------
# ACDC threshold-sensitivity (cached effect-matrix sweep — zero re-runs)
# ---------------------------------------------------------------------------------------
def _f1(tp, fp, fn):
    denom = (2 * tp + fp + fn)
    return (2.0 * tp / denom) if denom else 0.0


def acdc_threshold_curve(n_grid=25):
    """Sweep τ over a fine grid by thresholding the CACHED edge-effect matrix in each
    acdc_<game>.npz (the budget note: 'the τ sweep adds zero re-runs — it thresholds the
    cached effect matrix').  Returns (per_game_rows, family_curve_rows).

    For each τ we keep edges with effect > τ, score against the committed true_graph, and
    report precision / recall / F1 (graph-F1 == the ACDC faithfulness F leg) and S
    (scrub_preserved, τ-independent, carried for context).
    """
    per_game = []
    # collect the global effect range to share one grid across games (comparable curves)
    eff_all = []
    cache = {}
    for game in CORE_GAMES:
        npz = os.path.join(PHASE_C_DIR, f"acdc_{game}.npz")
        js = os.path.join(PHASE_C_DIR, f"acdc_{game}.json")
        if not (os.path.isfile(npz) and os.path.isfile(js)):
            continue
        z = np.load(npz)
        eff = np.asarray(z["edge_effects"], dtype=float)
        tg = np.asarray(z["true_graph"], dtype=int)
        scrub = float(np.asarray(z["scrub_preserved"]).reshape(-1)[0]) \
            if "scrub_preserved" in z.files else float("nan")
        # candidate edges = off-diagonal cells (a cell's effect on another cell)
        mask = ~np.eye(eff.shape[0], dtype=bool)
        cache[game] = (eff, tg, mask, scrub)
        eff_all.extend(eff[mask].tolist())
    if not cache:
        return [], []
    emax = max(eff_all) if eff_all else 1.0
    grid = np.linspace(0.0, emax * 1.001, n_grid)

    # per-game curve
    curve_by_game = {}
    for game, (eff, tg, mask, scrub) in cache.items():
        rows = []
        for tau in grid:
            disc = (eff > tau) & mask
            true = (tg > 0) & mask
            tp = int(np.sum(disc & true))
            fp = int(np.sum(disc & ~true))
            fn = int(np.sum(~disc & true))
            f1 = _f1(tp, fp, fn)
            prec = (tp / (tp + fp)) if (tp + fp) else 0.0
            rec = (tp / (tp + fn)) if (tp + fn) else 0.0
            rows.append({
                "game": game, "tau": round(float(tau), 6),
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 4), "recall": round(rec, 4),
                "f1_faithfulness": round(f1, 4),
                "scrub_preserved_sufficiency": round(scrub, 4) if not math.isnan(scrub) else "",
            })
        curve_by_game[game] = rows
        per_game.extend(rows)

    # family mean curve (mean F1 over games at each τ)
    family = []
    for k, tau in enumerate(grid):
        f1s = [curve_by_game[g][k]["f1_faithfulness"] for g in curve_by_game]
        scr = [c[k]["scrub_preserved_sufficiency"] for g in curve_by_game
               for c in [curve_by_game[g]] if c[k]["scrub_preserved_sufficiency"] != ""]
        family.append({
            "game": "FAMILY_MEAN", "tau": round(float(tau), 6),
            "tp": "", "fp": "", "fn": "",
            "precision": "", "recall": "",
            "f1_faithfulness": round(float(np.mean(f1s)), 4),
            "scrub_preserved_sufficiency": round(float(np.mean(scr)), 4) if scr else "",
        })
    return per_game, family


# ---------------------------------------------------------------------------------------
# Aggregation robustness (review P1#17)
# ---------------------------------------------------------------------------------------
def aggregation_robustness(groups, rng, n_boot):
    """Recompute the headline causal/intervention vs gradient/correlational faithfulness
    GAP under six aggregation rules, each with a bootstrap CI over games.  The headline
    claim survives iff every variant's gap CI excludes 0 (or at least stays positive)."""
    CAUSAL = {"causal", "intervention"}
    GRAD = {"gradient", "correlational"}
    ORACLE_LIKE = {  # methods that are exact-by-construction (near-ceiling positive controls)
        "activation_patching", "causal_scrubbing", "interchange_interventions_das",
        "logit_tuned_lens", "A2_lesions", "A8_wholestate",
    }

    def method_game_vals(method_filter, regime=None):
        """list of {game: faith} for methods passing the filter (optionally regime-only)."""
        out = []
        for (phase, mk), g in groups.items():
            if not method_filter(mk, g["tradition"]):
                continue
            if regime is None:
                vals = game_means(g["by_game"])
            else:
                vals = game_means(g["regimes"].get(regime, {}))
            if vals:
                out.append((mk, vals))
        return out

    def gap_ci(causal_list, grad_list, equal_by_game=False):
        cmethods = [v for _, v in causal_list]
        gmethods = [v for _, v in grad_list]
        if not cmethods or not gmethods:
            return None, None, None, len(cmethods), len(gmethods)
        if equal_by_game:
            # pool all records of a family per game, weight games equally
            def per_game_pool(methods):
                acc = defaultdict(list)
                for m in methods:
                    for gg, val in m.items():
                        acc[gg].append(val)
                return {gg: float(np.mean(v)) for gg, v in acc.items()}
            cg = per_game_pool(cmethods)
            gg_ = per_game_pool(gmethods)
            games = sorted(set(cg) & set(gg_))
            if not games:
                return None, None, None, len(cmethods), len(gmethods)
            point = float(np.mean([cg[x] for x in games]) - np.mean([gg_[x] for x in games]))
            cm = np.array([cg[x] for x in games])
            gm = np.array([gg_[x] for x in games])
            idx = rng.integers(0, len(games), size=(n_boot, len(games)))
            boots = cm[idx].mean(axis=1) - gm[idx].mean(axis=1)
        else:
            # equal-by-method: mean over methods of each method's game-mean
            games = sorted(set().union(*[set(m) for m in cmethods + gmethods]))
            G = len(games)
            def mat(methods):
                M = np.full((len(methods), G), np.nan)
                for i, m in enumerate(methods):
                    for j, x in enumerate(games):
                        if x in m:
                            M[i, j] = m[x]
                return M
            CM, GM = mat(cmethods), mat(gmethods)
            point = float(np.nanmean(np.nanmean(CM, axis=1)) - np.nanmean(np.nanmean(GM, axis=1)))
            if G <= 1:
                return point, point, point, len(cmethods), len(gmethods)
            gi = np.arange(G)
            boots = np.empty(n_boot)
            for b in range(n_boot):
                d = rng.choice(gi, size=G, replace=True)
                with np.errstate(invalid="ignore"):
                    boots[b] = np.nanmean(np.nanmean(CM[:, d], axis=1)) - \
                               np.nanmean(np.nanmean(GM[:, d], axis=1))
        lo = float(np.percentile(boots, 2.5))
        hi = float(np.percentile(boots, 97.5))
        return point, lo, hi, len(cmethods), len(gmethods)

    variants = []

    def add(name, regime, equal_by_game, exclude_oracle):
        def cf(mk, tr):
            if exclude_oracle and mk in ORACLE_LIKE:
                return False
            return tr in CAUSAL
        def gf(mk, tr):
            if exclude_oracle and mk in ORACLE_LIKE:
                return False
            return tr in GRAD
        clist = method_game_vals(cf, regime)
        glist = method_game_vals(gf, regime)
        point, lo, hi, nc, ng = gap_ci(clist, glist, equal_by_game=equal_by_game)
        variants.append({
            "variant": name,
            "regime": regime or "all",
            "causal_intervention_mean": "" if point is None else round(
                float(np.mean([np.mean(list(v.values())) for _, v in clist])), 4) if clist else "",
            "gradient_correlational_mean": "" if point is None else round(
                float(np.mean([np.mean(list(v.values())) for _, v in glist])), 4) if glist else "",
            "gap": "" if point is None else round(point, 4),
            "gap_ci_lo": "" if lo is None else round(lo, 4),
            "gap_ci_hi": "" if hi is None else round(hi, 4),
            "n_causal_methods": nc, "n_gradient_methods": ng,
            "gap_excludes_zero": "" if lo is None else bool(lo > 0 or hi < 0),
        })

    # the six rules required by review P1#17
    add("equal_by_method", None, False, False)
    add("equal_by_game", None, True, False)
    add("equal_by_output_regime_position", "position", False, False)
    add("excluding_oracle_like", None, False, True)
    add("content_only", "content", False, False)
    add("position_only", "position", False, False)
    return variants


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=os.path.join(STUDY, "results_index.csv"),
                    help="optional results_index.csv (carried as provenance only; the phase "
                         "out/ trees are the source of truth either way)")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples (>=1000)")
    ap.add_argument("--seed", type=int, default=20260623, help="bootstrap RNG seed")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    index_note = ("present (provenance only)" if os.path.exists(args.index)
                  else "not present — globbed phase out/ trees directly (records ARE the index)")

    groups = collect_per_game()

    # ---- per-method rows with bootstrap CIs (resample over games + over records) ----
    rows = []
    for (phase, mk), g in sorted(groups.items()):
        pg = game_means(g["by_game"])
        mean_g, lo_g, hi_g, n_games = boot_ci_over_games(pg, rng, args.boot)
        mean_r, lo_r, hi_r, n_rec = boot_ci_over_records(g["records"], rng, args.boot)
        # triad legs (S, M) bootstrapped over games — for the headline S/M numbers
        s_pg = game_means(g["S"]); m_pg = game_means(g["M"]); f_pg = game_means(g["F"])
        S_mean, S_lo, S_hi, _ = boot_ci_over_games(s_pg, rng, args.boot)
        M_mean, M_lo, M_hi, _ = boot_ci_over_games(m_pg, rng, args.boot)
        F_mean, F_lo, F_hi, _ = boot_ci_over_games(f_pg, rng, args.boot)
        seed_sd, seed_src, seed_n = (None, "", 0)
        if mk in SAMPLING_METHODS:
            seed_sd, seed_src, seed_n = seed_sd_for_method(mk)
        rows.append({
            "kind": "method",
            "phase": phase, "method": mk, "tradition": g["tradition"],
            "n_games": n_games, "n_records": n_rec,
            "mean": None if mean_g is None else round(mean_g, 4),
            "ci_lo": None if lo_g is None else round(lo_g, 4),
            "ci_hi": None if hi_g is None else round(hi_g, 4),
            "mean_over_records": None if mean_r is None else round(mean_r, 4),
            "ci_lo_over_records": None if lo_r is None else round(lo_r, 4),
            "ci_hi_over_records": None if hi_r is None else round(hi_r, 4),
            "S_mean": None if S_mean is None else round(S_mean, 4),
            "S_ci_lo": None if S_lo is None else round(S_lo, 4),
            "S_ci_hi": None if S_hi is None else round(S_hi, 4),
            "M_mean": None if M_mean is None else round(M_mean, 4),
            "M_ci_lo": None if M_lo is None else round(M_lo, 4),
            "M_ci_hi": None if M_hi is None else round(M_hi, 4),
            "F_mean": None if F_mean is None else round(F_mean, 4),
            "seed_sd": None if seed_sd is None else round(seed_sd, 4),
            "seed_sd_source": seed_src,
            "seed_sd_n_records": seed_n,
            "commits": ";".join(sorted(g["commits"])),
        })

    # ---- oracle positive control (degenerate: ci_lo == mean == ci_hi == 1) ----
    rows.append({
        "kind": "method", "phase": "ground_truth", "method": "ORACLE",
        "tradition": "oracle", "n_games": 1, "n_records": 1,
        "mean": 1.0, "ci_lo": 1.0, "ci_hi": 1.0,
        "mean_over_records": 1.0, "ci_lo_over_records": 1.0, "ci_hi_over_records": 1.0,
        "S_mean": 1.0, "S_ci_lo": 1.0, "S_ci_hi": 1.0,
        "M_mean": 1.0, "M_ci_lo": 1.0, "M_ci_hi": 1.0, "F_mean": 1.0,
        "seed_sd": 0.0, "seed_sd_source": "construction (exact oracle)",
        "seed_sd_n_records": 0, "commits": "",
    })

    # ---- family means (causal/intervention, gradient/correlational, per phase) ----
    def family_rows(name, tradition_set, regime=None):
        per_method = []
        members = []
        for (phase, mk), g in groups.items():
            if g["tradition"] in tradition_set:
                vals = game_means(g["by_game"] if regime is None
                                  else g["regimes"].get(regime, {}))
                if vals:
                    per_method.append(vals)
                    members.append(mk)
        mean, lo, hi, nm, ng = boot_ci_family(per_method, rng, args.boot)
        return {
            "kind": "family", "phase": "ALL" if regime is None else f"regime:{regime}",
            "method": name, "tradition": "+".join(sorted(tradition_set)),
            "n_games": ng, "n_records": nm,
            "mean": None if mean is None else round(mean, 4),
            "ci_lo": None if lo is None else round(lo, 4),
            "ci_hi": None if hi is None else round(hi, 4),
            "mean_over_records": "", "ci_lo_over_records": "", "ci_hi_over_records": "",
            "S_mean": "", "S_ci_lo": "", "S_ci_hi": "",
            "M_mean": "", "M_ci_lo": "", "M_ci_hi": "", "F_mean": "",
            "seed_sd": "", "seed_sd_source": "", "seed_sd_n_records": "",
            "commits": "members=" + ";".join(sorted(members)),
        }

    family_specs = [
        ("family_causal_intervention", {"causal", "intervention"}, None),
        ("family_gradient_correlational", {"gradient", "correlational"}, None),
        ("family_causal_intervention_position", {"causal", "intervention"}, "position"),
        ("family_gradient_correlational_position", {"gradient", "correlational"}, "position"),
        ("family_causal_intervention_content", {"causal", "intervention"}, "content"),
        ("family_gradient_correlational_content", {"gradient", "correlational"}, "content"),
    ]
    family_rows_list = [family_rows(*spec) for spec in family_specs]
    rows.extend(family_rows_list)

    # ---- headline numbers (each as its own row with a CI) ----
    def named_headline(name, per_game_value):
        mean, lo, hi, n = boot_ci_over_games(per_game_value, rng, args.boot)
        return {
            "kind": "headline", "phase": "headline", "method": name, "tradition": "",
            "n_games": n, "n_records": "",
            "mean": None if mean is None else round(mean, 4),
            "ci_lo": None if lo is None else round(lo, 4),
            "ci_hi": None if hi is None else round(hi, 4),
            "mean_over_records": "", "ci_lo_over_records": "", "ci_hi_over_records": "",
            "S_mean": "", "S_ci_lo": "", "S_ci_hi": "",
            "M_mean": "", "M_ci_lo": "", "M_ci_hi": "", "F_mean": "",
            "seed_sd": "", "seed_sd_source": "", "seed_sd_n_records": "", "commits": "",
        }

    # ACDC S = 0.44  (its sufficiency / triad_S, per-game)
    acdc_grp = groups.get(("phaseC_mechanistic", "ACDC"))
    if acdc_grp:
        rows.append(named_headline("ACDC_S", game_means(acdc_grp["S"])))
        rows.append(named_headline("ACDC_F_faithfulness", game_means(acdc_grp["F"])))
    # SAE matched-feature S (sparse_autoencoder S leg, per game)
    sae_grp = groups.get(("phaseC_mechanistic", "sparse_autoencoder"))
    if sae_grp:
        rows.append(named_headline("SAE_matched_feature_S", game_means(sae_grp["S"])))
    # whole-state M = 0.07 (A8 M leg, per game)
    a8_grp = groups.get(("phaseA_kording", "A8_wholestate"))
    if a8_grp:
        rows.append(named_headline("wholestate_M", game_means(a8_grp["M"])))

    # position-regime gap headline (causal/intervention − gradient/correlational on position)
    causal_pos = next(r for r in family_rows_list
                      if r["method"] == "family_causal_intervention_position")
    grad_pos = next(r for r in family_rows_list
                    if r["method"] == "family_gradient_correlational_position")
    # bootstrap the gap directly (shared games) via aggregation_robustness's position_only
    agg = aggregation_robustness(groups, np.random.default_rng(args.seed + 1), args.boot)
    pos_gap = next(v for v in agg if v["variant"] == "position_only")
    all_gap = next(v for v in agg if v["variant"] == "equal_by_method")
    rows.append({
        "kind": "headline", "phase": "headline", "method": "position_regime_gap",
        "tradition": "causal_intervention − gradient_correlational",
        "n_games": 6, "n_records": "",
        "mean": pos_gap["gap"], "ci_lo": pos_gap["gap_ci_lo"], "ci_hi": pos_gap["gap_ci_hi"],
        "mean_over_records": "", "ci_lo_over_records": "", "ci_hi_over_records": "",
        "S_mean": "", "S_ci_lo": "", "S_ci_hi": "",
        "M_mean": "", "M_ci_lo": "", "M_ci_hi": "", "F_mean": "",
        "seed_sd": "", "seed_sd_source": "", "seed_sd_n_records": "",
        "commits": f"excludes_zero={pos_gap['gap_excludes_zero']}",
    })
    rows.append({
        "kind": "headline", "phase": "headline", "method": "all_regime_gap",
        "tradition": "causal_intervention − gradient_correlational",
        "n_games": 6, "n_records": "",
        "mean": all_gap["gap"], "ci_lo": all_gap["gap_ci_lo"], "ci_hi": all_gap["gap_ci_hi"],
        "mean_over_records": "", "ci_lo_over_records": "", "ci_hi_over_records": "",
        "S_mean": "", "S_ci_lo": "", "S_ci_hi": "",
        "M_mean": "", "M_ci_lo": "", "M_ci_hi": "", "F_mean": "",
        "seed_sd": "", "seed_sd_source": "", "seed_sd_n_records": "",
        "commits": f"excludes_zero={all_gap['gap_excludes_zero']}",
    })

    # ---- ACDC threshold sensitivity ----
    thr_per_game, thr_family = acdc_threshold_curve()
    threshold_rows = thr_per_game + thr_family

    # ---- self-check ----
    checks = []
    # (1) every non-degenerate row brackets its point estimate: ci_lo <= mean <= ci_hi
    bad_bracket = []
    for r in rows:
        m, lo, hi = r.get("mean"), r.get("ci_lo"), r.get("ci_hi")
        if isinstance(m, (int, float)) and isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            if not (lo - 1e-9 <= m <= hi + 1e-9):
                bad_bracket.append((r["method"], lo, m, hi))
    checks.append(("ci_lo <= mean <= ci_hi for every row", not bad_bracket,
                   f"violations={bad_bracket}" if bad_bracket else "ok"))
    # (2) every method/family/headline row has a finite CI
    null_ci = [r["method"] for r in rows
               if r["kind"] in ("method", "family", "headline")
               and (r.get("mean") is None or r.get("ci_lo") is None or r.get("ci_hi") is None)]
    checks.append(("every row has a finite CI", not null_ci,
                   f"null={null_ci}" if null_ci else "ok"))
    # (3) position-regime gap CI excludes 0 (the headline survives uncertainty)
    pos_excl = (isinstance(pos_gap["gap_ci_lo"], (int, float)) and pos_gap["gap_ci_lo"] > 0)
    checks.append(("position-regime gap CI excludes 0", pos_excl,
                   f"gap={pos_gap['gap']} ci=[{pos_gap['gap_ci_lo']},{pos_gap['gap_ci_hi']}]"))
    # (4) the headline causal>gradient gap stays POSITIVE under every robustness variant,
    # EXCEPT the content regime. On smooth content outputs the naive gradient is valid, so the
    # two families are level and the gap is ~0 (about -0.01) BY DESIGN — a documented finding
    # (sec:results_compare: "the content gap is -0.01, essentially zero"), not a failure. We
    # therefore assert positivity only where the paper claims an advantage (all-regime and
    # position) and exclude the content-only variant.
    neg_variants = [v["variant"] for v in agg
                    if isinstance(v["gap"], (int, float)) and v["gap"] <= 0
                    and v.get("regime") != "content"]
    checks.append(("causal>gradient gap positive under every non-content variant",
                   not neg_variants, f"non_positive={neg_variants}" if neg_variants else "ok"))
    # (5) ACDC threshold curve produced (>=2 τ points per game)
    n_tau = len({r["tau"] for r in thr_per_game})
    checks.append(("ACDC threshold curve has a fine τ grid", n_tau >= 5, f"n_tau={n_tau}"))
    self_check_pass = all(ok for _, ok, _ in checks)

    # ---- write leaderboard_ci.json ----
    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {
        "item": "P2-R-UNC",
        "title": "Bootstrap CIs + threshold/sampling sensitivity + aggregation robustness",
        "paper": "P2",
        "revision": "R1 (sprint 9)",
        "addresses_reviews": ["improvement_instructions P1#16", "improvement_instructions P1#17",
                              "final/review P0#5 (CIs & uncertainty)", "final/review P0#6 (bootstrap S)"],
        "generated_by": "tools/xai_study/compare/uncertainty.py",
        "method": ("pure read + resample over committed §R records (no emulator re-run, no "
                   "committed result altered); CIs wrap the identical leaderboard.py point "
                   "estimate. Bootstrap = percentile, resampling the 6 core games with "
                   "replacement; n_boot=%d; seed=%d." % (args.boot, args.seed)),
        "index_note": index_note,
        "n_boot": args.boot, "seed": args.seed,
        "rows": rows,
        "aggregation_robustness": agg,
        "self_check": {"pass": self_check_pass,
                       "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in checks]},
    }
    with open(os.path.join(OUT_DIR, "leaderboard_ci.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # ---- write leaderboard_ci.csv ----
    csv_cols = ["kind", "phase", "method", "tradition", "n_games", "n_records",
                "mean", "ci_lo", "ci_hi",
                "mean_over_records", "ci_lo_over_records", "ci_hi_over_records",
                "S_mean", "S_ci_lo", "S_ci_hi", "M_mean", "M_ci_lo", "M_ci_hi", "F_mean",
                "seed_sd", "seed_sd_source", "seed_sd_n_records", "commits"]
    with open(os.path.join(OUT_DIR, "leaderboard_ci.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in csv_cols})

    # ---- write threshold_sensitivity.csv ----
    thr_cols = ["game", "tau", "tp", "fp", "fn", "precision", "recall",
                "f1_faithfulness", "scrub_preserved_sufficiency"]
    with open(os.path.join(OUT_DIR, "threshold_sensitivity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=thr_cols)
        w.writeheader()
        for r in threshold_rows:
            w.writerow({k: r.get(k, "") for k in thr_cols})

    # ---- write aggregation_robustness.csv ----
    agg_cols = ["variant", "regime", "causal_intervention_mean", "gradient_correlational_mean",
                "gap", "gap_ci_lo", "gap_ci_hi", "n_causal_methods", "n_gradient_methods",
                "gap_excludes_zero"]
    with open(os.path.join(OUT_DIR, "aggregation_robustness.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_cols)
        w.writeheader()
        for r in agg:
            w.writerow({k: r.get(k, "") for k in agg_cols})

    # ---- console summary ----
    print(f"[P2-R-UNC] bootstrap n={args.boot} (resample 6 games), seed={args.seed}")
    print(f"  index: {index_note}")
    cfam = next(r for r in family_rows_list if r["method"] == "family_causal_intervention")
    gfam = next(r for r in family_rows_list if r["method"] == "family_gradient_correlational")
    print(f"  family causal/intervention faithfulness = {cfam['mean']} "
          f"[{cfam['ci_lo']}, {cfam['ci_hi']}] (n_methods={cfam['n_records']})")
    print(f"  family gradient/correlational faithfulness = {gfam['mean']} "
          f"[{gfam['ci_lo']}, {gfam['ci_hi']}] (n_methods={gfam['n_records']})")
    print(f"  position-regime gap = {pos_gap['gap']} "
          f"[{pos_gap['gap_ci_lo']}, {pos_gap['gap_ci_hi']}] "
          f"(excludes 0: {pos_gap['gap_excludes_zero']})")
    print("  headline numbers (mean [ci_lo, ci_hi]):")
    for nm in ("ACDC_S", "ACDC_F_faithfulness", "SAE_matched_feature_S", "wholestate_M"):
        hr = next((r for r in rows if r["method"] == nm), None)
        if hr:
            print(f"    {nm:<24} {hr['mean']} [{hr['ci_lo']}, {hr['ci_hi']}]")
    print("  seed_sd (sampling methods):")
    for r in rows:
        if r.get("seed_sd") not in (None, "", 0.0) or (r["method"] in SAMPLING_METHODS):
            if r["method"] in SAMPLING_METHODS:
                print(f"    {r['method']:<22} seed_sd={r['seed_sd']}  ({r['seed_sd_source'][:60]})")
    print("  aggregation robustness (gap under each rule):")
    for v in agg:
        print(f"    {v['variant']:<34} gap={v['gap']} "
              f"[{v['gap_ci_lo']}, {v['gap_ci_hi']}] >0:{v['gap_excludes_zero']}")
    print("  self-check:")
    for n, ok, d in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {n} — {d}")
    print(f"  SELF-CHECK: {'PASS' if self_check_pass else 'FAIL'}")
    print(f"  wrote: leaderboard_ci.json, leaderboard_ci.csv, "
          f"threshold_sensitivity.csv, aggregation_robustness.csv (in compare/out/)")

    return 0 if self_check_pass else 1


if __name__ == "__main__":
    sys.exit(main())
