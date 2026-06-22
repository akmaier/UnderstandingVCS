#!/usr/bin/env python3
"""P2-E6-1 — Cross-tradition leaderboard (faithfulness vs plausibility).

A *pure read* over every committed §R record under
``tools/xai_study/{phaseA_kording,phaseB_attribution,phaseC_mechanistic}/out/*.json``
(SPEC §R schema). It places every Phase-A/B/C method on the two shared reporting axes of
``experiment_design.md`` §0 — **faithfulness** (X, scored against the §1 oracle ground
truth) vs **human-plausibility** (Y, a documented method-tradition proxy) — and emits one
comparable F∧S(∧M) score per method plus the discrepancy table that highlights the danger
zone (high plausibility, low faithfulness).

It does **not** re-run any experiment and does **not** write ``results_index.csv`` (the
index is SM-merged at Review). If the index exists it is used only as an optional source
list; otherwise the phase ``out/`` trees are globbed directly (the records ARE the index).

Run:
    python tools/xai_study/compare/leaderboard.py
    python tools/xai_study/compare/leaderboard.py --index tools/xai_study/results_index.csv

Outputs (to ``tools/xai_study/compare/out/``):
    leaderboard.json  — per-(phase,method) rows + per-phase rollups + headline contrast
    leaderboard.csv   — tidy one-row-per-(phase,method) leaderboard
    leaderboard.md    — short human-readable summary

----------------------------------------------------------------------------------------
Scoring conventions (every number traces to a committed §R record):

  Faithfulness (X, [0,1], higher = more faithful to the oracle):
    - Preferred source: each per-game record's ``extra.triad.F`` (or top-level
      ``mean_triad_F``) — the F leg of the correctness triad, which is ALREADY oriented
      "vs the §1 oracle, higher=better". Correlation-style F legs are clipped to [0,1]
      (a negative/anti-correlated map is no more faithful than a random one → 0).
    - Fallback (records with no triad block): the record's ``value``, re-oriented to
      "faithfulness vs oracle" via FAITHFULNESS_ORIENT below. Each entry there is keyed
      by the record's own ``metric_name`` and documents the orientation, so nothing is
      invented — we only re-orient committed numbers.

  Plausibility (Y, [0,1], higher = more "convincing-looking to a human"):
    - There is NO measured plausibility field in the records (a human-rating study is out
      of scope for P2). We therefore report a TRANSPARENT, method-tradition PROXY derived
      from §7's method matrix: methods that emit a clean, eye-pleasing artefact (saliency/
      gradient heatmaps; monosemantic SAE/dictionary stories; tidy probe accuracies) score
      high on *perceived* plausibility regardless of faithfulness — that is exactly the
      §0 "danger zone" the headline plot is built to expose. The proxy is a fixed prior
      per method (PLAUSIBILITY_PROXY), clearly labelled ``plausibility_proxy`` everywhere
      so it is never mistaken for a measurement.

  F / S / M triad: mean over the method's per-game ``extra.triad`` blocks (or the
    method's ``mean_triad_*`` summary). Correlation-style legs clipped to [0,1] for the
    composite; raw means also reported.

  Positive control (oracle-as-method): the ground-truth oracle, entered as a method with
    faithfulness == 1 by construction (it IS the oracle). Documented in the Phase-B
    counterfactual summary: "Oracle-as-method positive control corr=1/p@k=1".
----------------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)  # tools/xai_study
PHASE_DIRS = {
    "A": os.path.join(STUDY, "phaseA_kording", "out"),
    "B": os.path.join(STUDY, "phaseB_attribution", "out"),
    "C": os.path.join(STUDY, "phaseC_mechanistic", "out"),
}
PHASE_LONG = {
    "A": "phaseA_kording",
    "B": "phaseB_attribution",
    "C": "phaseC_mechanistic",
}
GROUND_TRUTH_DIR = os.path.join(STUDY, "ground_truth", "out")
OUT_DIR = os.path.join(HERE, "out")

# ----------------------------------------------------------------------------------------
# Method tradition labels (from experiment_design.md §7 method matrix + §4/§5/§6).
# tradition ∈ {causal, intervention, gradient, correlational, dim_reduction, probing,
#              descriptive, audit, oracle}.
# The "regime contrast" of interest is causal/intervention (near-ceiling) vs
# gradient/correlational (near-chance on the position/index outputs).
# ----------------------------------------------------------------------------------------
TRADITION = {
    # --- Phase A (Kording / neuroscience battery) ---
    "A1_connectomics": "intervention",                 # single-shot perturbation graph
    "A2_lesions": "intervention",                      # exhaustive single-unit lesions
    "A3_tuning": "correlational",                      # tuning curves (present≠used trap)
    "A4_spike_word": "correlational",                  # pairwise correlations
    "A5_local_field_potentials": "correlational",      # pooled-activity spectra (epiphenom.)
    "A6_granger": "correlational",                     # Granger (precedence≠causation)
    "A7_dim_reduction": "dim_reduction",               # NMF+PCA latent components
    "A8_wholestate": "descriptive",                    # record-everything baseline
    # --- Phase B (attribution / XAI) ---
    "vanilla_saliency": "gradient",
    "gradxinput_deeplift": "gradient",
    "guided_backprop": "gradient",
    "smoothgrad": "gradient",
    "integrated_gradients": "gradient",
    "expected_gradients": "gradient",
    "occlusion": "intervention",
    "extremal_perturbation": "intervention",
    "rise": "gradient",                                # random-masking saliency surrogate
    "lime": "gradient",                                # local linear surrogate (heatmap)
    "kernelshap": "gradient",                          # Shapley surrogate (heatmap)
    "on_distribution_counterfactual": "intervention",  # set-state-and-re-render
    "na_audit": "audit",                               # NN/policy-XAI do-not-apply writeup
    # --- Phase C (mechanistic) ---
    "activation_patching": "causal",
    "interchange_interventions_das": "causal",
    "attribution_patching": "gradient",                # cheap gradient approx of patching
    "path_patching": "causal",
    "ACDC": "causal",
    "sparse_autoencoder": "dim_reduction",
    "nmf_pca_dictionaries": "dim_reduction",
    "causal_scrubbing": "causal",
    "linear_probing_control_tasks": "probing",
    "logit_tuned_lens": "causal",                      # exact intermediate readout
    # --- positive control ---
    "ORACLE": "oracle",
}

# Plausibility PROXY per tradition (Y axis, [0,1]). A documented prior, NOT a measurement.
# Rationale: methods that produce a polished, intuitive artefact look convincing to a
# human reader irrespective of whether they recover the true cause (§0 danger zone).
PLAUSIBILITY_BY_TRADITION = {
    "gradient": 0.90,        # crisp heatmaps — the canonical "looks right" output
    "correlational": 0.80,   # tuning curves / correlation matrices read as mechanism
    "dim_reduction": 0.75,   # "interpretable" latent atoms / monosemantic features
    "probing": 0.85,         # a high decoding accuracy reads as "the concept is there"
    "intervention": 0.55,    # perturbation maps are plausible but less visually slick
    "causal": 0.50,          # circuits/patches are correct but read as technical
    "descriptive": 0.30,     # a full state dump explains nothing on its face
    "audit": float("nan"),   # not an explanation method (applicability writeup)
    "oracle": 1.00,          # ground truth: maximally faithful AND, trivially, plausible
}

# Faithfulness orientation for records that carry NO triad block. Keyed by the record's
# own metric_name. value -> faithfulness ([0,1], higher=better-vs-oracle). Every formula
# only re-orients a committed number; it invents nothing.
#   "clip01"   : corr/F1/fraction already higher=better → clip to [0,1]
#   "one_minus": an error/leakage rate where lower=better → 1 - clip01(value)
#   "exact_err": absolute recovery error vs exact (0 == perfect) → 1 if ==0 else 1-clip
#   "na"       : not a faithfulness number (audit) → excluded from the X axis
FAITHFULNESS_ORIENT = {
    "pearson_corr_with_oracle": "clip01",
    "deeplift_pearson_corr_with_oracle": "clip01",
    "corr_approx_vs_exact": "clip01",
    "interchange_accuracy_aligned": "clip01",
    "logit_lens_fidelity_true_intermediate": "clip01",
    "scrubbing_preserved_performance_true": "clip01",
    "path_circuit_F1_vs_true_routine": "clip01",
    "feature_variable_matched_fraction": "clip01",
    "nmf_matched_component_fraction_vs_known_vars": "clip01",
    "A7_nmf_matched_component_fraction_vs_known_vars": "clip01",
    "mean_selectivity": "clip01",                      # probe selectivity vs control task
    "A5_global_pool_clock_explained_variance_fraction_epiphenomenal": "one_minus",  # epiphenomenal share
    "max_abs_recovered_minus_exact": "exact_err",
    "n_methods_not_applicable": "na",
}


def clip01(x: float) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    return max(0.0, min(1.0, float(x)))


def orient_faithfulness(metric_name, value):
    """Re-orient a no-triad record's value to faithfulness-vs-oracle in [0,1]."""
    rule = FAITHFULNESS_ORIENT.get(metric_name)
    if rule is None or value is None:
        return None, rule
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None, rule
    if rule == "clip01":
        return clip01(v), rule
    if rule == "one_minus":
        return clip01(1.0 - v), rule
    if rule == "exact_err":
        return (1.0 if abs(v) < 1e-12 else clip01(1.0 - abs(v))), rule
    if rule == "na":
        return None, rule
    return None, rule


def method_key(rec):
    """Canonical short method id used for TRADITION/plausibility lookup."""
    m = (rec.get("method") or "").strip()
    # normalise the verbose method strings to a stable key
    table = [
        ("A1_connectomics", "A1_connectomics"),
        ("A2_lesions", "A2_lesions"),
        ("A3_tuning", "A3_tuning"),
        ("A4_spike_word", "A4_spike_word"),
        ("A5_local_field_potentials", "A5_local_field_potentials"),
        ("A6_granger", "A6_granger"),
        ("A7_dim_reduction", "A7_dim_reduction"),
        ("A8_wholestate", "A8_wholestate"),
        ("path_patching", "path_patching"),
        ("ACDC", "ACDC"),
        ("nmf_pca_dictionaries", "nmf_pca_dictionaries"),
        ("interchange_interventions_das", "interchange_interventions_das"),
        ("logit_tuned_lens", "logit_tuned_lens"),
        ("linear_probing_control_tasks", "linear_probing_control_tasks"),
        ("sparse_autoencoder", "sparse_autoencoder"),
        ("activation_patching", "activation_patching"),
        ("attribution_patching", "attribution_patching"),
        ("causal_scrubbing", "causal_scrubbing"),
        ("on_distribution_counterfactual", "on_distribution_counterfactual"),
        ("extremal_perturbation", "extremal_perturbation"),
        ("vanilla_saliency", "vanilla_saliency"),
        ("guided_backprop", "guided_backprop"),
        ("gradxinput_deeplift", "gradxinput_deeplift"),
        ("smoothgrad", "smoothgrad"),
        ("integrated_gradients", "integrated_gradients"),
        ("expected_gradients", "expected_gradients"),
        ("occlusion", "occlusion"),
        ("kernelshap", "kernelshap"),
        ("lime", "lime"),
        ("rise", "rise"),
        ("na_audit", "na_audit"),
    ]
    for needle, key in table:
        if m.startswith(needle) or needle in m:
            return key
    return m or "UNKNOWN"


def output_regime(rec):
    """content | position | other — from target_output / filename token."""
    t = (rec.get("target_output") or "").lower()
    st = (rec.get("state") or "")
    blob = (t + " " + str(st)).lower()
    if "ball_pixel" in blob or "position" in blob:
        return "position"
    if blob.startswith("content") or "content(" in blob or "content:" in blob or "colup" in blob:
        return "content"
    return "other"


def is_summary_record(path, rec):
    bn = os.path.basename(path)
    g = rec.get("game")
    return (
        "per_game" in rec
        or "core_summary" in bn
        or bn.endswith("_all.json")
        or bn.endswith("_combined.json")
        or g in ("core_set", "ALL_CORE", "combined", "all", None)
    )


def load_per_game_records():
    """Yield (phase, path, rec) for every per-game §R record across A/B/C."""
    for phase, d in PHASE_DIRS.items():
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                rec = json.load(open(path))
            except Exception as e:  # pragma: no cover
                print(f"WARN: could not parse {path}: {e}", file=sys.stderr)
                continue
            if not isinstance(rec, dict):
                continue
            if is_summary_record(path, rec):
                continue
            yield phase, path, rec


def triad_of(rec):
    """Return (F,S,M) for a per-game record, or (None,None,None)."""
    ex = rec.get("extra") or {}
    tri = ex.get("triad")
    if isinstance(tri, dict) and all(k in tri for k in ("F", "S", "M")):
        return tri.get("F"), tri.get("S"), tri.get("M")
    if all(k in rec for k in ("mean_triad_F", "mean_triad_S", "mean_triad_M")):
        return rec["mean_triad_F"], rec["mean_triad_S"], rec["mean_triad_M"]
    return None, None, None


def mean_ci(vals):
    """Mean and 95% normal-approx CI half-width over a list of floats (NaNs dropped)."""
    xs = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(xs)
    if n == 0:
        return None, None, 0
    mu = sum(xs) / n
    if n == 1:
        return mu, 0.0, 1
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    half = 1.96 * math.sqrt(var / n)
    return mu, half, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=os.path.join(STUDY, "results_index.csv"),
                    help="optional results_index.csv (used as a source list if present); "
                         "the phase out/ trees are the source of truth either way")
    args = ap.parse_args()

    index_note = "not present — globbed phase out/ trees directly (records ARE the index)"
    if os.path.exists(args.index):
        index_note = f"present at {os.path.relpath(args.index, STUDY)} (phase out/ trees still globbed as source of truth)"

    # ---- collect per-game rows, grouped by (phase, method) ----
    groups = defaultdict(lambda: {
        "faith": [], "F": [], "S": [], "M": [],
        "games": set(), "regimes": defaultdict(list), "records": [],
        "metric_names": set(), "faith_source": set(), "commits": set(),
    })
    skipped_na = []

    for phase, path, rec in load_per_game_records():
        mk = method_key(rec)
        if mk == "na_audit":
            skipped_na.append(os.path.basename(path))
            continue
        g = groups[(phase, mk)]
        g["games"].add(rec.get("game"))
        g["records"].append(os.path.basename(path))
        g["metric_names"].add(rec.get("metric_name"))
        if rec.get("commit"):
            g["commits"].add(rec.get("commit"))

        F, S, M = triad_of(rec)
        regime = output_regime(rec)

        # faithfulness for this record
        if F is not None:
            faith = clip01(F)
            g["faith_source"].add("triad.F")
        else:
            faith, _rule = orient_faithfulness(rec.get("metric_name"), rec.get("value"))
            if faith is not None:
                g["faith_source"].add(f"value:{rec.get('metric_name')}")
        if faith is not None:
            g["faith"].append(faith)
            g["regimes"][regime].append(faith)
        if F is not None:
            g["F"].append(clip01(F))
        if S is not None:
            g["S"].append(clip01(S))
        if M is not None:
            g["M"].append(clip01(M))

    # ---- build per-method leaderboard rows ----
    rows = []
    for (phase, mk), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        tradition = TRADITION.get(mk, "unknown")
        plaus = PLAUSIBILITY_BY_TRADITION.get(tradition, float("nan"))
        f_mu, f_ci, f_n = mean_ci(g["faith"])
        F_mu, _, _ = mean_ci(g["F"])
        S_mu, _, _ = mean_ci(g["S"])
        M_mu, _, _ = mean_ci(g["M"])
        # composite F∧S∧M (geometric-mean of the three legs where all present, else mean)
        legs = [x for x in (F_mu, S_mu, M_mu) if x is not None]
        if len(legs) == 3 and all(x >= 0 for x in legs):
            fsm = (legs[0] * legs[1] * legs[2]) ** (1.0 / 3.0)
        elif legs:
            fsm = sum(legs) / len(legs)
        else:
            fsm = None
        pos_faith = mean_ci(g["regimes"].get("position", []))[0]
        con_faith = mean_ci(g["regimes"].get("content", []))[0]
        other_faith = mean_ci(g["regimes"].get("other", []))[0]
        rows.append({
            "phase": PHASE_LONG[phase],
            "phase_code": phase,
            "method": mk,
            "tradition": tradition,
            "n_games": len([x for x in g["games"] if x]),
            "n_records": len(g["records"]),
            "faithfulness": None if f_mu is None else round(f_mu, 4),
            "faithfulness_ci95": None if f_ci is None else round(f_ci, 4),
            "faithfulness_n": f_n,
            "faithfulness_position_regime": None if pos_faith is None else round(pos_faith, 4),
            "faithfulness_content_regime": None if con_faith is None else round(con_faith, 4),
            "faithfulness_other_regime": None if other_faith is None else round(other_faith, 4),
            "plausibility_proxy": None if (isinstance(plaus, float) and math.isnan(plaus)) else plaus,
            "triad_F": None if F_mu is None else round(F_mu, 4),
            "triad_S": None if S_mu is None else round(S_mu, 4),
            "triad_M": None if M_mu is None else round(M_mu, 4),
            "FSM_composite": None if fsm is None else round(fsm, 4),
            "is_positive_control": 0,
            "faith_source": ";".join(sorted(g["faith_source"])) or "none",
            "metric_names": ";".join(sorted(str(x) for x in g["metric_names"] if x)),
            "commits": ";".join(sorted(g["commits"])),
        })

    # ---- positive control: oracle-as-method (faithfulness == 1 by construction) ----
    oracle_records = []
    oracle_commits = set()
    if os.path.isdir(GROUND_TRUTH_DIR):
        for p in sorted(glob.glob(os.path.join(GROUND_TRUTH_DIR, "*.json"))):
            try:
                r = json.load(open(p))
            except Exception:
                continue
            if isinstance(r, dict):
                oracle_records.append(os.path.basename(p))
                if r.get("commit"):
                    oracle_commits.add(r["commit"])
    rows.append({
        "phase": "ground_truth",
        "phase_code": "GT",
        "method": "ORACLE",
        "tradition": "oracle",
        "n_games": 1,
        "n_records": len(oracle_records),
        "faithfulness": 1.0,
        "faithfulness_ci95": 0.0,
        "faithfulness_n": len(oracle_records),
        "faithfulness_position_regime": 1.0,
        "faithfulness_content_regime": 1.0,
        "faithfulness_other_regime": None,
        "plausibility_proxy": PLAUSIBILITY_BY_TRADITION["oracle"],
        "triad_F": 1.0, "triad_S": 1.0, "triad_M": 1.0, "FSM_composite": 1.0,
        "is_positive_control": 1,
        "faith_source": "construction (oracle IS the §1 ground truth; "
                        "counterfactual_core_summary: oracle-as-method corr=1/p@k=1)",
        "metric_names": "intervention_oracle;gradient_oracle_content_path;oracle_xcheck",
        "commits": ";".join(sorted(oracle_commits)),
    })

    # ---- per-phase rollups (means ± CI over the phase's methods) ----
    phase_rollups = {}
    for phase in ("A", "B", "C"):
        ph_rows = [r for r in rows if r["phase_code"] == phase]
        f_mu, f_ci, f_n = mean_ci([r["faithfulness"] for r in ph_rows])
        p_mu, p_ci, p_n = mean_ci([r["plausibility_proxy"] for r in ph_rows])
        fsm_mu, fsm_ci, _ = mean_ci([r["FSM_composite"] for r in ph_rows])
        pos_mu, pos_ci, pos_n = mean_ci([r["faithfulness_position_regime"] for r in ph_rows])
        phase_rollups[PHASE_LONG[phase]] = {
            "n_methods": len(ph_rows),
            "faithfulness_mean": None if f_mu is None else round(f_mu, 4),
            "faithfulness_ci95": None if f_ci is None else round(f_ci, 4),
            "plausibility_proxy_mean": None if p_mu is None else round(p_mu, 4),
            "FSM_composite_mean": None if fsm_mu is None else round(fsm_mu, 4),
            "faithfulness_position_regime_mean": None if pos_mu is None else round(pos_mu, 4),
            "position_regime_n": pos_n,
        }

    # ---- headline contrast: causal/intervention vs gradient/correlational on POSITION ----
    def tradition_pos(traditions):
        vals = []
        for r in rows:
            if r["is_positive_control"]:
                continue
            if r["tradition"] in traditions and r["faithfulness_position_regime"] is not None:
                vals.append(r["faithfulness_position_regime"])
        return vals

    causal_pos = tradition_pos({"causal", "intervention"})
    grad_pos = tradition_pos({"gradient", "correlational"})
    c_mu, c_ci, c_n = mean_ci(causal_pos)
    g_mu, g_ci, g_n = mean_ci(grad_pos)

    # also an overall (all-regime) faithfulness contrast for the abstract
    def tradition_all(traditions):
        return [r["faithfulness"] for r in rows
                if (not r["is_positive_control"]) and r["tradition"] in traditions
                and r["faithfulness"] is not None]
    ca_mu, ca_ci, ca_n = mean_ci(tradition_all({"causal", "intervention"}))
    ga_mu, ga_ci, ga_n = mean_ci(tradition_all({"gradient", "correlational"}))

    headline = {
        "claim": "Intervention/causal methods are near-ceiling while gradient/correlational "
                 "methods collapse to near-chance on the position/index regime "
                 "(the discrete sprite-position outputs whose naive gradient is zero, §1).",
        "position_regime": {
            "causal_intervention_faithfulness_mean": None if c_mu is None else round(c_mu, 4),
            "causal_intervention_ci95": None if c_ci is None else round(c_ci, 4),
            "causal_intervention_n": c_n,
            "gradient_correlational_faithfulness_mean": None if g_mu is None else round(g_mu, 4),
            "gradient_correlational_ci95": None if g_ci is None else round(g_ci, 4),
            "gradient_correlational_n": g_n,
            "gap": None if (c_mu is None or g_mu is None) else round(c_mu - g_mu, 4),
        },
        "all_regimes": {
            "causal_intervention_faithfulness_mean": None if ca_mu is None else round(ca_mu, 4),
            "causal_intervention_n": ca_n,
            "gradient_correlational_faithfulness_mean": None if ga_mu is None else round(ga_mu, 4),
            "gradient_correlational_n": ga_n,
            "gap": None if (ca_mu is None or ga_mu is None) else round(ca_mu - ga_mu, 4),
        },
    }

    # ---- danger-zone discrepancy table (high plausibility, low faithfulness) ----
    danger = []
    for r in rows:
        if r["is_positive_control"] or r["plausibility_proxy"] is None or r["faithfulness"] is None:
            continue
        gap = round(r["plausibility_proxy"] - r["faithfulness"], 4)
        danger.append({
            "phase": r["phase"], "method": r["method"], "tradition": r["tradition"],
            "plausibility_proxy": r["plausibility_proxy"], "faithfulness": r["faithfulness"],
            "plausibility_minus_faithfulness": gap,
        })
    danger.sort(key=lambda d: d["plausibility_minus_faithfulness"], reverse=True)

    # ---- self-check ----
    checks = []
    # (1) every method present (no method dropped silently)
    all_methods = {r["method"] for r in rows}
    expected_methods = set(TRADITION) - {"na_audit"}  # na_audit is an audit, not a leaderboard method
    missing = sorted(expected_methods - all_methods)
    checks.append(("every expected method present", not missing,
                   f"missing={missing}" if missing else "ok"))
    # (2) no unexplained NaN/None in the faithfulness column (except the audit, excluded)
    nan_faith = [r["method"] for r in rows if r["faithfulness"] is None]
    checks.append(("no method with null faithfulness", not nan_faith,
                   f"null_faith={nan_faith}" if nan_faith else "ok"))
    # (3) positive-control column == 1
    pc = [r for r in rows if r["is_positive_control"] == 1]
    pc_ok = len(pc) == 1 and abs(pc[0]["faithfulness"] - 1.0) < 1e-9
    checks.append(("positive control present and == 1", pc_ok,
                   f"n_pc={len(pc)} faith={pc[0]['faithfulness'] if pc else None}"))
    # (4) headline gap is positive on the position regime (the whole point)
    pos_gap = headline["position_regime"]["gap"]
    checks.append(("position-regime gap (causal − gradient) > 0", (pos_gap or 0) > 0,
                   f"gap={pos_gap}"))
    # (5) plausibility proxy in [0,1] for every non-audit method
    bad_p = [r["method"] for r in rows
             if r["plausibility_proxy"] is not None and not (0.0 <= r["plausibility_proxy"] <= 1.0)]
    checks.append(("plausibility proxy in [0,1]", not bad_p,
                   f"bad={bad_p}" if bad_p else "ok"))
    self_check_pass = all(ok for _, ok, _ in checks)

    # ---- assemble + write ----
    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {
        "item": "P2-E6-1",
        "title": "Cross-tradition leaderboard (faithfulness vs plausibility)",
        "paper": "P2",
        "generated_by": "tools/xai_study/compare/leaderboard.py",
        "index_note": index_note,
        "axes": {
            "x_faithfulness": "vs the §1 oracle ground truth (triad.F preferred; "
                              "else value re-oriented via FAITHFULNESS_ORIENT). [0,1] higher=better.",
            "y_plausibility": "DOCUMENTED METHOD-TRADITION PROXY (PLAUSIBILITY_BY_TRADITION), "
                              "NOT a measurement — exposes the §0 danger zone. [0,1] higher=looks-more-convincing.",
        },
        "n_methods": len(rows),
        "n_per_game_records_aggregated": sum(r["n_records"] for r in rows if not r["is_positive_control"]),
        "na_audit_records_excluded": sorted(set(skipped_na)),
        "rows": rows,
        "phase_rollups": phase_rollups,
        "headline_contrast": headline,
        "danger_zone_discrepancy_table": danger,
        "self_check": {"pass": self_check_pass,
                       "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in checks]},
    }
    with open(os.path.join(OUT_DIR, "leaderboard.json"), "w") as f:
        json.dump(payload, f, indent=2)

    csv_cols = ["phase", "method", "tradition", "n_games", "n_records",
                "faithfulness", "faithfulness_ci95",
                "faithfulness_position_regime", "faithfulness_content_regime",
                "plausibility_proxy", "triad_F", "triad_S", "triad_M",
                "FSM_composite", "is_positive_control", "faith_source"]
    with open(os.path.join(OUT_DIR, "leaderboard.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in csv_cols})

    write_md(payload)

    # ---- console summary ----
    print(f"[P2-E6-1] aggregated {payload['n_per_game_records_aggregated']} per-game §R "
          f"records into {len(rows)} method rows (incl. 1 oracle positive control).")
    print(f"  index: {index_note}")
    hp = headline["position_regime"]
    print(f"  HEADLINE (position/index regime): causal/intervention "
          f"{hp['causal_intervention_faithfulness_mean']} (n={hp['causal_intervention_n']})  "
          f"vs gradient/correlational {hp['gradient_correlational_faithfulness_mean']} "
          f"(n={hp['gradient_correlational_n']})  gap={hp['gap']}")
    ha = headline["all_regimes"]
    print(f"  all-regime mean faithfulness: causal/intervention "
          f"{ha['causal_intervention_faithfulness_mean']} vs gradient/correlational "
          f"{ha['gradient_correlational_faithfulness_mean']}  gap={ha['gap']}")
    print("  top danger-zone (plausibility − faithfulness):")
    for d in danger[:5]:
        print(f"    {d['method']:<32} plaus={d['plausibility_proxy']} "
              f"faith={d['faithfulness']}  Δ={d['plausibility_minus_faithfulness']}")
    print("  self-check:")
    for n, ok, d in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {n} — {d}")
    print(f"  SELF-CHECK: {'PASS' if self_check_pass else 'FAIL'}")
    print(f"  wrote: {os.path.relpath(os.path.join(OUT_DIR,'leaderboard.json'), STUDY)}, "
          f"leaderboard.csv, leaderboard.md")

    return 0 if self_check_pass else 1


def write_md(payload):
    rows = payload["rows"]
    hp = payload["headline_contrast"]["position_regime"]
    ha = payload["headline_contrast"]["all_regimes"]
    lines = []
    lines.append("# Cross-tradition leaderboard — faithfulness vs plausibility (P2-E6-1)\n")
    lines.append("Pure read over every committed §R record under "
                 "`phaseA_kording/` · `phaseB_attribution/` · `phaseC_mechanistic/` `out/*.json`. "
                 "No experiment re-run. Generated by `compare/leaderboard.py`.\n")
    lines.append(f"- Records aggregated: **{payload['n_per_game_records_aggregated']}** per-game "
                 f"into **{payload['n_methods']}** method rows (incl. the oracle positive control).")
    lines.append(f"- Index: {payload['index_note']}")
    lines.append("")
    lines.append("## Axes")
    lines.append(f"- **Faithfulness (X)** — {payload['axes']['x_faithfulness']}")
    lines.append(f"- **Plausibility (Y)** — {payload['axes']['y_plausibility']}")
    lines.append("")
    lines.append("## Headline contrast")
    lines.append(f"On the **position/index regime** (discrete sprite-position outputs, naive "
                 f"gradient = 0): causal/intervention methods average "
                 f"**{hp['causal_intervention_faithfulness_mean']}** "
                 f"(±{hp['causal_intervention_ci95']}, n={hp['causal_intervention_n']}) "
                 f"faithfulness vs gradient/correlational methods at "
                 f"**{hp['gradient_correlational_faithfulness_mean']}** "
                 f"(±{hp['gradient_correlational_ci95']}, n={hp['gradient_correlational_n']}) — "
                 f"a gap of **{hp['gap']}**.")
    lines.append("")
    lines.append(f"Across **all output regimes**, causal/intervention average "
                 f"**{ha['causal_intervention_faithfulness_mean']}** (n={ha['causal_intervention_n']}) "
                 f"vs gradient/correlational **{ha['gradient_correlational_faithfulness_mean']}** "
                 f"(n={ha['gradient_correlational_n']}), gap **{ha['gap']}**.")
    lines.append("")
    lines.append("## Per-phase rollups (mean ± 95% CI over the phase's methods)")
    lines.append("| Phase | n methods | Faithfulness | Plausibility (proxy) | F∧S∧M | Faith (position regime) |")
    lines.append("|---|---|---|---|---|---|")
    for ph, r in payload["phase_rollups"].items():
        lines.append(f"| {ph} | {r['n_methods']} | {r['faithfulness_mean']} ± {r['faithfulness_ci95']} "
                     f"| {r['plausibility_proxy_mean']} | {r['FSM_composite_mean']} "
                     f"| {r['faithfulness_position_regime_mean']} (n={r['position_regime_n']}) |")
    lines.append("")
    lines.append("## Leaderboard (sorted by faithfulness, desc)")
    lines.append("| Phase | Method | Tradition | Faith | Faith(pos) | Plaus(proxy) | F | S | M | F∧S∧M | PC |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: (-(x["faithfulness"] if x["faithfulness"] is not None else -1))):
        lines.append("| {phase} | {method} | {tradition} | {faithfulness} | "
                     "{faithfulness_position_regime} | {plausibility_proxy} | {triad_F} | "
                     "{triad_S} | {triad_M} | {FSM_composite} | {pc} |".format(
                         pc="✔" if r["is_positive_control"] else "", **r))
    lines.append("")
    lines.append("## Danger zone — high plausibility, low faithfulness (top 10)")
    lines.append("| Phase | Method | Tradition | Plaus(proxy) | Faith | Δ (plaus − faith) |")
    lines.append("|---|---|---|---|---|---|")
    for d in payload["danger_zone_discrepancy_table"][:10]:
        lines.append(f"| {d['phase']} | {d['method']} | {d['tradition']} | {d['plausibility_proxy']} "
                     f"| {d['faithfulness']} | {d['plausibility_minus_faithfulness']} |")
    lines.append("")
    sc = payload["self_check"]
    lines.append(f"## Self-check: {'PASS' if sc['pass'] else 'FAIL'}")
    for c in sc["checks"]:
        lines.append(f"- [{'x' if c['pass'] else ' '}] {c['name']} — {c['detail']}")
    lines.append("")
    with open(os.path.join(OUT_DIR, "leaderboard.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
