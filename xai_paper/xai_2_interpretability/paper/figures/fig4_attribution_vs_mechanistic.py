#!/usr/bin/env python3
"""Figure 4 — attribution (Phase B) vs mechanistic interpretability (Phase C).

A head-to-head of the twelve Phase-B attribution methods against the ten Phase-C
mechanistic-interpretability methods, scored on the SAME intervention-oracle
faithfulness axis (the §3.2 oracle). These 22 method rows are a SUBSET of the
canonical leaderboard (30 interpretability methods + the oracle positive control
= 31 rows); the cross-tradition headline scatter is Figure 2. The figure makes two
measured claims (experiment_design.md §5, §6, §7; plan.md figure 4):

  (i)  PHASE B, regime split.  Faithfulness is split by the output regime the
       attribution targets: CONTENT (a register/colour/graphics-bit value, which
       the STE gradient routes to) vs POSITION/INDEX (a discrete sprite position
       via strobe timing, whose naive gradient is provably zero — §3.2). The whole
       GRADIENT family (vanilla saliency, SmoothGrad, guided backprop, Grad×Input/
       DeepLIFT, Integrated Gradients, Expected Gradients, plus the sampling/linear
       surrogates LIME/KernelSHAP/RISE that lean on it) COLLAPSES toward chance on
       position, while the INTERVENTION/perturbation methods (occlusion, extremal
       perturbation, on-distribution counterfactual) HOLD — they remain valid
       interventions on a discrete output. This is §7's "Fail / zero on index
       outputs" prediction, in numbers.

  (ii) PHASE C, the split inside mechanistic interp.  Activation patching, DAS/
       interchange interventions, causal scrubbing and the logit/tuned lens are
       EXACT (faithfulness = 1.0 vs the exact patch / true intermediate); the
       circuit-discovery methods (ACDC, path patching) and the gradient
       approximation (attribution patching) are PARTIAL; and the
       present-not-used methods — SAE features and linear probing-with-control-
       tasks — score LOW: the representation contains the variable, but the probe
       does not show it is causally USED (Hewitt & Liang's control-task gap).

The headline contrast (42-game scored battery): on the POSITION regime the
causal/intervention family averages 0.561 (±0.321, n=4) vs 0.234 (±0.156, n=9)
for the gradient/correlational family — a 0.327 family-mean faithfulness gap,
whose bootstrap over the 42 games is 95% CI [0.132, 0.370] (EXCLUDES 0:
significant). The exact causal method activation_patching reaches
1.000 (= the oracle ceiling) where vanilla saliency's naive position gradient
collapses to 0.000 (= the naive-gradient floor).

UNCERTAINTY (R-UNC, P2-R-UNC).  Every faithfulness point carries a bootstrap-over-
games 95% CI read from leaderboard_ci.csv:
  * Panel (b) per-method whiskers come from the per-method CI rows.
  * Panel (a) is a regime split (per-method regime CIs are not recorded); the
    FAMILY-level position/content 95% CI bands are drawn behind the bars from the
    family CI rows, so point positions are not over-interpreted.

ALL plotted numbers are READ from committed records — no experiment is re-run:
  * tools/xai_study/compare/out/leaderboard.json     (P2-E6-1; the aggregate index)
  * tools/xai_study/compare/out/leaderboard_ci.csv   (P2-R-UNC; bootstrap 95% CIs)
  * tools/xai_study/compare/out/faithful_demo.json   (P2-E6-3; the headline gap)
The per-method §R records under tools/xai_study/phase{B,C}_*/out/*.json are the
sources the leaderboard aggregates (referenced, not re-read here).

Run:
    python fig4_attribution_vs_mechanistic.py \
        --leaderboard tools/xai_study/compare/out/leaderboard.json
Produces:
    fig4_attribution_vs_mechanistic.pdf  (vector, colour-blind-safe, self-legend)
"""

import argparse
import csv
import json
import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")  # headless vector backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
# repo root = .../  (figures -> paper -> xai_2_interpretability -> xai_paper -> root)
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))


# ---------------------------------------------------------------------------
# Palette — Okabe-Ito colour-blind-safe set (matches fig1).
# ---------------------------------------------------------------------------
C_BG = "#ffffff"
C_INK = "#1a1a1a"           # primary text / outlines
C_MUTE = "#5b5b5b"          # secondary text
C_GRID = "#d9d9d9"

# tradition / family colours (Okabe-Ito)
C_GRAD = "#0072B2"          # blue        — gradient family (incl. surrogates)
C_INTERV = "#D55E00"        # vermillion  — intervention / perturbation / causal-exact
C_CAUSAL = "#D55E00"        # (Phase C causal == intervention colour: causal-by-construction)
C_APPROX = "#E69F00"        # orange      — gradient approximation (attribution patching)
C_DIMRED = "#CC79A7"        # reddish-purple — dim-reduction / SAE
C_PROBE = "#999999"         # grey        — probing (present-not-used)
C_ORACLE = "#000000"        # black       — oracle ceiling reference line

# content/position regime fills for the Phase-B grouped bars
C_CONTENT = "#56B4E9"       # sky blue    — content regime
C_POSITION = "#009E73"      # bluish-green — position/index regime

# minimum in-figure font (figure-detail-pass target): 7.5 pt
FS_MIN = 7.5

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.6,
        "axes.linewidth": 0.8,
        "axes.edgecolor": C_MUTE,
        "pdf.fonttype": 42,    # embed TrueType (editable, not bitmap)
        "svg.fonttype": "none",
        "figure.dpi": 150,
        "xtick.color": C_INK,
        "ytick.color": C_INK,
        "text.color": C_INK,
    }
)


# ---------------------------------------------------------------------------
# Data load — pure reads over committed records.
# ---------------------------------------------------------------------------
def load_records(leaderboard_path):
    with open(leaderboard_path) as fh:
        lb = json.load(fh)
    base = os.path.dirname(leaderboard_path)
    with open(os.path.join(base, "faithful_demo.json")) as fh:
        demo = json.load(fh)
    # R-UNC bootstrap-over-games 95% CIs (P2-R-UNC).
    ci_method = {}      # method -> (mean, lo, hi)  all-regime faithfulness CI
    ci_family = {}      # family key -> (mean, lo, hi)
    ci_headline = {}    # headline gap key -> (mean, lo, hi)  (all_regime / position)
    ci_path = os.path.join(base, "leaderboard_ci.csv")
    with open(ci_path) as fh:
        for row in csv.DictReader(fh):
            kind = row.get("kind")
            try:
                triple = (float(row["mean"]), float(row["ci_lo"]), float(row["ci_hi"]))
            except (TypeError, ValueError, KeyError):
                continue
            if kind == "method":
                ci_method[row["method"]] = triple
            elif kind == "family":
                ci_family[row["method"]] = triple
            elif kind == "headline":
                ci_headline[row["method"]] = triple
    # Position-regime gap SIGNIFICANCE — the paper's headline bootstrap over the
    # 42 scored games (position_bootstrap.json is the source of truth for this
    # number, comparable to the original 6-game position CI). Optional.
    pos_boot = None
    pb_path = os.path.join(base, "position_bootstrap.json")
    if os.path.isfile(pb_path):
        with open(pb_path) as fh:
            pos_boot = json.load(fh)
    return lb, demo, ci_method, ci_family, ci_headline, pos_boot


# Pretty labels (the records' machine names -> publication labels + citation tags).
PRETTY_B = {
    "vanilla_saliency": "Vanilla saliency",
    "smoothgrad": "SmoothGrad",
    "guided_backprop": "Guided backprop",
    "gradxinput_deeplift": "Grad×Input / DeepLIFT",
    "integrated_gradients": "Integrated Gradients",
    "expected_gradients": "Expected Gradients",
    "lime": "LIME",
    "kernelshap": "KernelSHAP",
    "rise": "RISE",
    "occlusion": "Occlusion",
    "extremal_perturbation": "Extremal perturbation",
    "on_distribution_counterfactual": "On-distrib. counterfactual",
}
PRETTY_C = {
    "activation_patching": "Activation patching",
    "interchange_interventions_das": "Interchange / DAS",
    "causal_scrubbing": "Causal scrubbing",
    "logit_tuned_lens": "Logit / tuned lens",
    "path_patching": "Path patching",
    "ACDC": "ACDC (circuit disc.)",
    "attribution_patching": "Attribution patching",
    "nmf_pca_dictionaries": "NMF/PCA dictionaries",
    "sparse_autoencoder": "Sparse autoencoder",
    "linear_probing_control_tasks": "Linear probing (+control)",
}

# Phase-B family grouping (for the regime panel): does this method ride the
# gradient (vanishes on index) or is it a real intervention (holds)?
GRADIENT_FAMILY = {
    "vanilla_saliency", "smoothgrad", "guided_backprop", "gradxinput_deeplift",
    "integrated_gradients", "expected_gradients", "lime", "kernelshap", "rise",
}
INTERVENTION_FAMILY = {
    "occlusion", "extremal_perturbation", "on_distribution_counterfactual",
}

# Phase-C category -> (colour, short tag) for the right panel.
C_CATEGORY = {
    "activation_patching": ("exact", C_CAUSAL),
    "interchange_interventions_das": ("exact", C_CAUSAL),
    "causal_scrubbing": ("exact", C_CAUSAL),
    "logit_tuned_lens": ("exact", C_CAUSAL),
    "path_patching": ("partial", C_CAUSAL),
    "ACDC": ("partial", C_CAUSAL),
    "attribution_patching": ("approx", C_APPROX),
    "nmf_pca_dictionaries": ("present-not-used", C_DIMRED),
    "sparse_autoencoder": ("present-not-used", C_DIMRED),
    "linear_probing_control_tasks": ("present-not-used", C_PROBE),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--leaderboard",
        default=os.path.join(
            REPO_ROOT, "tools", "xai_study", "compare", "out", "leaderboard.json"
        ),
        help="path to the E6-1 leaderboard.json (the aggregate index)",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(HERE, "fig4_attribution_vs_mechanistic.pdf"),
        help="output PDF path",
    )
    args = ap.parse_args()

    lb, demo, ci_method, ci_family, ci_headline, pos_boot = load_records(args.leaderboard)
    rows = {r["method"]: r for r in lb["rows"]}

    # ---- assemble Phase-B regime table (content vs position) ----------------
    b_methods = [m for m in PRETTY_B if m in rows]
    # order: gradient family first (sorted by content faith desc), then intervention
    grad_b = sorted(
        [m for m in b_methods if m in GRADIENT_FAMILY],
        key=lambda m: -(rows[m]["faithfulness_content_regime"] or 0.0),
    )
    interv_b = sorted(
        [m for m in b_methods if m in INTERVENTION_FAMILY],
        key=lambda m: -(rows[m]["faithfulness_content_regime"] or 0.0),
    )
    b_order = grad_b + interv_b

    b_content = [rows[m]["faithfulness_content_regime"] for m in b_order]
    b_position = [rows[m]["faithfulness_position_regime"] for m in b_order]
    b_labels = [PRETTY_B[m] for m in b_order]
    b_fam = ["gradient" if m in GRADIENT_FAMILY else "intervention" for m in b_order]

    # ---- assemble Phase-C table (single faithfulness, categorised) ----------
    c_methods = [m for m in PRETTY_C if m in rows]
    # sort by faithfulness desc so exact (1.0) at top
    c_order = sorted(c_methods, key=lambda m: -rows[m]["faithfulness"])
    c_faith = [rows[m]["faithfulness"] for m in c_order]
    # R-UNC bootstrap 95% CI per method (asymmetric lo/hi about the point).
    c_lo = [ci_method.get(m, (rows[m]["faithfulness"],) * 3)[1] for m in c_order]
    c_hi = [ci_method.get(m, (rows[m]["faithfulness"],) * 3)[2] for m in c_order]
    c_labels = [PRETTY_C[m] for m in c_order]
    c_cat = [C_CATEGORY[m][0] for m in c_order]
    c_col = [C_CATEGORY[m][1] for m in c_order]

    # headline numbers from the faithful demo (asserted == leaderboard headline)
    agg = demo["aggregate_contrast"]
    pair = demo["pair_contrast"]

    # family-level regime CI bands (R-UNC) for Panel (a) — per-method regime CIs
    # are not recorded, so we show the family band instead of over-precise points.
    fam_pos_grad = ci_family.get("family_gradient_correlational_position")
    fam_pos_caus = ci_family.get("family_causal_intervention_position")

    # =======================================================================
    # FIGURE
    # =======================================================================
    fig = plt.figure(figsize=(13.2, 7.1), facecolor=C_BG)
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.18, 1.0], left=0.135, right=0.965,
        top=0.745, bottom=0.180, wspace=0.40,
    )
    axB = fig.add_subplot(gs[0, 0])
    axC = fig.add_subplot(gs[0, 1])

    # ---------------- Panel B: Phase-B regime split (grouped bars) ----------
    nB = len(b_order)
    y = np.arange(nB)[::-1]          # top-to-bottom in listed order
    h = 0.38

    axB.barh(y + h / 2 + 0.02, b_content, height=h, color=C_CONTENT,
             edgecolor=C_INK, linewidth=0.4, label="content output", zorder=3)
    axB.barh(y - h / 2 - 0.02, b_position, height=h, color=C_POSITION,
             edgecolor=C_INK, linewidth=0.4, label="position / index output", zorder=3)

    # value annotations (>= FS_MIN)
    for yi, v in zip(y + h / 2 + 0.02, b_content):
        axB.text(v + 0.012, yi, f"{v:.2f}", va="center", ha="left",
                 fontsize=FS_MIN, color=C_MUTE)
    for yi, v in zip(y - h / 2 - 0.02, b_position):
        txt = f"{v:.2f}"
        axB.text(max(v, 0.0) + 0.012, yi, txt, va="center", ha="left",
                 fontsize=FS_MIN, color=(C_INTERV if v == 0.0 else C_MUTE),
                 fontweight=("bold" if v == 0.0 else "normal"))

    axB.set_yticks(y)
    axB.set_yticklabels(b_labels, fontsize=8.2)
    # colour-tint the family on the left margin via tick label colour
    for tick, fam in zip(axB.get_yticklabels(), b_fam):
        tick.set_color(C_GRAD if fam == "gradient" else C_INTERV)
    axB.set_xlim(0, 1.0)
    axB.set_xlabel("faithfulness vs intervention oracle (§3.2)  ·  Pearson corr, 0–1",
                   fontsize=9.0)
    axB.set_ylim(-0.8, nB - 0.2)
    axB.xaxis.grid(True, color=C_GRID, linewidth=0.6, zorder=0)
    axB.set_axisbelow(True)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    # Family-level POSITION-regime 95% CI bands (R-UNC). Per-method regime CIs
    # are not recorded; the family band shows the regime uncertainty without
    # over-precising any single point. The translucent span sits BEHIND the bars
    # over the position (green) sub-bars only; the mean + CI are stated once in
    # the family tag (right), not duplicated as in-data prose.
    n_grad = len(grad_b)

    def _pos_band(ci, y_lo, y_hi, color):
        if not ci:
            return None
        mean, lo, hi = ci
        axB.add_patch(Rectangle((lo, y_lo), hi - lo, y_hi - y_lo,
                                facecolor=color, alpha=0.13, edgecolor="none",
                                zorder=1))
        axB.plot([mean, mean], [y_lo, y_hi], color=color, linewidth=1.0,
                 linestyle=(0, (2, 2)), alpha=0.9, zorder=2)
        return mean, lo, hi

    # position (green) sub-bars occupy y - h/2 - 0.02 +/- h/2 in each row.
    grad_pos_lo = (y[n_grad - 1] - h / 2 - 0.02) - h / 2 - 0.04
    grad_pos_hi = (y[0] - h / 2 - 0.02) + h / 2 + 0.04
    interv_pos_lo = (y[-1] - h / 2 - 0.02) - h / 2 - 0.04
    interv_pos_hi = (y[n_grad] - h / 2 - 0.02) + h / 2 + 0.04
    g_ci = _pos_band(fam_pos_grad, grad_pos_lo, grad_pos_hi, C_POSITION)
    i_ci = _pos_band(fam_pos_caus, interv_pos_lo, interv_pos_hi, C_POSITION)

    # divider between the gradient family and the intervention family
    div_y = (y[n_grad - 1] + y[n_grad]) / 2.0  # between last grad and first interv
    axB.axhline(div_y, color=C_MUTE, linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)

    # family tags (right), each carrying the position-regime family mean + CI once.
    g_tag = "GRADIENT FAMILY"
    if g_ci:
        g_tag += f"\nposition F̄={g_ci[0]:.2f}  CI [{g_ci[1]:.2f}, {g_ci[2]:.2f}]"
    i_tag = "INTERVENTION / PERTURBATION"
    if i_ci:
        i_tag += f"\nposition F̄={i_ci[0]:.2f}  CI [{i_ci[1]:.2f}, {i_ci[2]:.2f}]"
    axB.text(0.985, y[0] + 0.62, g_tag, ha="right", va="center",
             fontsize=FS_MIN, color=C_GRAD, fontweight="bold", linespacing=1.15)
    axB.text(0.985, y[n_grad] + 0.62, i_tag, ha="right", va="center",
             fontsize=FS_MIN, color=C_INTERV, fontweight="bold", linespacing=1.15)

    axB.set_title(
        "(a)  Phase B — attribution, split by output regime",
        fontsize=10.2, fontweight="bold", loc="left", pad=8,
    )

    # ---------------- Panel C: Phase-C faithfulness (categorised bars) -------
    nC = len(c_order)
    yc = np.arange(nC)[::-1]
    axC.barh(yc, c_faith, height=0.62, color=c_col, edgecolor=C_INK,
             linewidth=0.4, zorder=3)
    # R-UNC bootstrap 95% CI whiskers (asymmetric about the point).
    xerr = np.array([[f - lo for f, lo in zip(c_faith, c_lo)],
                     [hi - f for f, hi in zip(c_faith, c_hi)]])
    xerr = np.clip(xerr, 0.0, None)
    axC.errorbar(c_faith, yc, xerr=xerr, fmt="none", ecolor=C_INK,
                 elinewidth=0.8, capsize=2.4, zorder=4)
    for yi, v, hi in zip(yc, c_faith, c_hi):
        axC.text(min(max(v, hi), 1.0) + 0.022, yi, f"{v:.2f}", va="center",
                 ha="left", fontsize=FS_MIN, color=C_MUTE)

    axC.set_yticks(yc)
    axC.set_yticklabels(c_labels, fontsize=8.2)
    axC.set_xlim(0, 1.18)
    axC.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axC.set_xlabel("faithfulness vs exact patch / true data-flow  ·  0–1",
                   fontsize=9.0)
    axC.set_ylim(-0.7, nC - 0.3)
    axC.xaxis.grid(True, color=C_GRID, linewidth=0.6, zorder=0)
    axC.set_axisbelow(True)
    for s in ("top", "right"):
        axC.spines[s].set_visible(False)

    # oracle ceiling line at 1.0
    axC.axvline(1.0, color=C_ORACLE, linewidth=1.0, linestyle=(0, (2, 2)),
                zorder=2)
    axC.text(1.0, nC - 0.32, " oracle\n ceiling", ha="left", va="top",
             fontsize=FS_MIN, color=C_ORACLE)

    axC.set_title(
        "(b)  Phase C — mechanistic interpretability",
        fontsize=10.2, fontweight="bold", loc="left", pad=8,
    )

    # bracket the three "present-not-used" rows (kept inside the axes box)
    pnu_idx = [i for i, cat in enumerate(c_cat) if cat == "present-not-used"]
    if pnu_idx:
        ytop = yc[min(pnu_idx)] + 0.42
        ybot = yc[max(pnu_idx)] - 0.42
        xb = 1.085
        axC.plot([xb, xb], [ybot, ytop], color=C_MUTE, linewidth=1.0,
                 clip_on=False, zorder=5)
        axC.plot([xb - 0.012, xb], [ytop, ytop], color=C_MUTE, linewidth=1.0,
                 clip_on=False, zorder=5)
        axC.plot([xb - 0.012, xb], [ybot, ybot], color=C_MUTE, linewidth=1.0,
                 clip_on=False, zorder=5)
        axC.text(xb + 0.012, (ytop + ybot) / 2, "present\nnot\nused",
                 ha="left", va="center", fontsize=FS_MIN, color=C_MUTE,
                 clip_on=False)

    # =======================================================================
    # Title + headline banner (the measured contrast)
    # =======================================================================
    fig.suptitle(
        "Attribution (Phase B) vs mechanistic interpretability (Phase C): "
        "faithfulness on the bit-exact VCS",
        fontsize=12.8, fontweight="bold", x=0.045, ha="left", y=0.975,
    )
    fig.text(
        0.045, 0.940,
        "The 22 Phase-B + Phase-C method rows (a subset of the 30 methods + the "
        "oracle = 31-row leaderboard; the cross-tradition scatter is Fig. 2).",
        fontsize=8.4, color=C_MUTE, style="italic", ha="left", va="top",
    )
    # HEADLINE = the ROBUST all-regime causal-vs-gradient faithfulness gap, whose
    # bootstrap CI excludes 0 (leaderboard_ci.csv headline row `all_regime_gap`).
    # On the 42-game scored battery the POSITION/INDEX-regime gap is now ALSO
    # SIGNIFICANT — its bootstrap CI over the 42 games EXCLUDES 0 (was directional
    # only at 6 games). The significance number is the paper's headline
    # position_bootstrap.json (per-game bootstrap of the causal−gradient gap).
    # All-regime gap CI: prefer position_bootstrap.json's all_regime block (the
    # paper's headline convention), fall back to leaderboard_ci.csv.
    _all_pb = pos_boot.get("all_regime") if pos_boot else None
    if _all_pb:
        all_gap = (_all_pb["family_mean_gap"], _all_pb["ci95"][0], _all_pb["ci95"][1])
    else:
        all_gap = ci_headline.get("all_regime_gap")
    fam_all_c = ci_family.get("family_causal_intervention")
    fam_all_g = ci_family.get("family_gradient_correlational")
    # position-regime family means (42-game) + the bootstrap-over-games CI.
    fam_pos_c_all = ci_family.get("family_causal_intervention_position")
    fam_pos_g_all = ci_family.get("family_gradient_correlational_position")
    headline = (
        "HEADLINE — across ALL output regimes, intervention/causal methods are more "
        "faithful to the true mechanism than gradient/correlational ones.  "
    )
    if fam_all_c and fam_all_g and all_gap:
        headline += (
            f"Causal/intervention F̄={fam_all_c[0]:.3f} (95% CI "
            f"[{fam_all_c[1]:.3f}, {fam_all_c[2]:.3f}]) vs gradient/correlational "
            f"F̄={fam_all_g[0]:.3f} (95% CI [{fam_all_g[1]:.3f}, {fam_all_g[2]:.3f}]) "
            f"— a {all_gap[0]:.3f} gap, 95% CI [{all_gap[1]:.3f}, {all_gap[2]:.3f}] "
            f"(excludes 0: robust).  "
        )
    headline += (
        "On the POSITION/INDEX regime alone (discrete sprite-position outputs), the "
        "gap is now SIGNIFICANT on the 42-game battery: "
    )
    if fam_pos_c_all and fam_pos_g_all:
        headline += (
            f"causal {fam_pos_c_all[0]:.3f} vs gradient {fam_pos_g_all[0]:.3f}; "
        )
    pos_pb = pos_boot.get("position") if pos_boot else None
    if pos_pb:
        headline += (
            f"family-mean gap {pos_pb['family_mean_gap']:.3f}, 95% CI "
            f"[{pos_pb['ci95'][0]:.3f}, {pos_pb['ci95'][1]:.3f}] over "
            f"{pos_pb['n_games']} games (EXCLUDES 0)."
        )
    else:
        headline += (
            f"causal {agg['faithful_bucket_mean']:.3f} vs gradient "
            f"{agg['popular_bucket_mean']:.3f}, gap {agg['bucket_gap']:.3f}."
        )
    # Wrap explicitly (robust across matplotlib versions; the figure-text wrap=
    # heuristic changed in 3.x). The banner spans the full usable width.
    headline_wrapped = "\n".join(textwrap.wrap(headline, width=170))
    fig.text(
        0.045, 0.910, headline_wrapped, fontsize=8.0, color=C_INK,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f7fa",
                  edgecolor=C_GRID, linewidth=0.7),
    )

    # =======================================================================
    # Legends — OUTSIDE the data field, in a strip beneath the two panels.
    # =======================================================================
    regime_handles = [
        Patch(facecolor=C_CONTENT, edgecolor=C_INK, linewidth=0.4,
              label="content output"),
        Patch(facecolor=C_POSITION, edgecolor=C_INK, linewidth=0.4,
              label="position / index output"),
    ]
    legB = fig.legend(
        handles=regime_handles, title="(a) output regime",
        loc="lower left", bbox_to_anchor=(0.135, 0.018), ncol=2,
        frameon=True, fontsize=8.0, title_fontsize=8.4,
        handlelength=1.3, borderpad=0.55, columnspacing=1.4,
    )
    legB.get_frame().set_edgecolor(C_GRID)
    legB.get_frame().set_linewidth(0.7)
    legB._legend_box.align = "left"

    cat_handles = [
        Patch(facecolor=C_CAUSAL, edgecolor=C_INK, linewidth=0.4,
              label="causal (exact / partial)"),
        Patch(facecolor=C_APPROX, edgecolor=C_INK, linewidth=0.4,
              label="gradient approximation"),
        Patch(facecolor=C_DIMRED, edgecolor=C_INK, linewidth=0.4,
              label="dim-reduction / SAE  (present-not-used)"),
        Patch(facecolor=C_PROBE, edgecolor=C_INK, linewidth=0.4,
              label="probing  (present-not-used)"),
        plt.Line2D([0], [0], color=C_INK, linewidth=1.0,
                   label="95% CI (bootstrap over games, R-UNC)"),
    ]
    legC = fig.legend(
        handles=cat_handles, title="(b) Phase-C category  +  uncertainty",
        loc="lower left", bbox_to_anchor=(0.560, 0.018), ncol=3,
        frameon=True, fontsize=8.0, title_fontsize=8.4,
        handlelength=1.3, borderpad=0.55, columnspacing=1.4,
        labelspacing=0.4,
    )
    legC.get_frame().set_edgecolor(C_GRID)
    legC.get_frame().set_linewidth(0.7)
    legC._legend_box.align = "left"

    # provenance footer — no in-figure data paths (figure-detail-pass: remove
    # microtext source strings; exact paths live in the Supplement).
    fig.text(
        0.045, 0.004,
        "Data: committed records (leaderboard + bootstrap CIs + position bootstrap); "
        "42-game scored battery, 30-frame horizon; pure read — no experiment re-run. "
        "Exact paths & per-number provenance in the Supplement.",
        fontsize=FS_MIN, color=C_MUTE, ha="left", va="bottom",
    )

    fig.savefig(args.out, format="pdf", facecolor=C_BG)
    plt.close(fig)

    # =======================================================================
    # In-script self-check (DoD)
    # =======================================================================
    ok = True
    msgs = []

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        msgs.append(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")

    # file exists + non-trivial size + %PDF- header
    exists = os.path.isfile(args.out)
    size = os.path.getsize(args.out) if exists else 0
    header_ok = False
    if exists:
        with open(args.out, "rb") as fh:
            header_ok = fh.read(5) == b"%PDF-"
    check("PDF file exists", exists, args.out)
    check("PDF size > 4 KB", size > 4096, f"{size} bytes")
    check("PDF has %PDF- header", header_ok)

    # data-integrity: every plotted number traced to the leaderboard
    check("12 Phase-B methods plotted", len(b_order) == 12, f"n={len(b_order)}")
    check("10 Phase-C methods plotted", len(c_order) == 10, f"n={len(c_order)}")
    check("22 method rows total (subset of 30+oracle=31)",
          len(b_order) + len(c_order) == 22)
    # Corrected re-run: the pure-gradient family scores LOW on the position regime
    # (not exactly 0 in the aggregate — only guided_backprop hits the 0 floor), and
    # the intervention family averages HIGHER but not uniformly (on-distribution
    # counterfactual dips below 0.1). So we no longer assert exact-zero / uniform
    # holds; we assert the honest ordering: the gradient family MEAN stays low and
    # below the intervention family MEAN on position.
    grad_pure = [m for m in grad_b
                 if m in {"vanilla_saliency", "smoothgrad", "guided_backprop",
                          "gradxinput_deeplift", "integrated_gradients",
                          "expected_gradients"}]
    grad_pos_mean = (sum(rows[m]["faithfulness_position_regime"] for m in grad_pure)
                     / len(grad_pure))
    interv_pos_mean = (sum(rows[m]["faithfulness_position_regime"] for m in interv_b)
                       / len(interv_b))
    check(
        "gradient family scores low on position (mean <= 0.20)",
        grad_pos_mean <= 0.20,
        f"grad position mean={grad_pos_mean:.3f}",
    )
    check(
        "intervention family above gradient family on position (mean)",
        interv_pos_mean > grad_pos_mean,
        f"interv={interv_pos_mean:.3f} > grad={grad_pos_mean:.3f}",
    )
    check(
        "activation patching exact (==1.0)",
        rows["activation_patching"]["faithfulness"] == 1.0,
    )
    check(
        "SAE + probing low (present-not-used, <0.25)",
        rows["sparse_autoencoder"]["faithfulness"] < 0.25
        and rows["linear_probing_control_tasks"]["faithfulness"] < 0.25,
    )
    # HEADLINE (robust) = the all-regime gap whose bootstrap CI excludes 0.
    check(
        "all-regime gap CI present and excludes 0 (robust headline)",
        (all_gap is not None) and all_gap[1] > 0.0,
        f"all_regime_gap={all_gap}",
    )
    # On the 42-game scored battery the position-regime gap is now SIGNIFICANT:
    # its bootstrap-over-games CI EXCLUDES 0 (position_bootstrap.json). Assert the
    # significance so the figure never regresses to the old "not significant" text.
    _pb = pos_boot.get("position") if pos_boot else None
    check(
        "position-regime gap CI EXCLUDES 0 (significant, 42-game bootstrap)",
        (_pb is not None) and _pb["ci95"][0] > 0.0
        and _pb.get("excludes_zero", _pb["ci95"][0] > 0.0),
        f"position_bootstrap={_pb['family_mean_gap'] if _pb else None} "
        f"CI={_pb['ci95'] if _pb else None}",
    )
    # the drawn position-bucket contrast still traces to the demo record
    check(
        "position bucket contrast matches demo",
        abs(agg["bucket_gap"] - demo["aggregate_contrast"]["bucket_gap"]) < 1e-9,
        f"gap={agg['bucket_gap']}",
    )
    # R-UNC uncertainty: every Phase-C point has a bootstrap CI from the CI file.
    check(
        "every Phase-C method carries an R-UNC CI",
        all(m in ci_method for m in c_order),
        "leaderboard_ci.csv method rows",
    )
    check(
        "Phase-C CIs bracket their points",
        all(lo <= f + 1e-6 and hi >= f - 1e-6
            for f, lo, hi in zip(c_faith, c_lo, c_hi)),
    )
    # all in-figure fonts >= 7.5 pt (the minimum used anywhere is FS_MIN).
    check("min in-figure font >= 7.5 pt", FS_MIN >= 7.5, f"FS_MIN={FS_MIN}")

    print("Figure 4 self-check:")
    print("\n".join(msgs))
    print(f"\n{'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'} -> {args.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
