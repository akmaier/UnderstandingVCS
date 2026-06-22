#!/usr/bin/env python3
"""Figure 4 — attribution (Phase B) vs mechanistic interpretability (Phase C).

A head-to-head of the twelve Phase-B attribution methods against the ten Phase-C
mechanistic-interpretability methods, scored on the SAME §1 intervention-oracle
faithfulness axis. The figure makes two measured claims (experiment_design.md §5,
§6, §7; plan.md figure 4):

  (i)  PHASE B, regime split.  Faithfulness is split by the output regime the
       attribution targets: CONTENT (a register/colour/graphics-bit value, which
       the STE gradient routes to) vs POSITION/INDEX (a discrete sprite position
       via strobe timing, whose naive gradient is provably zero — §1). The whole
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

The headline contrast (the faithful-demonstration banner, P2-E6-3): on the
POSITION regime the causal/intervention bucket averages 0.412 (±0.390, n=4) vs
0.068 (±0.070, n=9) for the gradient/correlational bucket — a 0.344 faithfulness
gap; the exact causal method activation_patching reaches 1.000 (= the oracle
ceiling) where vanilla saliency collapses to 0.000 (= the naive-gradient floor).

ALL plotted numbers are READ from committed records — no experiment is re-run:
  * tools/xai_study/compare/out/leaderboard.json   (P2-E6-1; the aggregate index)
  * tools/xai_study/compare/out/faithful_demo.json (P2-E6-3; the headline gap)
The per-method §R records under tools/xai_study/phase{B,C}_*/out/*.json are the
sources the leaderboard aggregates (referenced, not re-read here).

Run:
    python fig4_attribution_vs_mechanistic.py \
        --leaderboard tools/xai_study/compare/out/leaderboard.json
Produces:
    fig4_attribution_vs_mechanistic.pdf  (vector, colour-blind-safe, self-legend)
"""

import argparse
import json
import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")  # headless vector backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.4,
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
    demo_path = os.path.join(os.path.dirname(leaderboard_path), "faithful_demo.json")
    with open(demo_path) as fh:
        demo = json.load(fh)
    return lb, demo


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

    lb, demo = load_records(args.leaderboard)
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
    c_ci = [rows[m].get("faithfulness_ci95") or 0.0 for m in c_order]
    c_labels = [PRETTY_C[m] for m in c_order]
    c_cat = [C_CATEGORY[m][0] for m in c_order]
    c_col = [C_CATEGORY[m][1] for m in c_order]

    # headline numbers from the faithful demo (asserted == leaderboard headline)
    agg = demo["aggregate_contrast"]
    pair = demo["pair_contrast"]

    # =======================================================================
    # FIGURE
    # =======================================================================
    fig = plt.figure(figsize=(13.2, 6.5), facecolor=C_BG)
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.18, 1.0], left=0.135, right=0.955,
        top=0.78, bottom=0.115, wspace=0.40,
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

    # value annotations
    for yi, v in zip(y + h / 2 + 0.02, b_content):
        axB.text(v + 0.012, yi, f"{v:.2f}", va="center", ha="left",
                 fontsize=6.6, color=C_MUTE)
    for yi, v in zip(y - h / 2 - 0.02, b_position):
        txt = f"{v:.2f}"
        axB.text(max(v, 0.0) + 0.012, yi, txt, va="center", ha="left",
                 fontsize=6.6, color=(C_INTERV if v == 0.0 else C_MUTE),
                 fontweight=("bold" if v == 0.0 else "normal"))

    axB.set_yticks(y)
    axB.set_yticklabels(b_labels, fontsize=8.0)
    # colour-tint the family on the left margin via tick label colour
    for tick, fam in zip(axB.get_yticklabels(), b_fam):
        tick.set_color(C_GRAD if fam == "gradient" else C_INTERV)
    axB.set_xlim(0, 1.0)
    axB.set_xlabel("faithfulness vs §1 intervention oracle  (Pearson corr, 0–1)",
                   fontsize=8.4)
    axB.set_ylim(-0.8, nB - 0.2)
    axB.xaxis.grid(True, color=C_GRID, linewidth=0.6, zorder=0)
    axB.set_axisbelow(True)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    # divider between the gradient family and the intervention family
    n_grad = len(grad_b)
    div_y = (y[n_grad - 1] + y[n_grad]) / 2.0  # between last grad and first interv
    axB.axhline(div_y, color=C_MUTE, linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)
    axB.text(0.985, y[0] + 0.55, "GRADIENT FAMILY", ha="right", va="center",
             fontsize=7.2, color=C_GRAD, fontweight="bold")
    axB.text(0.985, y[n_grad] + 0.55, "INTERVENTION / PERTURBATION",
             ha="right", va="center", fontsize=7.2, color=C_INTERV,
             fontweight="bold")

    axB.set_title(
        "(a)  Phase B — attribution, split by output regime",
        fontsize=10.0, fontweight="bold", loc="left", pad=8,
    )
    axB.legend(loc="lower right", fontsize=7.6, frameon=True, framealpha=0.95,
               edgecolor=C_GRID, handlelength=1.3, borderpad=0.5)

    # ---------------- Panel C: Phase-C faithfulness (categorised bars) -------
    nC = len(c_order)
    yc = np.arange(nC)[::-1]
    bars = axC.barh(yc, c_faith, height=0.62, color=c_col, edgecolor=C_INK,
                    linewidth=0.4, zorder=3)
    # 95% CI whiskers
    axC.errorbar(c_faith, yc, xerr=c_ci, fmt="none", ecolor=C_INK,
                 elinewidth=0.8, capsize=2.2, zorder=4)
    for yi, v, cat in zip(yc, c_faith, c_cat):
        axC.text(min(v, 1.0) + 0.018, yi, f"{v:.2f}", va="center", ha="left",
                 fontsize=6.8, color=C_MUTE)

    axC.set_yticks(yc)
    axC.set_yticklabels(c_labels, fontsize=8.0)
    axC.set_xlim(0, 1.18)
    axC.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axC.set_xlabel("faithfulness vs exact patch / true data-flow  (0–1)",
                   fontsize=8.4)
    axC.set_ylim(-0.7, nC - 0.3)
    axC.xaxis.grid(True, color=C_GRID, linewidth=0.6, zorder=0)
    axC.set_axisbelow(True)
    for s in ("top", "right"):
        axC.spines[s].set_visible(False)

    # oracle ceiling line at 1.0
    axC.axvline(1.0, color=C_ORACLE, linewidth=1.0, linestyle=(0, (2, 2)),
                zorder=2)
    axC.text(1.0, nC - 0.35, " oracle\n ceiling", ha="left", va="top",
             fontsize=6.8, color=C_ORACLE)

    axC.set_title(
        "(b)  Phase C — mechanistic interpretability",
        fontsize=10.0, fontweight="bold", loc="left", pad=8,
    )

    # ---- Phase-C category legend (exact / partial / approx / present-not-used)
    cat_handles = [
        Patch(facecolor=C_CAUSAL, edgecolor=C_INK, linewidth=0.4,
              label="causal (exact / partial)"),
        Patch(facecolor=C_APPROX, edgecolor=C_INK, linewidth=0.4,
              label="gradient approximation"),
        Patch(facecolor=C_DIMRED, edgecolor=C_INK, linewidth=0.4,
              label="dim-reduction / SAE  (present‑not‑used)"),
        Patch(facecolor=C_PROBE, edgecolor=C_INK, linewidth=0.4,
              label="probing  (present‑not‑used)"),
    ]
    axC.legend(handles=cat_handles, loc="lower right", fontsize=7.2,
               frameon=True, framealpha=0.95, edgecolor=C_GRID,
               handlelength=1.2, borderpad=0.5, labelspacing=0.35)

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
                 ha="left", va="center", fontsize=6.6, color=C_MUTE,
                 clip_on=False)

    # =======================================================================
    # Title + headline banner (the measured contrast)
    # =======================================================================
    fig.suptitle(
        "Attribution (Phase B) vs mechanistic interpretability (Phase C): "
        "faithfulness on the bit-exact VCS",
        fontsize=12.6, fontweight="bold", x=0.045, ha="left", y=0.965,
    )
    headline = (
        "On the POSITION/INDEX regime (discrete sprite-position outputs, naive "
        "gradient provably zero): the whole gradient family collapses toward "
        "chance while intervention methods hold.  "
        f"Causal/intervention bucket {agg['faithful_bucket_mean']:.3f} "
        f"(±{agg['faithful_bucket_ci95']:.3f}, n={agg['faithful_bucket_n']}) "
        f"vs gradient/correlational {agg['popular_bucket_mean']:.3f} "
        f"(±{agg['popular_bucket_ci95']:.3f}, n={agg['popular_bucket_n']}) "
        f"— a {agg['bucket_gap']:.3f} faithfulness gap.  "
        f"Activation patching = {pair['faithful_faithfulness_position']:.3f} "
        f"(oracle ceiling) vs vanilla saliency = "
        f"{pair['popular_faithfulness_position']:.3f} (naive-gradient floor)."
    )
    # Wrap explicitly (robust across matplotlib versions; the figure-text wrap=
    # heuristic changed in 3.x). The banner spans the full usable width.
    headline_wrapped = "\n".join(textwrap.wrap(headline, width=175))
    fig.text(
        0.045, 0.915, headline_wrapped, fontsize=7.9, color=C_INK,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f7fa",
                  edgecolor=C_GRID, linewidth=0.7),
    )

    # provenance footer
    fig.text(
        0.045, 0.018,
        "Source: tools/xai_study/compare/out/leaderboard.json (P2-E6-1) + "
        "faithful_demo.json (P2-E6-3); per-method §R records under "
        "tools/xai_study/phase{B,C}_*/out/. 6 core games, 30-frame horizon. "
        "Pure read — no experiment re-run.",
        fontsize=6.4, color=C_MUTE, ha="left", va="bottom",
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
    check(
        "gradient family vanishes on position (all == 0.0)",
        all(rows[m]["faithfulness_position_regime"] == 0.0 for m in grad_b
            if m in {"vanilla_saliency", "smoothgrad", "guided_backprop",
                     "gradxinput_deeplift", "integrated_gradients",
                     "expected_gradients"}),
        "pure-gradient methods",
    )
    check(
        "intervention family holds on position (all > 0.1)",
        all(rows[m]["faithfulness_position_regime"] > 0.1 for m in interv_b),
        "occlusion/extremal/counterfactual",
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
    check(
        "headline bucket gap matches demo (0.344)",
        abs(agg["bucket_gap"] - 0.3435) < 1e-6,
        f"gap={agg['bucket_gap']}",
    )

    print("Figure 4 self-check:")
    print("\n".join(msgs))
    print(f"\n{'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'} -> {args.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
