#!/usr/bin/env python3
"""Figure 6 — the failure taxonomy.

Every distinct way an interpretability method goes wrong on the VCS, observed
across Phases A (neuroscience battery), B (attribution / XAI) and C (mechanistic
interpretability), organised into a clean TAXONOMY by *underlying mechanism* —
not by method family. The point of organising by mechanism is the paper's
"single strongest objection" rebuttal (plan.md, "Why a 1977 chip is a fair
test"): the VCS is a necessary-condition screen because it already *exhibits the
failure modes that make interpretation hard* (aliased / polysemantic RAM cells,
race-the-beam distributed timing, context-dependent registers, present-not-used
representations). A method that trips one of these mechanisms here has no business
being trusted on a neural net. Each leaf is annotated with the method(s) that
exhibit it and their MEASURED faithfulness against the §1 intervention oracle.

The five mechanism branches (plan.md figure 6; experiment_design.md §7 "Why"
column; §0 danger zone; §1 zero-gradient-on-index caveat):

  (1) GRADIENT PATHOLOGIES — vanishing gradient on the discrete index/position
      output (the naive position gradient is *provably zero*, §1), saturation,
      shattering, and model-invariance / sanity-check failures (Adebayo 2018).
        leaves: vanilla saliency, SmoothGrad, guided backprop, Grad×Input/
        DeepLIFT, Integrated Gradients, Expected Gradients.

  (2) SAMPLING / APPROXIMATION ERROR — surrogate methods whose estimate is a
      stochastic or baseline-dependent approximation that diverges from the true
      attribution (LIME/SHAP/RISE sampling variance; IG baseline sensitivity).
        leaves: LIME, KernelSHAP, RISE, (IG baseline-sweep, cross-ref branch 1).

  (3) CORRELATIONAL CONFOUNDS — the method reports a *correlation* (tuning curve,
      Granger edge, spike-word coupling, probe accuracy) that need not be a cause:
      the present-not-used trap (Hewitt & Liang control tasks, §2/§6).
        leaves: tuning curves (A3), spike-word/pairwise (A4), Granger (A6),
        linear probing + control tasks (C).

  (4) DECOMPOSITION NON-IDENTIFIABILITY — unsupervised factorisations whose atoms
      are not unique and need not align to the true variables: SAE
      polysemanticity / dead-or-mixed features, NMF/PCA rotation ambiguity.
        leaves: sparse autoencoder (C), NMF/PCA dictionaries (C), dim-reduction (A7).

  (5) COMPOSITIONAL BLIND-SPOTS — single-site analyses that miss distributed /
      interacting structure: single-unit lesions over-credit, attribution/edge
      patching's first-order approximation, automatic circuit discovery's
      greedy edge search miss higher-order interactions.
        leaves: single-unit lesions (A2), attribution patching (C), ACDC (C),
        path patching (C).

  + a control LEAF that is NOT a failure-of-method but a property of the
    SUBJECT: architecture-specific N/A — Grad-CAM/++, attention rollout, VIPER
    simply do not apply to a non-neural artifact (a *recorded* finding, §5/§7).
    Shown greyed, with no F (it was never run; there is nothing to score).

EVERY measured faithfulness number is READ from a committed record — no
experiment is re-run here:
  * tools/xai_study/compare/out/leaderboard.json   (P2-E6-1; the aggregate index)
  * tools/xai_study/compare/out/faithful_demo.json (P2-E6-3; the position/content
    regime split + the headline gap)
The conceptual mechanism->leaf mapping is grounded in experiment_design.md §7
("Why" column) and plan.md (representativeness rebuttal). The N/A leaf is the
"does not apply — recorded finding" row of experiment_design.md §5.

Run:
    python fig6_failure_taxonomy.py \
        --leaderboard tools/xai_study/compare/out/leaderboard.json
Produces:
    fig6_failure_taxonomy.pdf  (vector, colour-blind-safe, self-legend)
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless vector backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

HERE = os.path.dirname(os.path.abspath(__file__))
# repo root = .../  (figures -> paper -> xai_2_interpretability -> xai_paper -> root)
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))


# ---------------------------------------------------------------------------
# Palette — Okabe-Ito colour-blind-safe set (matches fig1/fig4).
# ---------------------------------------------------------------------------
C_BG = "#ffffff"
C_INK = "#1a1a1a"           # primary text / outlines
C_MUTE = "#5b5b5b"          # secondary text
C_FAINT = "#8a8a8a"
C_GRID = "#d9d9d9"

# Five mechanism-branch accent colours (Okabe-Ito; each branch a distinct hue).
BRANCH_COLORS = {
    "grad":   "#0072B2",   # blue          — (1) gradient pathologies
    "sample": "#E69F00",   # orange        — (2) sampling / approximation error
    "corr":   "#CC79A7",   # reddish-purple— (3) correlational confounds
    "decomp": "#009E73",   # bluish-green  — (4) decomposition non-identifiability
    "comp":   "#D55E00",   # vermillion    — (5) compositional blind-spots
    "na":     "#999999",   # grey          — control: architecture-specific N/A
}

# Sequential faithfulness ramp for the leaf F chips (colour-blind-safe;
# low-F = pale, high-F = dark teal). Distinct from the categorical branch hues.
FAITH_CMAP = LinearSegmentedColormap.from_list(
    "faith_cb", ["#f7f4ef", "#cfe8e0", "#7fc6b6", "#2a8f7e", "#0d4f47"]
)
FAITH_NORM = Normalize(vmin=0.0, vmax=1.0)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.4,
        "axes.linewidth": 0.8,
        "axes.edgecolor": C_MUTE,
        "pdf.fonttype": 42,    # embed TrueType (editable, not bitmap)
        "svg.fonttype": "none",
        "figure.dpi": 150,
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


# Publication labels for the leaf methods (machine name -> short label).
PRETTY = {
    "vanilla_saliency": "Vanilla saliency",
    "smoothgrad": "SmoothGrad",
    "guided_backprop": "Guided backprop",
    "gradxinput_deeplift": "Grad×Input / DeepLIFT",
    "integrated_gradients": "Integrated Gradients",
    "expected_gradients": "Expected Gradients",
    "lime": "LIME",
    "kernelshap": "KernelSHAP",
    "rise": "RISE",
    "A3_tuning": "Tuning curves (A3)",
    "A4_spike_word": "Spike-word / pairwise (A4)",
    "A6_granger": "Granger causality (A6)",
    "linear_probing_control_tasks": "Linear probing + control (C)",
    "sparse_autoencoder": "Sparse autoencoder (C)",
    "nmf_pca_dictionaries": "NMF/PCA dictionaries (C)",
    "A7_dim_reduction": "Dim-reduction NMF (A7)",
    "A2_lesions": "Single-unit lesions (A2)",
    "attribution_patching": "Attribution patching (C)",
    "ACDC": "ACDC circuit disc. (C)",
    "path_patching": "Path patching (C)",
}


# The taxonomy: ordered list of (branch_key, branch_title, branch_subtitle,
# [leaf method machine-names], extra-note). Methods listed are looked up in the
# leaderboard for their MEASURED F; the branch grouping is the conceptual map.
TAXONOMY = [
    (
        "grad",
        "1  Gradient pathologies",
        "vanishing on the index · saturation · model-invariance",
        ["vanilla_saliency", "smoothgrad", "guided_backprop",
         "gradxinput_deeplift", "integrated_gradients", "expected_gradients"],
        "naive position gradient = 0 (§1)",
    ),
    (
        "sample",
        "2  Sampling / approximation error",
        "surrogate variance · baseline sensitivity",
        ["lime", "kernelshap", "rise"],
        "IG baseline-sweep → branch 1",
    ),
    (
        "corr",
        "3  Correlational confounds",
        "present ≠ used · spurious tuning · spurious edges",
        ["A3_tuning", "A4_spike_word", "A6_granger",
         "linear_probing_control_tasks"],
        "Hewitt & Liang control-task gap",
    ),
    (
        "decomp",
        "4  Decomposition non-identifiability",
        "polysemantic / rotated atoms ≠ true variables",
        ["sparse_autoencoder", "nmf_pca_dictionaries", "A7_dim_reduction"],
        "factorisation atoms not unique",
    ),
    (
        "comp",
        "5  Compositional blind-spots",
        "single-site misses distributed / interacting structure",
        ["A2_lesions", "attribution_patching", "ACDC", "path_patching"],
        "1st-order / greedy-edge approx.",
    ),
]

# The control leaf — a property of the SUBJECT, not a method failure: methods
# that DO NOT APPLY to a non-neural artifact (experiment_design.md §5/§7 N/A row).
NA_LEAF = {
    "branch": "na",
    "title": "0  Architecture-specific N/A  (control — not a failure of method)",
    "subtitle": "needs conv maps / attention / a learned policy — does not apply",
    "methods": ["Grad-CAM / Grad-CAM++", "Attention rollout", "VIPER"],
    "note": "recorded finding (§5) — nothing to score; the subject lacks the substrate",
}


def faith_lookup(lb):
    return {r["method"]: r["faithfulness"] for r in lb["rows"]}


# ---------------------------------------------------------------------------
# Figure.
# ---------------------------------------------------------------------------
def build(lb, demo, out_path):
    F = faith_lookup(lb)

    # Geometry: one row per leaf; branches stacked top->bottom (+ N/A control).
    leaf_rows = []   # (branch_key, method_machine, label, F-or-None)
    branch_spans = []  # (branch_key, title, subtitle, note, y_top, y_bot)
    y = 0.0
    ROW_H = 1.0
    GAP_BRANCH = 0.9  # vertical gap between branches

    # iterate top (branch 1) downward
    for bkey, btitle, bsub, methods, bnote in TAXONOMY:
        y_top = y
        for m in methods:
            leaf_rows.append((bkey, m, PRETTY[m], F[m]))
            y += ROW_H
        branch_spans.append((bkey, btitle, bsub, bnote, y_top, y))
        y += GAP_BRANCH

    # N/A control leaf block (no F)
    na_top = y
    for m in NA_LEAF["methods"]:
        leaf_rows.append((NA_LEAF["branch"], None, m, None))
        y += ROW_H
    branch_spans.append((NA_LEAF["branch"], NA_LEAF["title"],
                         NA_LEAF["subtitle"], NA_LEAF["note"], na_top, y))

    total_h = y
    n_leaf = len(leaf_rows)

    fig_h = 0.40 * n_leaf + 2.9
    fig = plt.figure(figsize=(10.2, fig_h))
    ax = fig.add_axes([0.005, 0.05, 0.99, 0.86])
    ax.set_xlim(0, 12.0)
    ax.set_ylim(-0.5, total_h + 0.4)
    ax.invert_yaxis()  # branch 1 at the top
    ax.axis("off")

    # --- column x-anchors -----------------------------------------------------
    X_ROOT = 0.30
    X_BRANCH = 1.05      # where branch boxes start
    X_BRANCH_W = 3.55    # branch box width
    X_LEAF = 5.05        # leaf label start
    X_LEAFTXT_W = 4.35   # leaf label box width
    X_CHIP = 9.75        # measured-F chip centre
    X_BAR0 = 10.30       # F mini-bar baseline
    X_BAR_W = 1.55       # F mini-bar full width (F=1.0)

    # vertical centre helper
    def yc(i):
        return i + 0.5

    # --- root spine + title ---------------------------------------------------
    y_first = branch_spans[0][4] + 0.5
    y_last = branch_spans[-1][5] - 0.5
    ax.plot([X_ROOT, X_ROOT], [y_first, y_last], color=C_INK, lw=1.6,
            solid_capstyle="round", zorder=2)
    ax.text(X_ROOT - 0.06, (y_first + y_last) / 2.0,
            "INTERPRETABILITY\nFAILURE MODES\non the VCS",
            rotation=90, ha="center", va="center", fontsize=8.6,
            fontweight="bold", color=C_INK, linespacing=1.15)

    # --- branches -------------------------------------------------------------
    for bkey, btitle, bsub, bnote, yt, yb in branch_spans:
        col = BRANCH_COLORS[bkey]
        ymid = (yt + yb) / 2.0
        # connector root -> branch box
        ax.plot([X_ROOT, X_BRANCH], [ymid, ymid], color=col, lw=1.4, zorder=2)
        ax.plot([X_ROOT, X_ROOT], [yt + 0.5, yb - 0.5], color=col, lw=1.4,
                alpha=0.0)  # (kept for clarity; spine drawn above)

        # branch box
        box = FancyBboxPatch(
            (X_BRANCH, yt + 0.16), X_BRANCH_W, (yb - yt) - 0.32,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            linewidth=1.4, edgecolor=col,
            facecolor=col, alpha=0.10, zorder=2,
        )
        ax.add_patch(box)
        # left accent rule
        ax.add_patch(FancyBboxPatch(
            (X_BRANCH + 0.01, yt + 0.16), 0.07, (yb - yt) - 0.32,
            boxstyle="square,pad=0", linewidth=0, facecolor=col, zorder=3))

        ax.text(X_BRANCH + 0.22, ymid - 0.30, btitle, ha="left", va="center",
                fontsize=9.6, fontweight="bold",
                color=(C_FAINT if bkey == "na" else C_INK))
        ax.text(X_BRANCH + 0.22, ymid + 0.18, bsub, ha="left", va="center",
                fontsize=7.4, color=(C_FAINT if bkey == "na" else C_MUTE),
                style="italic")
        ax.text(X_BRANCH + 0.22, ymid + 0.62,
                "mechanism: " + bnote, ha="left", va="center",
                fontsize=6.7, color=col, fontweight="bold")

        # connector branch box -> leaf comb
        ax.plot([X_BRANCH + X_BRANCH_W, X_LEAF - 0.18], [ymid, ymid],
                color=col, lw=1.2, zorder=2)
        # vertical comb spanning this branch's leaves
        leaf_idx = [i for i, lr in enumerate(leaf_rows) if lr[0] == bkey]
        if leaf_idx:
            top_y = yc(min(leaf_idx))
            bot_y = yc(max(leaf_idx))
            ax.plot([X_LEAF - 0.18, X_LEAF - 0.18], [top_y, bot_y],
                    color=col, lw=1.2, zorder=2)

    # --- leaves ---------------------------------------------------------------
    for i, (bkey, m, label, fval) in enumerate(leaf_rows):
        col = BRANCH_COLORS[bkey]
        cy = yc(i)
        na = fval is None
        # twig
        ax.plot([X_LEAF - 0.18, X_LEAF - 0.02], [cy, cy], color=col, lw=1.0,
                zorder=2)
        # leaf label chip
        lbl_face = "#f4f4f4" if na else "#ffffff"
        ax.add_patch(FancyBboxPatch(
            (X_LEAF, cy - 0.34), X_LEAFTXT_W, 0.68,
            boxstyle="round,pad=0.015,rounding_size=0.10",
            linewidth=0.8, edgecolor=(C_GRID if na else col),
            facecolor=lbl_face, zorder=3))
        ax.text(X_LEAF + 0.16, cy, label, ha="left", va="center",
                fontsize=8.0, color=(C_FAINT if na else C_INK),
                style=("italic" if na else "normal"))

        if na:
            ax.text(X_CHIP + 0.05, cy, "N/A", ha="center", va="center",
                    fontsize=8.0, fontweight="bold", color=C_FAINT)
            ax.text(X_BAR0 + 0.05, cy, "does not apply", ha="left", va="center",
                    fontsize=6.8, color=C_FAINT, style="italic")
            continue

        # measured-F chip (numeric, coloured by F ramp)
        chip = FAITH_CMAP(FAITH_NORM(fval))
        # readable text colour on the chip
        txt_c = "#ffffff" if fval >= 0.45 else C_INK
        ax.add_patch(FancyBboxPatch(
            (X_CHIP - 0.40, cy - 0.26), 0.80, 0.52,
            boxstyle="round,pad=0.01,rounding_size=0.08",
            linewidth=0.7, edgecolor=C_MUTE, facecolor=chip, zorder=4))
        ax.text(X_CHIP, cy, f"{fval:.2f}", ha="center", va="center",
                fontsize=7.8, fontweight="bold", color=txt_c, zorder=5)

        # F mini-bar
        ax.add_patch(plt.Rectangle((X_BAR0, cy - 0.13), X_BAR_W, 0.26,
                                   facecolor="#eeeeee", edgecolor="none",
                                   zorder=3))
        ax.add_patch(plt.Rectangle((X_BAR0, cy - 0.13), X_BAR_W * fval, 0.26,
                                   facecolor=col, edgecolor="none", zorder=4))

    # --- column headers -------------------------------------------------------
    head_y = -0.95
    ax.text(X_BRANCH + 0.10, head_y, "MECHANISM BRANCH", ha="left",
            va="center", fontsize=7.6, fontweight="bold", color=C_MUTE)
    ax.text(X_LEAF + 0.10, head_y, "METHOD EXHIBITING IT (phase)", ha="left",
            va="center", fontsize=7.6, fontweight="bold", color=C_MUTE)
    ax.text(X_CHIP, head_y, "F", ha="center", va="center",
            fontsize=7.6, fontweight="bold", color=C_MUTE)
    ax.text(X_BAR0, head_y - 0.20,
            "measured faithfulness vs §1 oracle (0–1)", ha="left",
            va="center", fontsize=6.8, color=C_MUTE, style="italic")

    # oracle ceiling reference tick on the bar axis (stop above the caption row)
    grid_bot = branch_spans[-1][5] - 0.55
    for gx, gl in [(0.0, "0"), (0.5, "0.5"), (1.0, "1.0")]:
        bx = X_BAR0 + X_BAR_W * gx
        ax.plot([bx, bx], [head_y + 0.45, grid_bot],
                color=C_GRID, lw=0.5, zorder=0)
        ax.text(bx, head_y + 0.18, gl, ha="center", va="center",
                fontsize=6.2, color=C_FAINT)

    # --- title ----------------------------------------------------------------
    fig.suptitle(
        "Figure 6 — A taxonomy of interpretability failure modes on the VCS, "
        "by underlying mechanism",
        x=0.005, y=0.992, ha="left", fontsize=11.4, fontweight="bold",
        color=C_INK,
    )

    # --- footnote / caption banner -------------------------------------------
    pos_gap = demo["aggregate_contrast"]["bucket_gap"]
    pos_F = demo["popular_method"]["faithfulness_position_regime"]
    con_F = demo["popular_method"]["faithfulness_content_regime"]
    cap = (
        "Each leaf is a method exhibiting the branch's failure mechanism, "
        "annotated with its MEASURED faithfulness F against the §1 intervention "
        "oracle (read from leaderboard.json; 0 = chance, 1 = oracle ceiling). "
        "Branch (1) is sharpest on the discrete position/index regime, where "
        f"the naive gradient is provably zero: vanilla saliency F={pos_F:.2f} "
        f"there vs F={con_F:.2f} on the content regime (a {pos_gap:.2f} "
        "causal−gradient bucket gap, faithful_demo.json). The N/A control "
        "block is a property of the SUBJECT, not a method failure — Grad-CAM, "
        "attention and VIPER need NN substrate the VCS lacks (recorded finding, "
        "exp_design §5). Higher-order branches (3–5) name the deepest objections: "
        "present≠used probing, non-identifiable factorisations, single-site "
        "blind-spots. Mechanism map: plan.md representativeness rebuttal + "
        "exp_design §7."
    )
    fig.text(0.012, 0.004, _wrap(cap, 150), ha="left", va="bottom",
             fontsize=6.5, color=C_MUTE, linespacing=1.35)

    # --- legend ---------------------------------------------------------------
    leg_handles = [
        Patch(facecolor=BRANCH_COLORS["grad"], alpha=0.45,
              edgecolor=BRANCH_COLORS["grad"], label="(1) gradient pathologies"),
        Patch(facecolor=BRANCH_COLORS["sample"], alpha=0.45,
              edgecolor=BRANCH_COLORS["sample"],
              label="(2) sampling / approximation"),
        Patch(facecolor=BRANCH_COLORS["corr"], alpha=0.45,
              edgecolor=BRANCH_COLORS["corr"],
              label="(3) correlational confound"),
        Patch(facecolor=BRANCH_COLORS["decomp"], alpha=0.45,
              edgecolor=BRANCH_COLORS["decomp"],
              label="(4) decomposition non-id."),
        Patch(facecolor=BRANCH_COLORS["comp"], alpha=0.45,
              edgecolor=BRANCH_COLORS["comp"],
              label="(5) compositional blind-spot"),
        Patch(facecolor="#f4f4f4", edgecolor=C_GRID,
              label="(0) N/A — does not apply (control)"),
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor=FAITH_CMAP(FAITH_NORM(0.15)),
               markeredgecolor=C_MUTE, markersize=9,
               label="low F (pale chip)"),
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor=FAITH_CMAP(FAITH_NORM(0.95)),
               markeredgecolor=C_MUTE, markersize=9,
               label="high F (dark chip)"),
    ]
    fig.legend(handles=leg_handles, loc="upper right",
               bbox_to_anchor=(0.998, 0.955), ncol=2, fontsize=6.9,
               frameon=True, framealpha=0.96, edgecolor=C_GRID,
               handlelength=1.3, columnspacing=1.0, borderpad=0.5)

    fig.savefig(out_path, format="pdf", facecolor=C_BG)
    plt.close(fig)
    return leaf_rows, F


def _wrap(s, width):
    import textwrap
    return "\n".join(textwrap.wrap(s, width=width))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--leaderboard",
        default=os.path.join(REPO_ROOT,
                             "tools/xai_study/compare/out/leaderboard.json"),
    )
    ap.add_argument(
        "--out",
        default=os.path.join(HERE, "fig6_failure_taxonomy.pdf"),
    )
    args = ap.parse_args()

    if not os.path.exists(args.leaderboard):
        sys.exit(f"ERROR: leaderboard not found: {args.leaderboard}")

    lb, demo = load_records(args.leaderboard)
    leaf_rows, F = build(lb, demo, args.out)

    # -----------------------------------------------------------------------
    # In-script self-check (DoD).
    # -----------------------------------------------------------------------
    checks = []

    def chk(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    # 1. PDF exists, non-empty, %PDF- magic.
    ok_pdf = os.path.exists(args.out) and os.path.getsize(args.out) > 0
    magic = b""
    if ok_pdf:
        with open(args.out, "rb") as fh:
            magic = fh.read(5)
    chk("pdf written, non-empty, %PDF- header",
        ok_pdf and magic == b"%PDF-",
        f"size={os.path.getsize(args.out) if ok_pdf else 0}B magic={magic!r}")

    # 2. Every scored leaf's F was read from the leaderboard and lies in [0,1].
    scored = [lr for lr in leaf_rows if lr[3] is not None]
    all_in_range = all(0.0 <= lr[3] <= 1.0 for lr in scored)
    chk("every scored leaf F in [0,1] (read from leaderboard)",
        all_in_range and len(scored) >= 19,
        f"n_scored_leaves={len(scored)}")

    # 3. Each leaf's plotted F == the leaderboard's value for that method
    #    (no number invented in the figure).
    traced = all(abs(lr[3] - F[lr[1]]) < 1e-9 for lr in scored)
    chk("each plotted F traces exactly to leaderboard.json",
        traced, "ok" if traced else "MISMATCH")

    # 4. All five mechanism branches + the N/A control are present.
    branch_keys = {lr[0] for lr in leaf_rows}
    expect = {"grad", "sample", "corr", "decomp", "comp", "na"}
    chk("all 5 mechanism branches + N/A control present",
        branch_keys == expect, f"{sorted(branch_keys)}")

    # 5. The N/A control block has leaves and NO F (it was never scored).
    na_leaves = [lr for lr in leaf_rows if lr[0] == "na"]
    chk("N/A control leaves present and unscored",
        len(na_leaves) >= 3 and all(lr[3] is None for lr in na_leaves),
        f"n_na={len(na_leaves)}")

    # 6. Branch (1) gradient family is the low-F floor on position regime —
    #    sanity: its mean F is below the compositional/decomposition branches'
    #    best exact method (single-unit lesions ~0.99). Read from records.
    grad_F = [lr[3] for lr in leaf_rows if lr[0] == "grad"]
    chk("gradient branch mean-F below the exact lesion ceiling",
        np.mean(grad_F) < F["A2_lesions"],
        f"grad_mean={np.mean(grad_F):.3f} < lesion={F['A2_lesions']:.3f}")

    # 7. Position-regime gradient floor is exactly 0 in the demo record
    #    (the figure's headline mechanism claim).
    chk("position-regime gradient floor == 0 (faithful_demo)",
        load_records(args.leaderboard)[1]["popular_method"][
            "faithfulness_position_regime"] == 0.0,
        "ok")

    all_ok = all(c[1] for c in checks)
    print("=" * 64)
    print("fig6_failure_taxonomy.py  self-check")
    print("=" * 64)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail else ""))
    print("-" * 64)
    print(f"scored leaves: {len(scored)}  |  N/A leaves: {len(na_leaves)}  |  "
          f"out: {args.out}")
    print("RESULT:", "ALL CHECKS PASS" if all_ok else "FAILURES PRESENT")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
