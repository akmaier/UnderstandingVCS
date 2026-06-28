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
representations). A method that trips one of these mechanisms here should not be
trusted for stronger claims about a neural net without further evidence. Each
leaf is annotated with the method(s) that exhibit it and their MEASURED
faithfulness against the intervention oracle (§3.2).

The five mechanism branches (plan.md figure 6; experiment_design.md §7 "Why"
column; §0 danger zone; §3.2 zero-gradient-on-index caveat):

  (1) GRADIENT PATHOLOGIES — vanishing gradient on the discrete index/position
      output (the naive position gradient is *provably zero*, §3.2),
      saturation, shattering, and model-invariance / sanity-check failures
      (Adebayo 2018).
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

The leaf methods shown here are the failure-exhibiting subset of the benchmark.
The full benchmark scores 30 interpretability methods + the intervention oracle
positive control = 31 rows (the leaderboard); the near-ceiling causal methods
(activation patching, causal scrubbing, DAS, tuned lens) are not failure leaves
and are therefore not drawn here.

EVERY measured faithfulness number is READ from a committed record — no
experiment is re-run here. The plotted point estimate F is read from the
aggregate leaderboard; the 95% bootstrap confidence interval (whisker) is read
from the per-game CI record. The conceptual mechanism->leaf mapping is grounded
in experiment_design.md §7 ("Why" column) and plan.md (representativeness
rebuttal). The N/A leaf is the "does not apply — recorded finding" row of
experiment_design.md §5. Full data provenance is in the Supplement.

Two variants are produced (PO #4 — full taxonomy → Supplementary Information,
simplified version → main text):

  * fig6_failure_taxonomy.pdf       SIMPLIFIED main-text version
  * fig6_failure_taxonomy_full.pdf  FULL supplementary version (mechanism notes)

Run:
    python fig6_failure_taxonomy.py \
        --leaderboard tools/xai_study/compare/out/leaderboard.json
"""

import argparse
import csv
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

# Hard floor on every plotted glyph (figure style guide / fig_pass §4).
MIN_FONT = 8.0


# ---------------------------------------------------------------------------
# Palette — Okabe-Ito colour-blind-safe set (matches fig1/fig4).
# ---------------------------------------------------------------------------
C_BG = "#ffffff"
C_INK = "#1a1a1a"           # primary text / outlines
C_MUTE = "#5b5b5b"          # secondary text
C_FAINT = "#767676"         # darkened (was #8a8a8a) for legibility at 8 pt
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
        "font.size": MIN_FONT,
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
    base = os.path.dirname(leaderboard_path)
    with open(os.path.join(base, "faithful_demo.json")) as fh:
        demo = json.load(fh)
    # 95% bootstrap CIs (R-UNC): method -> (ci_lo, ci_hi) bracketing the
    # leaderboard point estimate. Optional — the figure degrades gracefully if
    # the CI record is absent (whiskers simply omitted).
    ci = {}
    ci_path = os.path.join(base, "leaderboard_ci.csv")
    if os.path.exists(ci_path):
        with open(ci_path) as fh:
            for r in csv.DictReader(fh):
                try:
                    ci[r["method"]] = (float(r["ci_lo"]), float(r["ci_hi"]))
                except (KeyError, ValueError):
                    pass
    return lb, demo, ci


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
        "naive position gradient = 0 (§3.2)",
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
# Figure builder.
#   variant="simple"  -> main-text version (no mechanism micro-notes, no
#                        bottom provenance banner; caption owns the prose)
#   variant="full"    -> supplementary version (keeps per-branch mechanism
#                        notes + a one-line provenance pointer)
# ---------------------------------------------------------------------------
def build(lb, demo, ci, out_path, variant="simple"):
    full = (variant == "full")
    F = faith_lookup(lb)

    # Geometry: one row per leaf; branches stacked top->bottom (+ N/A control).
    leaf_rows = []   # (branch_key, method_machine, label, F-or-None)
    branch_spans = []  # (branch_key, title, subtitle, note, y_top, y_bot)
    y = 0.0
    ROW_H = 1.0
    GAP_BRANCH = 0.95  # vertical gap between branches

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

    # Taller rows so every glyph clears 8 pt without overlap.
    fig_h = 0.52 * n_leaf + (3.1 if full else 2.4)
    fig = plt.figure(figsize=(12.6, fig_h))
    # In-image title removed (it duplicated the LaTeX caption); the axes top is
    # raised to fill the band the headline used to occupy.
    bottom = 0.045 if full else 0.035
    top = 0.945 if full else 0.955
    ax = fig.add_axes([0.005, bottom, 0.99, top])
    ax.set_xlim(0, 13.2)
    ax.set_ylim(-1.4, total_h + 0.4)
    ax.invert_yaxis()  # branch 1 at the top
    ax.axis("off")

    # --- column x-anchors -----------------------------------------------------
    X_ROOT = 0.32
    X_BRANCH = 1.10      # where branch boxes start
    X_BRANCH_W = 4.05    # branch box width
    X_LEAF = 5.55        # leaf label start
    X_LEAFTXT_W = 4.55   # leaf label box width
    X_CHIP = 10.45       # measured-F chip centre
    X_BAR0 = 11.05       # F mini-bar baseline
    X_BAR_W = 1.85       # F mini-bar full width (F=1.0)

    def yc(i):
        return i + 0.5

    # --- root spine + title ---------------------------------------------------
    y_first = branch_spans[0][4] + 0.5
    y_last = branch_spans[-1][5] - 0.5
    ax.plot([X_ROOT, X_ROOT], [y_first, y_last], color=C_INK, lw=1.6,
            solid_capstyle="round", zorder=2)
    ax.text(X_ROOT - 0.07, (y_first + y_last) / 2.0,
            "INTERPRETABILITY\nFAILURE MODES\non the VCS",
            rotation=90, ha="center", va="center", fontsize=8.6,
            fontweight="bold", color=C_INK, linespacing=1.18)

    # --- branches -------------------------------------------------------------
    for bkey, btitle, bsub, bnote, yt, yb in branch_spans:
        col = BRANCH_COLORS[bkey]
        ymid = (yt + yb) / 2.0
        ax.plot([X_ROOT, X_BRANCH], [ymid, ymid], color=col, lw=1.4, zorder=2)

        box = FancyBboxPatch(
            (X_BRANCH, yt + 0.16), X_BRANCH_W, (yb - yt) - 0.32,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            linewidth=1.4, edgecolor=col,
            facecolor=col, alpha=0.10, zorder=2,
        )
        ax.add_patch(box)
        ax.add_patch(FancyBboxPatch(
            (X_BRANCH + 0.01, yt + 0.16), 0.07, (yb - yt) - 0.32,
            boxstyle="square,pad=0", linewidth=0, facecolor=col, zorder=3))

        # In the FULL variant the box carries title + subtitle + mechanism note
        # stacked; in the SIMPLE variant only title + subtitle (note dropped).
        if full:
            ax.text(X_BRANCH + 0.24, ymid - 0.34, btitle, ha="left",
                    va="center", fontsize=10.4, fontweight="bold",
                    color=(C_FAINT if bkey == "na" else C_INK))
            ax.text(X_BRANCH + 0.24, ymid + 0.06, bsub, ha="left", va="center",
                    fontsize=8.4, color=(C_FAINT if bkey == "na" else C_MUTE),
                    style="italic")
            ax.text(X_BRANCH + 0.24, ymid + 0.50,
                    "mechanism: " + bnote, ha="left", va="center",
                    fontsize=8.0, color=col, fontweight="bold")
        else:
            ax.text(X_BRANCH + 0.24, ymid - 0.18, btitle, ha="left",
                    va="center", fontsize=10.4, fontweight="bold",
                    color=(C_FAINT if bkey == "na" else C_INK))
            ax.text(X_BRANCH + 0.24, ymid + 0.30, bsub, ha="left", va="center",
                    fontsize=8.2, color=(C_FAINT if bkey == "na" else C_MUTE),
                    style="italic")

        ax.plot([X_BRANCH + X_BRANCH_W, X_LEAF - 0.18], [ymid, ymid],
                color=col, lw=1.2, zorder=2)
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
        ax.plot([X_LEAF - 0.18, X_LEAF - 0.02], [cy, cy], color=col, lw=1.0,
                zorder=2)
        lbl_face = "#f4f4f4" if na else "#ffffff"
        ax.add_patch(FancyBboxPatch(
            (X_LEAF, cy - 0.36), X_LEAFTXT_W, 0.72,
            boxstyle="round,pad=0.015,rounding_size=0.10",
            linewidth=0.8, edgecolor=(C_GRID if na else col),
            facecolor=lbl_face, zorder=3))
        ax.text(X_LEAF + 0.18, cy, label, ha="left", va="center",
                fontsize=8.6, color=(C_FAINT if na else C_INK),
                style=("italic" if na else "normal"))

        if na:
            ax.text(X_CHIP + 0.05, cy, "N/A", ha="center", va="center",
                    fontsize=8.6, fontweight="bold", color=C_FAINT)
            ax.text(X_BAR0 + 0.05, cy, "does not apply", ha="left", va="center",
                    fontsize=8.0, color=C_FAINT, style="italic")
            continue

        # measured-F chip (numeric, coloured by F ramp)
        chip = FAITH_CMAP(FAITH_NORM(fval))
        txt_c = "#ffffff" if fval >= 0.45 else C_INK
        ax.add_patch(FancyBboxPatch(
            (X_CHIP - 0.42, cy - 0.28), 0.84, 0.56,
            boxstyle="round,pad=0.01,rounding_size=0.08",
            linewidth=0.7, edgecolor=C_MUTE, facecolor=chip, zorder=4))
        ax.text(X_CHIP, cy, f"{fval:.2f}", ha="center", va="center",
                fontsize=8.4, fontweight="bold", color=txt_c, zorder=5)

        # F mini-bar
        ax.add_patch(plt.Rectangle((X_BAR0, cy - 0.14), X_BAR_W, 0.28,
                                   facecolor="#eeeeee", edgecolor="none",
                                   zorder=3))
        ax.add_patch(plt.Rectangle((X_BAR0, cy - 0.14), X_BAR_W * fval, 0.28,
                                   facecolor=col, edgecolor="none", zorder=4))

        # 95% bootstrap CI whisker on the bar (R-UNC; read from CI record).
        if m in ci:
            lo, hi = ci[m]
            lo = max(0.0, min(1.0, lo))
            hi = max(0.0, min(1.0, hi))
            x_lo = X_BAR0 + X_BAR_W * lo
            x_hi = X_BAR0 + X_BAR_W * hi
            ax.plot([x_lo, x_hi], [cy, cy], color=C_INK, lw=1.0, zorder=6,
                    solid_capstyle="butt")
            for xx in (x_lo, x_hi):
                ax.plot([xx, xx], [cy - 0.10, cy + 0.10], color=C_INK,
                        lw=1.0, zorder=6)

    # --- column headers -------------------------------------------------------
    head_y = -1.05
    ax.text(X_BRANCH + 0.10, head_y, "MECHANISM BRANCH", ha="left",
            va="center", fontsize=8.4, fontweight="bold", color=C_MUTE)
    ax.text(X_LEAF + 0.10, head_y, "METHOD EXHIBITING IT (phase)", ha="left",
            va="center", fontsize=8.4, fontweight="bold", color=C_MUTE)
    ax.text(X_CHIP, head_y, "F", ha="center", va="center",
            fontsize=8.4, fontweight="bold", color=C_MUTE)
    ax.text(X_BAR0, head_y,
            "faithfulness F vs intervention oracle (0–1, ± 95% CI)", ha="left",
            va="center", fontsize=8.0, color=C_MUTE, style="italic")

    # faint reference grid on the bar axis (0 / 0.5 / 1.0)
    grid_bot = branch_spans[-1][5] - 0.55
    for gx, gl in [(0.0, "0"), (0.5, "0.5"), (1.0, "1.0")]:
        bx = X_BAR0 + X_BAR_W * gx
        ax.plot([bx, bx], [head_y + 0.45, grid_bot],
                color=C_GRID, lw=0.5, zorder=0)
        ax.text(bx, head_y + 0.40, gl, ha="center", va="center",
                fontsize=8.0, color=C_FAINT)

    # --- title ----------------------------------------------------------------
    # (In-image title removed — the LaTeX caption owns that prose.)

    # --- legend (outside the axes, top-right) ---------------------------------
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
        Line2D([0], [0], color=C_INK, lw=1.0, marker="|", markersize=8,
               label="95% bootstrap CI (whisker)"),
    ]
    fig.legend(handles=leg_handles, loc="upper right",
               bbox_to_anchor=(0.998, 0.992), ncol=3, fontsize=8.0,
               frameon=True, framealpha=0.97, edgecolor=C_GRID,
               handlelength=1.4, columnspacing=1.1, borderpad=0.55)

    # --- provenance pointer (FULL variant only; no data paths in figure) ------
    if full:
        prov = ("F = point estimate; whisker = 95% bootstrap CI over games. "
                "All numbers read from committed records; full data provenance "
                "in the Supplement. The full benchmark is 30 interpretability "
                "methods + the intervention oracle positive control = 31 rows; "
                "only the failure-exhibiting subset is drawn here.")
        fig.text(0.012, 0.006, _wrap(prov, 168), ha="left", va="bottom",
                 fontsize=8.0, color=C_MUTE, linespacing=1.3)

    fig.savefig(out_path, format="pdf", facecolor=C_BG)
    plt.close(fig)
    return leaf_rows, F


def _wrap(s, width):
    import textwrap
    return "\n".join(textwrap.wrap(s, width=width))


# ---------------------------------------------------------------------------
def _self_check(out_path, leaf_rows, F, lb, demo, ci, label):
    """Self-check for one rendered variant; returns (all_ok, lines)."""
    checks = []

    def chk(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    ok_pdf = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    magic = b""
    size = os.path.getsize(out_path) if ok_pdf else 0
    if ok_pdf:
        with open(out_path, "rb") as fh:
            magic = fh.read(5)
    chk("pdf written, %PDF- header, > 4 KB",
        ok_pdf and magic == b"%PDF-" and size > 4096,
        f"size={size}B magic={magic!r}")

    scored = [lr for lr in leaf_rows if lr[3] is not None]
    all_in_range = all(0.0 <= lr[3] <= 1.0 for lr in scored)
    chk("every scored leaf F in [0,1] (read from leaderboard)",
        all_in_range and len(scored) >= 19, f"n_scored_leaves={len(scored)}")

    traced = all(abs(lr[3] - F[lr[1]]) < 1e-9 for lr in scored)
    chk("each plotted F traces exactly to leaderboard.json",
        traced, "ok" if traced else "MISMATCH")

    branch_keys = {lr[0] for lr in leaf_rows}
    expect = {"grad", "sample", "corr", "decomp", "comp", "na"}
    chk("all 5 mechanism branches + N/A control present",
        branch_keys == expect, f"{sorted(branch_keys)}")

    na_leaves = [lr for lr in leaf_rows if lr[0] == "na"]
    chk("N/A control leaves present and unscored",
        len(na_leaves) >= 3 and all(lr[3] is None for lr in na_leaves),
        f"n_na={len(na_leaves)}")

    grad_F = [lr[3] for lr in leaf_rows if lr[0] == "grad"]
    chk("gradient branch mean-F below the exact lesion ceiling",
        np.mean(grad_F) < F["A2_lesions"],
        f"grad_mean={np.mean(grad_F):.3f} < lesion={F['A2_lesions']:.3f}")

    chk("position-regime gradient floor == 0 (faithful_demo)",
        demo["popular_method"]["faithfulness_position_regime"] == 0.0, "ok")

    # CIs that were drawn bracket their leaderboard point estimate.
    ci_ok = True
    n_ci = 0
    for lr in scored:
        m = lr[1]
        if m in ci:
            n_ci += 1
            lo, hi = ci[m]
            if not (lo - 1e-6 <= F[m] <= hi + 1e-6):
                ci_ok = False
    chk("every drawn CI brackets its leaderboard point estimate",
        ci_ok and n_ci >= 19, f"n_ci_whiskers={n_ci}")

    all_ok = all(c[1] for c in checks)
    lines = [f"  [{'PASS' if okx else 'FAIL'}] {name}"
             + (f"  ({detail})" if detail else "")
             for name, okx, detail in checks]
    lines.append(f"  scored leaves: {len(scored)}  |  N/A leaves: "
                 f"{len(na_leaves)}  |  CI whiskers: {n_ci}  |  out: {out_path}")
    return all_ok, lines


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
        help="output path for the SIMPLIFIED main-text PDF",
    )
    ap.add_argument(
        "--out-full",
        default=os.path.join(HERE, "fig6_failure_taxonomy_full.pdf"),
        help="output path for the FULL supplementary PDF",
    )
    args = ap.parse_args()

    if not os.path.exists(args.leaderboard):
        sys.exit(f"ERROR: leaderboard not found: {args.leaderboard}")

    lb, demo, ci = load_records(args.leaderboard)

    # Simplified main-text version + full supplementary version (PO #4).
    rows_s, F = build(lb, demo, ci, args.out, variant="simple")
    rows_f, _ = build(lb, demo, ci, args.out_full, variant="full")

    print("=" * 70)
    print("fig6_failure_taxonomy.py  self-check")
    print("=" * 70)
    all_ok = True
    for label, out, rows in [("SIMPLE (main)", args.out, rows_s),
                             ("FULL (supplement)", args.out_full, rows_f)]:
        ok, lines = _self_check(out, rows, F, lb, demo, ci, label)
        all_ok = all_ok and ok
        print(f"-- {label} --")
        for ln in lines:
            print(ln)
    print("-" * 70)
    print("RESULT:", "ALL CHECKS PASS" if all_ok else "FAILURES PRESENT")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
