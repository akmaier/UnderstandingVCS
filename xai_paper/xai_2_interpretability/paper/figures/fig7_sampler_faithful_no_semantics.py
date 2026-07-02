#!/usr/bin/env python3
"""Figure 7 — the keystone: the bilinear sampler makes the gradient methods FAITHFUL
on the position/index regime, yet none of them names a game concept (semantic recovery is yes/no; the answer is no).

This is the sharpened thesis's foreclosing experiment (plan.md storyline step 4;
honesty-contract claim 4; backlog P2-R7-EXP-sampler). On the discrete position/index
regime — a framebuffer pixel placed by round/argmax of a sprite-position byte, whose
NAIVE gradient is provably zero (Prop. prop:zero) — turning Paper-1's bilinear
index-boundary sampler ON (the SAME differentiable surrogate as the SI joystick tool,
tools/xai_si_gradient/si_joystick_gradient.jl) restores a real position gradient. The
gradient/correlational attribution methods then become FAITHFUL: their per-cause map
correlates with the intervention oracle's true |Δ position-pixel|, off the zero floor.

The point the figure makes in one image: faithfulness is REPAIRABLE (0 -> >0) but
SEMANTICS is NOT (semantic recovery is yes/no; the answer stays no). Repairing the gradient does not make an attribution map
emit or name a single T3 game concept (missile, collision, score, restart); the only
semantic label in play is the imported T3 annotation, used to CHECK localization,
never PRODUCED by the method. So the universal semantic gap is not a fixable technical
artifact of the vanishing gradient — it survives the fix.

Two panels:
  (a)  Per-method faithfulness, naive -> sampler, on the position regime. For each of
       the five gradient methods we plot the naive bar (== 0, the §1 floor) and the
       sampler bar (the mean over the games where the sampler restored a gradient),
       with the per-game sampler points overlaid so the rise is not a single number.
  (b)  The semantics row: no method names a game concept (yes/no; "no" for every method) in BOTH
       conditions — a row of "no" markers, the foreclosure.

ALL numbers are READ from the committed keystone record — no experiment is re-run:
  * tools/xai_study/compare/out/sampler_faithfulness.csv   (P2-R7-EXP-sampler aggregate)
The per-method/per-game §R records under
tools/xai_study/phaseB_attribution/sampler_on/out/*.json are the sources the CSV
aggregates (referenced via the CSV's record_path column, not re-read here).

Run:
    python fig7_sampler_faithful_no_semantics.py
Produces:
    fig7_sampler_faithful_no_semantics.pdf  (vector, colour-blind-safe, self-legend)
"""

import argparse
import csv
import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")  # headless vector backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
# repo root = .../  (figures -> paper -> xai_2_interpretability -> xai_paper -> root)
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))

# ---------------------------------------------------------------------------
# Palette — Okabe-Ito colour-blind-safe set (matches fig4).
# ---------------------------------------------------------------------------
C_BG = "#ffffff"
C_INK = "#1a1a1a"
C_MUTE = "#5b5b5b"
C_GRID = "#d9d9d9"

C_NAIVE = "#999999"        # grey         — naive gradient (the §1 floor, == 0)
C_SAMPLER = "#0072B2"      # blue         — bilinear sampler ON (faithfulness restored)
C_SEM = "#D55E00"          # vermillion   — semantics (flat at 0)
C_POINT = "#005a8d"        # darker blue  — per-game sampler points

FS_MIN = 7.5

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.6,
        "axes.linewidth": 0.8,
        "axes.edgecolor": C_MUTE,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
        "xtick.color": C_INK,
        "ytick.color": C_INK,
        "text.color": C_INK,
    }
)

PRETTY = {
    "vanilla_saliency": "Vanilla saliency",
    "gradxinput": "Grad×Input",
    "smoothgrad": "SmoothGrad",
    "integrated_gradients": "Integrated Gradients",
    "expected_gradients": "Expected Gradients",
}
METHOD_ORDER = [
    "vanilla_saliency", "gradxinput", "smoothgrad",
    "integrated_gradients", "expected_gradients",
]


def load_rows(csv_path):
    rows = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            rows.append(
                {
                    "method": r["method"],
                    "game": r["game"],
                    "regime": r["regime"],
                    "faithfulness_naive": float(r["faithfulness_naive"]),
                    "faithfulness_sampler": float(r["faithfulness_sampler"]),
                    "semantic_recovery": int(r["semantic_recovery"]),
                    "record_path": r["record_path"],
                }
            )
    return rows


def load_headline_gaps(ci_csv_path):
    """Read the committed all-regime (robust) + position (directional) gap CIs
    from leaderboard_ci.csv. Returns {key: (mean, lo, hi)} or {} if absent."""
    gaps = {}
    if not os.path.isfile(ci_csv_path):
        return gaps
    with open(ci_csv_path) as fh:
        for r in csv.DictReader(fh):
            if r.get("kind") == "headline":
                try:
                    gaps[r["method"]] = (float(r["mean"]), float(r["ci_lo"]),
                                         float(r["ci_hi"]))
                except (KeyError, ValueError):
                    pass
    return gaps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        default=os.path.join(
            REPO_ROOT, "tools", "xai_study", "compare", "out",
            "sampler_faithfulness.csv",
        ),
        help="path to the P2-R7-EXP-sampler aggregate CSV",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(HERE, "fig7_sampler_faithful_no_semantics.pdf"),
        help="output PDF path",
    )
    args = ap.parse_args()

    rows = load_rows(args.csv)
    # The aggregate cross-method contrast this figure references is the ROBUST
    # all-regime causal-vs-gradient gap (CI excludes 0), NOT a position-regime
    # win. The position gap is shown honestly (directional, CI includes 0).
    ci_csv = os.path.join(
        REPO_ROOT, "tools", "xai_study", "compare", "out", "leaderboard_ci.csv"
    )
    headline_gaps = load_headline_gaps(ci_csv)

    # per-method aggregates over the position regime ------------------------------
    by_method = {m: [] for m in METHOD_ORDER}
    for r in rows:
        if r["regime"] == "position" and r["method"] in by_method:
            by_method[r["method"]].append(r)

    methods = [m for m in METHOD_ORDER if by_method[m]]
    labels = [PRETTY[m] for m in methods]

    # HONEST aggregate: the sampler bar is the mean over ALL position games (not
    # only the games that rose). Averaging over the rose games alone would inflate
    # the bar to ~0.53–0.79 and falsely read as "the sampler makes the gradient
    # faithful"; the corrected message is that the sampler restores a NONZERO
    # gradient (mechanism) whose faithfulness is LOW/MIXED — nonzero in only a
    # minority of games, and low on average. The per-game points show that spread.
    naive_mean = []          # the naive floor (== 0)
    sampler_mean = []        # mean over ALL position games (low/mixed, honest)
    sampler_pts = []         # per-game sampler points (all games, incl. the zeros)
    n_rose = []              # how many games rose, per method
    for m in methods:
        rs = by_method[m]
        naive_mean.append(float(np.mean([r["faithfulness_naive"] for r in rs])))
        allvals = [r["faithfulness_sampler"] for r in rs]
        rose = [v for v in allvals if v > 1e-9]
        sampler_pts.append(allvals)
        n_rose.append(len(rose))
        sampler_mean.append(float(np.mean(allvals)) if allvals else 0.0)

    # headline numbers (measured, from the CSV)
    rose_pairs = [
        (r["method"], r["game"], r["faithfulness_sampler"])
        for r in rows
        if r["regime"] == "position"
        and r["faithfulness_sampler"] > r["faithfulness_naive"] + 1e-9
    ]
    best = max(rose_pairs, key=lambda t: t[2]) if rose_pairs else ("—", "—", 0.0)
    all_sem_zero = all(r["semantic_recovery"] == 0 for r in rows)
    n_records = len(rows)

    # =======================================================================
    # FIGURE
    # =======================================================================
    fig = plt.figure(figsize=(11.6, 5.9), facecolor=C_BG)
    # In-image title + headline banner removed (they duplicated the LaTeX
    # caption).  The two panel (a)/(b) sub-titles were also removed for
    # decluttering — the panel meaning now lives in the LaTeX caption — so the
    # panel block top is raised and the inter-panel gap tightened.
    gs = fig.add_gridspec(
        2, 1, height_ratios=[3.0, 0.62], left=0.115, right=0.965,
        top=0.965, bottom=0.135, hspace=0.22,
    )
    axF = fig.add_subplot(gs[0, 0])
    axS = fig.add_subplot(gs[1, 0])

    # ---------------- Panel (a): faithfulness naive -> sampler --------------
    nM = len(methods)
    x = np.arange(nM)
    w = 0.38

    axF.bar(x - w / 2 - 0.02, naive_mean, width=w, color=C_NAIVE,
            edgecolor=C_INK, linewidth=0.4, label="naive gradient (sampler off)",
            zorder=3)
    axF.bar(x + w / 2 + 0.02, sampler_mean, width=w, color=C_SAMPLER,
            edgecolor=C_INK, linewidth=0.4,
            label="bilinear sampler ON (mean over all position games)",
            zorder=3)

    # per-game sampler points (jittered) so the rise is shown, not just a mean
    rng = np.random.default_rng(0)
    for xi, pts in zip(x + w / 2 + 0.02, sampler_pts):
        jit = (rng.random(len(pts)) - 0.5) * (w * 0.7)
        axF.scatter(np.full(len(pts), xi) + jit, pts, s=18, color=C_POINT,
                    edgecolor="white", linewidth=0.5, zorder=5)

    # value annotations
    for xi, v in zip(x - w / 2 - 0.02, naive_mean):
        axF.text(xi, v + 0.018, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=FS_MIN, color=C_INTERV_or_mute(v))
    for xi, v, nr in zip(x + w / 2 + 0.02, sampler_mean, n_rose):
        txt = f"{v:.2f}" + (f"\n(≠0 in {nr}/6)" if nr else "")
        axF.text(xi, v + 0.018, txt, ha="center", va="bottom",
                 fontsize=FS_MIN, color=C_MUTE, linespacing=1.05)

    axF.set_xticks(x)
    axF.set_xticklabels(labels, fontsize=8.4)
    # Headroom above 1.0 for the honest aggregate-contrast banner (drawn in the
    # 1.0–1.25 band so it never overlaps the bars or the per-game points).
    axF.set_ylim(0, 1.25)
    axF.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axF.set_ylabel("faithfulness vs intervention oracle\n(Pearson corr, 0–1)",
                   fontsize=9.0)
    axF.yaxis.grid(True, color=C_GRID, linewidth=0.6, zorder=0)
    axF.set_axisbelow(True)
    for s in ("top", "right"):
        axF.spines[s].set_visible(False)

    # the §1 floor line at 0 with a label
    axF.axhline(0.0, color=C_NAIVE, linewidth=1.0, zorder=2)
    axF.text(-0.35, 0.86,
             "naive-gradient floor = 0  (Prop. prop:zero — the position gradient vanishes)",
             ha="left", va="bottom", fontsize=FS_MIN, color=C_MUTE, style="italic")

    # (Panel (a) sub-title removed for decluttering — meaning lives in caption.)

    # ---------------- Panel (b): semantic recovery is a yes/no question -------------
    # Categorical, NOT a number. For every method, in BOTH conditions, no concept is
    # named, so the answer is "no". We render "no" markers, not a value on a scale.
    axS.set_xlim(axF.get_xlim())
    for xi in x:
        axS.text(xi - w / 2 - 0.02, 0.5, "no", ha="center", va="center",
                 fontsize=8.8, color=C_SEM, fontweight="bold")
        axS.text(xi + w / 2 + 0.02, 0.5, "no", ha="center", va="center",
                 fontsize=8.8, color=C_SEM, fontweight="bold")
    axS.set_xticks(x)
    axS.set_xticklabels(labels, fontsize=8.4)
    axS.set_ylim(0, 1.0)
    axS.set_yticks([])
    axS.set_ylabel("names a\nconcept?", fontsize=9.0)
    for s in ("top", "right", "left"):
        axS.spines[s].set_visible(False)
    # (Panel (b) sub-title removed for decluttering — meaning lives in caption.)

    # =======================================================================
    # (In-image title + headline banner removed — the LaTeX caption owns that
    #  prose.  `best`/`n_records` are still computed above for the self-check.)
    # =======================================================================

    # =======================================================================
    # Legend — beneath the panels.
    # =======================================================================
    handles = [
        Patch(facecolor=C_NAIVE, edgecolor=C_INK, linewidth=0.4,
              label="naive gradient (sampler off) — the §1 floor = 0"),
        Patch(facecolor=C_SAMPLER, edgecolor=C_INK, linewidth=0.4,
              label="bilinear sampler ON — gradient restored (nonzero), "
                    "faithfulness low/mixed"),
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=C_POINT, markeredgecolor="white",
                   markersize=6, label="per-game sampler faithfulness (all 6 games)"),
        plt.Line2D([0], [0], marker="$\\mathsf{no}$", color="none",
                   markerfacecolor=C_SEM, markeredgecolor=C_SEM, markersize=12,
                   label="names no game concept (every method, both conditions)"),
    ]
    leg = fig.legend(
        handles=handles, loc="lower left", bbox_to_anchor=(0.115, 0.012),
        ncol=2, frameon=True, fontsize=8.0, handlelength=1.4, borderpad=0.55,
        columnspacing=1.6, labelspacing=0.5,
    )
    leg.get_frame().set_edgecolor(C_GRID)
    leg.get_frame().set_linewidth(0.7)
    leg._legend_box.align = "left"

    # Honest aggregate-contrast line: lead with the ROBUST all-regime gap (CI
    # excludes 0); report the position gap as directional (CI includes 0). This
    # keeps the figure from asserting a large significant position-regime win.
    ag = headline_gaps.get("all_regime_gap")
    pg = headline_gaps.get("position_regime_gap")
    if ag and pg:
        agg_line = (
            f"Aggregate contrast (all methods): causal−gradient faithfulness gap "
            f"{ag[0]:.3f}, 95% CI [{ag[1]:.3f}, {ag[2]:.3f}] over ALL regimes "
            f"(robust, excludes 0).\n"
            f"Position regime alone: gap {pg[0]:.3f}, "
            f"95% CI [{pg[1]:.3f}, {pg[2]:.3f}] (directional, includes 0)."
        )
        # Placed in the headroom at the top of panel (a), above the bars, so it
        # never collides with the bottom legend strip. Two centred lines so the
        # box stays inside the panel width.
        axF.text(
            (nM - 1) / 2.0, 1.235, agg_line, fontsize=FS_MIN, color=C_INK,
            ha="center", va="top", style="italic", linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f7fa",
                      edgecolor=C_GRID, linewidth=0.6),
        )

    fig.text(
        0.965, 0.058,
        "Data: committed keystone record sampler_faithfulness.csv + leaderboard_ci.csv; "
        "6 core games, 120+30-frame state; pure read — no experiment re-run.",
        fontsize=FS_MIN, color=C_MUTE, ha="right", va="bottom",
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

    exists = os.path.isfile(args.out)
    size = os.path.getsize(args.out) if exists else 0
    header_ok = False
    if exists:
        with open(args.out, "rb") as fh:
            header_ok = fh.read(5) == b"%PDF-"
    check("PDF file exists", exists, args.out)
    check("PDF size > 4 KB", size > 4096, f"{size} bytes")
    check("PDF has %PDF- header", header_ok)

    check("records loaded from CSV", n_records > 0, f"n={n_records}")
    check("naive faithfulness is the floor (all == 0)",
          all(r["faithfulness_naive"] == 0.0 for r in rows))
    check(">=1 method rose under the sampler", len(rose_pairs) >= 1,
          f"{len(rose_pairs)} (method,game) pair(s)")
    check("semantic recovery is 'no' for all records, both conditions",
          all_sem_zero)
    check("min in-figure font >= 7.5 pt", FS_MIN >= 7.5, f"FS_MIN={FS_MIN}")

    print("Figure 7 self-check:")
    print("\n".join(msgs))
    print(f"\n{'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'} -> {args.out}")
    sys.exit(0 if ok else 1)


def C_INTERV_or_mute(v):
    # zero naive bars annotated in vermillion+bold-equivalent mute for emphasis
    return "#D55E00" if v == 0.0 else C_MUTE


if __name__ == "__main__":
    main()
