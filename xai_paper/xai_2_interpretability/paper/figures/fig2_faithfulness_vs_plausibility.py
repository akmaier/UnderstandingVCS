#!/usr/bin/env python3
"""Figure 2 — THE HEADLINE: A-C on the shared faithfulness-vs-plausibility axes (P2-E7-2).

Every interpretability method from the three traditions (Phase A neuroscience/Kording,
Phase B attribution/XAI, Phase C mechanistic-interp) placed on the paper's two reporting
axes (experiment_design.md §0):

    X = FAITHFULNESS    — vs the intervention oracle (§3.2 ground truth). [0,1], higher
                          = closer to the TRUE causes. Horizontal whiskers are the
                          bootstrap-over-games 95% CI (leaderboard_ci.csv, P2-R-UNC).
    Y = PLAUSIBILITY    — the documented per-tradition plausibility PROXY (how convincing
                          the explanation LOOKS to a reader). [0,1]. A documented proxy,
                          NOT a human-subjects measurement — it is exactly what exposes
                          the §0 DANGER ZONE: high plausibility, low faithfulness.

The danger zone (upper-left: looks-convincing-but-isn't-true) is drawn explicitly. A
compact key-results table (right panel, no callout boxes) carries the four headline rows;
the legends sit OUTSIDE the plotting area in a strip beneath the axes.

LAYOUT (figure-detail-pass redraw):
  Panel A (left)  — the scatter only: 30 methods + the oracle positive control = 31 rows,
                    each with a horizontal 95% CI whisker; the danger zone washed; legends
                    in a strip below the axes (never inside the data field).
  Panel B (right) — a readable key-results table (finding / number / interpretation),
                    replacing the old floating callout boxes and the in-figure contrast box.

DATA — read-only, every plotted number traces to a committed record (no re-run, no
fabrication):
  * tools/xai_study/compare/out/leaderboard.json     (P2-E6-1) — 30 methods + ORACLE = 31
    rows; each row's faithfulness, plausibility_proxy, tradition, phase.
  * tools/xai_study/compare/out/leaderboard_ci.csv   (P2-R-UNC) — per-method bootstrap-
    over-games 95% CI (ci_lo / ci_hi) attached as the horizontal whiskers.
  * tools/xai_study/compare/out/faithful_demo.json   (P2-E6-3) — the near-ceiling
    (activation patching, F=1.0) vs near-chance (vanilla saliency, F=0.267) contrast,
    used for the table's headline contrast row and to cross-check plotted endpoints.

Run:
    python fig2_faithfulness_vs_plausibility.py
  (all input paths default to the committed records, resolved relative to the repo root.)

Produces:
    fig2_faithfulness_vs_plausibility.pdf   (vector, colour-blind-safe, self-contained)
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless vector backend
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Locate the committed data records (E6-1 leaderboard, R-UNC CIs, E6-3 demo).
# The repo root is the ancestor that contains tools/xai_study/.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))


def find_repo_root(start):
    d = start
    for _ in range(12):
        if os.path.isdir(os.path.join(d, "tools", "xai_study")):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    raise SystemExit("could not locate repo root (tools/xai_study not found above figure)")


REPO = find_repo_root(HERE)
DEFAULT_LB = os.path.join(REPO, "tools", "xai_study", "compare", "out", "leaderboard.json")
DEFAULT_CI = os.path.join(REPO, "tools", "xai_study", "compare", "out", "leaderboard_ci.csv")
DEFAULT_DEMO = os.path.join(REPO, "tools", "xai_study", "compare", "out", "faithful_demo.json")

ap = argparse.ArgumentParser(description="Figure 2 — faithfulness vs plausibility (headline).")
ap.add_argument("--leaderboard", default=DEFAULT_LB,
                help="path to E6-1 leaderboard.json (default: committed record)")
ap.add_argument("--ci", default=DEFAULT_CI,
                help="path to R-UNC leaderboard_ci.csv (default: committed record)")
ap.add_argument("--demo", default=DEFAULT_DEMO,
                help="path to E6-3 faithful_demo.json (default: committed record)")
args = ap.parse_args()

with open(args.leaderboard) as fh:
    LB = json.load(fh)
with open(args.demo) as fh:
    DEMO = json.load(fh)

# Per-method bootstrap-over-games 95% CI on faithfulness (R-UNC record).
CI = {}
with open(args.ci) as fh:
    for row in csv.DictReader(fh):
        if row.get("kind") != "method":
            continue
        try:
            CI[row["method"]] = (float(row["ci_lo"]), float(row["ci_hi"]))
        except (TypeError, ValueError):
            pass

ROWS = LB["rows"]
assert ROWS, "leaderboard has no rows"

# ---------------------------------------------------------------------------
# Palette — Okabe-Ito colour-blind-safe, one colour per method TRADITION.
# ---------------------------------------------------------------------------
C_BG = "#ffffff"
C_INK = "#1a1a1a"
C_MUTE = "#5b5b5b"
C_DANGER = "#CC79A7"   # reddish-purple — the danger-zone wash + flag
C_ORACLE = "#000000"   # the oracle ceiling star

# Okabe-Ito hues mapped to traditions. Colour = WHY a method is (un)faithful by
# construction; marker shape = which experimental PHASE/battery it came from.
TRAD_COLOR = {
    "causal":        "#0072B2",  # blue        — causal mediation / patching (faithful by construction)
    "intervention":  "#56B4E9",  # sky-blue    — do(u)-based attribution (occlusion, counterfactual, lesions)
    "gradient":      "#D55E00",  # vermillion  — gradient/saliency family (vanishes on index outputs)
    "correlational": "#E69F00",  # orange      — tuning/Granger/spike-word (correlational neuroscience)
    "probing":       "#CC79A7",  # red-purple  — linear probing (present != used)
    "dim_reduction": "#009E73",  # green       — NMF/PCA/SAE dictionaries
    "descriptive":   "#999999",  # grey        — whole-state recording baseline
    "oracle":        C_ORACLE,
}
TRAD_LABEL = {
    "causal": "causal mediation / patching",
    "intervention": "intervention / occlusion",
    "gradient": "gradient / saliency",
    "correlational": "correlational (tuning, Granger, …)",
    "probing": "linear probing",
    "dim_reduction": "dim-reduction (NMF/PCA/SAE)",
    "descriptive": "descriptive baseline",
}

# Marker = experimental PHASE / battery.
PHASE_MARKER = {
    "phaseA_kording": "o",      # circle   — Phase A (neuroscience / Kording battery)
    "phaseB_attribution": "s",  # square   — Phase B (attribution / XAI)
    "phaseC_mechanistic": "^",  # triangle — Phase C (mechanistic interp)
    "ground_truth": "*",        # star     — the oracle ceiling
}
PHASE_LABEL = {
    "phaseA_kording": "Phase A — neuroscience battery (Kording)",
    "phaseB_attribution": "Phase B — attribution / XAI",
    "phaseC_mechanistic": "Phase C — mechanistic interpretability",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,   # embed TrueType (editable, not bitmap)
        "svg.fonttype": "none",
        "figure.dpi": 150,
    }
)

# ---------------------------------------------------------------------------
# Pull every row into a tidy record. Faithfulness/plausibility come straight
# from the committed leaderboard; the 95% CI from the R-UNC record. No value
# is computed here.
# ---------------------------------------------------------------------------
def pretty(method):
    """Human-readable method name (strip the A#_/phase prefixes, tidy)."""
    m = method
    # strip "A1_".."A8_" Phase-A prefixes
    if len(m) > 3 and m[0] == "A" and m[1].isdigit() and m[2] == "_":
        m = m[3:]
    m = m.replace("_", " ")
    fixes = {
        "connectomics": "connectomics",
        "lesions": "single-unit lesions",
        "tuning": "tuning curves",
        "spike word": "spike-word corr.",
        "local field potentials": "local field potentials",
        "granger": "Granger causality",
        "dim reduction": "NMF/PCA dim-reduction",
        "wholestate": "whole-state recording",
        "expected gradients": "expected gradients",
        "extremal perturbation": "extremal perturbation",
        "gradxinput deeplift": "Grad×Input / DeepLIFT",
        "guided backprop": "guided backprop",
        "integrated gradients": "integrated gradients",
        "kernelshap": "KernelSHAP",
        "lime": "LIME",
        "occlusion": "occlusion",
        "on distribution counterfactual": "on-distrib. counterfactual",
        "rise": "RISE",
        "smoothgrad": "SmoothGrad",
        "vanilla saliency": "vanilla saliency",
        "ACDC": "ACDC",
        "activation patching": "activation patching",
        "attribution patching": "attribution patching",
        "causal scrubbing": "causal scrubbing",
        "interchange interventions das": "interchange / DAS",
        "linear probing control tasks": "linear probing (+control)",
        "logit tuned lens": "logit / tuned lens",
        "nmf pca dictionaries": "NMF/PCA dictionaries",
        "path patching": "path patching",
        "sparse autoencoder": "sparse autoencoder",
    }
    return fixes.get(m, m)


methods = []   # dict(faith, plaus, tradition, phase, method, name, ci_lo, ci_hi)
oracle = None
for r in ROWS:
    f = r.get("faithfulness")
    p = r.get("plausibility_proxy")
    if f is None or p is None:
        continue
    lo, hi = CI.get(r["method"], (float(f), float(f)))
    rec = dict(faith=float(f), plaus=float(p), tradition=r["tradition"],
               phase=r["phase"], method=r["method"], name=pretty(r["method"]),
               ci_lo=lo, ci_hi=hi)
    if r["phase"] == "ground_truth":
        oracle = rec
    else:
        methods.append(rec)

n_methods = len(methods)
N_TOTAL = n_methods + (1 if oracle is not None else 0)   # canonical: 30 methods + oracle = 31
assert n_methods == 30, f"expected 30 method rows (canonical count), got {n_methods}"
assert N_TOTAL == 31, f"expected 31 rows total (30 methods + oracle), got {N_TOTAL}"

# ---------------------------------------------------------------------------
# Canvas — two columns: Panel A scatter (left) + Panel B key-results table
# (right). Legends live in a strip BELOW the axes (outside the data field).
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 11.6, 8.0
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C_BG)

# Panel A: the scatter.  (left, bottom, width, height) in figure fractions.
# Bottom raised to clear room for the x-axis label AND the legend strip below it.
ax = fig.add_axes([0.066, 0.300, 0.575, 0.590])
ax.set_xlim(-0.03, 1.07)
ax.set_ylim(0.20, 1.05)
ax.set_xlabel("FAITHFULNESS  →  recovers the TRUE causes\n"
              "(vs the intervention oracle, §3.2;  1.0 = oracle ceiling)",
              fontsize=10.0, labelpad=7)
ax.set_ylabel("PLAUSIBILITY  →  how convincing it LOOKS\n"
              "(documented per-tradition proxy — not a measurement)",
              fontsize=10.0, labelpad=7)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.tick_params(labelsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_axisbelow(True)
ax.grid(True, color="#e9e9e9", linewidth=0.8, zorder=0)

# ---------------------------------------------------------------------------
# THE DANGER ZONE — high plausibility, low faithfulness (the §0 trap).
# Threshold: plausibility >= 0.70 and faithfulness <= 0.40 (the upper-left box).
# ---------------------------------------------------------------------------
DZ_X = 0.40   # faithfulness ceiling of the danger zone
DZ_Y = 0.70   # plausibility floor of the danger zone
ax.add_patch(Rectangle((-0.03, DZ_Y), DZ_X - (-0.03), 1.05 - DZ_Y,
                        facecolor=C_DANGER, alpha=0.10, edgecolor=C_DANGER,
                        linewidth=1.4, linestyle=(0, (5, 3)), zorder=1))
ax.text(0.385, 1.030, "DANGER ZONE", ha="right", va="top",
        fontsize=12.0, fontweight="bold", color=C_DANGER, zorder=6)
ax.text(0.385, 0.990, "looks convincing  ·  isn't faithful", ha="right", va="top",
        fontsize=8.4, style="italic", color=C_DANGER, zorder=6)

# faithful corner cue (lower-right, near the ceiling)
ax.text(1.055, 0.305, "FAITHFUL\nCLUSTER", ha="right", va="bottom",
        fontsize=9.0, fontweight="bold", color=TRAD_COLOR["causal"], zorder=6,
        linespacing=1.0)

# ---------------------------------------------------------------------------
# Scatter every method. Colour = tradition, marker = phase.
# A faint jitter ONLY on overlapping (faith,plaus) coincidences so the dense
# causal-ceiling cluster (four methods at exactly (1.0, 0.5)) is all visible;
# jitter is cosmetic and < 0.012 and never crosses a zone boundary.
# Horizontal whiskers = the bootstrap 95% CI (R-UNC), anchored on the plotted
# faithfulness (the CI mean equals it to rounding for all rows).
# ---------------------------------------------------------------------------
from collections import defaultdict
coincid = defaultdict(list)
for i, m in enumerate(methods):
    coincid[(round(m["faith"], 4), round(m["plaus"], 4))].append(i)

JIT = 0.011
plotted = {}
for key, idxs in coincid.items():
    k = len(idxs)
    for j, i in enumerate(idxs):
        m = methods[i]
        dx = dy = 0.0
        if k > 1:
            import math
            ang = 2 * math.pi * j / k
            dx = JIT * math.cos(ang)
            dy = JIT * math.sin(ang)
        plotted[i] = (m["faith"] + dx, m["plaus"] + dy)

# CI whiskers first (under markers).
for i, m in enumerate(methods):
    x, y = plotted[i]
    lo, hi = m["ci_lo"], m["ci_hi"]
    if hi - lo > 1e-6:
        ax.plot([lo, hi], [y, y], color=TRAD_COLOR.get(m["tradition"], "#777777"),
                linewidth=1.1, alpha=0.45, solid_capstyle="butt", zorder=3)
        for xc in (lo, hi):
            ax.plot([xc, xc], [y - 0.009, y + 0.009],
                    color=TRAD_COLOR.get(m["tradition"], "#777777"),
                    linewidth=1.1, alpha=0.45, zorder=3)

for i, m in enumerate(methods):
    x, y = plotted[i]
    ax.scatter(x, y, s=120, marker=PHASE_MARKER[m["phase"]],
               facecolor=TRAD_COLOR.get(m["tradition"], "#777777"),
               edgecolor="white", linewidth=1.1, alpha=0.97, zorder=5)

# the oracle ceiling
if oracle is not None:
    ax.scatter(oracle["faith"], oracle["plaus"], s=400, marker="*",
               facecolor="#ffffff", edgecolor=C_ORACLE, linewidth=1.8, zorder=7)
    ax.annotate("ORACLE\nceiling (1, 1)", (oracle["faith"], oracle["plaus"]),
                xytext=(oracle["faith"] - 0.010, oracle["plaus"] - 0.050),
                ha="right", va="top", fontsize=8.2, fontweight="bold",
                color=C_INK, zorder=8, linespacing=1.0)

# ---------------------------------------------------------------------------
# Annotate the worst danger-zone offenders + the faithful cluster.
# Each label's coordinates ARE the plotted (data) coordinates → no fabrication.
# ---------------------------------------------------------------------------
def idx_of(name):
    for i, m in enumerate(methods):
        if m["method"] == name:
            return i
    raise KeyError(name)


def label(name, dx, dy, ha="left", va="center", col=None, weight="normal",
          arrow=True, fs=8.0):
    i = idx_of(name)
    m = methods[i]
    px, py = plotted[i]
    tx, ty = px + dx, py + dy
    c = col if col is not None else TRAD_COLOR.get(m["tradition"], C_INK)
    txt = f"{m['name']}\nF={m['faith']:.3f}"
    if arrow:
        ax.annotate(txt, (px, py), xytext=(tx, ty), ha=ha, va=va,
                    fontsize=fs, color=c, fontweight=weight, linespacing=0.95,
                    zorder=9,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.8,
                                    shrinkA=2, shrinkB=4, alpha=0.8))
    else:
        ax.text(tx, ty, txt, ha=ha, va=va, fontsize=fs, color=c,
                fontweight=weight, linespacing=0.95, zorder=9)


# DANGER-ZONE offenders: expected_gradients, A3 tuning, A6 granger, vanilla
# saliency, guided_backprop, linear_probing. Labels fanned out around the
# upper-left cluster with leader lines, each (faith) is the committed record.
label("expected_gradients",            dx=0.010, dy=0.075, ha="center", va="bottom",
      weight="bold")
label("A3_tuning",                      dx=-0.085, dy=0.052, ha="right", va="bottom")
label("A6_granger",                     dx=-0.070, dy=0.006, ha="right", va="center")
label("vanilla_saliency",              dx=-0.060, dy=-0.066, ha="right", va="top",
      weight="bold")
label("guided_backprop",               dx=0.090, dy=-0.030, ha="left",  va="center")
label("linear_probing_control_tasks",  dx=0.100, dy=-0.060, ha="left",  va="top",
      weight="bold")

# FAITHFUL cluster (near ceiling): activation patching, occlusion, counterfactual.
# activation patching is placed to the LEFT so its label never clips the right edge.
label("activation_patching",           dx=-0.030, dy=0.082, ha="right", va="bottom",
      weight="bold")
label("occlusion",                     dx=0.060, dy=0.018, ha="left", va="center")
label("on_distribution_counterfactual", dx=0.060, dy=-0.050, ha="left", va="center")

# ---------------------------------------------------------------------------
# Title + subtitle (figure-level, above Panel A).
# ---------------------------------------------------------------------------
fig.text(0.066, 0.963,
         "Plausible ≠ faithful: every interpretability method scored where the "
         "truth is known exactly",
         ha="left", fontsize=12.0, fontweight="bold", color=C_INK)
fig.text(0.066, 0.928,
         f"All {N_TOTAL} rows (30 methods + the oracle positive control) across the three "
         "traditions, vs the intervention oracle — the most plausible-looking methods are "
         "the least faithful.",
         ha="left", fontsize=8.6, color=C_MUTE, style="italic")

# ---------------------------------------------------------------------------
# Legends — OUTSIDE the plotting area, in a strip beneath Panel A.
# Two figure-level legends side by side (tradition = colour, battery = marker).
# ---------------------------------------------------------------------------
trad_present = [t for t in ["causal", "intervention", "gradient", "correlational",
                            "probing", "dim_reduction", "descriptive"]
                if any(m["tradition"] == t for m in methods)]
trad_handles = [
    Line2D([0], [0], marker="o", linestyle="none", markersize=9.0,
           markerfacecolor=TRAD_COLOR[t], markeredgecolor="white",
           markeredgewidth=1.0, label=TRAD_LABEL[t])
    for t in trad_present
]
leg1 = fig.legend(handles=trad_handles, title="tradition (colour)",
                  loc="lower left", bbox_to_anchor=(0.066, 0.018),
                  ncol=4, frameon=True, fontsize=8.0, title_fontsize=8.6,
                  handletextpad=0.4, columnspacing=1.4, labelspacing=0.45,
                  borderpad=0.6)
leg1.get_frame().set_edgecolor("#cccccc")
leg1.get_frame().set_linewidth(0.8)
leg1._legend_box.align = "left"

phase_handles = [
    Line2D([0], [0], marker=PHASE_MARKER[p], linestyle="none", markersize=9.0,
           markerfacecolor="#777777", markeredgecolor="white",
           markeredgewidth=1.0, label=PHASE_LABEL[p])
    for p in ["phaseA_kording", "phaseB_attribution", "phaseC_mechanistic"]
] + [
    Line2D([0], [0], marker="*", linestyle="none", markersize=12,
           markerfacecolor="#ffffff", markeredgecolor=C_ORACLE,
           markeredgewidth=1.4, label="oracle ceiling"),
] + [
    Line2D([0], [0], color="#777777", linewidth=1.2, alpha=0.55,
           label="95% CI (bootstrap over games)"),
]
leg2 = fig.legend(handles=phase_handles, title="experimental battery (marker)  +  uncertainty",
                  loc="lower left", bbox_to_anchor=(0.066, 0.140),
                  ncol=3, frameon=True, fontsize=8.0, title_fontsize=8.6,
                  handletextpad=0.4, columnspacing=1.4, labelspacing=0.45,
                  borderpad=0.6)
leg2.get_frame().set_edgecolor("#cccccc")
leg2.get_frame().set_linewidth(0.8)
leg2._legend_box.align = "left"

# ---------------------------------------------------------------------------
# Panel B — the key-results TABLE (replaces the floating callout boxes and the
# in-figure contrast box).  Columns: finding · number · interpretation.
# Every number reads from the committed records (leaderboard.json / R-UNC CI /
# faithful_demo.json) — no value is invented.
# ---------------------------------------------------------------------------
def m_by(name):
    return next(m for m in methods if m["method"] == name)


def famF(family_method):
    """Read a family mean faithfulness from the R-UNC family rows."""
    with open(args.ci) as fh:
        for row in csv.DictReader(fh):
            if row.get("kind") == "family" and row.get("method") == family_method \
                    and row.get("phase") == "ALL":
                return float(row["mean"])
    return None


fam_faithful = famF("family_causal_intervention")          # causal+intervention mean
fam_popular = famF("family_gradient_correlational")        # gradient+correlational mean
ap_rec = m_by("activation_patching")
vs_rec = m_by("vanilla_saliency")
eg_rec = m_by("expected_gradients")

# Build the table rows: (finding, number, interpretation).
rows_tbl = [
    ("Faithful cluster",
     f"F̄ = {fam_faithful:.3f}" if fam_faithful is not None else "—",
     "causal / intervention\nmethods recover the\ntrue causes"),
    ("Danger zone",
     f"F̄ = {fam_popular:.3f}" if fam_popular is not None else "—",
     "gradient / correlational:\nfamiliar yet least\nfaithful"),
    ("Big contrast",
     f"{vs_rec['faith']:.2f} → {ap_rec['faith']:.2f}",
     "vanilla saliency vs\nactivation patching\n(default → faithful)"),
    ("Worst offender",
     f"F = {eg_rec['faith']:.3f}",
     "expected gradients:\nhigh plausibility,\nnear-zero F"),
    ("Oracle control",
     "F = 1.000",
     "intervention oracle\nanchors the ceiling\nat (1, 1)"),
]

# Table panel axes (right column), no spines.  Tracks Panel A's box.
tax = fig.add_axes([0.675, 0.300, 0.315, 0.590])
tax.set_xlim(0, 1)
tax.set_ylim(0, 1)
tax.axis("off")

tax.text(0.0, 1.005, "KEY RESULTS", ha="left", va="top",
         fontsize=10.5, fontweight="bold", color=C_INK)
tax.text(0.0, 0.965, "every number reads from a committed record",
         ha="left", va="top", fontsize=7.6, style="italic", color=C_MUTE)

# column header.  Three columns: finding (x=0.035) · number (x=0.34) ·
# interpretation (left-aligned at x=0.575) — wide gap so they never abut.
CX_FIND, CX_NUM, CX_INT = 0.035, 0.385, 0.640
hy = 0.905
tax.text(CX_FIND, hy, "finding", ha="left", va="bottom", fontsize=8.4,
         fontweight="bold", color=C_MUTE)
tax.text(CX_NUM, hy, "number", ha="left", va="bottom", fontsize=8.4,
         fontweight="bold", color=C_MUTE)
tax.text(CX_INT, hy, "interpretation", ha="left", va="bottom", fontsize=8.4,
         fontweight="bold", color=C_MUTE)
tax.plot([0.0, 1.0], [hy - 0.012, hy - 0.012], color="#bbbbbb", linewidth=1.0)

row_h = 0.165
top = hy - 0.05
ROW_TINT = {
    0: ("#eaf3fa", TRAD_COLOR["causal"]),       # faithful cluster — blue
    1: ("#fbeee6", TRAD_COLOR["gradient"]),     # danger zone — vermillion
    2: ("#f3f3f3", C_INK),                       # contrast — neutral
    3: ("#f7eef4", C_DANGER),                    # worst offender — danger purple
    4: ("#ededed", C_ORACLE),                    # oracle — black
}
for k, (finding, number, interp) in enumerate(rows_tbl):
    yc = top - k * row_h
    tint, accent = ROW_TINT[k]
    tax.add_patch(Rectangle((0.0, yc - row_h + 0.022), 1.0, row_h - 0.028,
                            facecolor=tint, edgecolor="none", zorder=0))
    # accent rule on the left
    tax.add_patch(Rectangle((0.0, yc - row_h + 0.022), 0.012, row_h - 0.028,
                            facecolor=accent, edgecolor="none", zorder=1))
    tax.text(CX_FIND, yc, finding, ha="left", va="top", fontsize=8.5,
             fontweight="bold", color=C_INK, zorder=2)
    tax.text(CX_NUM, yc, number, ha="left", va="top", fontsize=8.7,
             fontweight="bold", color=accent, zorder=2, family="DejaVu Sans")
    tax.text(CX_INT, yc, interp, ha="left", va="top", fontsize=7.7,
             color=C_INK, zorder=2, linespacing=1.08)

# self-contained provenance pointer (no source path drawn): supplement only.
tax.text(0.0, top - len(rows_tbl) * row_h + 0.018,
         "Means: causal/intervention vs gradient/correlational families;\n"
         "contrast & worst-offender numbers cross-checked against the\n"
         "Phase-A/B/C records (see Supplement number-provenance table).",
         ha="left", va="top", fontsize=7.2, style="italic", color=C_MUTE,
         linespacing=1.15)

# ---------------------------------------------------------------------------
# Save (vector PDF)
# ---------------------------------------------------------------------------
OUT = os.path.join(HERE, "fig2_faithfulness_vs_plausibility.pdf")
fig.savefig(OUT, format="pdf", facecolor=C_BG)
plt.close(fig)

# ---------------------------------------------------------------------------
# Self-check — file exists, non-empty, %PDF- header, size > 4 KB; and the
# headline numbers I drew match the committed records.
# ---------------------------------------------------------------------------
sz = os.path.getsize(OUT)
assert sz > 4000, f"PDF suspiciously small: {sz} bytes"
with open(OUT, "rb") as fh:
    head = fh.read(5)
assert head == b"%PDF-", f"not a PDF (header={head!r})"

# canonical count: 30 methods + oracle = 31 rows.
assert n_methods == 30 and N_TOTAL == 31, "canonical count broken"

# cross-check: the contrast endpoints I drew are exactly the E6-3 records.
assert abs(ap_rec["faith"] - DEMO["faithful_method"]["faithfulness_all_regimes"]) < 1e-9, \
    "activation_patching faithfulness disagrees between leaderboard and faithful_demo"
assert abs(vs_rec["faith"] - DEMO["popular_method"]["faithfulness_all_regimes"]) < 1e-9, \
    "vanilla_saliency faithfulness disagrees between leaderboard and faithful_demo"
# danger zone really contains its named offenders
for nm in ("expected_gradients", "linear_probing_control_tasks", "A3_tuning",
           "A6_granger", "guided_backprop"):
    m = next(mm for mm in methods if mm["method"] == nm)
    assert m["plaus"] >= DZ_Y and m["faith"] <= DZ_X, \
        f"{nm} not actually in danger zone (F={m['faith']}, P={m['plaus']})"
# faithful cluster really near ceiling
m = m_by("activation_patching")
assert m["faith"] >= 0.95, f"activation_patching not near ceiling (F={m['faith']})"
# every plotted method carries a CI from the R-UNC record (or a degenerate one)
for m in methods:
    assert m["ci_lo"] <= m["faith"] + 1e-6 and m["ci_hi"] >= m["faith"] - 1e-6, \
        f"{m['method']} faithfulness outside its CI ({m['ci_lo']},{m['ci_hi']})"

print(f"[OK] wrote {OUT}  ({sz} bytes, header {head!r})")
print(f"[OK] plotted 30 methods + oracle = {N_TOTAL} rows; danger zone = F<={DZ_X}, P>={DZ_Y}")
print(f"[OK] contrast (E6-3): vanilla saliency F={vs_rec['faith']:.3f} "
      f"→ activation patching F={ap_rec['faith']:.3f}")
print(f"[OK] family means (R-UNC): faithful={fam_faithful}, popular={fam_popular}")
