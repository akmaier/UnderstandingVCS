#!/usr/bin/env python3
"""Figure 2 — THE HEADLINE: A-C on the shared faithfulness-vs-plausibility axes (P2-E7-2).

Every interpretability method from the three traditions (Phase A neuroscience/Kording,
Phase B attribution/XAI, Phase C mechanistic-interp) placed on the paper's two reporting
axes (experiment_design.md §0):

    X = FAITHFULNESS   — vs the §1 exact intervention oracle (ground truth). [0,1], higher
                          = closer to the TRUE causes. (triad.F where available, else the
                          method's re-oriented vs-oracle value; see leaderboard provenance.)
    Y = PLAUSIBILITY   — the documented per-tradition plausibility proxy (how convincing the
                          explanation LOOKS to a human reader). [0,1]. This is a documented
                          proxy, NOT a measurement — it is exactly what exposes the §0
                          DANGER ZONE: high plausibility, low faithfulness.

The danger zone (upper-left: looks-convincing-but-isn't-true) is drawn explicitly, and its
worst offenders are annotated — expected_gradients, linear_probing(+control), A3 tuning
curves, A6 Granger, guided_backprop — alongside the faithful cluster near the oracle
ceiling: activation_patching (F=1.000), occlusion and on-distribution counterfactual
(faithful intervention-based attribution). The oracle itself sits at (1,1) as the ceiling.

This is the figure the whole paper builds toward: plausible != faithful, in numbers, on a
system where the answer is known exactly.

DATA — read-only, every plotted number traces to a committed record (no re-run, no
fabrication):
  * tools/xai_study/compare/out/leaderboard.json   (P2-E6-1)  — all 31 rows (30 methods +
    ORACLE), each row's faithfulness, plausibility_proxy, tradition, phase, faith_source.
  * tools/xai_study/compare/out/faithful_demo.json  (P2-E6-3) — the near-ceiling (activation
    patching, F=1.0) vs near-chance (vanilla saliency, F=0.267) contrast, used to draw the
    faithful<->popular contrast arrow and to cross-check the plotted endpoints.

Run:
    python fig2_faithfulness_vs_plausibility.py \
        --leaderboard tools/xai_study/compare/out/leaderboard.json
  (the --leaderboard path defaults to the committed E6-1 record, resolved relative to the
   repo root that contains this figure's tree, so the script also runs argument-free.)

Produces:
    fig2_faithfulness_vs_plausibility.pdf   (vector, colour-blind-safe, self-contained)
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless vector backend
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch

# ---------------------------------------------------------------------------
# Locate the committed data records (E6-1 leaderboard, E6-3 faithful demo).
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
DEFAULT_DEMO = os.path.join(REPO, "tools", "xai_study", "compare", "out", "faithful_demo.json")

ap = argparse.ArgumentParser(description="Figure 2 — faithfulness vs plausibility (headline).")
ap.add_argument("--leaderboard", default=DEFAULT_LB,
                help="path to E6-1 leaderboard.json (default: committed record)")
ap.add_argument("--demo", default=DEFAULT_DEMO,
                help="path to E6-3 faithful_demo.json (default: committed record)")
args = ap.parse_args()

with open(args.leaderboard) as fh:
    LB = json.load(fh)
with open(args.demo) as fh:
    DEMO = json.load(fh)

ROWS = LB["rows"]
assert ROWS, "leaderboard has no rows"

# ---------------------------------------------------------------------------
# Palette — Okabe-Ito colour-blind-safe, one colour per method TRADITION.
# (matches Figure 1's accent vocabulary: blue=substrate-ish, vermillion=oracle,
#  green=gradient companion, etc.)
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
    "dim_reduction": "dimensionality reduction (NMF/PCA/SAE)",
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
# from the committed leaderboard; no value is computed here.
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


methods = []   # (faith, plaus, tradition, phase, method, pretty)
oracle = None
for r in ROWS:
    f = r.get("faithfulness")
    p = r.get("plausibility_proxy")
    if f is None or p is None:
        continue
    rec = dict(faith=float(f), plaus=float(p), tradition=r["tradition"],
               phase=r["phase"], method=r["method"], name=pretty(r["method"]))
    if r["phase"] == "ground_truth":
        oracle = rec
    else:
        methods.append(rec)

n_methods = len(methods)
assert n_methods >= 28, f"expected ~30 method rows, got {n_methods}"

# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 10.4, 7.4
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C_BG)
ax = fig.add_axes([0.082, 0.085, 0.895, 0.80])
ax.set_xlim(-0.02, 1.06)
ax.set_ylim(0.20, 1.04)
ax.set_xlabel("FAITHFULNESS  →  recovers the TRUE causes\n"
              "(vs the exact §1 intervention oracle;  1.0 = oracle ceiling)",
              fontsize=10.5, labelpad=8)
ax.set_ylabel("PLAUSIBILITY  →  how convincing it LOOKS\n"
              "(documented per-tradition proxy — not a measurement)",
              fontsize=10.5, labelpad=8)
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
ax.add_patch(Rectangle((-0.02, DZ_Y), DZ_X - (-0.02), 1.04 - DZ_Y,
                        facecolor=C_DANGER, alpha=0.10, edgecolor=C_DANGER,
                        linewidth=1.4, linestyle=(0, (5, 3)), zorder=1))
ax.text(0.385, 1.012, "DANGER ZONE", ha="right", va="top",
        fontsize=12.5, fontweight="bold", color=C_DANGER, zorder=6)
ax.text(0.385, 0.972, "looks convincing  ·  isn't faithful", ha="right", va="top",
        fontsize=8.6, style="italic", color=C_DANGER, zorder=6)

# faithful corner cue (lower-right, near the ceiling)
ax.text(1.045, 0.305, "FAITHFUL\nCLUSTER", ha="right", va="bottom",
        fontsize=9.0, fontweight="bold", color=TRAD_COLOR["causal"], zorder=6,
        linespacing=1.0)

# ---------------------------------------------------------------------------
# Scatter every method. Colour = tradition, marker = phase.
# A faint jitter ONLY on overlapping (faith,plaus) coincidences so the dense
# causal-ceiling cluster (four methods at exactly (1.0, 0.5)) is all visible;
# jitter is cosmetic and < 0.012, annotated as such, and never crosses a zone
# boundary.
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
            # spread coincident points on a tiny circle (cosmetic only)
            import math
            ang = 2 * math.pi * j / k
            dx = JIT * math.cos(ang)
            dy = JIT * math.sin(ang)
        plotted[i] = (m["faith"] + dx, m["plaus"] + dy)

for i, m in enumerate(methods):
    x, y = plotted[i]
    ax.scatter(x, y, s=128, marker=PHASE_MARKER[m["phase"]],
               facecolor=TRAD_COLOR.get(m["tradition"], "#777777"),
               edgecolor="white", linewidth=1.1, alpha=0.95, zorder=5)

# the oracle ceiling
if oracle is not None:
    ax.scatter(oracle["faith"], oracle["plaus"], s=420, marker="*",
               facecolor="#ffffff", edgecolor=C_ORACLE, linewidth=1.8, zorder=7)
    ax.annotate("ORACLE\nceiling (1, 1)", (oracle["faith"], oracle["plaus"]),
                xytext=(oracle["faith"] - 0.005, oracle["plaus"] - 0.052),
                ha="center", va="top", fontsize=8.4, fontweight="bold",
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
          arrow=True, fs=8.3):
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


# DANGER-ZONE offenders (named in the brief): expected_gradients, linear_probing,
# A3 tuning, A6 granger, guided_backprop. Labels are fanned out around the
# upper-left cluster (P~0.8-0.9, F~0.03-0.27) with leader lines so each ties to
# exactly one marker; the (faith,plaus) in each label is the committed record.
label("expected_gradients",            dx=-0.022, dy=0.092, ha="center", va="bottom",
      weight="bold")
label("A3_tuning",                      dx=-0.075, dy=0.060, ha="right", va="bottom")
label("A6_granger",                     dx=-0.060, dy=0.012, ha="right", va="center")
label("vanilla_saliency",              dx=-0.072, dy=-0.052, ha="right", va="top",
      weight="bold")
label("guided_backprop",               dx=0.072, dy=-0.030, ha="left",  va="center")
label("linear_probing_control_tasks",  dx=0.090, dy=-0.052, ha="left",  va="top",
      weight="bold")

# FAITHFUL cluster (near ceiling): activation patching, occlusion, counterfactual.
label("activation_patching",           dx=0.012, dy=0.078, ha="center", va="bottom",
      weight="bold")
label("occlusion",                     dx=0.062, dy=0.014, ha="left", va="center")
label("on_distribution_counterfactual", dx=0.062, dy=-0.052, ha="left", va="center")

# ---------------------------------------------------------------------------
# The E6-3 contrast arrow: the field's DEFAULT tool (vanilla saliency, near
# chance) vs the FAITHFUL causal method (activation patching, near ceiling) —
# the same per-method gap the paper headlines (faithful_demo.json).
# ---------------------------------------------------------------------------
pc = DEMO["pair_contrast"]
fa = DEMO["faithful_method"]   # activation_patching
po = DEMO["popular_method"]    # vanilla_saliency
ax_i = idx_of(fa["method"])
po_i = idx_of(po["method"])
fx, fy = plotted[ax_i]
ppx, ppy = plotted[po_i]
ax.add_patch(FancyArrowPatch(
    (ppx, ppy), (fx, fy),
    connectionstyle="arc3,rad=-0.22", arrowstyle="-|>", mutation_scale=16,
    linewidth=1.6, color=C_MUTE, linestyle=(0, (4, 2)), zorder=4, alpha=0.9))
midx, midy = (ppx + fx) / 2 + 0.10, (ppy + fy) / 2 + 0.085
ax.text(midx, midy,
        f"the field's DEFAULT tool → a FAITHFUL one\n"
        f"vanilla saliency  F={po['faithfulness_all_regimes']:.3f}  "
        f"→  activation patching  F={fa['faithfulness_all_regimes']:.3f}\n"
        f"(position/index regime: {pc['popular_faithfulness_position']:.0f} → "
        f"{pc['faithful_faithfulness_position']:.0f}, gap {pc['position_gap']:.0f})",
        ha="center", va="center", fontsize=7.8, color=C_MUTE, style="italic",
        linespacing=1.15, zorder=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fbfbfb",
                  edgecolor="#d8d8d8", linewidth=0.8))

# (vanilla saliency is labelled in the danger-zone offender group above.)

# ---------------------------------------------------------------------------
# Title + subtitle
# ---------------------------------------------------------------------------
fig.text(0.082, 0.966,
         "Plausible ≠ faithful: every interpretability method scored where the "
         "truth is known exactly",
         ha="left", fontsize=12.2, fontweight="bold", color=C_INK)
fig.text(0.082, 0.930,
         f"All {n_methods} methods across the three traditions on the bit-exact VCS, "
         "vs the §1 intervention oracle.  The most plausible-looking methods are the "
         "least faithful.",
         ha="left", fontsize=8.8, color=C_MUTE, style="italic")

# ---------------------------------------------------------------------------
# Legend — two groups: tradition (colour) and phase (marker shape).
# ---------------------------------------------------------------------------
trad_present = [t for t in ["causal", "intervention", "gradient", "correlational",
                            "probing", "dim_reduction", "descriptive"]
                if any(m["tradition"] == t for m in methods)]
trad_handles = [
    Line2D([0], [0], marker="o", linestyle="none", markersize=9.5,
           markerfacecolor=TRAD_COLOR[t], markeredgecolor="white",
           markeredgewidth=1.0, label=TRAD_LABEL[t])
    for t in trad_present
]
leg1 = ax.legend(handles=trad_handles, title="tradition (colour)",
                 loc="lower left", bbox_to_anchor=(0.355, 0.005),
                 frameon=True, fontsize=7.6, title_fontsize=8.2,
                 handletextpad=0.4, labelspacing=0.45, borderpad=0.7)
leg1.get_frame().set_edgecolor("#cccccc")
leg1.get_frame().set_linewidth(0.8)
leg1._legend_box.align = "left"
ax.add_artist(leg1)

phase_handles = [
    Line2D([0], [0], marker=PHASE_MARKER[p], linestyle="none", markersize=9.5,
           markerfacecolor="#777777", markeredgecolor="white",
           markeredgewidth=1.0, label=PHASE_LABEL[p])
    for p in ["phaseA_kording", "phaseB_attribution", "phaseC_mechanistic"]
] + [
    Line2D([0], [0], marker="*", linestyle="none", markersize=13,
           markerfacecolor="#ffffff", markeredgecolor=C_ORACLE,
           markeredgewidth=1.4, label="oracle ceiling"),
]
leg2 = ax.legend(handles=phase_handles, title="experimental battery (marker)",
                 loc="lower left", bbox_to_anchor=(0.685, 0.005),
                 frameon=True, fontsize=7.6, title_fontsize=8.2,
                 handletextpad=0.4, labelspacing=0.45, borderpad=0.7)
leg2.get_frame().set_edgecolor("#cccccc")
leg2.get_frame().set_linewidth(0.8)
leg2._legend_box.align = "left"

# provenance footnote
fig.text(0.085, 0.012,
         "Data (read-only, no re-run): faithfulness & plausibility per row from "
         "tools/xai_study/compare/out/leaderboard.json (P2-E6-1); "
         "faithful↔popular contrast from faithful_demo.json (P2-E6-3).  "
         "Coincident points jittered <0.011 (cosmetic; never crosses a zone boundary).",
         ha="left", fontsize=6.2, color="#9a9a9a", style="italic")

# ---------------------------------------------------------------------------
# Save (vector PDF)
# ---------------------------------------------------------------------------
OUT = os.path.join(HERE, "fig2_faithfulness_vs_plausibility.pdf")
fig.savefig(OUT, format="pdf", facecolor=C_BG)
plt.close(fig)

# ---------------------------------------------------------------------------
# Self-check — file exists, non-empty, %PDF- header, size > 4 KB; and the
# headline numbers I asserted in the plot match the committed records.
# ---------------------------------------------------------------------------
sz = os.path.getsize(OUT)
assert sz > 4000, f"PDF suspiciously small: {sz} bytes"
with open(OUT, "rb") as fh:
    head = fh.read(5)
assert head == b"%PDF-", f"not a PDF (header={head!r})"

# cross-check: the contrast endpoints I drew are exactly the E6-3 records.
ap_rec = next(m for m in methods if m["method"] == "activation_patching")
vs_rec = next(m for m in methods if m["method"] == "vanilla_saliency")
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
for nm in ("activation_patching",):
    m = next(mm for mm in methods if mm["method"] == nm)
    assert m["faith"] >= 0.95, f"{nm} not near ceiling (F={m['faith']})"

print(f"[OK] wrote {OUT}  ({sz} bytes, header {head!r})")
print(f"[OK] plotted {n_methods} methods + oracle ceiling; "
      f"danger zone = F<={DZ_X}, P>={DZ_Y}")
print(f"[OK] contrast (E6-3): vanilla saliency F={vs_rec['faith']:.3f} "
      f"→ activation patching F={ap_rec['faith']:.3f}")
