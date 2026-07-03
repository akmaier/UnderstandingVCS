#!/usr/bin/env python3
"""Figure 4 — THE HEADLINE: A-C on the shared faithfulness-vs-plausibility plane.

Every interpretability method from the three traditions (Phase A neuroscience/Kording,
Phase B attribution/XAI, Phase C mechanistic-interp) placed on the paper's two reporting
axes (experiment_design.md §0):

    X = FAITHFULNESS   — vs the intervention oracle (§3.2 ground truth). [0,1], higher
                          = closer to the TRUE causes. (triad.F where available, else the
                          method's re-oriented vs-oracle value; see leaderboard provenance.)
    Y = PLAUSIBILITY PROXY — the documented per-tradition plausibility proxy (how convincing
                          the explanation LOOKS to a human reader). [0,1]. This is a
                          documented proxy, NOT a measurement — it is exactly what exposes
                          the §0 DANGER ZONE: high plausibility, low faithfulness.

The danger zone (upper-left: looks-convincing-but-isn't-true) is drawn explicitly with a
dashed offset boundary, and its worst offenders are annotated. The oracle sits at (1,1) as
a ceiling star that NO method reaches — the gap, made a picture.

This is the clean plane: the dashed offset boundary + the unreached oracle ceiling carry the
story. Per-method 95% bootstrap CIs are reported in the Phase-A/B battery figures (Fig 2/3),
the supplement, and leaderboard_ci — they are intentionally NOT drawn here, so the offset
boundary and the unreached oracle read cleanly.

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

ap = argparse.ArgumentParser(description="Figure 4 — faithfulness vs plausibility (headline).")
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
# ---------------------------------------------------------------------------
C_BG = "#ffffff"
C_INK = "#1a1a1a"
C_MUTE = "#5b5b5b"
C_DANGER = "#CC79A7"   # reddish-purple — the danger-zone wash + dashed offset boundary
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
assert oracle is not None, "leaderboard has no oracle (ground_truth) row"

# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 10.4, 7.4
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C_BG)
# In-image title/subtitle removed (they duplicated the LaTeX caption); the axes
# top is raised to fill the band the headline used to occupy.
ax = fig.add_axes([0.082, 0.085, 0.895, 0.875])
ax.set_xlim(-0.02, 1.06)
ax.set_ylim(0.20, 1.04)
ax.set_xlabel("FAITHFULNESS  →  recovers the TRUE causes\n"
              "(vs the exact intervention oracle, §3.2;  1.0 = oracle ceiling)",
              fontsize=10.5, labelpad=8)
ax.set_ylabel("PLAUSIBILITY PROXY  →  how convincing it LOOKS\n"
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
# Drawn as a dashed OFFSET boundary (upper-left box) — the visual of the gap.
# Threshold: plausibility >= 0.70 and faithfulness <= 0.40.
# ---------------------------------------------------------------------------
DZ_X = 0.40   # faithfulness ceiling of the danger zone
DZ_Y = 0.70   # plausibility floor of the danger zone
ax.add_patch(Rectangle((-0.02, DZ_Y), DZ_X - (-0.02), 1.04 - DZ_Y,
                        facecolor=C_DANGER, alpha=0.10, edgecolor=C_DANGER,
                        linewidth=1.4, linestyle=(0, (5, 3)), zorder=1))
ax.text(0.385, 1.012, "DANGER ZONE", ha="right", va="top",
        fontsize=12.5, fontweight="bold", color=C_DANGER, zorder=6)
ax.text(0.385, 0.972, "looks convincing  ·  isn't faithful", ha="right", va="top",
        fontsize=9.0, style="italic", color=C_DANGER, zorder=6)

# faithful corner cue (right side, above the causal ceiling cluster, clear of legends)
ax.text(1.045, 0.66, "FAITHFUL\nCLUSTER", ha="right", va="bottom",
        fontsize=9.0, fontweight="bold", color=TRAD_COLOR["causal"], zorder=6,
        linespacing=1.0)

# ---------------------------------------------------------------------------
# Scatter every method. Colour = tradition, marker = phase.
# A faint jitter ONLY on overlapping (faith,plaus) coincidences so the dense
# causal-ceiling cluster (methods at exactly the same coordinate) is all visible;
# jitter is cosmetic and < 0.012, and never crosses a zone boundary.
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

# the oracle ceiling — a star at (1,1) that no method reaches
ax.scatter(oracle["faith"], oracle["plaus"], s=420, marker="*",
           facecolor="#ffffff", edgecolor=C_ORACLE, linewidth=1.8, zorder=7)
ax.annotate("ORACLE\nceiling (1, 1)", (oracle["faith"], oracle["plaus"]),
            xytext=(oracle["faith"] - 0.005, oracle["plaus"] - 0.052),
            ha="center", va="top", fontsize=8.4, fontweight="bold",
            color=C_INK, zorder=8, linespacing=1.0)

# ---------------------------------------------------------------------------
# Annotate ~6-8 ANCHOR points: the danger-zone offenders + the faithful one.
# Each label's leader ties to exactly one plotted (data) coordinate → no
# fabrication. Kept to <=8 method labels for legibility (oracle labelled above).
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


# DANGER-ZONE offenders: expected_gradients, A3 tuning, A6 granger,
# guided backprop, linear probing — the plausible-but-unfaithful cluster.
# (Under the corrected re-run leaderboard, vanilla_saliency's all-regime
# faithfulness rose to 0.42, i.e. OUT of the danger zone (F<=0.40) — so the
# annotated gradient offender is now guided_backprop, F=0.34, which is still
# inside the zone. vanilla_saliency remains the contrast-arrow ORIGIN below.)
# Labels fanned around the upper-left cluster with leader lines; each (F) is the
# committed record. (5 anchors)
label("expected_gradients",            dx=-0.022, dy=0.092, ha="center", va="bottom",
      weight="bold")
label("A3_tuning",                      dx=-0.075, dy=0.060, ha="right", va="bottom")
label("A6_granger",                     dx=-0.060, dy=0.012, ha="right", va="center")
label("guided_backprop",               dx=-0.072, dy=-0.052, ha="right", va="top",
      weight="bold")
label("linear_probing_control_tasks",  dx=0.090, dy=-0.052, ha="left",  va="top",
      weight="bold")

# The one FAITHFUL anchor (near ceiling): activation patching at F=1.0. (1 anchor)
# Label fanned LEFT so the text stays inside the right margin (no clip).
label("activation_patching",           dx=-0.075, dy=0.095, ha="right", va="bottom",
      weight="bold")
# (6 method anchors + the oracle = 7 labelled points — within the <=8 budget.)

# ---------------------------------------------------------------------------
# The E6-3 contrast arc: the field's DEFAULT tool (vanilla saliency, near
# chance) -> the FAITHFUL causal method (activation patching, near ceiling) —
# the same per-method gap the paper headlines (faithful_demo.json). Drawn as a
# dashed offset arc — the gap, made a picture. (The wordy boxed text label was
# removed for decluttering; the vanilla-saliency -> activation-patching contrast
# is discussed in the body text. The E6-3 records are still cross-checked against
# the leaderboard in the self-check block below.)
# (In-image title/subtitle removed — the LaTeX caption owns that prose.)
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

# ---------------------------------------------------------------------------
# Legend — two groups: tradition (colour) and phase (marker shape). Both sit
# OUTSIDE the dense data field (lower-centre/right), never over a marker.
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
                 frameon=True, fontsize=8.0, title_fontsize=8.4,
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
           markeredgewidth=1.4, label="oracle ceiling (unreached)"),
]
leg2 = ax.legend(handles=phase_handles, title="experimental battery (marker)",
                 loc="lower left", bbox_to_anchor=(0.685, 0.005),
                 frameon=True, fontsize=8.0, title_fontsize=8.4,
                 handletextpad=0.4, labelspacing=0.45, borderpad=0.7)
leg2.get_frame().set_edgecolor("#cccccc")
leg2.get_frame().set_linewidth(0.8)
leg2._legend_box.align = "left"

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

# cross-check: the contrast endpoints I drew are the SAME methods E6-3 names
# (activation_patching = the faithful method; vanilla_saliency = the popular one).
# The plotted coordinates come from the 42-game leaderboard (the source of truth),
# not from faithful_demo.json — the demo's `faithfulness_all_regimes` fields are a
# frozen 6-core-game record and no longer byte-equal to the re-run 42-game
# leaderboard, so we assert method IDENTITY + the qualitative faithful>popular
# ordering, not a stale byte match.
ap_rec = next(m for m in methods if m["method"] == "activation_patching")
vs_rec = next(m for m in methods if m["method"] == "vanilla_saliency")
assert DEMO["faithful_method"]["method"] == "activation_patching", \
    "faithful_demo faithful method is no longer activation_patching"
assert DEMO["popular_method"]["method"] == "vanilla_saliency", \
    "faithful_demo popular method is no longer vanilla_saliency"
assert ap_rec["faith"] > vs_rec["faith"], \
    "leaderboard: faithful endpoint (activation patching) not above popular (vanilla saliency)"
# danger zone really contains its named offenders (corrected leaderboard: the
# gradient offender inside the zone is guided_backprop; vanilla_saliency rose to
# F=0.42, out of the zone, and is only the contrast-arrow origin).
for nm in ("expected_gradients", "linear_probing_control_tasks", "A3_tuning",
           "A6_granger", "guided_backprop"):
    m = next(mm for mm in methods if mm["method"] == nm)
    assert m["plaus"] >= DZ_Y and m["faith"] <= DZ_X, \
        f"{nm} not actually in danger zone (F={m['faith']}, P={m['plaus']})"
# the oracle is the unreached ceiling: at (1,1) and strictly above every method
assert abs(oracle["faith"] - 1.0) < 1e-9 and abs(oracle["plaus"] - 1.0) < 1e-9, \
    f"oracle not at (1,1): ({oracle['faith']},{oracle['plaus']})"
for m in methods:
    assert not (m["faith"] >= 1.0 and m["plaus"] >= 1.0), \
        f"{m['method']} reaches the oracle ceiling (F={m['faith']}, P={m['plaus']})"
# faithful anchor really near ceiling
m = next(mm for mm in methods if mm["method"] == "activation_patching")
assert m["faith"] >= 0.95, f"activation_patching not near ceiling (F={m['faith']})"

print(f"[OK] wrote {OUT}  ({sz} bytes, header {head!r})")
print(f"[OK] plotted {n_methods} methods + oracle ceiling (30 methods + oracle = 31 rows); "
      f"danger zone = F<={DZ_X}, P>={DZ_Y}")
print(f"[OK] contrast (E6-3): vanilla saliency F={vs_rec['faith']:.3f} "
      f"→ activation patching F={ap_rec['faith']:.3f}")
print(f"[OK] oracle ceiling (1,1) unreached by all {n_methods} methods")
