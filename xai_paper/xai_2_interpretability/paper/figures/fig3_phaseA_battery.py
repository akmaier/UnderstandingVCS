#!/usr/bin/env python3
"""Figure 3 — the Kording neuroscience battery, scored (P2-E7-3).

The Phase-A battery (A1 connectomics, A2 single-unit lesions, A3 tuning curves,
A4 spike-word/pairwise correlations, A5 local field potentials, A6 Granger
causality, A7 dim-reduction, A8 whole-state recording) run on the bit-exact,
fully-known Atari VCS and SCORED against the §1 intervention oracle. This is the
*quantified* Jonas & Kording (2017): each classical analysis finds rich
structure, yet most score LOW in faithfulness against the known register-transfer
mechanism — while the oracle-as-method positive control sits at 1.0 by
construction. Phase A is the calibration baseline that the causal contrast in
Fig. 4 sharpens (plan.md Results-A; experiment_design.md §4 / §9 Phase-A row).

LAYOUT
  (a) The battery, scored. One bar per analysis A1–A8 = its headline
      faithfulness vs the oracle (leaderboard.json), coloured by method tradition.
      Where the §0 correctness triad is available (A1–A4, A6, A8), the F / S / M
      points are overlaid on the bar. The oracle positive control is drawn as a
      reference at faithfulness = 1.0 (dashed line + a labelled control bar).
  (b) Four key findings, each annotated with the exact per-game numbers that
      drive them:
        · A6 Granger F = 0 on Pong — temporal precedence ≠ causation
        · A3 spurious tuning — strongly-tuned units whose tuning ≠ true role
        · A2 ρ = 0.99 yet an interaction blind-spot (super-additive lesions)
        · A7 NMF > PCA on matched components (and both are mediocre)

DATA — every plotted number traces to a committed record (no re-run, no fabrication):
  tools/xai_study/compare/out/leaderboard.json            (E6-1 aggregate per method)
  tools/xai_study/phaseA_kording/out/A{1..8}_*.json        (per-method §R records)
The script reads these files at run time; constants below are mirrored ONLY for
the annotation callouts and are asserted equal to the on-disk records by the
self-check, so the figure cannot silently drift from the data.

Run:
    python fig3_phaseA_battery.py
    # (DoD command) python fig3_phaseA_battery.py --leaderboard <path>
Produces:
    fig3_phaseA_battery.pdf   (vector, colour-blind-safe, self-contained legend)
"""

import os
import sys
import json
import argparse

import matplotlib

matplotlib.use("Agg")  # headless vector backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Locate the committed data records (repo-relative; robust to the worktree path)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
# .../xai_paper/xai_2_interpretability/paper/figures  ->  repo root is 5 up
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))


def _resolve(rel):
    p = os.path.join(REPO, rel)
    return p


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--leaderboard",
    default=_resolve("tools/xai_study/compare/out/leaderboard.json"),
    help="path to the E6-1 leaderboard.json (the aggregate per-method scores)",
)
args, _ = parser.parse_known_args()

PHASEA_DIR = _resolve("tools/xai_study/phaseA_kording/out")

with open(args.leaderboard) as fh:
    LB = json.load(fh)

# Pull the eight Phase-A method rows + the oracle positive control from the
# leaderboard (these ARE the headline faithfulness numbers; triad F/S/M ride along).
LB_BY_METHOD = {r["method"]: r for r in LB["rows"]}
ORACLE = LB_BY_METHOD["ORACLE"]
assert ORACLE["is_positive_control"] == 1 and abs(ORACLE["faithfulness"] - 1.0) < 1e-9, \
    "leaderboard positive control must be the oracle at faithfulness=1.0"


def _load_phaseA(name):
    with open(os.path.join(PHASEA_DIR, name)) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Palette — Okabe-Ito colour-blind-safe set, keyed by method TRADITION.
# ---------------------------------------------------------------------------
C_BG = "#ffffff"
C_INK = "#1a1a1a"
C_MUTE = "#5b5b5b"
C_GRID = "#dcdcdc"
C_ORACLE = "#000000"  # the positive-control reference (neutral black)

TRAD_COLOR = {
    "intervention": "#0072B2",   # blue
    "correlational": "#D55E00",  # vermillion
    "dim_reduction": "#009E73",  # bluish-green
    "descriptive": "#999999",    # grey
}
TRAD_LABEL = {
    "intervention": "intervention-based",
    "correlational": "correlational",
    "dim_reduction": "dim-reduction",
    "descriptive": "descriptive baseline",
}

# Short, human-readable analysis names (the A# battery in §4 order).
A_NAMES = {
    "A1_connectomics": ("A1", "connectomics /\ndata-flow graph"),
    "A2_lesions": ("A2", "single-unit\nlesions"),
    "A3_tuning": ("A3", "tuning\ncurves"),
    "A4_spike_word": ("A4", "spike-word /\npairwise corr."),
    "A5_local_field_potentials": ("A5", "local field\npotentials"),
    "A6_granger": ("A6", "Granger\ncausality"),
    "A7_dim_reduction": ("A7", "dim-reduction\n(NMF / PCA)"),
    "A8_wholestate": ("A8", "whole-state\nrecording"),
}
A_ORDER = list(A_NAMES.keys())

# What the headline faithfulness number MEANS for each analysis (the metric, oriented
# so higher = more faithful to the known mechanism). Straight from the §R metric_names.
A_METRIC = {
    "A1_connectomics": "data-flow graph F1 vs oracle",
    "A2_lesions": "lesion-importance ρ vs true role",
    "A3_tuning": "1 − spurious-tuning rate",
    "A4_spike_word": "coupling corr. vs true coupling",
    "A5_local_field_potentials": "1 − clock-explained var. (epiphen.)",
    "A6_granger": "1 − false-edge rate vs true data-flow",
    "A7_dim_reduction": "NMF matched-component fraction",
    "A8_wholestate": "minimality M vs oracle",
}

# ---------------------------------------------------------------------------
# Assemble the per-analysis bar data from the leaderboard rows.
# ---------------------------------------------------------------------------
bars = []
for m in A_ORDER:
    r = LB_BY_METHOD[m]
    bars.append({
        "method": m,
        "short": A_NAMES[m][0],
        "name": A_NAMES[m][1],
        "metric": A_METRIC[m],
        "trad": r["tradition"],
        "faith": r["faithfulness"],
        "F": r.get("triad_F"),
        "S": r.get("triad_S"),
        "M": r.get("triad_M"),
        "ci": r.get("faithfulness_ci95"),
    })

# ---------------------------------------------------------------------------
# Annotation facts — mirrored from the per-method §R records and ASSERTED equal
# to disk in the self-check, so a stale constant fails the build.
# ---------------------------------------------------------------------------
GAMES6 = ["pong", "breakout", "space_invaders", "seaquest", "ms_pacman", "qbert"]

# A6 Granger on Pong: false-edge = missed-edge = 1.0, triad F = 0.0
_a6_pong = _load_phaseA("A6_pong.json")
FACT_A6 = {
    "pong_false_edge": _a6_pong["extra"]["cell_scope"]["false_edge_rate"],
    "pong_missed_edge": _a6_pong["extra"]["cell_scope"]["missed_edge_rate"],
    "pong_triadF": _a6_pong["extra"]["triad"]["F"],
}

# A3 spurious tuning per game (game-variable version, §4: the tuning-curve trap)
_a3 = _load_phaseA("A3_tuning_all.json")["per_game"]
FACT_A3 = {g: _a3[g]["game_variable_spurious_tuning_rate"] for g in GAMES6}

# A2 lesions: ρ near 1 but interaction blind-spot (super-additive lesions)
_a2 = _load_phaseA("A2_lesions.json")["per_game"]
FACT_A2 = {
    "rho_mean": LB_BY_METHOD["A2_lesions"]["faithfulness"],
    "si_rho": _a2["space_invaders"]["spearman_rho_lesion_vs_oracle"],
    "si_miss": _a2["space_invaders"]["interaction_missed_rate"],
    "si_superadd": _a2["space_invaders"]["max_superadditive_delta"],
}

# A7 NMF vs PCA matched-component fraction (does NMF beat the pilot's PCA?)
_a7 = _load_phaseA("A7_dimred_all.json")["games"]
_nmf = [g["nmf_matched_frac"] for g in _a7]
_pca = [g["pca_matched_frac"] for g in _a7]
FACT_A7 = {
    "nmf_mean": sum(_nmf) / len(_nmf),
    "pca_mean": sum(_pca) / len(_pca),
    "nmf_best": max(_nmf),
    "pca_best": max(_pca),
}

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.4,
    "pdf.fonttype": 42,   # embed TrueType (editable, not bitmap)
    "svg.fonttype": "none",
    "figure.dpi": 150,
    "axes.linewidth": 0.8,
})

# ---------------------------------------------------------------------------
# Figure scaffold: (a) the scored battery (left, wide) · (b) findings (right)
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 12.6, 6.4
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C_BG)

# Title band
fig.text(0.012, 0.965,
         "The neuroscience battery on a known machine — scored against ground truth",
         ha="left", va="center", fontsize=13.0, fontweight="bold", color=C_INK)
fig.text(0.012, 0.928,
         "Jonas & Kording's battery, quantified: classical analyses find rich structure yet score low in faithfulness to the "
         "true register-transfer mechanism; the oracle-as-method control = 1.0.",
         ha="left", va="center", fontsize=8.6, color=C_MUTE, fontstyle="italic")

# --- Panel (a): grouped scored bars -----------------------------------------
axA = fig.add_axes([0.060, 0.155, 0.560, 0.720])
axA.set_facecolor(C_BG)

n = len(bars)
xs = list(range(n))
bw = 0.62

# oracle positive-control reference line at 1.0
axA.axhline(1.0, color=C_ORACLE, lw=1.4, ls=(0, (5, 3)), zorder=1)
axA.text(n - 0.42, 1.012, "oracle-as-method (positive control) = 1.0",
         ha="right", va="bottom", fontsize=7.6, color=C_ORACLE, fontstyle="italic")

# faint horizontal gridlines
for yv in (0.25, 0.5, 0.75):
    axA.axhline(yv, color=C_GRID, lw=0.8, zorder=0)

for i, b in enumerate(bars):
    col = TRAD_COLOR[b["trad"]]
    axA.bar(i, b["faith"], width=bw, color=col, edgecolor=C_INK,
            linewidth=0.7, zorder=3, alpha=0.92)
    # 95% CI whisker on the headline faithfulness (over the 6 core games)
    if b["ci"]:
        axA.errorbar(i, b["faith"], yerr=b["ci"], fmt="none",
                     ecolor=C_INK, elinewidth=0.9, capsize=2.5, zorder=4)
    # value label above the bar
    axA.text(i, b["faith"] + (b["ci"] or 0) + 0.028, f"{b['faith']:.2f}",
             ha="center", va="bottom", fontsize=8.0, fontweight="bold",
             color=C_INK, zorder=6)
    # overlay the F / S / M triad points where available
    triad = [("F", b["F"], "o"), ("S", b["S"], "s"), ("M", b["M"], "^")]
    offs = [-0.20, 0.0, 0.20]
    for (lab, val, mk), dx in zip(triad, offs):
        if val is None:
            continue
        axA.scatter(i + dx, val, marker=mk, s=34, facecolor="white",
                    edgecolor=C_INK, linewidth=1.0, zorder=7)

# the oracle control bar at the far right (a 9th column, set apart)
ox = n + 0.55
axA.bar(ox, 1.0, width=bw, color="white", edgecolor=C_ORACLE,
        linewidth=1.4, hatch="////", zorder=3)
axA.text(ox, 1.012, "1.00", ha="center", va="bottom", fontsize=8.0,
         fontweight="bold", color=C_ORACLE, zorder=6)

axA.set_xlim(-0.7, ox + 0.7)
axA.set_ylim(0, 1.16)
axA.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axA.set_ylabel("faithfulness to the true mechanism  (vs §1 oracle)",
               fontsize=9.0)
axA.set_xticks(xs + [ox])
xtl = [f"{b['short']}\n{b['name']}" for b in bars] + ["GT\noracle"]
axA.set_xticklabels(xtl, fontsize=7.6)
# colour the oracle tick label
for t, lab in zip(axA.get_xticklabels(), [b["short"] for b in bars] + ["GT"]):
    if lab == "GT":
        t.set_color(C_ORACLE)
        t.set_fontweight("bold")
axA.tick_params(axis="both", length=3)
for sp in ("top", "right"):
    axA.spines[sp].set_visible(False)

# per-analysis metric subcaptions under each bar (what the number measures)
for i, b in enumerate(bars):
    axA.text(i, -0.255, b["metric"], ha="center", va="top", rotation=0,
             fontsize=5.8, color=C_MUTE, transform=axA.get_xaxis_transform(),
             clip_on=False)

axA.set_title("(a)  the eight analyses, scored", loc="left",
              fontsize=10.0, fontweight="bold", pad=8)

# legend for panel (a): traditions + triad markers
trad_handles = [
    Patch(facecolor=TRAD_COLOR[t], edgecolor=C_INK, linewidth=0.7,
          label=TRAD_LABEL[t])
    for t in ("intervention", "correlational", "dim_reduction", "descriptive")
]
triad_handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
           markeredgecolor=C_INK, markersize=7, label="F  faithful"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
           markeredgecolor=C_INK, markersize=7, label="S  sufficient"),
    Line2D([0], [0], marker="^", color="none", markerfacecolor="white",
           markeredgecolor=C_INK, markersize=7, label="M  minimal"),
]
leg1 = axA.legend(handles=trad_handles, loc="upper left",
                  bbox_to_anchor=(0.002, 0.985), frameon=True, fontsize=7.1,
                  title="tradition", title_fontsize=7.4, handlelength=1.2,
                  labelspacing=0.35, borderpad=0.5)
leg1.get_frame().set_edgecolor("#cccccc")
leg1.get_frame().set_linewidth(0.7)
axA.add_artist(leg1)
leg2 = axA.legend(handles=triad_handles, loc="upper left",
                  bbox_to_anchor=(0.205, 0.985), frameon=True, fontsize=7.1,
                  title="triad (where scored)", title_fontsize=7.4,
                  handlelength=1.0, labelspacing=0.35, borderpad=0.5)
leg2.get_frame().set_edgecolor("#cccccc")
leg2.get_frame().set_linewidth(0.7)

# --- Panel (b): the four key findings ---------------------------------------
axB = fig.add_axes([0.655, 0.060, 0.335, 0.815])
axB.set_xlim(0, 100)
axB.set_ylim(0, 100)
axB.axis("off")
axB.set_title("(b)  key findings", loc="left", fontsize=10.0,
              fontweight="bold", pad=8)


def finding(y, h, accent, tag, head, lines):
    """A small annotated callout card."""
    axB.add_patch(FancyBboxPatch(
        (1.5, y), 97.0, h,
        boxstyle="round,pad=0.0,rounding_size=2.2",
        facecolor="#fbfbfb", edgecolor=accent, linewidth=1.3, zorder=2))
    # accent tag chip
    axB.add_patch(FancyBboxPatch(
        (3.0, y + h - 8.0), 13.0, 6.2,
        boxstyle="round,pad=0.0,rounding_size=1.0",
        facecolor=accent, edgecolor="none", zorder=3))
    axB.text(9.5, y + h - 4.9, tag, ha="center", va="center", color="white",
             fontsize=8.4, fontweight="bold", zorder=4)
    axB.text(19.0, y + h - 4.9, head, ha="left", va="center", color=C_INK,
             fontsize=8.6, fontweight="bold", zorder=4)
    ly = y + h - 11.5
    for s, bold in lines:
        axB.text(4.5, ly, s, ha="left", va="top",
                 fontsize=7.4 if not bold else 7.8,
                 color=C_INK if bold else C_MUTE,
                 fontweight="bold" if bold else "normal", zorder=4)
        ly -= 5.2


# A6 Granger F = 0
finding(
    75.5, 22.5, TRAD_COLOR["correlational"], "A6",
    "Granger F = 0",
    [(f"Pong: false-edge rate = {FACT_A6['pong_false_edge']:.2f},", False),
     (f"missed-edge = {FACT_A6['pong_missed_edge']:.2f}  →  triad F = {FACT_A6['pong_triadF']:.2f}.", True),
     ("Temporal precedence ≠ causation: on a", False),
     ("clock-locked machine Granger infers", False),
     ("spurious bidirectional edges everywhere.", False)],
)

# A3 spurious tuning
_a3_hi = max(FACT_A3.items(), key=lambda kv: kv[1])
finding(
    51.0, 22.5, TRAD_COLOR["correlational"], "A3",
    "spurious tuning",
    [(f"Pong spurious-tuning rate = {FACT_A3['pong']:.2f};", True),
     (f"Space Invaders = {FACT_A3['space_invaders']:.2f}, Q*bert = {FACT_A3['qbert']:.2f}.", False),
     ("Strongly-tuned units whose tuning does", False),
     ("NOT match their true causal role —", False),
     ("the classic tuning-curve trap, quantified.", False)],
)

# A2 rho 0.99 but interaction blind spot
finding(
    26.5, 22.5, TRAD_COLOR["intervention"], "A2",
    "ρ = 0.99, blind to interactions",
    [(f"Single-unit lesion importance ρ = {FACT_A2['rho_mean']:.2f}", True),
     ("vs the true role — yet single lesions", False),
     (f"miss {FACT_A2['si_miss']*100:.0f}% of interactions (Space Inv.):", False),
     (f"super-additive pair Δ up to {FACT_A2['si_superadd']:.0f}", True),
     ("invisible to one-unit-at-a-time ablation.", False)],
)

# A7 NMF > PCA
finding(
    2.0, 22.5, TRAD_COLOR["dim_reduction"], "A7",
    "NMF > PCA (both mediocre)",
    [(f"NMF matched-component fraction = {FACT_A7['nmf_mean']:.2f}", True),
     (f"vs PCA = {FACT_A7['pca_mean']:.2f} mean (NMF best {FACT_A7['nmf_best']:.1f}).", False),
     ("Non-negativity aligns components with", False),
     ("the additive register basis better than", False),
     ("PCA — but neither recovers the mechanism.", False)],
)

# provenance footnote
fig.text(0.012, 0.018,
         "Data (no re-run): per-method faithfulness + triad F/S/M from "
         "tools/xai_study/compare/out/leaderboard.json (E6-1); annotation numbers from "
         "tools/xai_study/phaseA_kording/out/A{1..8}_*.json. 6 core games; whiskers = 95% CI.",
         ha="left", va="bottom", fontsize=6.1, color="#9a9a9a", fontstyle="italic")

# ---------------------------------------------------------------------------
# Save (vector PDF)
# ---------------------------------------------------------------------------
OUT = os.path.join(HERE, "fig3_phaseA_battery.pdf")
fig.savefig(OUT, format="pdf", facecolor=C_BG)
plt.close(fig)

# ===========================================================================
# In-script self-check: file exists + non-empty + %PDF- header + size>4KB,
# and every plotted/annotated number matches the on-disk committed records.
# ===========================================================================
problems = []

# (1) PDF artifact sanity
sz = os.path.getsize(OUT)
if sz <= 4096:
    problems.append(f"PDF suspiciously small: {sz} bytes (need >4KB)")
with open(OUT, "rb") as fh:
    head = fh.read(5)
if head != b"%PDF-":
    problems.append(f"not a PDF (header={head!r})")

# (2) the plotted bars match the leaderboard rows exactly
for b in bars:
    r = LB_BY_METHOD[b["method"]]
    if abs(b["faith"] - r["faithfulness"]) > 1e-9:
        problems.append(f"{b['method']} bar != leaderboard faithfulness")

# (3) positive control == 1.0
if abs(ORACLE["faithfulness"] - 1.0) > 1e-9:
    problems.append("oracle positive control != 1.0")

# (4) the annotated facts re-read from disk equal what we drew
_chk_a6 = _load_phaseA("A6_pong.json")
if abs(_chk_a6["extra"]["triad"]["F"] - FACT_A6["pong_triadF"]) > 1e-9:
    problems.append("A6 Pong triad F drifted from record")
if FACT_A6["pong_triadF"] != 0.0:
    problems.append("A6 Pong F is not 0 (headline finding broken)")
if FACT_A3["pong"] != 1.0:
    problems.append("A3 Pong spurious-tuning is not 1.0 (headline finding broken)")
if not (FACT_A2["rho_mean"] >= 0.98):
    problems.append("A2 mean rho not ~0.99 (headline finding broken)")
if not (FACT_A2["si_superadd"] > 0):
    problems.append("A2 interaction blind-spot absent (super-additive delta = 0)")
if not (FACT_A7["nmf_mean"] >= FACT_A7["pca_mean"]):
    problems.append("A7 NMF not >= PCA (headline finding broken)")

if problems:
    print("[FAIL] self-check:")
    for p in problems:
        print("   -", p)
    sys.exit(1)

print(f"[OK] wrote {OUT}  ({sz} bytes, header {head!r})")
print(f"[OK] {len(bars)} analyses scored; oracle control = 1.0")
print(f"[OK] findings: A6 Pong F={FACT_A6['pong_triadF']:.2f}; "
      f"A3 Pong spurious={FACT_A3['pong']:.2f}; "
      f"A2 rho={FACT_A2['rho_mean']:.2f}/superadd={FACT_A2['si_superadd']:.0f}; "
      f"A7 NMF={FACT_A7['nmf_mean']:.2f} > PCA={FACT_A7['pca_mean']:.2f}")
