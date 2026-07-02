#!/usr/bin/env python3
"""Phase-A neuroscience battery, scored — embedded as **paper Figure 2** (P2-E7-3).

(Filename ≠ caption number: this file is `fig3_phaseA_battery.py` but it is the
bar-chart battery that compiles as **Figure 2** of the paper via
`sections/04_results_A.tex`. The R2 revision applies the figure_detail_pass
"Fig. 2 — Kording neuroscience battery" block, NOT the "Fig. 3" attribution block
— that one is `fig4_attribution_vs_mechanistic.py`.)

The Phase-A battery (A1 connectomics, A2 single-unit lesions, A3 tuning curves,
A4 spike-word/pairwise correlations, A5 local field potentials, A6 Granger
causality, A7 dim-reduction, A8 whole-state recording) run on the bit-exact,
fully-known Atari VCS and SCORED against the intervention oracle (§3.2). This is
the *quantified* Jonas & Kording (2017): each classical analysis finds rich
structure, yet most score LOW in faithfulness against the known register-transfer
mechanism — while the oracle-as-method positive control sits at 1.0 by
construction. Phase A is the calibration baseline that the causal contrast in the
next figure sharpens (plan.md Results-A; experiment_design.md §4 / §9 Phase-A row).

LAYOUT (figure_detail_pass "Fig. 2" recommended redraw: bar chart + key-results
table, legend OUTSIDE the data field, one numeric label per bar):
  (a) The battery, scored. One bar per analysis A1–A8 = its headline faithfulness
      vs the oracle (leaderboard.json), coloured by method tradition. ONE numeric
      label per bar, placed consistently above the upper CI whisker. The oracle
      positive control is a 9th, set-apart hatched bar at 1.0 with a dashed
      reference line. Tradition legend is a horizontal strip BELOW the axes.
  (b) A compact key-results TABLE (replaces the old callout boxes that overlapped:
      fig_pass blocker). Rows A6 / A3 / A2 / A7; columns finding · number ·
      interpretation. Every number is read from / asserted equal to the committed
      §R per-method records.

DATA — every plotted number traces to a committed record (no re-run, no fabrication):
  tools/xai_study/compare/out/leaderboard.json            (E6-1 aggregate per method)
  tools/xai_study/compare/out/leaderboard_ci.csv          (R-UNC 95% bootstrap CIs)
  tools/xai_study/phaseA_kording/out/A{1..8}_*.json        (per-method §R records)
The script reads these files at run time; constants below are mirrored ONLY for
the table cells and are asserted equal to the on-disk records by the self-check, so
the figure cannot silently drift from the data. The whiskers are the asymmetric
95% bootstrap CIs from leaderboard_ci.csv (R-UNC); every Phase-A bar has a CI row,
so every bar gets a whisker (none are fabricated). The exact data paths are NOT
printed inside the figure (they live in the Supplement provenance table); the
figure body carries only the takeaway.

Run:
    python fig3_phaseA_battery.py
    # (DoD command) python fig3_phaseA_battery.py --leaderboard <path>
Produces:
    fig3_phaseA_battery.pdf   (vector, colour-blind-safe, fonts embedded)
"""

import os
import sys
import csv
import json
import argparse

import matplotlib

matplotlib.use("Agg")  # headless vector backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
parser.add_argument(
    "--leaderboard-ci",
    default=_resolve("tools/xai_study/compare/out/leaderboard_ci.csv"),
    help="path to the R-UNC leaderboard_ci.csv (95%% bootstrap CIs per method)",
)
args, _ = parser.parse_known_args()

PHASEA_DIR = _resolve("tools/xai_study/phaseA_kording/out")

with open(args.leaderboard) as fh:
    LB = json.load(fh)

# Pull the eight Phase-A method rows + the oracle positive control from the
# leaderboard (these ARE the headline faithfulness numbers).
LB_BY_METHOD = {r["method"]: r for r in LB["rows"]}
ORACLE = LB_BY_METHOD["ORACLE"]
assert ORACLE["is_positive_control"] == 1 and abs(ORACLE["faithfulness"] - 1.0) < 1e-9, \
    "leaderboard positive control must be the oracle at faithfulness=1.0"

# R-UNC: asymmetric 95% bootstrap CIs (ci_lo, ci_hi) keyed by method, from the
# committed leaderboard_ci.csv. The whiskers below read straight from this record
# (uncertainty source of truth for the revision); no CI is recomputed here.
CI_BY_METHOD = {}
with open(args.leaderboard_ci) as fh:
    for _row in csv.DictReader(fh):
        if _row.get("kind") == "method":
            CI_BY_METHOD[_row["method"]] = (
                float(_row["ci_lo"]), float(_row["ci_hi"]))


def _load_phaseA(name):
    with open(os.path.join(PHASEA_DIR, name)) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Palette — Okabe-Ito colour-blind-safe set, keyed by method TRADITION.
# ---------------------------------------------------------------------------
C_BG = "#ffffff"
C_INK = "#1a1a1a"
C_MUTE = "#4a4a4a"
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
    "descriptive": "descriptive",
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
# Kept for provenance + the Supplement handoff; not printed in the figure body.
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
    faith = r["faithfulness"]
    # Asymmetric 95% bootstrap CI whiskers from R-UNC's leaderboard_ci.csv,
    # expressed as (lower, upper) error-bar lengths relative to the bar height.
    # Every Phase-A method has a CI row, so every bar gets a whisker (we never
    # fabricate one). Clamp to >=0 so a tiny mean/headline mismatch never flips it.
    ci_lohi = CI_BY_METHOD.get(m)
    if ci_lohi:
        lo, hi = ci_lohi
        yerr = [max(0.0, faith - lo), max(0.0, hi - faith)]
        has_ci = True
    else:
        yerr = None
        has_ci = False
    bars.append({
        "method": m,
        "short": A_NAMES[m][0],
        "name": A_NAMES[m][1],
        "metric": A_METRIC[m],
        "trad": r["tradition"],
        "faith": faith,
        "yerr": yerr,            # asymmetric [lo, hi] from leaderboard_ci.csv
        "has_ci": has_ci,
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
# rcParams — 8 pt minimum font everywhere; embed TrueType fonts (pdf.fonttype=42).
# ---------------------------------------------------------------------------
FS_MIN = 8.0
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FS_MIN,
    "pdf.fonttype": 42,   # embed TrueType (editable, not bitmap) — fonts embedded
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 150,
    "axes.linewidth": 0.8,
})

# ---------------------------------------------------------------------------
# Figure scaffold: (a) the scored battery (left, wide) · (b) findings TABLE (right)
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 12.6, 6.0
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C_BG)

# Title band
fig.text(0.012, 0.965,
         "The neuroscience battery on a known machine — scored against ground truth",
         ha="left", va="center", fontsize=12.5, fontweight="bold", color=C_INK)
fig.text(0.012, 0.927,
         "Jonas & Kording's battery, quantified: classical analyses find rich structure yet "
         "score low in faithfulness to the true mechanism.",
         ha="left", va="center", fontsize=8.6, color=C_MUTE, fontstyle="italic")

# --- Panel (a): scored bars --------------------------------------------------
# Bottom raised to leave a clean tradition-legend strip BELOW the axes, so the
# legend never sits over the bars (fig_pass: legends outside the plotting area).
# The panel bottom (0.255) reserves a band for the two-line x-tick labels + the
# legend strip + the footnote, with no collision/clipping at the figure edge.
axA = fig.add_axes([0.060, 0.255, 0.560, 0.585])
axA.set_facecolor(C_BG)

n = len(bars)
xs = list(range(n))
bw = 0.64

# faint horizontal gridlines (drawn first, behind everything)
for yv in (0.25, 0.5, 0.75):
    axA.axhline(yv, color=C_GRID, lw=0.8, zorder=0)

# oracle positive-control reference line at 1.0
axA.axhline(1.0, color=C_ORACLE, lw=1.3, ls=(0, (5, 3)), zorder=1)
# label sits in the headroom above the line, left-anchored so it never overflows
# the right edge (fig_pass: oracle ceiling squeezed against the edge).
axA.text(-0.55, 1.108, "oracle-as-method (positive control) = 1.0",
         ha="left", va="bottom", fontsize=8.0, color=C_ORACLE, fontstyle="italic")

for i, b in enumerate(bars):
    col = TRAD_COLOR[b["trad"]]
    axA.bar(i, b["faith"], width=bw, color=col, edgecolor=C_INK,
            linewidth=0.7, zorder=3, alpha=0.95)
    # asymmetric 95% bootstrap-CI whisker from leaderboard_ci.csv (R-UNC).
    top = b["faith"]
    if b["yerr"]:
        yl, yh = b["yerr"]
        axA.errorbar(i, b["faith"], yerr=[[yl], [yh]], fmt="none",
                     ecolor=C_INK, elinewidth=1.0, capsize=3.0, zorder=4)
        top = b["faith"] + yh
    # ONE numeric label per bar, consistently above the upper whisker
    # (fig_pass: "Keep only one numeric label per bar, placed consistently above").
    # Black text on coloured bars for contrast.
    axA.text(i, top + 0.030, f"{b['faith']:.2f}",
             ha="center", va="bottom", fontsize=8.4, fontweight="bold",
             color=C_INK, zorder=6)

# the oracle control bar at the far right (a 9th column, set apart)
ox = n + 0.55
axA.bar(ox, 1.0, width=bw, color="white", edgecolor=C_ORACLE,
        linewidth=1.4, hatch="////", zorder=3)
axA.text(ox, 1.018, "1.00", ha="center", va="bottom", fontsize=8.4,
         fontweight="bold", color=C_ORACLE, zorder=6)

axA.set_xlim(-0.7, ox + 0.7)
axA.set_ylim(0, 1.18)
axA.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axA.tick_params(axis="y", labelsize=8.0)
axA.set_ylabel("faithfulness to the true mechanism\n(vs intervention oracle, §3.2)",
               fontsize=9.0)
axA.set_xticks(xs + [ox])
xtl = [f"{b['short']}\n{b['name']}" for b in bars] + ["GT\noracle"]
axA.set_xticklabels(xtl, fontsize=8.0)
# colour the oracle tick label
for t, lab in zip(axA.get_xticklabels(), [b["short"] for b in bars] + ["GT"]):
    if lab == "GT":
        t.set_color(C_ORACLE)
        t.set_fontweight("bold")
axA.tick_params(axis="x", length=3)
for sp in ("top", "right"):
    axA.spines[sp].set_visible(False)

axA.set_title("(a)  the eight analyses, scored", loc="left",
              fontsize=10.0, fontweight="bold", pad=6)

# Tradition legend — a single horizontal strip BELOW the axes (outside the data
# field). Placed as a FIGURE-level legend at an absolute y so it sits cleanly in
# the band between the two-line x-tick labels and the footnote: nothing over the
# bars, nothing clipping the figure edge (fig_pass: legends outside the plotting
# area). Only one legend now — the triad-marker overlay was removed so each bar
# carries exactly one numeric label (fig_pass: "one numeric label per bar").
trad_handles = [
    Patch(facecolor=TRAD_COLOR[t], edgecolor=C_INK, linewidth=0.7,
          label=TRAD_LABEL[t])
    for t in ("intervention", "correlational", "dim_reduction", "descriptive")
]
leg = fig.legend(handles=trad_handles, loc="center",
                 bbox_to_anchor=(0.34, 0.085), frameon=True, fontsize=8.0,
                 title="method tradition", title_fontsize=8.2, handlelength=1.3,
                 ncol=4, columnspacing=1.4, handletextpad=0.5, borderpad=0.5)
leg.get_frame().set_edgecolor("#cccccc")
leg.get_frame().set_linewidth(0.7)

# --- Panel (b): the four key findings as a COMPACT TABLE ---------------------
# fig_pass "Fig. 2" blocker fix: the old callout boxes overlapped (head over body
# text). Replaced with a clean three-column table (finding · number ·
# interpretation), no boxes over data, all text >= 8 pt.
axB = fig.add_axes([0.660, 0.060, 0.332, 0.785])
axB.set_xlim(0, 100)
axB.set_ylim(0, 100)
axB.axis("off")
axB.set_title("(b)  key findings", loc="left", fontsize=10.0,
              fontweight="bold", pad=6)

# Column anchors (x in 0..100). tag chip | head + driving number + interpretation.
# ROW_H is the vertical pitch; each row's content fits well inside it so the bottom
# interpretation line never reaches the next row's chip (the overlap blocker fix).
COL_TAG = 2.0
COL_BODY = 16.0
ROW_TOP = 97.0
ROW_H = 24.0      # vertical pitch between rows (4 rows × 24 = 96 < ROW_TOP)
HEAD_FS = 8.8
NUM_FS = 8.4
BODY_FS = 8.0

# faint row separators (one per row, just under the head band)
for k in range(4):
    yref = ROW_TOP - k * ROW_H + 3.0
    axB.plot([0, 100], [yref, yref], color=C_GRID, lw=0.7, zorder=1)


def finding_row(k, accent, tag, head, number_line, interp_lines):
    """One compact table row: colour chip + tag, bold finding head, the driving
    number (bold), then a 2-line interpretation. No box — avoids the overlap
    blocker. Content depth (~18) stays well inside ROW_H (24)."""
    y0 = ROW_TOP - k * ROW_H
    # accent colour chip + tag
    axB.add_patch(plt.Rectangle((COL_TAG, y0 - 5.4), 11.5, 5.4,
                                facecolor=accent, edgecolor="none", zorder=3))
    axB.text(COL_TAG + 5.75, y0 - 2.7, tag, ha="center", va="center",
             color="white", fontsize=HEAD_FS, fontweight="bold", zorder=4)
    # finding head
    axB.text(COL_BODY, y0 - 2.7, head, ha="left", va="center", color=C_INK,
             fontsize=HEAD_FS, fontweight="bold", zorder=4)
    # the driving number (bold, ink)
    ly = y0 - 9.2
    axB.text(COL_BODY - 1.0, ly, number_line, ha="left", va="top",
             color=C_INK, fontsize=NUM_FS, fontweight="bold", zorder=4)
    ly -= 4.7
    # interpretation (muted, normal weight) — at most 2 lines to keep the row short
    for s in interp_lines:
        axB.text(COL_BODY - 1.0, ly, s, ha="left", va="top",
                 color=C_MUTE, fontsize=BODY_FS, zorder=4)
        ly -= 4.3


# Row A6 — Granger F = 0
finding_row(
    0, TRAD_COLOR["correlational"], "A6", "Granger F = 0",
    f"Pong false-edge {FACT_A6['pong_false_edge']:.2f}, missed {FACT_A6['pong_missed_edge']:.2f} → F = {FACT_A6['pong_triadF']:.2f}",
    ["Temporal precedence ≠ causation: a clock-locked",
     "machine yields spurious bidirectional edges."],
)

# Row A3 — spurious tuning
finding_row(
    1, TRAD_COLOR["correlational"], "A3", "spurious tuning",
    f"Pong rate {FACT_A3['pong']:.2f}; Space Inv. {FACT_A3['space_invaders']:.2f}; Q*bert {FACT_A3['qbert']:.2f}",
    ["Strongly-tuned units whose tuning ≠ true causal",
     "role — the tuning-curve trap, quantified."],
)

# Row A2 — rho ~ 0.99 but interaction blind spot
finding_row(
    2, TRAD_COLOR["intervention"], "A2", "ρ = 0.99, blind to interactions",
    f"ρ = {FACT_A2['rho_mean']:.2f} vs true role; misses {FACT_A2['si_miss']*100:.0f}% of interactions",
    [f"Super-additive pair Δ up to {FACT_A2['si_superadd']:.0f} (Space Inv.) is",
     "invisible to one-unit-at-a-time ablation."],
)

# Row A7 — NMF ~ PCA (both mediocre)
finding_row(
    3, TRAD_COLOR["dim_reduction"], "A7", "NMF ≈ PCA (both mediocre)",
    f"NMF matched frac {FACT_A7['nmf_mean']:.2f} ≈ PCA {FACT_A7['pca_mean']:.2f} (best {FACT_A7['nmf_best']:.1f})",
    ["Near-tied on matched components; neither",
     "recovers the true basis."],
)

# provenance footnote — no data-path microtext in the figure body (fig_pass #6);
# exact paths live in the Supplement provenance table.
fig.text(0.012, 0.018,
         "Data: committed §R records (6 core games); whiskers = 95% bootstrap CI "
         "(every bar has a CI record). See Supplement provenance table.",
         ha="left", va="bottom", fontsize=8.0, color=C_MUTE, fontstyle="italic")

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

# (2b) every Phase-A bar has a CI whisker (no fabricated, none missing)
for b in bars:
    if not b["has_ci"] or b["yerr"] is None:
        problems.append(f"{b['method']} has no CI row in leaderboard_ci.csv")

# (3) positive control == 1.0
if abs(ORACLE["faithfulness"] - 1.0) > 1e-9:
    problems.append("oracle positive control != 1.0")

# (4) the annotated facts re-read from disk equal what we drew
_chk_a6 = _load_phaseA("A6_pong.json")
if abs(_chk_a6["extra"]["triad"]["F"] - FACT_A6["pong_triadF"]) > 1e-9:
    problems.append("A6 Pong triad F drifted from record")
if FACT_A6["pong_triadF"] != 0.0:
    problems.append("A6 Pong F is not 0 (headline finding broken)")
# A3: the tuning-curve trap = a nonzero spurious-tuning rate somewhere in the
# battery. Under the corrected re-run leaderboard the per-game rates dropped
# (Pong 0.50, not the old 1.0); the finding is now "spurious tuning up to ~0.53"
# (q*bert), read live into the table cell. Assert the trap is still present, not
# a stale point value.
if max(FACT_A3.values()) <= 0.0:
    problems.append("A3 spurious-tuning absent in every game (tuning-curve trap gone)")
if not (FACT_A2["rho_mean"] >= 0.98):
    problems.append("A2 mean rho not ~0.99 (headline finding broken)")
if not (FACT_A2["si_superadd"] > 0):
    problems.append("A2 interaction blind-spot absent (super-additive delta = 0)")
# A7: NMF ~ PCA, both mediocre. Under the corrected re-run they are near-tied
# (NMF 0.60, PCA 0.57) rather than exactly equal; the finding is the near-tie +
# mediocrity, not exact equality. Assert both are mediocre and close.
if not (abs(FACT_A7["nmf_mean"] - FACT_A7["pca_mean"]) < 0.10
        and FACT_A7["nmf_mean"] < 0.75 and FACT_A7["pca_mean"] < 0.75):
    problems.append("A7 NMF/PCA no longer near-tied-and-mediocre (headline finding broken)")

if problems:
    print("[FAIL] self-check:")
    for p in problems:
        print("   -", p)
    sys.exit(1)

print(f"[OK] wrote {OUT}  ({sz} bytes, header {head!r})")
print(f"[OK] {len(bars)} analyses scored; oracle control = 1.0; "
      f"{sum(b['has_ci'] for b in bars)}/{len(bars)} bars have CI whiskers")
print(f"[OK] findings: A6 Pong F={FACT_A6['pong_triadF']:.2f}; "
      f"A3 Pong spurious={FACT_A3['pong']:.2f}; "
      f"A2 rho={FACT_A2['rho_mean']:.2f}/superadd={FACT_A2['si_superadd']:.0f}; "
      f"A7 NMF={FACT_A7['nmf_mean']:.2f} = PCA={FACT_A7['pca_mean']:.2f}")
