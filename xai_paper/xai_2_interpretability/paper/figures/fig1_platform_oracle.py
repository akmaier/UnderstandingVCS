#!/usr/bin/env python3
"""Figure 1 — platform & ground-truth oracle schematic (P2-E7-1).

A publication-quality, vector-PDF schematic of Paper-2's measurement platform:

    differentiable VCS  →  ground-truth oracle  →  Faithful ∧ Sufficient ∧ Minimal

Three stages, left to right:

  (1) THE SUBSTRATE — the differentiable VCS (jutari/jaxtari), bit-exact to the
      xitari reference (Paper-1: 64/64 games, RAM + screen). It gives us four
      capabilities no neural network grants for free: exact forward, exact
      interventions do(u), full state observability, and content-path gradients.

  (2) THE ORACLE — built ON the substrate (experiment_design.md §1 / oracle
      README). Two independent constructions:
        * the EXACT intervention oracle (primary):  do(u:=v') → bit-exact re-run
          → Δy(u). No world model is assumed; every Δ is a real re-run.
        * the content-path GRADIENT companion: ∂y/∂u through the SOFT-STE path.
      The caveat that recurs through Phases B/C is made visually explicit: the
      STE gradient routes to the CONTENT path (colour/graphics-bit values); a
      DISCRETE INDEX (sprite position via strobe timing) has zero naive gradient,
      so for position/index/event outputs the intervention oracle is the SOLE
      ground truth and gradient methods are scored as methods under test.

  (3) THE SCORE — any XAI method's explanation Ê is scored against the oracle by
      the §0 correctness triad: Faithful ∧ Sufficient ∧ Minimal. The two reporting
      axes (faithfulness vs human-plausibility) expose the danger zone — high
      plausibility, low faithfulness.

This is a SCHEMATIC: it draws no experiment data (no E6 dependency). It is grounded
in committed artifacts only by reference (the oracle code, the leaderboard, game_set).

Run:
    python fig1_platform_oracle.py
Produces:
    fig1_platform_oracle.pdf   (vector, colour-blind-safe, self-contained legend)
"""

import os
import matplotlib

matplotlib.use("Agg")  # headless vector backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Palette — Okabe-Ito colour-blind-safe set. Two ramps only (per design budget):
#   a neutral grey ramp for substrate/structure, an accent ramp for the oracle.
# ---------------------------------------------------------------------------
C_BG = "#ffffff"
C_INK = "#1a1a1a"          # primary text / outlines
C_MUTE = "#5b5b5b"         # secondary text
C_SUBSTRATE = "#0072B2"    # blue  — the differentiable VCS substrate
C_SUBSTRATE_FILL = "#e7f1f8"
C_ORACLE = "#D55E00"       # vermillion — the ground-truth oracle (the instrument)
C_ORACLE_FILL = "#fdece0"
C_GRAD = "#009E73"         # bluish-green — the gradient companion (content path)
C_GRAD_FILL = "#e3f5ee"
C_SCORE = "#56260f"        # dark — the triad scoring
C_SCORE_FILL = "#f3eae4"
C_CAVEAT = "#CC79A7"       # reddish-purple — the index-caveat (danger flag)
C_PANEL = "#fafafa"        # faint stage backplate

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.6,
        "axes.linewidth": 0.0,
        "pdf.fonttype": 42,   # embed TrueType (editable, not bitmap)
        "svg.fonttype": "none",
        "figure.dpi": 150,
    }
)

# ---------------------------------------------------------------------------
# Canvas (data coords 0..100 x, 0..62 y; A4-ish landscape figure)
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 11.4, 6.6
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C_BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 62)
ax.axis("off")


# ---------------------------------------------------------------------------
# small drawing helpers
# ---------------------------------------------------------------------------
def box(x, y, w, h, fc, ec, lw=1.2, rad=0.9, z=2, ls="-"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rad}",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=z,
        mutation_aspect=1.0,
    )
    ax.add_patch(p)
    return p


def panel(x, y, w, h, label, col):
    """Faint stage backplate with a top tab label."""
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor=C_PANEL, edgecolor=col,
                  linewidth=1.0, zorder=1, joinstyle="round")
    )
    ax.add_patch(
        FancyBboxPatch((x + 0.6, y + h - 4.0), 21.5, 3.2,
                       boxstyle="round,pad=0.0,rounding_size=0.6",
                       facecolor=col, edgecolor="none", zorder=3)
    )
    ax.text(x + 1.6, y + h - 2.4, label, ha="left", va="center",
            color="white", fontsize=9.4, fontweight="bold", zorder=4)


def txt(x, y, s, ha="center", va="center", size=8.6, col=C_INK, weight="normal",
        style="normal", z=5):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=col,
            fontweight=weight, fontstyle=style, zorder=z)


def arrow(x0, y0, x1, y1, col=C_INK, lw=1.8, rad=0.0, ls="-", z=4,
          mut=14, head="-|>"):
    ax.add_patch(
        FancyArrowPatch((x0, y0), (x1, y1),
                        connectionstyle=f"arc3,rad={rad}",
                        arrowstyle=head, mutation_scale=mut,
                        linewidth=lw, color=col, linestyle=ls, zorder=z,
                        shrinkA=0, shrinkB=0)
    )


# ===========================================================================
# Title
# ===========================================================================
txt(50, 60.4,
    "A differentiable, bit-exact game console as a ground-truth testbed for interpretability",
    size=11.6, weight="bold")
txt(50, 57.7,
    "The subject IS the substrate — a fully known program — so every explanation can be scored against the true causes.",
    size=8.6, col=C_MUTE, style="italic")

# Stage backplates
panel(1.0, 6.0, 31.0, 49.0, "(1) THE SUBSTRATE", C_SUBSTRATE)
panel(34.5, 6.0, 33.0, 49.0, "(2) THE ORACLE", C_ORACLE)
panel(70.0, 6.0, 29.0, 49.0, "(3) THE SCORE", C_SCORE)

# ===========================================================================
# STAGE 1 — the differentiable VCS substrate
# ===========================================================================
# central substrate block
box(3.5, 33.0, 26.0, 14.0, C_SUBSTRATE_FILL, C_SUBSTRATE, lw=1.8, rad=1.2)
txt(16.5, 44.6, "Differentiable VCS", size=11.0, weight="bold", col=C_SUBSTRATE)
txt(16.5, 41.7, "jutari (JAX)  ·  jaxtari (JAX)", size=8.6, col=C_INK)
txt(16.5, 38.4, "bit-exact to xitari C++ reference",
    size=8.2, col=C_MUTE, style="italic")
txt(16.5, 35.6, "Paper-1: 64 / 64 games  (RAM + screen)",
    size=8.6, col=C_SUBSTRATE, weight="bold")

# ROM -> emulator input chip
box(3.5, 48.4, 12.0, 3.6, "#ffffff", C_SUBSTRATE, lw=1.1, rad=0.7)
txt(9.5, 50.2, "Atari 2600 ROM", size=8.0, col=C_INK)
box(17.5, 48.4, 12.0, 3.6, "#ffffff", C_SUBSTRATE, lw=1.1, rad=0.7)
txt(23.5, 50.2, "joystick do(action)", size=8.0, col=C_INK)
arrow(9.5, 48.4, 11.5, 47.2, col=C_SUBSTRATE, lw=1.3)
arrow(23.5, 48.4, 21.5, 47.2, col=C_SUBSTRATE, lw=1.3)

# the four capabilities the substrate grants
caps = [
    ("Exact forward", "deterministic, bit-exact re-run"),
    ("Exact interventions", "do(u) on RAM / register / ROM / input"),
    ("Full observability", "every RAM cell, register, opcode, pixel"),
    ("Content-path gradients", "∂y/∂u via SOFT-STE (exact-forward)"),
]
cy = 29.5
for i, (h, sub) in enumerate(caps):
    y = cy - i * 5.5
    box(3.5, y - 2.0, 26.0, 4.6, "#ffffff", C_SUBSTRATE, lw=1.0, rad=0.7)
    txt(5.0, y + 0.9, "✓", ha="left", size=10.5, col=C_SUBSTRATE, weight="bold")
    txt(7.2, y + 0.9, h, ha="left", size=8.6, col=C_INK, weight="bold")
    txt(7.2, y - 1.0, sub, ha="left", size=7.4, col=C_MUTE)

# substrate -> oracle arrow (between panels)
arrow(32.2, 30.5, 34.3, 30.5, col=C_INK, lw=2.4, mut=18)
txt(33.25, 32.4, "outputs y", size=7.4, col=C_MUTE)
txt(33.25, 28.7, "+ causes u", size=7.4, col=C_MUTE)

# ===========================================================================
# STAGE 2 — the ground-truth oracle (two constructions)
# ===========================================================================
# (2a) the PRIMARY intervention oracle
box(36.0, 33.0, 30.0, 17.5, C_ORACLE_FILL, C_ORACLE, lw=1.9, rad=1.2)
txt(51.0, 48.0, "Exact intervention oracle", size=10.4, weight="bold", col=C_ORACLE)
txt(51.0, 45.6, "— the PRIMARY ground truth —", size=7.8, col=C_MUTE, style="italic")

# the do->rerun->Δy mini-pipeline inside the oracle
def chip(x, w, top, bot, fc=C_ORACLE_FILL, ec=C_ORACLE):
    box(x, 38.0, w, 4.7, "#ffffff", ec, lw=1.1, rad=0.7)
    txt(x + w / 2, 41.0, top, size=8.4, col=C_INK, weight="bold")
    txt(x + w / 2, 38.9, bot, size=7.0, col=C_MUTE)

chip(37.0, 8.6, "do(u := v′)", "occlude · clamp")
chip(46.8, 8.6, "bit-exact", "re-run ROM")
chip(56.6, 8.6, "Δy(u)", "= y′ − y₀")
arrow(45.6, 40.35, 46.8, 40.35, col=C_ORACLE, lw=1.6, mut=12)
arrow(55.4, 40.35, 56.6, 40.35, col=C_ORACLE, lw=1.6, mut=12)
txt(51.0, 35.2, "true causal map  { Δy(u) }   ·   no world model assumed",
    size=7.6, col=C_ORACLE, weight="bold")

# (2b) the GRADIENT companion (content path only) + the index caveat
box(36.0, 16.5, 30.0, 14.0, C_GRAD_FILL, C_GRAD, lw=1.6, rad=1.1)
txt(51.0, 28.2, "Content-path gradient companion", size=9.4, weight="bold", col=C_GRAD)
txt(51.0, 25.5, "∂y / ∂u   via SOFT-STE   (raw grad · Integrated Gradients)",
    size=8.0, col=C_INK)

# the explicit content vs index split — the recurring caveat
box(37.0, 18.0, 13.6, 5.6, "#ffffff", C_GRAD, lw=1.1, rad=0.7)
txt(43.8, 22.2, "CONTENT output", size=7.8, col=C_GRAD, weight="bold")
txt(43.8, 20.4, "colour / graphics-bit", size=7.0, col=C_MUTE)
txt(43.8, 18.9, "gradient valid ✓", size=7.2, col=C_GRAD, weight="bold")

box(51.4, 18.0, 14.0, 5.6, "#ffffff", C_CAVEAT, lw=1.3, rad=0.7, ls="--")
txt(58.4, 22.2, "INDEX output", size=7.8, col=C_CAVEAT, weight="bold")
txt(58.4, 20.4, "sprite position (strobe)", size=6.8, col=C_MUTE)
txt(58.4, 18.9, "naive gradient = 0  ✗", size=7.2, col=C_CAVEAT, weight="bold")

# arrows from substrate's two output kinds INTO the two oracle constructions
arrow(34.3, 41.0, 36.0, 41.0, col=C_ORACLE, lw=1.6, rad=0.0, mut=13)
arrow(34.3, 23.5, 36.0, 23.5, col=C_GRAD, lw=1.6, rad=0.0, mut=13)

# cross-check link between the two constructions
arrow(51.0, 33.0, 51.0, 30.5, col=C_MUTE, lw=1.3, rad=0.0, mut=11, head="<|-|>")
txt(64.2, 31.6, "cross-check", size=6.9, col=C_MUTE, style="italic", ha="right")
txt(64.2, 30.1, "(1)↔(2) corr.", size=6.6, col=C_MUTE, ha="right")

# the caveat banner — index outputs => intervention is the sole truth
box(36.0, 8.7, 30.0, 6.0, "#ffffff", C_CAVEAT, lw=1.3, rad=0.8, ls="--")
txt(51.0, 12.5, "For position / index / event outputs:", size=7.6,
    col=C_CAVEAT, weight="bold")
txt(51.0, 10.6, "the intervention oracle is the SOLE ground truth;",
    size=7.4, col=C_INK)
txt(51.0, 9.3, "gradient methods are scored AS methods under test.",
    size=7.4, col=C_INK)
arrow(58.4, 18.0, 55.0, 14.7, col=C_CAVEAT, lw=1.1, ls="--", mut=10)

# oracle -> score arrow
arrow(67.6, 41.0, 69.8, 41.0, col=C_INK, lw=2.4, mut=18)
txt(68.7, 43.1, "ground", size=7.2, col=C_MUTE)
txt(68.7, 41.6, "truth", size=7.2, col=C_MUTE)

# ===========================================================================
# STAGE 3 — scoring any XAI method by F ∧ S ∧ M
# ===========================================================================
# the candidate explanation entering
box(71.5, 46.6, 26.0, 5.0, "#ffffff", C_SCORE, lw=1.2, rad=0.8)
txt(84.5, 49.8, "any XAI method →  explanation Ê", size=8.6, col=C_INK, weight="bold")
txt(84.5, 47.7, "saliency · IG · SHAP · lesions · SAE · patching · probes",
    size=6.9, col=C_MUTE)
arrow(84.5, 46.6, 84.5, 45.0, col=C_SCORE, lw=1.6, mut=13)

# the triad
triad = [
    ("F  Faithful", "claimed causes = TRUE causes", "precision · recall · sign · magnitude"),
    ("S  Sufficient", "predicts y under held-out do(u)", "scored vs the exact oracle, not a method"),
    ("M  Minimal", "parsimonious, right level", "registers / RAM / opcodes; module match"),
]
ty = 41.5
for i, (h, mid, sub) in enumerate(triad):
    y = ty - i * 6.4
    box(71.5, y - 2.4, 26.0, 5.4, C_SCORE_FILL, C_SCORE, lw=1.2, rad=0.8)
    txt(73.2, y + 1.1, h, ha="left", size=9.0, col=C_SCORE, weight="bold")
    txt(73.2, y - 0.6, mid, ha="left", size=7.4, col=C_INK)
    txt(73.2, y - 2.0, sub, ha="left", size=6.6, col=C_MUTE, style="italic")
    if i < 2:
        txt(95.6, y - 2.85, "∧", ha="center", size=12, col=C_SCORE, weight="bold")

# the verdict
box(71.5, 14.5, 26.0, 5.2, C_SCORE, "none", lw=0, rad=0.9)
txt(84.5, 17.8, "right  =  F ∧ S ∧ M", size=11.0, col="white", weight="bold")
txt(84.5, 15.5, "verified against the true causes", size=7.6, col="#f0e6df", style="italic")
arrow(84.5, 22.7, 84.5, 19.9, col=C_SCORE, lw=1.6, mut=13)

# the two reporting axes / danger-zone callout
box(71.5, 7.0, 26.0, 5.6, "#ffffff", C_CAVEAT, lw=1.3, rad=0.8)
txt(84.5, 10.9, "two reporting axes:  faithfulness  vs.  plausibility",
    size=7.4, col=C_INK, weight="bold")
txt(84.5, 8.7, "danger zone = looks convincing, isn't faithful",
    size=7.2, col=C_CAVEAT, weight="bold")

# ===========================================================================
# Self-contained legend (bottom strip)
# ===========================================================================
legend_handles = [
    Line2D([0], [0], marker="s", color="none", markerfacecolor=C_SUBSTRATE_FILL,
           markeredgecolor=C_SUBSTRATE, markersize=11,
           label="substrate — differentiable VCS (exact forward + interventions)"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor=C_ORACLE_FILL,
           markeredgecolor=C_ORACLE, markersize=11,
           label="oracle — exact intervention map Δy(u) (primary ground truth)"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor=C_GRAD_FILL,
           markeredgecolor=C_GRAD, markersize=11,
           label="oracle — content-path gradient companion ∂y/∂u"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor="#ffffff",
           markeredgecolor=C_CAVEAT, markersize=11,
           label="caveat — index/position output: naive gradient = 0; intervention only"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor=C_SCORE_FILL,
           markeredgecolor=C_SCORE, markersize=11,
           label="score — correctness triad  Faithful ∧ Sufficient ∧ Minimal"),
]
leg = ax.legend(
    handles=legend_handles, loc="lower center", ncol=2,
    bbox_to_anchor=(0.5, 0.003), frameon=True, fontsize=7.4,
    handletextpad=0.5, columnspacing=1.6, borderpad=0.7, labelspacing=0.55,
)
leg.get_frame().set_edgecolor("#cccccc")
leg.get_frame().set_linewidth(0.8)
leg.set_zorder(20)

# provenance footnote (grounding the schematic in committed artifacts)
txt(1.2, 0.9,
    "Schematic (no experiment data). Pipeline per experiment_design.md §0/§1/§3; "
    "oracle: tools/xai_study/ground_truth/oracle_intervene.{jl,py} + common/jutari_oracle.jl.",
    ha="left", size=6.2, col="#9a9a9a", style="italic")

# ---------------------------------------------------------------------------
# Save (vector PDF)
# ---------------------------------------------------------------------------
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "fig1_platform_oracle.pdf")
fig.savefig(OUT, format="pdf", facecolor=C_BG, bbox_inches=None)
plt.close(fig)

# self-check: the file exists and is a non-empty PDF
sz = os.path.getsize(OUT)
assert sz > 4000, f"PDF suspiciously small: {sz} bytes"
with open(OUT, "rb") as fh:
    head = fh.read(5)
assert head == b"%PDF-", f"not a PDF (header={head!r})"
print(f"[OK] wrote {OUT}  ({sz} bytes, header {head!r})")
