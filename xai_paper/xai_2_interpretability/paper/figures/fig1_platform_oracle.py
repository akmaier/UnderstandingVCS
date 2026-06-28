#!/usr/bin/env python3
"""Figure 1 — measurement platform & ground-truth oracle schematic.

A publication-quality, vector-PDF schematic of Paper-2's measurement platform,
redrawn for the revision (P2-R-F1) per the figure_detail_pass "Fig. 1" notes and
the RESOLVED PO decisions in revision_plan.md:

    differentiable VCS  →  intervention oracle (§3.2)  →  F ∧ S ∧ M scoring

Three clean columns, left to right, each carrying ONE visual idea:

  (1) THE SUBSTRATE — the differentiable VCS (jutari / jaxtari), bit-exact to the
      xitari C++ reference (Paper-1: 64/64 games, RAM + screen). It grants four
      capabilities no neural network gives for free: exact forward, exact
      interventions do(u), full state observability, content-path gradients.

  (2) THE ORACLE — built ON the substrate (§3.2 oracle). The PRIMARY ground truth
      is the EXACT intervention oracle: do(u:=v') → bit-exact re-run → Δy. A
      content-path gradient companion (∂y/∂u via SOFT-STE) is a secondary check.
      A single caveat callout states the discrete-index limitation: for
      position / index / event outputs the naive gradient is zero, so the
      intervention oracle is the SOLE ground truth there.

  (3) THE SCORE — any XAI method's explanation Ê is scored against the oracle by
      the correctness triad Faithful ∧ Sufficient ∧ Minimal. The two reporting
      axes — faithfulness vs the plausibility PROXY (not a human measurement) —
      expose the danger zone: high plausibility proxy, low faithfulness.

Design rules applied (figure_detail_pass global + Fig. 1):
  - one idea per column; short labels only; no prose paragraphs inside boxes;
  - colour identity carried by direct panel headers (no bottom legend strip);
  - oracle (vermillion, solid, heavy) vs method-under-test (grey, dashed) are
    visually distinct;
  - the discrete-index caveat is a single callout under the oracle column;
  - all in-figure text ≥ 8 pt; Okabe-Ito colour-blind-safe palette;
  - no source paths / data-provenance microtext in the figure body (it lives in
    the caption / supplement);
  - vector PDF with embedded TrueType fonts.

This is a SCHEMATIC: it draws no experiment data (no leaderboard dependency).

Run:
    python fig1_platform_oracle.py
Produces:
    fig1_platform_oracle.pdf   (vector, colour-blind-safe)
"""

import os
import matplotlib

matplotlib.use("Agg")  # headless vector backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# ---------------------------------------------------------------------------
# Palette — Okabe-Ito colour-blind-safe set. One neutral grey ramp for
# structure + three accent hues that key the three columns.
# ---------------------------------------------------------------------------
C_BG = "#ffffff"
C_INK = "#1a1a1a"           # primary text / outlines
C_MUTE = "#4a4a4a"          # secondary text (darkened for contrast)
C_SUBSTRATE = "#0072B2"     # blue       — the differentiable VCS substrate
C_SUBSTRATE_FILL = "#e7f1f8"
C_ORACLE = "#D55E00"        # vermillion — the ground-truth oracle (the instrument)
C_ORACLE_FILL = "#fdece0"
C_GRAD = "#009E73"          # bluish-green — the gradient companion (content path)
C_GRAD_FILL = "#e3f5ee"
C_SCORE = "#56260f"         # dark brown — the triad scoring
C_SCORE_FILL = "#f3eae4"
C_CAVEAT = "#9b3d7a"        # reddish-purple (darkened) — the index caveat / danger flag
C_CAVEAT_FILL = "#f6e7f0"
C_PANEL = "#fafafa"         # faint stage backplate

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.linewidth": 0.0,
        "pdf.fonttype": 42,   # embed TrueType (editable, not bitmap)
        "svg.fonttype": "none",
        "figure.dpi": 150,
    }
)

# ---------------------------------------------------------------------------
# Canvas (data coords 0..100 x, 0..56 y; landscape figure, full text width)
# The in-image title/subtitle were removed (they duplicated the LaTeX caption);
# the canvas top is tightened to the panel block (top edge at y=54.5) so there
# is no empty band where the headline used to be.
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 11.4, 5.8
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C_BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 56)
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
    """Faint stage backplate with a coloured top-tab header.

    The tab IS the colour key for the column (replaces the old bottom legend)."""
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor=C_PANEL, edgecolor=col,
                  linewidth=1.2, zorder=1, joinstyle="round")
    )
    ax.add_patch(
        FancyBboxPatch((x + 0.6, y + h - 4.2), w - 1.2, 3.4,
                       boxstyle="round,pad=0.0,rounding_size=0.6",
                       facecolor=col, edgecolor="none", zorder=3)
    )
    ax.text(x + w / 2, y + h - 2.5, label, ha="center", va="center",
            color="white", fontsize=10.4, fontweight="bold", zorder=4)


def txt(x, y, s, ha="center", va="center", size=9.0, col=C_INK, weight="normal",
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


# ---------------------------------------------------------------------------
# Three column backplates. Tabs carry the colour key (no separate legend).
# (In-image title/subtitle removed — the LaTeX caption owns that prose.)
# ---------------------------------------------------------------------------
panel(1.0, 3.5, 31.0, 51.0, "1  SUBSTRATE", C_SUBSTRATE)
panel(34.5, 3.5, 33.0, 51.0, "2  ORACLE  (§3.2)", C_ORACLE)
panel(70.0, 3.5, 29.0, 51.0, "3  SCORE", C_SCORE)

# ===========================================================================
# STAGE 1 — the differentiable VCS substrate
# ===========================================================================
# central substrate block
box(3.5, 41.5, 26.0, 7.6, C_SUBSTRATE_FILL, C_SUBSTRATE, lw=1.8, rad=1.2)
txt(16.5, 46.7, "Differentiable VCS", size=11.5, weight="bold", col=C_SUBSTRATE)
txt(16.5, 43.7, "jutari (Julia) · jaxtari (JAX)", size=9.0, col=C_INK)

# Paper-1 conformance badge (kept — it is the load-bearing fact)
box(3.5, 36.6, 26.0, 3.8, "#ffffff", C_SUBSTRATE, lw=1.1, rad=0.7)
txt(16.5, 38.5, "bit-exact to xitari · 64/64 games", size=8.6, col=C_SUBSTRATE,
    weight="bold")

# the four capabilities the substrate grants — short labels, no prose paragraphs
caps = [
    ("Exact forward", "deterministic re-run"),
    ("Exact interventions", "do(u) on RAM / register / ROM / input"),
    ("Full observability", "every cell, register, opcode, pixel"),
    ("Content-path gradients", "∂y/∂u via SOFT-STE"),
]
cy = 31.0
for i, (h, sub) in enumerate(caps):
    y = cy - i * 6.2
    box(3.5, y - 2.2, 26.0, 5.0, "#ffffff", C_SUBSTRATE, lw=1.0, rad=0.7)
    txt(5.2, y + 0.7, "✓", ha="left", size=11.0, col=C_SUBSTRATE, weight="bold")
    txt(7.6, y + 0.9, h, ha="left", size=9.2, col=C_INK, weight="bold")
    txt(7.6, y - 1.1, sub, ha="left", size=8.0, col=C_MUTE)

# substrate -> oracle arrow (clean lane, single short label)
arrow(32.2, 35.0, 34.3, 35.0, col=C_INK, lw=2.4, mut=18)
txt(33.25, 37.2, "outputs y", size=8.0, col=C_MUTE)
txt(33.25, 32.8, "+ causes u", size=8.0, col=C_MUTE)

# ===========================================================================
# STAGE 2 — the ground-truth oracle (primary + companion + single caveat)
# ===========================================================================
# (2a) the PRIMARY intervention oracle — vermillion, solid, heavy (= "the oracle")
box(36.0, 35.5, 30.0, 13.6, C_ORACLE_FILL, C_ORACLE, lw=2.2, rad=1.2)
txt(51.0, 46.8, "Exact intervention oracle", size=10.8, weight="bold", col=C_ORACLE)
txt(51.0, 44.4, "PRIMARY ground truth", size=8.4, col=C_MUTE, style="italic")


# the do -> re-run -> Δy mini-pipeline inside the oracle (short labels)
def chip(x, w, top, bot, ec=C_ORACLE):
    box(x, 38.0, w, 4.6, "#ffffff", ec, lw=1.2, rad=0.7)
    txt(x + w / 2, 40.9, top, size=8.8, col=C_INK, weight="bold")
    txt(x + w / 2, 38.9, bot, size=8.0, col=C_MUTE)


chip(37.0, 8.6, "do(u:=v')", "occlude")
chip(46.8, 8.6, "Exact re-run", "bit-exact")
chip(56.6, 8.6, "Δy", "= y' − y₀")
arrow(45.6, 40.3, 46.8, 40.3, col=C_ORACLE, lw=1.7, mut=12)
arrow(55.4, 40.3, 56.6, 40.3, col=C_ORACLE, lw=1.7, mut=12)
txt(51.0, 36.6, "true causal map { Δy(u) } · no world model assumed",
    size=8.0, col=C_ORACLE, weight="bold")

# (2b) the GRADIENT companion (content path only) — green, secondary
box(36.0, 22.5, 30.0, 11.0, C_GRAD_FILL, C_GRAD, lw=1.5, rad=1.1)
txt(51.0, 31.4, "Content-path gradient companion", size=9.6, weight="bold",
    col=C_GRAD)
txt(51.0, 29.0, "∂y/∂u via SOFT-STE · raw grad · IG", size=8.4, col=C_INK)
txt(51.0, 26.7, "secondary check — cross-validates the oracle", size=8.0,
    col=C_MUTE, style="italic")
txt(51.0, 24.4, "valid for CONTENT (colour / graphics-bit) outputs", size=8.0,
    col=C_GRAD, weight="bold")

# arrows from substrate's two output kinds INTO the two oracle constructions
arrow(34.3, 42.3, 36.0, 42.3, col=C_ORACLE, lw=1.7, rad=0.0, mut=13)
arrow(34.3, 28.0, 36.0, 28.0, col=C_GRAD, lw=1.7, rad=0.0, mut=13)

# the single discrete-index caveat callout (separate, under the oracle column)
box(36.0, 5.0, 30.0, 15.5, C_CAVEAT_FILL, C_CAVEAT, lw=1.5, rad=0.9, ls="--")
txt(51.0, 18.2, "Caveat — discrete-index outputs", size=9.0, col=C_CAVEAT,
    weight="bold")
txt(51.0, 15.6, "sprite position / index / event", size=8.2, col=C_INK)
txt(51.0, 13.2, "naive gradient = 0  ✗", size=8.8, col=C_CAVEAT, weight="bold")
txt(51.0, 10.4, "→ the intervention oracle is the SOLE", size=8.0, col=C_INK)
txt(51.0, 8.6, "ground truth there; gradient methods", size=8.0, col=C_INK)
txt(51.0, 6.8, "are scored AS methods under test.", size=8.0, col=C_INK)

# oracle -> score arrow (clean lane, single short label)
arrow(66.2, 42.3, 69.8, 42.3, col=C_INK, lw=2.4, mut=18)
txt(68.0, 44.3, "ground truth", size=8.0, col=C_MUTE)

# ===========================================================================
# STAGE 3 — scoring any XAI method by F ∧ S ∧ M
# ===========================================================================
# the candidate explanation entering (grey/dashed = "method under test")
box(71.5, 44.5, 26.0, 5.0, "#ffffff", C_MUTE, lw=1.4, rad=0.8, ls="--")
txt(84.5, 47.7, "any XAI method → explanation Ê", size=9.0, col=C_INK,
    weight="bold")
txt(84.5, 45.6, "method under test", size=8.0, col=C_MUTE, style="italic")
arrow(84.5, 44.5, 84.5, 43.1, col=C_SCORE, lw=1.7, mut=13)

# the triad — short labels (no internal prose paragraphs)
triad = [
    ("F: causes", "claimed causes = TRUE causes"),
    ("S: behavior", "predicts y under held-out do(u)"),
    ("M: parsimony", "minimal, right level of state"),
]
ty = 39.8
for i, (h, mid) in enumerate(triad):
    y = ty - i * 6.6
    box(71.5, y - 2.5, 26.0, 5.3, C_SCORE_FILL, C_SCORE, lw=1.3, rad=0.8)
    txt(73.4, y + 0.9, h, ha="left", size=9.8, col=C_SCORE, weight="bold")
    txt(73.4, y - 1.3, mid, ha="left", size=8.0, col=C_INK)
    if i < 2:
        txt(95.6, y - 3.3, "∧", ha="center", size=13, col=C_SCORE, weight="bold")

# the verdict
box(71.5, 13.0, 26.0, 5.4, C_SCORE, "none", lw=0, rad=0.9)
txt(84.5, 16.4, "right = F ∧ S ∧ M", size=11.0, col="white", weight="bold")
txt(84.5, 14.1, "verified against the true causes", size=8.2, col="#f0e6df",
    style="italic")
arrow(84.5, 20.2, 84.5, 18.4, col=C_SCORE, lw=1.7, mut=13)

# the two reporting axes / danger-zone callout (plausibility PROXY)
box(71.5, 5.0, 26.0, 6.6, C_CAVEAT_FILL, C_CAVEAT, lw=1.5, rad=0.8)
txt(84.5, 9.6, "axes: faithfulness vs plausibility proxy", size=8.2,
    col=C_INK, weight="bold")
txt(84.5, 6.8, "danger zone: convincing but not faithful", size=8.2,
    col=C_CAVEAT, weight="bold")

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
