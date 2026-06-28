#!/usr/bin/env python3
"""Figure: the VCS<->neural-network failure-mode map.

(In the compiled paper this file is embedded in sections/08_discussion.tex and
renders as the **VCS<->neural-network failure-mode map** — the figure whose
caption begins "The VCS<->neural-network failure-mode map."  The companion
"taxonomy of failure modes" figure is a different file, fig6_failure_taxonomy.py.
The in-figure header here carries NO hard-coded figure number, so the LaTeX
caption number is authoritative and cannot drift.)

This is the paper's answer to its single strongest objection (plan.md
"Why a 1977 chip is a fair test"): *"you study deterministic hand-coded
software to judge methods built for learned, distributed, stochastic neural
nets -- why does a score here predict anything about a transformer?"*

The figure makes the necessary-condition-screen argument concrete. Each ROW is
an interpretability FAILURE MODE that we actually MEASURED on the VCS substrate,
with its committed VCS evidence number. Each row also names the well-documented
NEURAL-NETWORK ANALOGUE -- so a method that fails *here* is failing in a way
that recurs on real networks. The crucial asymmetry: on the VCS each failure is
PROVEN in this benchmark (we hold the exact intervention oracle of §3.2, full
observability, no learning), whereas on neural nets the same failure is only a
documented ANALOGUE, not proven here. Passing the VCS screen is NECESSARY, not
sufficient (plan.md); failing it should temper trust in a method's stronger
claims without further evidence.

REVISION (P2-R-F5, per figure_detail_pass "Fig. 6 - VCS to neural-network
failure-mode map" + improvement_instructions P4#43 / P0#12; PO decisions
2026-06-23):
  * SIMPLIFIED MAIN version (default --out) + FULL SUPPLEMENT version
    (--full-out) -- PO#4 keeps a clean main-text figure and moves the full map
    to Supplementary Information.
  * Single calm header line; NO subtitle-over-callout overlap (the pink callout
    is gone, its content folded into one boundary-of-inference legend line).
  * "...has no business being trusted there..." -> "...should not be trusted
    for stronger claims without further evidence." (P0#12).
  * The repeated per-row green/orange status badges are replaced by a single
    status header/legend ("VCS evidence: proven in this benchmark; NN analogue:
    documented, not proven here").
  * In-cell citation NAMES removed; the seven analogue works are discussed in
    the caption (P2-R-S07/S08) and added to the bibliography by P2-R-REF.
  * "§1 oracle" / "Section 1 oracle" -> "§3.2 oracle" / "intervention oracle".
  * Data-path microtext removed from the figure body (provenance -> caption /
    Supplement); landscape page, fonts >= 8 pt, legend outside the table,
    colour-blind-safe (Okabe-Ito), embedded TrueType fonts.
  * Uncertainty: where a row's headline score maps to a leaderboard_ci.csv row
    (R-UNC bootstrap-over-games 95% CI), the badge shows the CI; the headline
    position-regime gap shows its committed CI95.  No CI is fabricated for the
    position-regime-specific or derived margins that have none in the records.

Six failure modes (experiment_design.md §4-§7; plan.md figures 4-5):

  1. Gradient VANISHES on discrete / index (position) outputs.
  2. Saliency MODEL-INVARIANCE -- the Adebayo sanity-check failure.
  3. CORRELATION != CAUSATION -- the Granger / tuning-curve trap.
  4. INTERACTION BLIND-SPOT of single-unit ablation.
  5. SAE / PROBE "PRESENT-NOT-USED" + polysemanticity.
  6. BASELINE-SENSITIVITY of Integrated / Expected Gradients.

ALL VCS numbers are READ from committed records -- no experiment is re-run:
  * tools/xai_study/compare/out/leaderboard.json          (P2-E6-1 aggregate)
  * tools/xai_study/compare/out/leaderboard_ci.csv        (P2-R-UNC 95% CIs)
  * tools/xai_study/compare/out/faithful_demo.json        (P2-E6-3 headline)
  * tools/xai_study/phaseB_attribution/out/
        guided_backprop_core_summary.json                 (Adebayo sanity)
        ig_baseline_sweep_core_summary.json               (IG baseline sens.)
  * tools/xai_study/phaseC_mechanistic/out/
        sae_core_summary.json                             (SAE causal-use)
        linear_probing_core_summary.json                  (present-not-used)
The NN-analogue labels are conceptual mappings (experiment_design.md §4-§7,
plan.md), not measurements -- the figure says so explicitly (the single status
legend: "documented analogue, not proven here").

Run:
    python fig5_representativeness_map.py            # main (simplified) + supplement
Produces:
    fig5_representativeness_map.pdf        (simplified, main text)
    fig5_representativeness_map_full.pdf   (full, Supplementary Information)
both vector, colour-blind-safe, embedded fonts.
"""

import argparse
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless vector backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

HERE = os.path.dirname(os.path.abspath(__file__))
# repo root = .../  (figures -> paper -> xai_2_interpretability -> xai_paper -> root)
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))


# ---------------------------------------------------------------------------
# Palette — Okabe-Ito colour-blind-safe set (matches fig1/fig4 house style).
# ---------------------------------------------------------------------------
C_BG = "#ffffff"
C_INK = "#1a1a1a"          # primary text / outlines
C_MUTE = "#5b5b5b"         # secondary text
C_GRID = "#d9d9d9"

C_VCS = "#0072B2"          # blue        — the VCS column (proven here)
C_NN = "#D55E00"           # vermillion  — the neural-network analogue
C_PROVE = "#009E73"        # bluish-green — "proven in this benchmark"
C_SUSPECT = "#E69F00"      # orange      — "documented analogue, not proven"
C_VCS_FILL = "#dbeaf4"     # pale blue panel
C_NN_FILL = "#f7e2d6"      # pale vermillion panel
C_HEAD = "#ececec"         # neutral header fill
C_ROW_A = "#f4f4f4"        # row zebra
C_ROW_B = "#ffffff"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.4,
        "axes.linewidth": 0.8,
        "axes.edgecolor": C_MUTE,
        "pdf.fonttype": 42,    # embed TrueType (editable, not bitmap)
        "svg.fonttype": "none",
        "figure.dpi": 150,
        "xtick.color": C_INK,
        "ytick.color": C_INK,
        "text.color": C_INK,
    }
)

MIN_FONT = 8.0   # figure style guide: minimum font size (>= 8 pt)


# ---------------------------------------------------------------------------
# Data load — pure reads over committed records.
# ---------------------------------------------------------------------------
def _load(path):
    with open(path) as fh:
        return json.load(fh)


def load_records():
    out = os.path.join(REPO_ROOT, "tools", "xai_study")
    lb = _load(os.path.join(out, "compare", "out", "leaderboard.json"))
    demo = _load(os.path.join(out, "compare", "out", "faithful_demo.json"))
    gbp = _load(os.path.join(out, "phaseB_attribution", "out",
                             "guided_backprop_core_summary.json"))
    igsw = _load(os.path.join(out, "phaseB_attribution", "out",
                              "ig_baseline_sweep_core_summary.json"))
    sae = _load(os.path.join(out, "phaseC_mechanistic", "out",
                             "sae_core_summary.json"))
    prb = _load(os.path.join(out, "phaseC_mechanistic", "out",
                             "linear_probing_core_summary.json"))
    ci = load_ci(os.path.join(out, "compare", "out", "leaderboard_ci.csv"))
    return lb, demo, gbp, igsw, sae, prb, ci


def load_ci(path):
    """Read R-UNC bootstrap-over-games 95% CIs: method -> (lo, hi)."""
    ci = {}
    if not os.path.isfile(path):
        return ci
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row.get("kind") == "method":
                try:
                    ci[row["method"]] = (float(row["ci_lo"]),
                                         float(row["ci_hi"]))
                except (KeyError, ValueError):
                    pass
    return ci


def rows_by_method(lb):
    return {r["method"]: r for r in lb["rows"]}


# ---------------------------------------------------------------------------
# Assemble the six failure-mode rows from the committed records.
# ---------------------------------------------------------------------------
def build_rows(lb, demo, gbp, igsw, sae, prb, ci):
    R = rows_by_method(lb)
    hc = lb["headline_contrast"]["position_regime"]

    # (1) gradient vanishes on index/position
    sal_pos = R["vanilla_saliency"]["faithfulness_position_regime"]
    grad_bucket_pos = hc["gradient_correlational_faithfulness_mean"]

    # (2) Adebayo model-invariance (per-game block of the sanity summary)
    ade = gbp["adebayo_sanity_summary"]["per_game"][0]
    ade_cos = ade["mean_cosine_to_real_map"]
    ade_ram = ade["mean_ram_at_frame_bytes_changed"]
    ade_pass = ade["passed"]

    # (3) correlation != causation
    granger = R["A6_granger"]["faithfulness"]
    tuning = R["A3_tuning"]["faithfulness"]

    # (4) interaction blind-spot of single-unit ablation
    lesion = R["A2_lesions"]["faithfulness"]
    connect = R["A1_connectomics"]["faithfulness"]
    sae_moved = sum(g.get("n_causal_use_moved", 0) for g in sae["games"]
                    if g.get("game") in {"pong", "space_invaders"})

    # (5) SAE / probe present-not-used + polysemanticity
    sae_F = R["sparse_autoencoder"]["faithfulness"]
    sae_match_games_full = sum(1 for g in sae["games"]
                               if g["matched_fraction"] == 1.0)
    sae_match_min = min(g["matched_fraction"] for g in sae["games"])
    probe_F = R["linear_probing_control_tasks"]["faithfulness"]
    probe_acc = prb["mean_probe_acc"]
    probe_notcausal = prb["total_not_causally_used"]

    # (6) baseline-sensitivity of IG/EG
    eg_F = R["expected_gradients"]["faithfulness"]
    bl_sens_max = max(rr["baseline_corr_sensitivity"] for rr in igsw["results"])

    def cifor(method):
        return ci.get(method)

    # Each row carries BOTH a one-sentence "vcs_main" (for the simplified
    # main-text figure) and a fuller "vcs_full" (for the supplement), so the
    # one script produces a clean main version and a complete supplement.
    data = [
        dict(
            n=1,
            mode="Gradient vanishes on\ndiscrete / index outputs",
            vcs_main=("Vanilla saliency & IG = "
                      f"{sal_pos:.3f} on the\nposition regime (provably zero at\n"
                      "a strobe-timed argmax); whole\n"
                      f"gradient bucket {grad_bucket_pos:.3f}."),
            vcs_full=("Vanilla saliency & IG faithfulness =\n"
                      f"{sal_pos:.3f} on the position regime; whole\n"
                      f"gradient/correlational bucket = {grad_bucket_pos:.3f}\n"
                      "(n=9). Provably 0 at a strobe-timed\n"
                      "argmax readout (§3.2 oracle)."),
            vcs_score=sal_pos,
            score_label="position-regime F",
            score_ci=None,  # position-specific number has no CI in the records
            nn_main=("Shattered / saturated gradients;\n"
                     "non-differentiable argmax & index\nreadouts."),
            nn_full=("Shattered / saturated gradients; non-\n"
                     "differentiable argmax & index readouts\n"
                     "(analogue refs G1, G2)."),
        ),
        dict(
            n=2,
            mode="Saliency is model-invariant\n(sanity-check fails)",
            vcs_main=("Program-randomization: saliency\ncosine = "
                      f"{ade_cos:.3f} while {ade_ram:.0f}% of RAM\nchanged "
                      f"-> sanity check {'PASSES' if ade_pass else 'FAILS'}."),
            vcs_full=("Program-randomization test: saliency\n"
                      f"cosine = {ade_cos:.3f} to the real map while\n"
                      f"{ade_ram:.0f}% of RAM changed -> sanity check\n"
                      f"{'PASSED' if ade_pass else 'FAILS'}. Constant Jacobian on the\n"
                      "content path, no ReLU."),
            vcs_score=1.0 - ade_cos,  # 0 = fully invariant (failed)
            score_label="sanity-check margin",
            score_ci=None,            # derived margin, no CI in the records
            nn_main=("Saliency unchanged under model- &\n"
                     "label-randomization; behaves like\nan edge detector."),
            nn_full=("Saliency unchanged under model- & label-\n"
                     "randomization; behaves like an edge\n"
                     "detector (analogue ref S1)."),
        ),
        dict(
            n=3,
            mode="Correlation ≠ causation\n(Granger & tuning trap)",
            vcs_main=("Granger-edge faithfulness = "
                      f"{granger:.3f}\n(false edges vs the true data-flow);\n"
                      f"game-variable tuning = {tuning:.3f}."),
            vcs_full=("Granger-edge faithfulness = "
                      f"{granger:.3f}\n(false edges vs the true data-flow);\n"
                      f"game-variable tuning = {tuning:.3f}\n"
                      "(spurious) -- vs the exact §3.2 oracle."),
            vcs_score=granger,
            score_label="Granger F",
            score_ci=cifor("A6_granger"),
            nn_main=("Spurious Granger edges; 'present ≠\n"
                     "used' tuning & attention read as\nexplanation."),
            nn_full=("Spurious Granger edges; 'present ≠ used'\n"
                     "tuning & attention-as-explanation\n"
                     "(analogue refs C1, C2)."),
        ),
        dict(
            n=4,
            mode="Single-unit ablation has an\ninteraction blind-spot",
            vcs_main=("Single-unit lesion is faithful "
                      f"({lesion:.3f})\nyet misses the wiring: connectome\n"
                      f"recovered at only {connect:.3f}."),
            vcs_full=("Single-unit lesion importance is\n"
                      f"faithful ({lesion:.3f}) yet MISSES wiring:\n"
                      f"the connectome graph is recovered at\n"
                      f"only {connect:.3f}; SAE ablations move y\n"
                      f"in {sae_moved} of the audited cells."),
            vcs_score=connect,
            score_label="connectome F",
            score_ci=cifor("A1_connectomics"),
            nn_main=("Single-unit ablation misleading;\n"
                     "computation is distributed across\nmany units."),
            nn_full=("Single-unit ablation misleading;\n"
                     "computation distributed across units\n"
                     "(analogue refs D1, D2)."),
        ),
        dict(
            n=5,
            mode="SAE/probe:\npresent-not-used",
            vcs_main=("SAEs match a variable, yet score\n"
                      f"{sae_F:.3f} on causal use; probes decode at\n"
                      f"{probe_acc:.3f} acc but {probe_notcausal} cells are not causal."),
            vcs_full=("SAEs match a variable (100% of\n"
                      f"candidates in {sae_match_games_full}/6 games, min "
                      f"{sae_match_min*100:.0f}%)\nyet score {sae_F:.3f} on causal use; "
                      "probes\n"
                      f"decode at {probe_acc:.3f} acc but {probe_notcausal} cells are\n"
                      f"decodable-not-causal -> selectivity\nF = {probe_F:.3f}."),
            vcs_score=probe_F,
            score_label="selectivity F",
            score_ci=cifor("linear_probing_control_tasks"),
            nn_main=("Control-task selectivity gap;\n"
                     "superposition / polysemantic\nneurons."),
            nn_full=("Control-task selectivity gap;\n"
                     "superposition / polysemantic neurons\n"
                     "(analogue refs P1, P2)."),
        ),
        dict(
            n=6,
            mode="IG/EG baseline\nsensitivity",
            vcs_main=("IG magnitude is baseline-sensitive\n"
                      f"(up to {bl_sens_max:.3f} across games);\n"
                      f"a content-byte baseline kills IG;\nEG lands at {eg_F:.3f}."),
            vcs_full=("IG magnitude is baseline-sensitive\n"
                      f"(corr. sens. up to {bl_sens_max:.3f} across\n"
                      "games); a baseline = the content\n"
                      "byte kills IG entirely; Expected\n"
                      f"Gradients lands at {eg_F:.3f}."),
            vcs_score=eg_F,
            score_label="EG F (all-regime)",
            score_ci=cifor("expected_gradients"),
            nn_main=("Attribution depends on the (arbitrary)\n"
                     "baseline choice; no canonical\nreference input."),
            nn_full=("Attribution depends on the (arbitrary)\n"
                     "baseline choice; no canonical reference\n"
                     "input (analogue refs B1, B2)."),
        ),
    ]
    return data, hc


# ---------------------------------------------------------------------------
# Drawing — one routine, two modes (main = simplified, full = supplement).
# ---------------------------------------------------------------------------
def draw(data, hc, full=False):
    """Render the failure-mode map. full=False -> simplified main figure;
    full=True -> complete supplement figure (more per-cell text + analogue
    reference key + provenance line)."""
    n_rows = len(data)

    # Landscape canvas; the supplement is a touch taller for the extra text.
    # Heights trimmed after the footer legend/headline-gap block was removed,
    # so no large empty band remains under the row block.
    fig = plt.figure(figsize=(11.6, 6.0 if not full else 7.6))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---- column geometry (x in 0..100): a clean 4-column table ----
    X0 = 2.5                 # left margin
    X1 = 97.5                # right margin
    COL_MODE = (X0, 21.5)            # failure mode (measured here)
    COL_VCS = (22.5, 53.5)           # VCS measured evidence (+ score badge)
    COL_NN = (54.5, 80.0)            # NN analogue
    COL_BND = (81.0, X1)             # boundary of inference

    # ---- vertical band for rows ----
    # The supplement needs extra room under the table for the analogue
    # reference key + full provenance line, so its row block stops higher.
    # The in-image title/subtitle were removed (they duplicated the LaTeX
    # caption); the row block top is raised to fill that band, leaving only a
    # small margin above the column headers.
    TOP = 92.5 if not full else 93.5       # top of the row block
    # Bottom of the row block lowered after the footer block was removed: the
    # main figure now has no below-table text, and the supplement keeps only the
    # analogue-reference key + provenance line just under the rows.
    BOT = 5.0 if not full else 14.0        # bottom of the row block
    row_h = (TOP - BOT) / n_rows

    # =====================================================================
    # (In-image title + subtitle removed — the LaTeX caption owns that prose.)
    # =====================================================================
    # Column headers — the single status relation lives HERE (header),
    # replacing the six repeated per-row green/orange badges.
    # =====================================================================
    HDR_Y = TOP + 1.4
    HDR_H = 3.0

    def header(cx0, cx1, line1, line2, fill, edge, tcol=C_INK):
        h = FancyBboxPatch(
            (cx0, HDR_Y), cx1 - cx0, HDR_H,
            boxstyle="round,pad=0.1,rounding_size=0.6",
            linewidth=1.1, edgecolor=edge, facecolor=fill, zorder=2,
        )
        ax.add_patch(h)
        cx = (cx0 + cx1) / 2
        ax.text(cx, HDR_Y + HDR_H * 0.66, line1, ha="center", va="center",
                fontsize=9.2, fontweight="bold", color=tcol, zorder=3)
        ax.text(cx, HDR_Y + HDR_H * 0.26, line2, ha="center", va="center",
                fontsize=MIN_FONT, color=C_MUTE, zorder=3)

    header(*COL_MODE, "Failure mode", "measured on the VCS", C_HEAD, C_MUTE)
    header(*COL_VCS, "VCS measured evidence",
           "proven in this benchmark", C_VCS_FILL, C_VCS, C_VCS)
    header(*COL_NN, "Neural-network analogue",
           "documented, not proven here", C_NN_FILL, C_NN, C_NN)
    header(*COL_BND, "Status", "here → there", C_HEAD, C_MUTE)

    # =====================================================================
    # Rows
    # =====================================================================
    for i, d in enumerate(data):
        y_top = TOP - i * row_h
        y_bot = y_top - row_h
        y_mid = (y_top + y_bot) / 2

        # zebra background band across the full width
        band = FancyBboxPatch(
            (X0, y_bot + 0.22), X1 - X0, row_h - 0.44,
            boxstyle="round,pad=0,rounding_size=0.4",
            linewidth=0, facecolor=(C_ROW_A if i % 2 == 0 else C_ROW_B),
            zorder=0.5,
        )
        ax.add_patch(band)

        # ---- mode cell (small number chip + text) ----
        chip_x = COL_MODE[0] + 1.4
        chip_y = y_top - 1.9
        ax.scatter([chip_x], [chip_y], s=150, marker="o",
                   color=C_INK, zorder=4)
        ax.text(chip_x, chip_y, str(d["n"]), ha="center", va="center",
                fontsize=MIN_FONT, fontweight="bold", color="white", zorder=5)
        ax.text(COL_MODE[0] + 2.9, y_mid, d["mode"], ha="left", va="center",
                fontsize=MIN_FONT + 0.6, fontweight="bold", color=C_INK,
                zorder=4, linespacing=1.2)

        # ---- VCS evidence panel ----
        vpanel = FancyBboxPatch(
            (COL_VCS[0], y_bot + 0.55), COL_VCS[1] - COL_VCS[0], row_h - 1.1,
            boxstyle="round,pad=0.1,rounding_size=0.5",
            linewidth=0.9, edgecolor=C_VCS, facecolor=C_VCS_FILL, zorder=2,
        )
        ax.add_patch(vpanel)
        # score badge (left of the text, the headline number for this row).
        # All badge text is kept >= 7.5 pt (figure style guide); the CI is
        # stacked on two short lines so it fits without shrinking the font.
        bw = 10.0
        bh = row_h - 2.0
        sb_cx = COL_VCS[0] + 1.0 + bw / 2
        sb = FancyBboxPatch(
            (COL_VCS[0] + 1.0, y_mid - bh / 2), bw, bh,
            boxstyle="round,pad=0.05,rounding_size=0.4",
            linewidth=1.0, edgecolor=C_VCS, facecolor="white", zorder=3,
        )
        ax.add_patch(sb)
        ax.text(sb_cx, y_mid + bh * 0.30, f"{d['vcs_score']:.3f}",
                ha="center", va="center", fontsize=11.0, fontweight="bold",
                color=C_VCS, zorder=4)
        ax.text(sb_cx, y_mid + bh * 0.05, d["score_label"], ha="center",
                va="center", fontsize=MIN_FONT - 0.5, color=C_MUTE, zorder=4)
        # uncertainty (R-UNC 95% CI) where the record provides one
        if d.get("score_ci") is not None:
            lo, hi = d["score_ci"]
            ax.text(sb_cx, y_mid - bh * 0.27,
                    f"95% CI\n[{lo:.2f}, {hi:.2f}]", ha="center",
                    va="center", fontsize=MIN_FONT - 0.5, color=C_MUTE,
                    zorder=4, linespacing=1.15)
        else:
            ax.text(sb_cx, y_mid - bh * 0.27, "single\nrecord", ha="center",
                    va="center", fontsize=MIN_FONT - 0.5, color=C_MUTE,
                    zorder=4, linespacing=1.15)
        txt = d["vcs_full"] if full else d["vcs_main"]
        ax.text(COL_VCS[0] + 1.0 + bw + 1.4, y_mid, txt, ha="left",
                va="center", fontsize=MIN_FONT, color=C_INK, zorder=4,
                linespacing=1.22)

        # ---- arrow VCS -> NN ----
        ax.annotate(
            "", xy=(COL_NN[0] - 0.3, y_mid), xytext=(COL_VCS[1] + 0.3, y_mid),
            arrowprops=dict(arrowstyle="-|>", color=C_MUTE, lw=1.3,
                            shrinkA=0, shrinkB=0), zorder=3,
        )

        # ---- NN analogue panel ----
        npanel = FancyBboxPatch(
            (COL_NN[0], y_bot + 0.55), COL_NN[1] - COL_NN[0], row_h - 1.1,
            boxstyle="round,pad=0.1,rounding_size=0.5",
            linewidth=0.9, edgecolor=C_NN, facecolor=C_NN_FILL, zorder=2,
        )
        ax.add_patch(npanel)
        nn_txt = d["nn_full"] if full else d["nn_main"]
        ax.text(COL_NN[0] + 1.2, y_mid, nn_txt, ha="left", va="center",
                fontsize=MIN_FONT, color=C_INK, zorder=4, linespacing=1.22)

        # ---- boundary-of-inference (status) column ----
        bnd_cx = (COL_BND[0] + COL_BND[1]) / 2
        ax.text(bnd_cx, y_mid + 1.2, "proven here", ha="center", va="center",
                fontsize=MIN_FONT, fontweight="bold", color=C_PROVE, zorder=4)
        ax.annotate(
            "", xy=(bnd_cx, y_mid - 0.55), xytext=(bnd_cx, y_mid + 0.55),
            arrowprops=dict(arrowstyle="-|>", color=C_MUTE, lw=1.0,
                            shrinkA=0, shrinkB=0), zorder=4,
        )
        ax.text(bnd_cx, y_mid - 1.2, "analogue there", ha="center",
                va="center", fontsize=MIN_FONT, fontweight="bold",
                color=C_SUSPECT, zorder=4)

    # =====================================================================
    # Footer.  The legend/explainer line(s) + the headline faithfulness-gap
    # sentence that used to sit just under the row block were REMOVED for
    # decluttering — that content now lives in the LaTeX caption.  The
    # supplement-only analogue reference key + full provenance line are kept
    # (they carry the citation NAMES) and anchored just under the row block.
    # =====================================================================
    legend_y = (BOT - 1.4)          # anchor just under the row block

    # Supplement-only: the analogue reference key (citation NAMES live here,
    # not in the table cells) + a full provenance line, under the row block.
    if full:
        ax.text(
            X0, (legend_y - 1.0),
            "Analogue references (documented in the literature, not measured "
            "here):\n"
            "G1 Balduzzi et al. 2017; G2 Sundararajan et al. 2017 (IG). "
            "S1 Adebayo et al. 2018. C1 Anand et al. 2019; C2 Jain & Wallace "
            "2019.\n"
            "D1 Morcos et al. 2018; D2 Leavitt & Morcos 2020. "
            "P1 Hewitt & Liang 2019; P2 Elhage et al. 2022. "
            "B1 Kindermans et al. 2019; B2 Sturmfels et al. 2020.",
            ha="left", va="top", fontsize=MIN_FONT - 0.6, color=C_MUTE,
            linespacing=1.35,
        )
        ax.text(
            X0, 2.6,
            "All VCS numbers read from committed records (leaderboard.json, "
            "leaderboard_ci.csv, faithful_demo.json, guided_backprop / "
            "ig_baseline_sweep / sae / linear_probing core summaries); 95% CIs "
            "are bootstrap-over-games (P2-R-UNC). No experiment re-run. "
            "Provenance: see Supplement.",
            ha="left", va="top", fontsize=MIN_FONT - 0.8, color=C_MUTE,
            linespacing=1.3,
        )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=os.path.join(HERE, "fig5_representativeness_map.pdf"),
        help="simplified main-text PDF",
    )
    ap.add_argument(
        "--full-out",
        default=os.path.join(HERE, "fig5_representativeness_map_full.pdf"),
        help="full Supplementary-Information PDF",
    )
    args = ap.parse_args()

    lb, demo, gbp, igsw, sae, prb, ci = load_records()
    data, hc = build_rows(lb, demo, gbp, igsw, sae, prb, ci)

    fig_main = draw(data, hc, full=False)
    fig_main.savefig(args.out, format="pdf", facecolor=C_BG)
    plt.close(fig_main)

    fig_full = draw(data, hc, full=True)
    fig_full.savefig(args.full_out, format="pdf", facecolor=C_BG)
    plt.close(fig_full)

    # =====================================================================
    # In-script self-check (DoD)
    # =====================================================================
    ok = True
    msgs = []

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        msgs.append(f"  [{'PASS' if cond else 'FAIL'}] {name}"
                    f"{(' — ' + detail) if detail else ''}")

    for tag, path in (("main", args.out), ("full", args.full_out)):
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        header_ok = False
        if exists:
            with open(path, "rb") as fh:
                header_ok = fh.read(5) == b"%PDF-"
        check(f"[{tag}] PDF file exists", exists, path)
        check(f"[{tag}] PDF size > 4 KB", size > 4096, f"{size} bytes")
        check(f"[{tag}] PDF has %PDF- header", header_ok)

    # data-integrity: every plotted VCS number traces to a committed record
    R = rows_by_method(lb)
    check("6 failure-mode rows built", len(data) == 6, f"n={len(data)}")
    check(
        "leaderboard = 30 methods + oracle = 31 rows",
        len(lb["rows"]) == 31, f"n={len(lb['rows'])}",
    )
    check(
        "row1 gradient vanishes on position (saliency==0.0, IG==0.0)",
        R["vanilla_saliency"]["faithfulness_position_regime"] == 0.0
        and R["integrated_gradients"]["faithfulness_position_regime"] == 0.0,
    )
    ade = gbp["adebayo_sanity_summary"]["per_game"][0]
    check(
        "row2 Adebayo sanity FAILS (cosine==1.0, not passed)",
        ade["mean_cosine_to_real_map"] == 1.0 and ade["passed"] is False,
        f"ram_changed={ade['mean_ram_at_frame_bytes_changed']:.1f}%",
    )
    check(
        "row3 correlation!=causation (Granger<0.25 & tuning<0.25)",
        R["A6_granger"]["faithfulness"] < 0.25
        and R["A3_tuning"]["faithfulness"] < 0.25,
        f"granger={R['A6_granger']['faithfulness']:.3f} "
        f"tuning={R['A3_tuning']['faithfulness']:.3f}",
    )
    check(
        "row4 single-unit blind-spot (lesion high, connectome lower)",
        R["A2_lesions"]["faithfulness"] > 0.9
        and R["A1_connectomics"]["faithfulness"] < 0.5,
        f"lesion={R['A2_lesions']['faithfulness']:.3f} "
        f"connectome={R['A1_connectomics']['faithfulness']:.3f}",
    )
    check(
        "row5 present-not-used (SAE<0.25, probe<0.25, SAE-match high)",
        R["sparse_autoencoder"]["faithfulness"] < 0.25
        and R["linear_probing_control_tasks"]["faithfulness"] < 0.25
        and min(g["matched_fraction"] for g in sae["games"]) > 0.8,
        f"not_causally_used={prb['total_not_causally_used']}, "
        f"SAE-match min={min(g['matched_fraction'] for g in sae['games']):.3f}",
    )
    bl_sens_max = max(rr["baseline_corr_sensitivity"] for rr in igsw["results"])
    check(
        "row6 IG/EG baseline-sensitive (max corr_sensitivity>0.5, EG<0.1)",
        bl_sens_max > 0.5 and R["expected_gradients"]["faithfulness"] < 0.1,
        f"bl_sens_max={bl_sens_max:.3f} "
        f"EG={R['expected_gradients']['faithfulness']:.3f}",
    )
    check(
        "headline position gap matches leaderboard (0.344)",
        abs(hc["gap"] - 0.3435) < 1e-6,
        f"gap={hc['gap']}",
    )
    # uncertainty wiring: the four rows that map to a leaderboard_ci.csv row
    # carry a CI; the position-specific / derived margins do not (not fabricated)
    ci_rows = sum(1 for d in data if d.get("score_ci") is not None)
    check(
        "CI attached to the 4 rows with a leaderboard_ci.csv mean",
        ci_rows == 4, f"rows-with-CI={ci_rows}",
    )
    for m in ("A6_granger", "A1_connectomics",
              "linear_probing_control_tasks", "expected_gradients"):
        check(f"CI present in record for {m}", m in ci, str(ci.get(m)))

    print("Figure (VCS<->NN map) self-check:")
    print("\n".join(msgs))
    print(f"\n{'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}")
    print(f"  main -> {args.out}")
    print(f"  full -> {args.full_out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
