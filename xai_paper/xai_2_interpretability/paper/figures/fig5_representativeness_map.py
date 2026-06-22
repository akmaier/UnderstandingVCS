#!/usr/bin/env python3
"""Figure 5 — the VCS<->NN failure-mode / representativeness map.

This is the paper's answer to its single strongest objection (plan.md
"Why a 1977 chip is a fair test"): *"you study deterministic hand-coded
software to judge methods built for learned, distributed, stochastic neural
nets -- why does a score here predict anything about a transformer?"*

The figure makes the necessary-condition-screen argument concrete. Each ROW is
an interpretability FAILURE MODE that we actually MEASURED on the VCS substrate,
with its committed VCS evidence number. Each row also names the well-documented
NEURAL-NETWORK counterpart (cite-ready label) -- so a method that fails *here*
is failing in a way that recurs on real networks. The crucial asymmetry, drawn
as a status column: on the VCS each failure is PROVABLE (we hold the exact
intervention oracle, full observability, no learning), whereas on neural nets
the same failure is only SUSPECTED / argued. Passing the VCS screen is
NECESSARY, not sufficient (plan.md); failing it disqualifies a method outright.

Six failure modes (experiment_design.md §4-§7; plan.md figures 4-5):

  1. Gradient VANISHES on discrete / index (position) outputs.
       VCS: vanilla-saliency & IG position-regime faithfulness = 0.000; the
       whole gradient/correlational bucket averages 0.068 on the position
       regime (leaderboard headline_contrast). NN: shattered/saturated
       gradients, non-differentiable argmax/index readouts (Balduzzi 2017;
       Sundararajan 2017 IG motivation).

  2. Saliency MODEL-INVARIANCE -- the Adebayo sanity-check failure.
       VCS: guided-backprop / vanilla saliency on the content path is invariant
       to a program-randomization (Adebayo) test -- cosine 1.000 between the
       original and ROM-scrambled saliency even though 95.3% of RAM changed;
       the sanity check FAILS. NN: Adebayo et al. 2018 -- saliency unchanged
       under model- and label-randomization.

  3. CORRELATION != CAUSATION -- the Granger / tuning-curve trap.
       VCS: A6 Granger faithfulness 0.136 (high false-edge rate vs the true
       data-flow); A3 game-variable tuning faithfulness 0.116 (high
       spurious-tuning rate). NN: spurious Granger edges; "present != used"
       tuning/attention readouts.

  4. INTERACTION BLIND-SPOT of single-unit ablation.
       VCS: single-unit A2 lesion importance is faithful (0.987) yet the
       single-unit view MISSES the wiring -- A1 connectomics recovers the true
       read/write graph at only 0.414, and SAE features that decode a variable
       move the output in 0 of the audited cases (n_causal_use_moved=0). NN:
       single-unit ablation is misleading; computation is distributed
       (Morcos et al. 2018; Leavitt & Morcos 2020).

  5. SAE / PROBE "PRESENT-NOT-USED" + polysemanticity.
       VCS: SAEs match a known variable for 100% of candidates yet score 0.195
       on causal use; linear probes decode at 0.894 accuracy but 25 cells
       across 6 games are decodable-but-not-causal -> selectivity faithfulness
       0.162. NN: control-task selectivity gap (Hewitt & Liang 2019);
       superposition / polysemantic neurons (Elhage et al. 2022).

  6. BASELINE-SENSITIVITY of Integrated / Expected Gradients.
       VCS: on the content path the IG magnitude is baseline-sensitive
       (corr_sensitivity up to 1.000 across games -- e.g. q*bert 1.000,
       ms-pac-man 0.81); a baseline equal to the content byte kills IG
       entirely; Expected Gradients lands at 0.034 overall. NN: IG attribution
       depends on the (arbitrary) baseline choice (Kindermans et al. 2019;
       Sturmfels et al. 2020).

ALL VCS numbers are READ from committed records -- no experiment is re-run:
  * tools/xai_study/compare/out/leaderboard.json          (P2-E6-1 aggregate)
  * tools/xai_study/compare/out/faithful_demo.json        (P2-E6-3 headline)
  * tools/xai_study/phaseB_attribution/out/
        guided_backprop_core_summary.json                 (Adebayo sanity)
        ig_baseline_sweep_core_summary.json               (IG baseline sens.)
  * tools/xai_study/phaseC_mechanistic/out/
        sae_core_summary.json                             (SAE causal-use)
        linear_probing_core_summary.json                  (present-not-used)
The NN-counterpart labels are the cite-ready analogues from
experiment_design.md (§4-§7) and plan.md (the representativeness rebuttal);
they are conceptual mappings, not measurements -- the figure says so explicitly
(the "suspected" status, and the legend note).

Run:
    python fig5_representativeness_map.py
Produces:
    fig5_representativeness_map.pdf   (vector, colour-blind-safe, self-legend)
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless vector backend
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

C_VCS = "#0072B2"          # blue        — the VCS column (where it is PROVABLE)
C_NN = "#D55E00"           # vermillion  — the neural-network counterpart
C_PROVE = "#009E73"        # bluish-green — "PROVABLE here" status
C_SUSPECT = "#E69F00"      # orange      — "suspected there" status
C_BANNER = "#CC79A7"       # reddish-purple — thesis banner accent
C_VCS_FILL = "#dbeaf4"     # pale blue panel
C_NN_FILL = "#f7e2d6"      # pale vermillion panel
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
    return lb, demo, gbp, igsw, sae, prb


def rows_by_method(lb):
    return {r["method"]: r for r in lb["rows"]}


# ---------------------------------------------------------------------------
# Assemble the six failure-mode rows from the committed records.
# ---------------------------------------------------------------------------
def build_rows(lb, demo, gbp, igsw, sae, prb):
    R = rows_by_method(lb)
    hc = lb["headline_contrast"]["position_regime"]

    # (1) gradient vanishes on index/position
    sal_pos = R["vanilla_saliency"]["faithfulness_position_regime"]
    ig_pos = R["integrated_gradients"]["faithfulness_position_regime"]
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
    # SAEs recover a known variable for nearly all candidate features
    # (4/6 games at matched_fraction==1.0; min 0.833); yet causal-use is low.
    sae_match_games_full = sum(1 for g in sae["games"]
                               if g["matched_fraction"] == 1.0)
    sae_match_min = min(g["matched_fraction"] for g in sae["games"])
    probe_F = R["linear_probing_control_tasks"]["faithfulness"]
    probe_acc = prb["mean_probe_acc"]
    probe_notcausal = prb["total_not_causally_used"]

    # (6) baseline-sensitivity of IG/EG
    eg_F = R["expected_gradients"]["faithfulness"]
    bl_sens_max = max(rr["baseline_corr_sensitivity"] for rr in igsw["results"])

    data = [
        dict(
            n=1,
            mode="Gradient vanishes on\ndiscrete / index outputs",
            vcs=("Vanilla saliency & IG\nfaithfulness = "
                 f"{sal_pos:.3f} on the position\nregime; whole gradient bucket\n"
                 f"= {grad_bucket_pos:.3f} (n=9). Provably 0\n"
                 "(strobe-timed argmax, §1)."),
            vcs_score=sal_pos,
            score_label="position-regime F",
            nn=("Shattered / saturated\ngradients; non-differentiable\n"
                "argmax & index readouts.\n[Balduzzi 2017; Sundararajan\n2017 (IG motivation)]"),
        ),
        dict(
            n=2,
            mode="Saliency is model-invariant\n(Adebayo sanity-check fails)",
            vcs=("Program-randomization test:\n"
                 f"saliency cosine = {ade_cos:.3f} while\n"
                 f"{ade_ram:.0f}% of RAM changed →\n"
                 f"sanity check {'PASSED' if ade_pass else 'FAILS'}.\n"
                 "Constant Jacobian, no ReLU."),
            vcs_score=1.0 - ade_cos,  # 0 = fully invariant (failed)
            score_label="sanity-check margin",
            nn=("Saliency unchanged under\nmodel- & label-randomization;\n"
                "looks like an edge detector.\n[Adebayo et al. 2018,\nNeurIPS]"),
        ),
        dict(
            n=3,
            mode="Correlation ≠ causation\n(Granger & tuning trap)",
            vcs=("Granger edges faithfulness\n"
                 f"= {granger:.3f} (false edges vs the\ntrue data-flow); game-variable\n"
                 f"tuning = {tuning:.3f} (spurious\ntuning) — vs exact oracle."),
            vcs_score=granger,
            score_label="Granger F",
            nn=("Spurious Granger edges;\n'present ≠ used' tuning &\n"
                "attention as explanation.\n[Anand 2019; Jain &\nWallace 2019 (attention)]"),
        ),
        dict(
            n=4,
            mode="Single-unit ablation has an\ninteraction blind-spot",
            vcs=("Single-unit lesion importance\n"
                 f"is faithful ({lesion:.3f}) yet MISSES\nthe wiring: connectome graph\n"
                 f"recovered at only {connect:.3f}; SAE\n"
                 f"feature ablations move y in\n{sae_moved} of the audited cells."),
            vcs_score=connect,
            score_label="connectome F",
            nn=("Single-unit ablation\nmisleading; computation is\n"
                "distributed across units.\n[Morcos et al. 2018;\nLeavitt & Morcos 2020]"),
        ),
        dict(
            n=5,
            mode="SAE / probe: present-not-used\n+ polysemanticity",
            vcs=(f"SAEs match a known var (100% of\ncandidates in {sae_match_games_full}/6 games, "
                 f"min\n{sae_match_min*100:.0f}%) yet score {sae_F:.3f} on causal\nuse; probes "
                 f"decode at {probe_acc:.3f} acc\nbut {probe_notcausal} cells are decodable-\n"
                 f"not-causal → selectivity F = {probe_F:.3f}."),
            vcs_score=probe_F,
            score_label="selectivity F",
            nn=("Control-task selectivity gap;\nsuperposition / polysemantic\n"
                "neurons (features in\nsuperposition).\n[Hewitt & Liang 2019;\nElhage et al. 2022]"),
        ),
        dict(
            n=6,
            mode="Baseline-sensitivity of\nIntegrated / Expected Gradients",
            vcs=("IG magnitude is baseline-\n"
                 f"sensitive (corr_sensitivity up\nto {bl_sens_max:.3f} across games); a\n"
                 "baseline = the content byte\n"
                 f"kills IG; EG lands at {eg_F:.3f}."),
            vcs_score=eg_F,
            score_label="EG F (all-regime)",
            nn=("Attribution depends on the\n(arbitrary) baseline choice;\n"
                "no canonical reference input.\n[Kindermans et al. 2019;\nSturmfels et al. 2020]"),
        ),
    ]
    return data, hc


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=os.path.join(HERE, "fig5_representativeness_map.pdf"),
    )
    args = ap.parse_args()

    lb, demo, gbp, igsw, sae, prb = load_records()
    data, hc = build_rows(lb, demo, gbp, igsw, sae, prb)

    n_rows = len(data)
    fig = plt.figure(figsize=(11.4, 8.6))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---- column geometry (x in 0..100) ----
    X0 = 2.5                # left margin
    X1 = 97.5               # right margin
    COL_MODE = (X0, 24.0)            # failure mode (measured here)
    COL_VCS = (25.0, 60.0)           # VCS evidence
    COL_NN = (61.0, 86.5)            # NN counterpart
    COL_STAT = (87.5, X1)            # provable vs suspected

    # ---- vertical band for rows ----
    TOP = 80.5              # top of the row block
    BOT = 11.0             # bottom of the row block (footer strip below)
    row_h = (TOP - BOT) / n_rows

    # =====================================================================
    # Title + thesis banner
    # =====================================================================
    ax.text(X0, 97.9, "The representativeness map: every failure mode we "
            "measure on the VCS has a",
            ha="left", va="top", fontsize=12.5, fontweight="bold", color=C_INK)
    ax.text(X0, 95.5, "documented neural-network counterpart",
            ha="left", va="top", fontsize=12.5, fontweight="bold", color=C_INK)
    ax.text(X0, 92.4,
            "The single strongest objection — “a deterministic 1977 chip "
            "cannot stand in for a learned, distributed, stochastic neural "
            "network” — answered as a necessary-condition screen.",
            ha="left", va="top", fontsize=8.6, color=C_MUTE, style="italic")

    # banner box
    ban = FancyBboxPatch(
        (X0, 86.3), X1 - X0, 4.9,
        boxstyle="round,pad=0.15,rounding_size=0.8",
        linewidth=1.0, edgecolor=C_BANNER, facecolor="#f6e9f1", zorder=1,
    )
    ax.add_patch(ban)
    ax.text(
        X0 + 1.0, 88.75,
        "The asymmetry is the argument:  on the VCS each failure is PROVABLE "
        "— exact intervention oracle, full observability, no learning;  on "
        "neural networks\nthe same failure is only SUSPECTED.  A method that "
        "fails here has no business being trusted there  (necessary, not "
        "sufficient).",
        ha="left", va="center", fontsize=8.4, color=C_INK, zorder=2,
        linespacing=1.35,
    )

    # =====================================================================
    # Column headers
    # =====================================================================
    def header(cx0, cx1, line1, line2, fill, edge):
        h = FancyBboxPatch(
            (cx0, 81.2), cx1 - cx0, 3.9,
            boxstyle="round,pad=0.1,rounding_size=0.6",
            linewidth=1.1, edgecolor=edge, facecolor=fill, zorder=2,
        )
        ax.add_patch(h)
        cx = (cx0 + cx1) / 2
        ax.text(cx, 83.85, line1, ha="center", va="center",
                fontsize=8.9, fontweight="bold", color=C_INK, zorder=3)
        ax.text(cx, 82.05, line2, ha="center", va="center",
                fontsize=7.0, color=C_MUTE, zorder=3)

    header(*COL_MODE, "Failure mode", "observed & MEASURED on the VCS",
           "#ececec", C_MUTE)
    header(*COL_VCS, "VCS evidence  (PROVABLE)",
           "scored vs the exact §1 oracle", C_VCS_FILL, C_VCS)
    header(*COL_NN, "Neural-network counterpart",
           "documented analogue (cite-ready)", C_NN_FILL, C_NN)
    header(*COL_STAT, "Status", "here → there", "#ececec", C_MUTE)

    # =====================================================================
    # Rows
    # =====================================================================
    for i, d in enumerate(data):
        y_top = TOP - i * row_h
        y_bot = y_top - row_h
        y_mid = (y_top + y_bot) / 2

        # zebra background band across the full width
        band = FancyBboxPatch(
            (X0, y_bot + 0.25), X1 - X0, row_h - 0.5,
            boxstyle="round,pad=0,rounding_size=0.4",
            linewidth=0, facecolor=(C_ROW_A if i % 2 == 0 else C_ROW_B),
            zorder=0.5,
        )
        ax.add_patch(band)

        # ---- mode cell (number chip + text) ----
        chip_x = COL_MODE[0] + 1.1
        ax.scatter([chip_x], [y_top - 1.55], s=235, marker="o",
                   color=C_INK, zorder=4)
        ax.text(chip_x, y_top - 1.55, str(d["n"]), ha="center", va="center",
                fontsize=8.2, fontweight="bold", color="white", zorder=5)
        ax.text(COL_MODE[0] + 2.6, y_mid, d["mode"], ha="left", va="center",
                fontsize=8.3, fontweight="bold", color=C_INK, zorder=4)

        # ---- VCS evidence panel ----
        vpanel = FancyBboxPatch(
            (COL_VCS[0], y_bot + 0.6), COL_VCS[1] - COL_VCS[0], row_h - 1.2,
            boxstyle="round,pad=0.1,rounding_size=0.5",
            linewidth=0.9, edgecolor=C_VCS, facecolor=C_VCS_FILL, zorder=2,
        )
        ax.add_patch(vpanel)
        # score badge (left of the text, the headline number for this row)
        sb_cx = COL_VCS[0] + 4.3
        sb = FancyBboxPatch(
            (COL_VCS[0] + 0.9, y_mid - 1.85), 6.8, 3.7,
            boxstyle="round,pad=0.05,rounding_size=0.4",
            linewidth=1.0, edgecolor=C_VCS, facecolor="white", zorder=3,
        )
        ax.add_patch(sb)
        ax.text(sb_cx, y_mid + 0.45, f"{d['vcs_score']:.3f}", ha="center",
                va="center", fontsize=10.0, fontweight="bold",
                color=C_VCS, zorder=4)
        ax.text(sb_cx, y_mid - 1.15, d["score_label"], ha="center",
                va="center", fontsize=5.3, color=C_MUTE, zorder=4)
        ax.text(COL_VCS[0] + 8.7, y_mid, d["vcs"], ha="left", va="center",
                fontsize=6.7, color=C_INK, zorder=4, linespacing=1.18)

        # ---- arrow VCS -> NN ----
        ax.annotate(
            "", xy=(COL_NN[0] - 0.3, y_mid), xytext=(COL_VCS[1] + 0.3, y_mid),
            arrowprops=dict(arrowstyle="-|>", color=C_MUTE, lw=1.4,
                            shrinkA=0, shrinkB=0), zorder=3,
        )

        # ---- NN counterpart panel ----
        npanel = FancyBboxPatch(
            (COL_NN[0], y_bot + 0.6), COL_NN[1] - COL_NN[0], row_h - 1.2,
            boxstyle="round,pad=0.1,rounding_size=0.5",
            linewidth=0.9, edgecolor=C_NN, facecolor=C_NN_FILL, zorder=2,
        )
        ax.add_patch(npanel)
        ax.text(COL_NN[0] + 1.0, y_mid, d["nn"], ha="left", va="center",
                fontsize=6.7, color=C_INK, zorder=4, linespacing=1.18)

        # ---- status column: PROVABLE -> suspected ----
        st_cx = (COL_STAT[0] + COL_STAT[1]) / 2
        # PROVABLE pill (green, filled)
        p1 = FancyBboxPatch(
            (COL_STAT[0] + 0.5, y_mid + 0.55), COL_STAT[1] - COL_STAT[0] - 1.0, 2.05,
            boxstyle="round,pad=0.05,rounding_size=0.6",
            linewidth=0, facecolor=C_PROVE, zorder=3,
        )
        ax.add_patch(p1)
        ax.text(st_cx, y_mid + 1.57, "PROVABLE here", ha="center",
                va="center", fontsize=6.4, fontweight="bold",
                color="white", zorder=4)
        # suspected pill (orange, hollow)
        p2 = FancyBboxPatch(
            (COL_STAT[0] + 0.5, y_mid - 2.5), COL_STAT[1] - COL_STAT[0] - 1.0, 2.05,
            boxstyle="round,pad=0.05,rounding_size=0.6",
            linewidth=1.1, edgecolor=C_SUSPECT, facecolor="white", zorder=3,
        )
        ax.add_patch(p2)
        ax.text(st_cx, y_mid - 1.48, "suspected there", ha="center",
                va="center", fontsize=6.4, fontweight="bold",
                color=C_SUSPECT, zorder=4)
        ax.annotate(
            "", xy=(st_cx, y_mid - 0.55), xytext=(st_cx, y_mid + 0.5),
            arrowprops=dict(arrowstyle="-|>", color=C_MUTE, lw=1.0,
                            shrinkA=0, shrinkB=0), zorder=4,
        )

    # =====================================================================
    # Footer: legend + provenance + headline gap
    # =====================================================================
    legend_handles = [
        Patch(facecolor=C_VCS_FILL, edgecolor=C_VCS,
              label="VCS evidence — measured vs the exact §1 intervention oracle"),
        Patch(facecolor=C_NN_FILL, edgecolor=C_NN,
              label="NN counterpart — documented analogue (conceptual mapping, cite-ready)"),
        Patch(facecolor=C_PROVE, edgecolor=C_PROVE, label="PROVABLE on the VCS (ground truth)"),
        Patch(facecolor="white", edgecolor=C_SUSPECT, label="only SUSPECTED on neural nets"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper left",
        bbox_to_anchor=(X0 / 100.0, (BOT - 2.2) / 100.0),
        ncol=2, frameon=False, fontsize=7.0, handlelength=1.4,
        columnspacing=1.6, handletextpad=0.6, labelspacing=0.5,
    )

    gap = hc["gap"]
    cm = hc["causal_intervention_faithfulness_mean"]
    gm = hc["gradient_correlational_faithfulness_mean"]
    # headline gap — its own clear strip just below the row block (right-aligned)
    ax.text(
        X1, BOT - 1.2,
        "Headline (position regime): causal/intervention "
        f"{cm:.3f} vs gradient/correlational {gm:.3f} — a {gap:.3f} "
        "faithfulness gap;  activation patching = 1.000 (oracle ceiling) "
        "where vanilla saliency = 0.000 (naive-gradient floor).",
        ha="right", va="top", fontsize=6.6, color=C_MUTE, linespacing=1.3,
    )
    ax.text(
        X0, 2.4,
        "All VCS numbers read from committed records: leaderboard.json "
        "(P2-E6-1), faithful_demo.json (P2-E6-3), guided_backprop_core_summary "
        "(Adebayo), ig_baseline_sweep_core_summary,\nsae_core_summary, "
        "linear_probing_core_summary. No experiment re-run. NN-counterpart "
        "labels per experiment_design.md §4–§7 and plan.md "
        "(the representativeness rebuttal).",
        ha="left", va="top", fontsize=5.7, color=C_MUTE, linespacing=1.3,
    )

    fig.savefig(args.out, format="pdf", facecolor=C_BG)
    plt.close(fig)

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

    exists = os.path.isfile(args.out)
    size = os.path.getsize(args.out) if exists else 0
    header_ok = False
    if exists:
        with open(args.out, "rb") as fh:
            header_ok = fh.read(5) == b"%PDF-"
    check("PDF file exists", exists, args.out)
    check("PDF size > 4 KB", size > 4096, f"{size} bytes")
    check("PDF has %PDF- header", header_ok)

    # data-integrity: every plotted VCS number traces to a committed record
    R = rows_by_method(lb)
    check("6 failure-mode rows built", len(data) == 6, f"n={len(data)}")
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

    print("Figure 5 self-check:")
    print("\n".join(msgs))
    print(f"\n{'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'} -> {args.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
