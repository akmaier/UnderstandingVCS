#!/usr/bin/env python3
"""Generate one explanatory figure per Paper-2 method from its committed record.

Phase B methods get a Paper-1-style image-domain view: the method's attribution
and the oracle's true causal effect are painted onto the actual game frame, using
each RAM cell's screen footprint (docs/cell_footprints.jl). Phase A/C get
method-specific, fully-labelled figures (matrices, scatters, spectra, bars) with
real RAM-cell labels. Run from the repo root:

    python3 docs/gen_method_figures.py
"""
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import manifest as M  # noqa: E402
from breakout_video import decode_palette, load_ntsc_palette  # noqa: E402

XS = os.path.join(REPO, "tools", "xai_study")
IMG = os.path.join(HERE, "assets", "methods")
os.makedirs(IMG, exist_ok=True)
RECDIR = {"A": "phaseA_kording", "B": "phaseB_attribution",
          "NA": "phaseB_attribution", "C": "phaseC_mechanistic"}

BG, FG, DIM, ACC, ACC2, BAD = "#1b212b", "#e6edf3", "#9aa7b4", "#5ec8f8", "#7ee787", "#ff7b72"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": DIM,
    "xtick.color": DIM, "ytick.color": DIM, "axes.titlecolor": FG,
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
})
H, W = 210, 160

# --- pong cell footprints (image-domain overlay data) ----------------------
_FP = {}
def footprints():
    if _FP:
        return _FP
    try:
        cells = [int(x) for x in open(os.path.join(IMG, "fp_pong_cells.txt"))]
        base = np.frombuffer(open(os.path.join(IMG, "fp_pong_base.raw"), "rb").read(),
                             np.uint8).reshape(H, W)
        fp = np.frombuffer(open(os.path.join(IMG, "fp_pong.raw"), "rb").read(),
                           np.float32).reshape(len(cells), H, W)
        rgb = decode_palette(base, load_ntsc_palette()).astype(np.float32)
        _FP.update(cells=cells, fp=fp, rgb=rgb)
    except Exception as e:
        print("  (no footprints:", e, ")")
    return _FP


def hexlab(i):
    return "$%02X" % int(i)


def load(meth):
    base = os.path.join(XS, RECDIR[meth["phase"]], "out", meth["record"])
    rec = json.load(open(base + ".json"))
    npz = dict(np.load(base + ".npz", allow_pickle=True)) if os.path.exists(base + ".npz") else {}
    return rec, npz


def title_for(rec, meth):
    v = rec.get("value")
    vs = ("%.3f" % v) if isinstance(v, (int, float)) else str(v)
    return "%s        %s = %s" % (meth["title"], rec.get("metric_name", "score"), vs)


def scene_ax(ax, game="pong"):
    p = os.path.join(IMG, "scene_%s.png" % game)
    if os.path.exists(p):
        ax.imshow(plt.imread(p))
    ax.set_title("game frame (%s)" % game); ax.set_xticks([]); ax.set_yticks([])


def overlay_ax(ax, cells, weights, title):
    """Paint per-cell weights onto the pong frame via screen footprints.
    `cells` are the RAM indices `weights` is indexed by; we match them to the
    footprint cells by VALUE (not position)."""
    F = footprints()
    if not F:
        ax.axis("off"); ax.text(0.5, 0.5, "(no footprints)", ha="center", color=DIM); return
    fcells, fp, rgb = F["cells"], F["fp"], F["rgb"]
    w = {int(c): float(weights[i]) for i, c in enumerate(cells) if i < len(weights)}
    heat = np.zeros((H, W))
    for fi, fc in enumerate(fcells):
        heat += w.get(int(fc), 0.0) * fp[fi]
    if heat.max() > 0:
        heat /= heat.max()
    ax.imshow((rgb * 0.45).astype(np.uint8))
    rgba = cm.get_cmap("inferno")(heat); rgba[..., 3] = np.clip(heat, 0, 1)
    ax.imshow(rgba)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])


def parse_cells(cause_names):
    """Unique RAM cell indices appearing in the cause names, in order."""
    out = []
    for nm in cause_names or []:
        m = re.search(r"ram\[(\d+)\]", str(nm))
        if m and int(m.group(1)) not in out:
            out.append(int(m.group(1)))
    return out


def cells_of(npz):
    for k in ("candidate_ram_indices", "candidate_cells", "cell_index"):
        if k in npz:
            return [int(x) for x in np.asarray(npz[k]).ravel()]
    return list(range(8))


def per_cell_from_causes(vals, cause_names, cells):
    """Aggregate a per-cause vector onto the candidate cells by |value|."""
    w = np.zeros(len(cells))
    for i, nm in enumerate(cause_names or []):
        m = re.search(r"ram\[(\d+)\]", str(nm))
        if m and int(m.group(1)) in cells and i < len(vals):
            w[cells.index(int(m.group(1)))] += abs(float(vals[i]))
    return w


def paired_cell_bars(ax, oracle, method, cells, tl="oracle |Δy| (truth)", ml="method"):
    o = np.abs(np.asarray(oracle, float)); m = np.abs(np.asarray(method, float))
    n = min(len(o), len(m), len(cells)); o, m, cc = o[:n], m[:n], cells[:n]
    o = o / (o.max() or 1); m = m / (m.max() or 1)
    x = np.arange(n); wbar = 0.4
    ax.bar(x - wbar / 2, o, wbar, color=ACC2, label=tl)
    ax.bar(x + wbar / 2, m, wbar, color=ACC, label=ml)
    ax.set_xticks(x); ax.set_xticklabels(["RAM %s" % hexlab(c) for c in cc], rotation=45, fontsize=6, ha="right")
    ax.set_xlabel("candidate RAM cell"); ax.set_ylabel("normalised importance")
    ax.legend(loc="upper right")


def curves(ax, npz):
    dk = next((k for k in npz if "deletion" in k and "curve" in k), None)
    ik = next((k for k in npz if "insertion" in k and "curve" in k), None)
    if not dk and not ik:
        return False
    if dk:
        ax.plot(np.linspace(0, 1, len(npz[dk])), npz[dk], color=BAD, label="deletion")
    if ik:
        ax.plot(np.linspace(0, 1, len(npz[ik])), npz[ik], color=ACC2, label="insertion")
    ax.set_xlabel("fraction of ranked causes perturbed"); ax.set_ylabel("model output  y")
    ax.set_title("faithfulness curves"); ax.legend(loc="best")
    return True


def matrix_pair(axs, true, rec, cells, titles=("true graph", "recovered graph"),
                xlab="effect cell", ylab="cause cell"):
    labs = ["%s" % hexlab(c) for c in cells]
    for ax, mat, t in zip(axs, (true, rec), titles):
        im = ax.imshow(np.asarray(mat, float), cmap="magma", aspect="equal", vmin=0)
        ax.set_title(t)
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=5, rotation=90)
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=5)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)


def scatter(ax, x, y, xl, yl, t="recovered vs exact (diagonal = perfect)"):
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    ax.scatter(x, y, s=16, color=ACC, alpha=0.8)
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], "--", color=DIM, lw=0.8)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(t)


# --- per-method attribution-array aliases (Phase B) ------------------------
B_ALIAS = {"saliency": "saliency", "gradxinput": "gradxinput", "guided_backprop": "gbp",
           "smoothgrad": "sg", "ig_baseline_sweep": "headline", "expected_gradients": "headline",
           "occlusion": "occlusion", "perturbation": "extremal", "rise": "rise",
           "lime": "lime", "kernelshap": "shap", "counterfactual": "cf"}


# ===========================================================================
def render_B(meth, rec, npz):
    cause_names = (rec.get("extra") or {}).get("cause_names")
    cells = parse_cells(cause_names) or cells_of(npz)
    oracle_v = npz.get("oracle_abs_delta")
    ak = B_ALIAS.get(meth["key"], "") + "_attr_per_cause"
    if ak not in npz:
        ak = next((k for k in npz if k.endswith("attr_per_cause")
                   and not k.startswith(("oracle", "vanilla", "saliency"))), None) \
            or next((k for k in npz if k.endswith("attr_per_cause")), None)
    method_v = npz.get(ak)
    o_cell = per_cell_from_causes(oracle_v, cause_names, cells) if oracle_v is not None else np.zeros(len(cells))
    m_cell = per_cell_from_causes(method_v, cause_names, cells) if method_v is not None else np.zeros(len(cells))

    fig = plt.figure(figsize=(11, 6.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0])
    scene_ax(fig.add_subplot(gs[0, 0]))
    overlay_ax(fig.add_subplot(gs[0, 1]), cells, o_cell, "ORACLE: true causal region")
    overlay_ax(fig.add_subplot(gs[0, 2]), cells, m_cell, "%s: attributed region" % meth["title"][:22])
    paired_cell_bars(fig.add_subplot(gs[1, 0:2]), o_cell, m_cell, cells, ml=meth["title"][:18])
    axc = fig.add_subplot(gs[1, 2])
    if not curves(axc, npz):
        axc.axis("off"); axc.text(0.5, 0.5, "(no del/ins curves)", ha="center", color=DIM)
    fig.suptitle(title_for(rec, meth), fontsize=10)
    return fig


def render_NA(meth, rec, npz):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2), gridspec_kw={"width_ratios": [1, 1.5]})
    scene_ax(ax[0])
    applies = npz.get("applies_per_method")
    ax[1].axis("off")
    txt = ("Grad-CAM / Grad-CAM++  — needs convolutional feature maps\n"
           "Attention rollout        — needs attention layers\n"
           "VIPER                    — needs a learned tree policy\n\n"
           "The VCS has none of these. Recorded as not-applicable (%s of %s),\n"
           "rather than forced into a misleading number."
           % (int(rec.get("value", 0)), len(applies) if applies is not None else 6))
    ax[1].text(0.0, 0.5, txt, va="center", ha="left", color=FG, fontsize=9, family="monospace")
    ax[1].set_title("why these methods do not apply")
    fig.suptitle(title_for(rec, meth), fontsize=10)
    return fig


def render_A(meth, rec, npz):
    k = meth["key"]; cells = cells_of(npz)
    if k == "A1_connectomics":
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
        scene_ax(ax[0]); matrix_pair(ax[1:], npz["true_graph"], npz["recovered_graph"], cells,
                                     ("true data-flow graph", "recovered graph"))
    elif k == "A2_lesions":
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.4]})
        scene_ax(ax[0]); overlay_ax(ax[1], cells_of(npz), np.abs(npz["lesion_importance"]), "lesion importance on the frame")
        paired_cell_bars(ax[2], npz["oracle_role"], npz["lesion_importance"], cells,
                         tl="true causal role", ml="lesion importance")
        ax[2].set_title("recovered importance vs true role")
    elif k == "A3_tuning":
        fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw={"width_ratios": [1, 1.5]})
        scene_ax(ax[0])
        oi = np.abs(npz["oracle_importance"]); oi = oi / (oi.max() or 1)
        tun = np.abs(npz["gvar_selectivity"]); tun = tun / (tun.max() or 1)
        spur = np.asarray(npz["gvar_spurious"]).astype(bool)
        ax[1].scatter(oi[~spur], tun[~spur], s=30, color=ACC2, label="genuine")
        ax[1].scatter(oi[spur], tun[spur], s=34, color=BAD, label="spurious (tuned, not causal)")
        for i, c in enumerate(cells):
            ax[1].annotate("RAM %s" % hexlab(c), (oi[i], tun[i]), fontsize=5, color=DIM,
                           xytext=(2, 2), textcoords="offset points")
        ax[1].set_xlabel("true causal importance (oracle)")
        ax[1].set_ylabel("tuning strength to a game variable")
        ax[1].set_title("tuning ≠ causation"); ax[1].legend(loc="upper center")
    elif k == "A4_correlations":
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
        scene_ax(ax[0]); matrix_pair(ax[1:], npz["oracle_coupling"], npz["corr_matrix"], cells,
                                     ("true coupling (oracle)", "measured correlation"),
                                     xlab="RAM cell", ylab="RAM cell")
    elif k == "A5_lfp":
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1, 1.7]})
        scene_ax(ax[0])
        f = np.asarray(npz["global_freqs"]).ravel(); p = np.asarray(npz["global_power"]).ravel()
        ax[1].semilogy(f, p + 1e-9, color=ACC)
        ax[1].set_xlabel("frequency  (cycles / frame)"); ax[1].set_ylabel("pooled-activity power")
        ax[1].set_title("LFP spectrum — the peaks are the known clocks")
    elif k == "A6_granger":
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
        scene_ax(ax[0]); matrix_pair(ax[1:], npz["true_dataflow_adj"], npz["cell_inferred_adj"], cells,
                                     ("true data-flow", "Granger-inferred edges"))
    elif k == "A7_dimred":
        fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw={"width_ratios": [1, 1.5]})
        scene_ax(ax[0])
        paired_cell_bars(ax[1], npz["oracle_importance"], npz["nmf_recovery"], cells,
                         tl="true importance", ml="NMF recovery")
        ax[1].set_title("NMF component ↔ known-variable match")
    else:  # A8_wholestate
        fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4), gridspec_kw={"width_ratios": [1, 1.7]})
        scene_ax(ax[0])
        var = np.asarray(npz["ram_cell_var"]).ravel()
        ax[1].bar(range(len(var)), var / (var.max() or 1), color=DIM, width=1.0)
        for c in [int(x) for x in np.asarray(npz.get("oracle_causal_mask", [])).ravel() if x]:
            pass
        for c in cells:
            ax[1].axvline(c, color=ACC2, lw=0.8, alpha=0.7)
        ax[1].set_xlabel("RAM cell index (0–127)"); ax[1].set_ylabel("activity (variance, norm.)")
        ax[1].set_title("whole-state dump — green = the few truly causal cells")
    fig.suptitle(title_for(rec, meth), fontsize=10)
    return fig


def render_C(meth, rec, npz):
    k = meth["key"]; cells = cells_of(npz)
    if k in ("activation_patching",):
        fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.5), gridspec_kw={"width_ratios": [1, 1.4]})
        scene_ax(ax[0]); scatter(ax[1], npz["exact"], npz["recovered"],
                                 "exact patch effect  Δy (oracle)", "recovered patch effect")
    elif k == "attribution_patching":
        fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.5), gridspec_kw={"width_ratios": [1, 1.4]})
        scene_ax(ax[0]); scatter(ax[1], npz["exact"], npz["approx"],
                                 "exact patch effect (oracle)", "gradient-approx effect",
                                 "approx vs exact (diagonal = perfect)")
    elif k == "das":
        fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.5), gridspec_kw={"width_ratios": [1, 1.4]})
        scene_ax(ax[0]); scatter(ax[1], npz["exact_effect"], npz["interchange_effect"],
                                 "exact interchange effect (oracle)", "DAS interchange effect")
    elif k == "path_patching":
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
        scene_ax(ax[0]); matrix_pair(ax[1:], npz["true_graph"], npz["recovered_path_graph"], cells,
                                     ("true routine (data-flow)", "recovered path circuit"))
    elif k == "acdc":
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
        scene_ax(ax[0]); matrix_pair(ax[1:], npz["true_graph"], npz["discovered_at_best_tau"], cells,
                                     ("true data-flow", "ACDC circuit (best τ)"))
    elif k == "sae":
        fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw={"width_ratios": [1, 1.5]})
        scene_ax(ax[0]); paired_cell_bars(ax[1], npz["oracle_importance"], npz["feature_best_corr"],
                                          cells, tl="true variable importance", ml="best feature match")
        ax[1].set_title("SAE feature ↔ known variable")
    elif k == "dictionaries":
        fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw={"width_ratios": [1, 1.5]})
        scene_ax(ax[0]); paired_cell_bars(ax[1], npz["nmf_causal_effect"], npz["pca_causal_effect"],
                                          cells, tl="NMF component effect", ml="PCA component effect")
        ax[1].set_title("dictionary component causal effect")
    elif k == "causal_scrubbing":
        fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4))
        scene_ax(ax[0])
        for a, key, t in zip(ax[1:], ("true_preserve_matrix", "wrong_preserve_matrix"),
                             ("true circuit → preserved", "wrong circuit → broken")):
            im = a.imshow(np.asarray(npz[key], float), cmap="magma", aspect="auto", vmin=0, vmax=1)
            a.set_title(t); a.set_xlabel("scrubbed node"); a.set_ylabel("output cell")
    elif k == "linear_probing":
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), gridspec_kw={"width_ratios": [1, 1.6]})
        scene_ax(ax[0])
        met = np.asarray(npz["metrics"], float)  # (8,6); selectivity ~ col
        sel = met[:, -1] if met.shape[1] else np.zeros(len(cells))
        notused = np.asarray(npz.get("not_causally_used", np.zeros(len(cells)))).astype(bool)
        cols = [BAD if nu else ACC for nu in notused[:len(sel)]]
        ax[1].bar(range(len(sel)), sel, color=cols)
        ax[1].set_xticks(range(len(cells))); ax[1].set_xticklabels(["RAM %s" % hexlab(c) for c in cells],
                                                                   rotation=45, fontsize=6, ha="right")
        ax[1].set_xlabel("labelled RAM cell"); ax[1].set_ylabel("selectivity (probe − control)")
        ax[1].set_title("decodable, but red = present-not-used")
    else:  # logit_lens
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), gridspec_kw={"width_ratios": [1, 1.6]})
        scene_ax(ax[0])
        fid = np.asarray(npz["logit_fidelity_true_intermediate"], float)  # (8,31)
        ax[1].plot(fid.mean(0), color=ACC2, label="readout fidelity")
        ax[1].set_xlabel("decode step"); ax[1].set_ylabel("fidelity  R²  vs true value")
        ax[1].set_title("state is linearly readable at the right site"); ax[1].legend()
    fig.suptitle(title_for(rec, meth), fontsize=10)
    return fig


def render(meth):
    rec, npz = load(meth)
    fn = {"A": render_A, "B": render_B, "C": render_C, "NA": render_NA}[meth["phase"]]
    fig = fn(meth, rec, npz)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(IMG, meth["key"] + ".png")
    fig.savefig(out, dpi=125); plt.close(fig)
    return out


def main():
    ok = 0
    for meth in M.P2_METHODS:
        try:
            out = render(meth)
            print("  %-22s -> %.0f KB" % (meth["key"], os.path.getsize(out) / 1024)); ok += 1
        except Exception as e:
            import traceback
            print("  !! %-22s FAILED: %s" % (meth["key"], e))
            traceback.print_exc()
    print("generated %d/%d method figures" % (ok, len(M.P2_METHODS)))


if __name__ == "__main__":
    main()
