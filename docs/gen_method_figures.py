#!/usr/bin/env python3
"""Generate one explanatory figure per Paper-2 method from its committed record.

For each method in manifest.P2_METHODS, load its §R record (.json + .npz) under
tools/xai_study/<phase>/out/ and render a figure that pairs a rendered game
screenshot with the method's result vs the ground-truth oracle. Saves
docs/assets/methods/<key>.png.

Screens are produced once by docs/render_scenes.jl (committed as
docs/assets/methods/scene_<game>.png). Run from the repo root:
    python3 docs/gen_method_figures.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import manifest as M  # noqa: E402

XS = os.path.join(REPO, "tools", "xai_study")
IMG = os.path.join(HERE, "assets", "methods")
os.makedirs(IMG, exist_ok=True)
RECDIR = {"A": "phaseA_kording", "B": "phaseB_attribution",
          "NA": "phaseB_attribution", "C": "phaseC_mechanistic"}

# dark style to match the site
BG, FG, DIM, ACC, ACC2, BAD = "#1b212b", "#e6edf3", "#9aa7b4", "#5ec8f8", "#7ee787", "#ff7b72"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": DIM,
    "xtick.color": DIM, "ytick.color": DIM, "axes.titlecolor": FG,
    "font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7,
})


def load(meth):
    d = os.path.join(XS, RECDIR[meth["phase"]], "out")
    base = os.path.join(d, meth["record"])
    rec = json.load(open(base + ".json"))
    npz = dict(np.load(base + ".npz", allow_pickle=True)) if os.path.exists(base + ".npz") else {}
    return rec, npz


def scene_ax(ax, game):
    p = os.path.join(IMG, "scene_%s.png" % game)
    if os.path.exists(p):
        ax.imshow(plt.imread(p))
    ax.set_title("game frame (%s)" % game)
    ax.set_xticks([]); ax.set_yticks([])


def _find(npz, *subs, exclude=()):
    for k in npz:
        kl = k.lower()
        if all(s in kl for s in subs) and not any(e in kl for e in exclude):
            return k
    return None


def paired_bars(ax, truth, method, labels=None, tl="oracle |Δy|", ml="method"):
    """Top-N causes by truth, oracle vs method (each normalised to its own max)."""
    truth = np.abs(np.asarray(truth, float)).ravel()
    method = np.abs(np.asarray(method, float)).ravel()
    n = min(len(truth), len(method))
    truth, method = truth[:n], method[:n]
    order = np.argsort(truth)[::-1][:8][::-1]
    t = truth[order] / (truth.max() or 1)
    m = method[order] / (method.max() or 1)
    y = np.arange(len(order)); h = 0.38
    ax.barh(y + h / 2, t, h, color=ACC2, label=tl)
    ax.barh(y - h / 2, m, h, color=ACC, label=ml)
    if labels is not None and len(labels) >= max(order) + 1:
        ax.set_yticks(y); ax.set_yticklabels([str(labels[i])[:14] for i in order], fontsize=6)
    else:
        ax.set_yticks(y); ax.set_yticklabels(["c%d" % i for i in order], fontsize=6)
    ax.set_xlabel("normalised importance"); ax.legend(loc="lower right")


def curves(ax, npz):
    dele = npz.get("deletion_curve"); ins = npz.get("insertion_curve")
    if dele is None and ins is None:
        return False
    if dele is not None:
        ax.plot(np.linspace(0, 1, len(dele)), dele, color=BAD, label="deletion")
    if ins is not None:
        ax.plot(np.linspace(0, 1, len(ins)), ins, color=ACC2, label="insertion")
    ax.set_xlabel("fraction perturbed"); ax.set_ylabel("output"); ax.legend(loc="best")
    ax.set_title("faithfulness curves")
    return True


def heat(ax, mat, title):
    ax.imshow(np.asarray(mat, float), cmap="magma", aspect="auto")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])


def scatter_vs(ax, exact, recovered, xl="exact effect", yl="recovered effect"):
    e = np.asarray(exact, float).ravel(); r = np.asarray(recovered, float).ravel()
    n = min(len(e), len(r)); e, r = e[:n], r[:n]
    ax.scatter(e, r, s=14, color=ACC, alpha=0.8)
    lo, hi = float(min(e.min(), r.min())), float(max(e.max(), r.max()))
    ax.plot([lo, hi], [lo, hi], "--", color=DIM, lw=0.8)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title("recovered vs exact (diagonal = perfect)")


def title_for(rec, meth):
    v = rec.get("value")
    mn = rec.get("metric_name", "")
    vs = ("%.3f" % v) if isinstance(v, (int, float)) else str(v)
    return "%s\n%s = %s" % (meth["title"], mn, vs)


# --------------------------------------------------------------------------
def render(meth):
    rec, npz = load(meth)
    phase, key, game = meth["phase"], meth["key"], meth["game"]

    if phase == "NA":
        fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2),
                               gridspec_kw={"width_ratios": [1, 1.3]})
        scene_ax(ax[0], game)
        ax[1].axis("off")
        ax[1].text(0.5, 0.5, "Does not apply.\nGrad-CAM / attention rollout / VIPER\n"
                   "need conv-net or attention layers, or a\nlearned policy — the VCS has none.\n\n"
                   "Recorded honestly, not forced.", ha="center", va="center",
                   color=FG, fontsize=10)
        ax[1].set_title("N/A audit")
    elif phase == "B":
        fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2),
                               gridspec_kw={"width_ratios": [1, 1.25, 1.25]})
        scene_ax(ax[0], game)
        truth = npz.get("oracle_abs_delta")
        mkey = _find(npz, "attr_per_cause", exclude=("oracle",)) or _find(npz, "over_ram", exclude=("oracle",))
        if truth is not None and mkey is not None:
            paired_bars(ax[1], truth, npz[mkey], rec.get("extra", {}).get("cause_names"),
                        ml=meth["title"][:16])
            ax[1].set_title("attribution vs true causal map")
        else:
            ax[1].axis("off"); ax[1].text(0.5, 0.5, "(arrays unavailable)", ha="center", color=DIM)
        if not curves(ax[2], npz):
            ax[2].axis("off"); ax[2].text(0.5, 0.5, "(no del/ins curves)", ha="center", color=DIM)
        fig.suptitle(title_for(rec, meth), color=FG, fontsize=9)
    elif phase == "A":
        tg, rg = npz.get("true_graph"), npz.get("recovered_graph")
        if tg is not None and rg is not None:
            fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
            scene_ax(ax[0], game); heat(ax[1], tg, "true data-flow graph")
            heat(ax[2], rg, "recovered graph")
        else:
            fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.2),
                                   gridspec_kw={"width_ratios": [1, 1.4]})
            scene_ax(ax[0], game)
            ok = _find(npz, "oracle")
            mk = (_find(npz, "lesion", "importance") or _find(npz, "recovered")
                  or _find(npz, "selectivity") or _find(npz, "matched"))
            if ok and mk and np.ndim(npz[ok]) == 1:
                paired_bars(ax[1], npz[ok], npz[mk], tl="true role", ml=meth["title"][:16])
                ax[1].set_title("recovered vs true (per cell)")
            elif ok and np.ndim(npz[ok]) == 2:
                heat(ax[1], npz[ok], "structure")
            else:
                oned = [k for k in npz if np.asarray(npz[k]).ndim == 1 and len(npz[k]) <= 32]
                if oned:
                    ax[1].bar(range(len(npz[oned[0]])), npz[oned[0]], color=ACC)
                    ax[1].set_title(oned[0])
                else:
                    ax[1].axis("off"); ax[1].text(0.5, 0.5, "(descriptive)", ha="center", color=DIM)
        fig.suptitle(title_for(rec, meth), color=FG, fontsize=9)
    else:  # phase C
        fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.2),
                               gridspec_kw={"width_ratios": [1, 1.4]})
        scene_ax(ax[0], game)
        ex, rv = npz.get("exact"), npz.get("recovered")
        if ex is not None and rv is not None:
            scatter_vs(ax[1], ex, rv)
        else:
            ok = _find(npz, "oracle")
            mk = (_find(npz, "feature", "matched") or _find(npz, "feature", "corr")
                  or _find(npz, "recovery") or _find(npz, "selectivity") or _find(npz, "scores"))
            if ok and mk and np.ndim(npz[ok]) == 1 and np.ndim(npz[mk]) == 1:
                paired_bars(ax[1], npz[ok], npz[mk], tl="oracle importance", ml=meth["title"][:16])
                ax[1].set_title("recovered vs true (per variable)")
            else:
                cand = mk or ok
                if cand is not None:
                    a = np.asarray(npz[cand], float).ravel()
                    ax[1].bar(range(len(a)), a, color=ACC); ax[1].set_title(cand)
                else:
                    ax[1].axis("off"); ax[1].text(0.5, 0.5, "(scalar result)", ha="center", color=DIM)
        fig.suptitle(title_for(rec, meth), color=FG, fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(IMG, key + ".png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    ok = 0
    for meth in M.P2_METHODS:
        try:
            out = render(meth)
            print("  %-22s -> %s (%.0f KB)" % (meth["key"], os.path.basename(out),
                                               os.path.getsize(out) / 1024))
            ok += 1
        except Exception as e:
            print("  !! %-22s FAILED: %s" % (meth["key"], e))
    print("generated %d/%d method figures" % (ok, len(M.P2_METHODS)))


if __name__ == "__main__":
    main()
