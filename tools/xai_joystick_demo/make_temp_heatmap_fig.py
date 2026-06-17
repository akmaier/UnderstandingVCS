#!/usr/bin/env python3
"""Soft-select temperature heatmap (supplementary).

Shows that the soft_select (softmax) temperature T turns a HARD pick into a
DISTRIBUTION over candidate cannon columns.  Same setup as
tools/xai_joystick_demo/alpha_temp_demo.jl:

    CAND   = 30:8:110                       (11 candidate cannon columns)
    target = 70
    logits_k = -((CAND[k] - 70) / 6)^2
    w_k(T)   = softmax(logits / T)

We sweep a fine log-spaced grid of T from 3.0 (high, spread) down to 0.08
(low, one-hot) and render a heatmap: x = candidate column, y = T (log scale),
colour = softmax weight.  An annotation gives the effective number of columns
N_eff(T) = 1 / sum(w^2) at a few temperatures to quantify the spread.

Run:  python3 tools/xai_joystick_demo/make_temp_heatmap_fig.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIG = os.path.abspath(os.path.join(HERE, "..", "..", "jutari_paper", "paper",
                                   "figures", "fig_temp_heatmap.pdf"))
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

STEEL = "#35618f"
ORANGE = "#c8641a"

# ---- soft_select setup (identical to alpha_temp_demo.jl) -------------------
CAND = np.arange(30, 111, 8, dtype=float)      # 30,38,...,110  (K = 11)
TARGET = 70.0
LOGITS = -((CAND - TARGET) / 6.0) ** 2


def softmax(logits, T):
    z = (logits - logits.max()) / T
    e = np.exp(z)
    return e / e.sum()


def neff(w):
    """Effective number of columns: inverse participation ratio."""
    return 1.0 / np.sum(w ** 2)


def main():
    # Fine log-spaced T grid: high T (spread) at TOP, low T (one-hot) at BOTTOM.
    nT = 50
    T_hi, T_lo = 3.0, 0.08
    Tgrid = np.logspace(np.log10(T_hi), np.log10(T_lo), nT)   # descending

    W = np.vstack([softmax(LOGITS, T) for T in Tgrid])        # (nT, K)

    fig, ax = plt.subplots(figsize=(4.6, 3.4))

    # log-scaled y in data coords: use the log10(T) values as cell centres so
    # pcolormesh spaces rows correctly on a log axis.  Build edges in log space.
    logT = np.log10(Tgrid)
    # cell edges (nT+1) midway between successive log-T centres
    edges_y = np.empty(nT + 1)
    edges_y[1:-1] = 0.5 * (logT[:-1] + logT[1:])
    edges_y[0] = logT[0] + 0.5 * (logT[0] - logT[1])
    edges_y[-1] = logT[-1] - 0.5 * (logT[-2] - logT[-1])
    # x edges centred on candidate columns (spacing 8)
    dx = 8.0
    edges_x = np.concatenate([CAND - dx / 2, [CAND[-1] + dx / 2]])

    pcm = ax.pcolormesh(edges_x, edges_y, W, cmap="magma",
                        vmin=0.0, vmax=W.max(), shading="flat", rasterized=True)

    # y axis shows actual T values (we plotted log10 T as the coordinate).
    ytick_T = np.array([3.0, 2.0, 1.0, 0.5, 0.3, 0.2, 0.1])
    ax.set_yticks(np.log10(ytick_T))
    ax.set_yticklabels([f"{t:g}" for t in ytick_T])
    ax.set_ylabel("soft-select temperature $T$  (log scale)")
    ax.set_xlabel("candidate cannon column")
    ax.set_xticks(CAND[::2].astype(int))

    # Mark the target column.
    ax.axvline(TARGET, color="white", lw=0.8, ls="--", alpha=0.75)
    ax.text(TARGET + 1.5, edges_y[0], "target = 70", color="white",
            fontsize=6.5, va="top", ha="left", rotation=90)

    ax.set_title("Soft select: temperature $T$ spreads a hard pick\n"
                 "into a distribution over columns", fontsize=8.5)

    cb = fig.colorbar(pcm, ax=ax, pad=0.14, fraction=0.046)
    cb.set_label("softmax weight $w_k$", fontsize=8)
    cb.ax.tick_params(length=2)
    cb.outline.set_linewidth(0.5)

    # ---- N_eff annotation at a few temperatures ---------------------------
    # Right-hand twin axis labelled with effective #columns at marker rows.
    mark_T = np.array([2.0, 1.0, 0.5, 0.2, 0.1])
    mark_logT = np.log10(mark_T)
    mark_neff = np.array([neff(softmax(LOGITS, t)) for t in mark_T])

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(mark_logT)
    ax2.set_yticklabels([f"{n:.1f}" for n in mark_neff], fontsize=6.5,
                        color=STEEL)
    ax2.set_ylabel(r"effective \#columns  $1/\sum_k w_k^2$",
                   fontsize=7, color=STEEL)
    ax2.tick_params(axis="y", length=2, colors=STEEL)
    for s in ax2.spines.values():
        s.set_visible(False)

    fig.savefig(FIG, bbox_inches="tight", dpi=300)
    print("wrote", FIG)
    # console summary of N_eff at the annotated T values
    for t, n in zip(mark_T, mark_neff):
        print(f"  T={t:4.2f}  N_eff={n:.2f}  max w={softmax(LOGITS,t).max():.3f}")


if __name__ == "__main__":
    main()
