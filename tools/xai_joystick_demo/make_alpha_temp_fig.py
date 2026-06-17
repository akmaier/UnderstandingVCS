#!/usr/bin/env python3
"""Render the alpha/temperature gradient-map figure (supplementary) from the
maps written by alpha_temp_demo.jl.  python3 make_alpha_temp_fig.py"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
FIG = os.path.abspath(os.path.join(HERE, "..", "..", "jutari_paper", "paper",
                                   "figures", "fig_alpha_temp.pdf"))
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# Zoom window around the cannon (rows, cols).
CR = (slice(16, 60), slice(26, 122))


def load():
    maps = {}
    with open(os.path.join(OUT, "alpha_temp_manifest.txt")) as f:
        for line in f:
            name, r, c = line.split()
            a = np.fromfile(os.path.join(OUT, name + ".bin"), dtype="<f4")
            maps[name] = a.reshape(int(r), int(c))
    return maps


def panel(ax, scene, grad, title, vmax):
    sc = scene[CR[0], CR[1]]
    gd = grad[CR[0], CR[1]]
    peak = np.abs(gd).max()
    # faint cannon (grey) underneath, gradient (diverging) on top where nonzero.
    # A shared `vmax` per row makes magnitude differences visible: the gradient
    # is brightest at moderate sharpness and FADES as the relaxation saturates.
    ax.imshow(sc, cmap="gray", vmin=0, vmax=1.6, aspect="equal",
              interpolation="none")
    gm = np.ma.masked_where(np.abs(gd) < 0.06 * vmax, gd)
    ax.imshow(gm, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal",
              interpolation="none")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(f"peak $|\\partial|$ = {peak/vmax:.2f}", fontsize=6.5)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    m = load()
    fig, ax = plt.subplots(2, 3, figsize=(5.6, 3.6))

    av = max(np.abs(m[f"at_branch_grad{j+1}"]).max() for j in range(3)) + 1e-9
    for j, a in enumerate(("2", "6", "20")):
        panel(ax[0, j], m[f"at_branch_scene{j+1}"], m[f"at_branch_grad{j+1}"],
              rf"$\alpha = {a}$", av)
    tv = max(np.abs(m[f"at_select_grad{j+1}"]).max() for j in range(3)) + 1e-9
    for j, t in enumerate(("2.0", "0.5", "0.1")):
        panel(ax[1, j], m[f"at_select_scene{j+1}"], m[f"at_select_grad{j+1}"],
              rf"$T = {t}$", tv)

    ax[0, 0].set_ylabel("soft branch\n(sharpness $\\alpha$)", fontsize=8)
    ax[1, 0].set_ylabel("soft select\n(temperature $T$)", fontsize=8)
    fig.suptitle(r"$\partial$(screen)$/\partial$(control) vs. relaxation "
                 r"strength (vanishes in the hard limit $\alpha\to\infty$, "
                 r"$T\to0$)", fontsize=8.5, y=0.99)
    fig.tight_layout(pad=0.4, w_pad=0.4, h_pad=0.6, rect=[0, 0.08, 1, 0.96])

    # Signed colour key (per-row-max units): blue = darkens, red = brightens.
    sm = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(-1, 1), cmap="RdBu_r")
    cax = fig.add_axes([0.30, 0.045, 0.40, 0.030])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    cb.ax.set_xticklabels(["$-$ (darkens)", "0", "$+$ (brightens)"], fontsize=6.5)
    cb.set_label(r"$\partial$pixel$/\partial$control (per-row max)", fontsize=6.5)
    cb.ax.tick_params(length=2)
    cb.outline.set_linewidth(0.5)

    fig.savefig(FIG, bbox_inches="tight", dpi=300)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
