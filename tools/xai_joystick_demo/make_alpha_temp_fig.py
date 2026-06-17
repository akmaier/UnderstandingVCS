#!/usr/bin/env python3
"""Soft-branch alpha gradient-map figure (supplementary): the screen-space
saliency d(screen)/d(control) as the sharpness alpha is swept, from the maps
written by alpha_temp_demo.jl.

Single row (soft branch only) on a black background; the soft-select temperature
effect is shown separately as a pixel-space occupancy heatmap
(make_temp_heatmap_fig.py), so it is not duplicated here.

    python3 make_alpha_temp_fig.py
"""
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

CR = (slice(22, 54), slice(44, 104))      # zoom around the two placements


def load():
    maps = {}
    with open(os.path.join(OUT, "alpha_temp_manifest.txt")) as f:
        for line in f:
            name, r, c = line.split()
            a = np.fromfile(os.path.join(OUT, name + ".bin"), dtype="<f4")
            maps[name] = a.reshape(int(r), int(c))
    return maps


def main():
    m = load()
    fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.0))
    fig.subplots_adjust(left=0.03, right=0.86, top=0.80, bottom=0.06, wspace=0.08)

    vmax = max(np.abs(m[f"at_branch_grad{j+1}"][CR]).max() for j in range(3)) + 1e-9
    titles = (r"$\alpha = 2$", r"$\alpha = 6$", r"$\alpha = 20$")
    im = None
    for j in range(3):
        sc = m[f"at_branch_scene{j+1}"][CR]
        gd = m[f"at_branch_grad{j+1}"][CR]
        # black background; faint grey sprite; diverging gradient where nonzero.
        ax[j].imshow(sc, cmap="gray", vmin=0.0, vmax=2.5, aspect="equal",
                     interpolation="nearest")
        gm = np.ma.masked_where(np.abs(gd) < 0.04 * vmax, gd)
        im = ax[j].imshow(gm, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          aspect="equal", interpolation="nearest")
        ax[j].set_title(titles[j], fontsize=8.5)
        ax[j].set_xlabel(rf"peak $|\partial|$ = {np.abs(gd).max()/vmax:.2f}",
                         fontsize=6.8)
        ax[j].set_xticks([]); ax[j].set_yticks([])
        for s in ax[j].spines.values():
            s.set_visible(False)

    cax = fig.add_axes([0.885, 0.12, 0.015, 0.60])
    cb = fig.colorbar(im, cax=cax, ticks=[-vmax, 0, vmax])
    cb.ax.set_yticklabels([rf"$-{vmax:.2f}$", "0", rf"$+{vmax:.2f}$"], fontsize=5.8)
    cb.set_label("blue: darkens\nred: brightens", fontsize=6)
    cb.ax.tick_params(length=2)
    cb.outline.set_linewidth(0.5)

    fig.suptitle(r"Soft branch: $\partial$(screen)$/\partial z$ vs. sharpness "
                 r"$\alpha$ (a dipole at moderate $\alpha$, vanishing as "
                 r"$\alpha \to \infty$)", fontsize=8.5, y=0.99)
    fig.savefig(FIG, bbox_inches="tight", dpi=300)
    print("wrote", FIG, f"(vmax={vmax:.3f})")


if __name__ == "__main__":
    main()
