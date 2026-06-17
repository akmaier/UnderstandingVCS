#!/usr/bin/env python3
"""Render the alpha/temperature gradient-map figure (supplementary, 2-column)
from the maps written by alpha_temp_demo.jl.

Design notes (addressing reviewer feedback):
  * Each ROW has its OWN colour scale (the soft-branch and soft-select gradients
    are with respect to different controls and differ in magnitude), shown by an
    individual colour bar per row in absolute units.
  * Zero gradient maps to the SAME neutral in every panel: the diverging colour
    is alpha-composited over a light, faint-cannon canvas, so a pixel with no
    sensitivity looks identical (light grey) in both rows -- no white-vs-grey
    inconsistency between rows.
  * Wider aspect for a two-column (figure*) placement.

    python3 make_alpha_temp_fig.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
FIG = os.path.abspath(os.path.join(HERE, "..", "..", "jutari_paper", "paper",
                                   "figures", "fig_alpha_temp.pdf"))
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# Zoom window around the cannon (rows, cols).
CR = (slice(16, 60), slice(26, 122))
DIV = matplotlib.colormaps["RdBu_r"]


def load():
    maps = {}
    with open(os.path.join(OUT, "alpha_temp_manifest.txt")) as f:
        for line in f:
            name, r, c = line.split()
            a = np.fromfile(os.path.join(OUT, name + ".bin"), dtype="<f4")
            maps[name] = a.reshape(int(r), int(c))
    return maps


def composite(scene, grad, vmax):
    """Light canvas + faint grey cannon + alpha-composited diverging gradient.
    A pixel with |grad|->0 shows the canvas (the same light grey in every panel),
    so the neutral point is identical across rows; |grad|->vmax saturates to
    blue (darkens) / red (brightens)."""
    sc = scene[CR]
    gd = grad[CR]
    spr = sc / (sc.max() + 1e-9)                 # 0..1 cannon mask
    canvas = np.repeat((0.97 - 0.22 * spr)[..., None], 3, axis=2)  # faint cannon
    g = np.clip(gd / vmax, -1.0, 1.0)
    rgb = DIV(0.5 + 0.5 * g)[..., :3]            # diverging colour, white at 0
    a = np.clip(np.abs(gd) / vmax, 0.0, 1.0)[..., None]  # opacity = |grad|
    return (1.0 - a) * canvas + a * rgb


def row(fig, axes, m, prefix, vals, vmax, cbar_box):
    for j, ax in enumerate(axes):
        gd = m[f"{prefix}_grad{j+1}"][CR]
        ax.imshow(composite(m[f"{prefix}_scene{j+1}"], m[f"{prefix}_grad{j+1}"],
                            vmax), aspect="equal", interpolation="none")
        ax.set_xlabel(rf"peak $|\partial|$ = {np.abs(gd).max()/vmax:.2f}", fontsize=6.3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(vals[j], fontsize=8)
    # individual colour bar for this row, in absolute units
    cax = fig.add_axes(cbar_box)
    sm = matplotlib.cm.ScalarMappable(norm=colors.Normalize(-vmax, vmax),
                                      cmap="RdBu_r")
    cb = fig.colorbar(sm, cax=cax, ticks=[-vmax, 0, vmax])
    cb.ax.set_yticklabels([rf"$-{vmax:.2f}$", "0", rf"$+{vmax:.2f}$"], fontsize=5.6)
    cb.ax.tick_params(length=2)
    cb.outline.set_linewidth(0.5)


def main():
    m = load()
    fig, ax = plt.subplots(2, 3, figsize=(7.1, 3.5))
    fig.subplots_adjust(left=0.075, right=0.86, top=0.83, bottom=0.10,
                        wspace=0.08, hspace=0.42)

    av = max(np.abs(m[f"at_branch_grad{j+1}"][CR]).max() for j in range(3)) + 1e-9
    tv = max(np.abs(m[f"at_select_grad{j+1}"][CR]).max() for j in range(3)) + 1e-9
    row(fig, ax[0], m, "at_branch", (r"$\alpha = 2$", r"$\alpha = 6$",
        r"$\alpha = 20$"), av, [0.875, 0.55, 0.016, 0.26])
    row(fig, ax[1], m, "at_select", (r"$T = 2.0$", r"$T = 0.5$", r"$T = 0.1$"),
        tv, [0.875, 0.14, 0.016, 0.26])

    ax[0, 0].set_ylabel("soft branch\n(sharpness $\\alpha$)", fontsize=8)
    ax[1, 0].set_ylabel("soft select\n(temperature $T$)", fontsize=8)
    fig.suptitle(r"$\partial$(screen)$/\partial$(control) vs. relaxation "
                 r"strength: largest at moderate relaxation, vanishing toward the "
                 r"hard limit ($\alpha\to\infty$, $T\to0$)", fontsize=8.5, y=0.97)

    fig.savefig(FIG, bbox_inches="tight", dpi=300)
    print("wrote", FIG, f"(branch vmax={av:.3f}, select vmax={tv:.3f})")


if __name__ == "__main__":
    main()
