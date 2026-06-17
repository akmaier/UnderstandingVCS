#!/usr/bin/env python3
"""Render the qualitative joystick/gradient XAI figure from the maps written
by joystick_grad_demo.jl. Outputs a vector PDF into the paper's figures dir.

  python3 make_joystick_fig.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
FIG = os.path.abspath(os.path.join(HERE, "..", "..", "jutari_paper", "paper",
                                   "figures", "fig_xai_joystick.pdf"))

plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

INK = "#1a1a1a"; STEEL = "#35618f"; ORANGE = "#c8641a"
# Single-hue dark->orange map: a binary saliency renders as a uniform orange
# object on a dark field (no confusing white-in-yellow magma highlights).
SAL = LinearSegmentedColormap.from_list("sal", ["#0c0c16", "#e07a1e"])


def load():
    maps = {}
    with open(os.path.join(OUT, "manifest.txt")) as f:
        for line in f:
            name, r, c = line.split()
            arr = np.fromfile(os.path.join(OUT, name + ".bin"), dtype="<f4")
            maps[name] = arr.reshape(int(r), int(c))
    grad = {}
    with open(os.path.join(OUT, "p3_grad.txt")) as f:
        for line in f:
            k, v = line.split()
            grad[k] = float(v)
    return maps, grad


# Square cut-out around the scene: zooms in 2x so the 8-px sprites display
# twice as large, and gives quadratic panels (80x80 -> square pixels).
CROP = (slice(8, 88), slice(20, 100))


def crop(A):
    return A[CROP[0], CROP[1]]


def imshow(ax, A, cmap, title, vmin=None, vmax=None):
    ax.imshow(crop(A), cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
              interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    m, grad = load()
    fig, ax = plt.subplots(2, 4, figsize=(7.0, 3.7))

    # Row 1 — genuine gradients in jutari's unmodified soft renderer.
    imshow(ax[0, 0], m["p1_frame"], "gray", "rendered scene")
    imshow(ax[0, 1], m["p1_sal_colup0"], SAL,
           r"$\partial$screen$/\partial$COLUP0 (cannon)")
    imshow(ax[0, 2], m["p1_sal_colup1"], SAL,
           r"$\partial$screen$/\partial$COLUP1 (invaders)")
    imshow(ax[0, 3], m["p1_sal_colubk"], SAL,
           r"$\partial$screen$/\partial$COLUBK (bg)")

    # Row 2 — stored switches (graphics) + soft sampler (joystick).
    # One graphics bit (white) shown within the faint cannon sprite outline.
    can = m["p2_cannon"]; can = can / (can.max() + 1e-6) * 0.4
    rgb2 = np.stack([can, can, can], axis=-1)
    rgb2[m["p2_sal_bit"] > 0] = [1.0, 1.0, 1.0]
    ax[1, 0].imshow(rgb2[CROP[0], CROP[1]], aspect="equal",
                    interpolation="nearest")
    ax[1, 0].set_title(r"$\partial$screen$/\partial$(one graphics bit)",
                       fontsize=8)
    ax[1, 0].set_xticks([]); ax[1, 0].set_yticks([])
    ax[1, 0].set_xlabel("white bit within the\ngrey cannon sprite",
                        fontsize=6.5)

    # scene with the player cannon (blue) and the target invader (orange).
    fr = m["p3_frame_neutral"].copy()
    base = fr / (fr.max() + 1e-6) * 0.5               # dim invaders for context
    rgb = np.stack([base, base, base], axis=-1)
    rgb[m["p3_goal"] > 0] = [0.78, 0.39, 0.10]        # target invader (orange)
    rgb[m["p3_cannon"] > 0.3] = [0.21, 0.38, 0.56]    # player cannon (blue)
    ax[1, 1].imshow(rgb[CROP[0], CROP[1]], aspect="equal",
                    interpolation="nearest")
    ax[1, 1].set_title("player cannon + target invader", fontsize=8)
    ax[1, 1].set_xticks([]); ax[1, 1].set_yticks([])
    ax[1, 1].set_xlabel("cannon (blue) slides under\nthe target invader (orange)",
                        fontsize=6.0)

    # joystick saliency: signed directional derivative (right + up combined)
    sal = m["p3_sal_right"] - m["p3_sal_left"] + 0  # right-sense map
    v = np.abs(sal).max() + 1e-6
    imshow(ax[1, 2], sal, "RdBu_r",
           r"$\partial$screen$/\partial$joystick$_{\rightarrow}$",
           vmin=-v, vmax=v)

    # joystick gradient bar chart + inferred direction
    a = ax[1, 3]
    order = ["up", "down", "left", "right"]
    vals = [grad[k] for k in order]
    cols = [ORANGE if v > 0 else STEEL for v in vals]
    a.bar(range(4), vals, color=cols, width=0.7)
    a.axhline(0, color=INK, lw=0.6)
    a.set_xticks(range(4)); a.set_xticklabels(["U", "D", "L", "R"], fontsize=7)
    a.set_title(r"$\partial$objective$/\partial$joystick", fontsize=8)
    a.tick_params(labelsize=6)
    pos = [order[i].upper() for i in range(4) if vals[i] > 0]
    a.set_xlabel("push: " + "+".join(pos), fontsize=6.5, color=ORANGE)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

    fig.tight_layout(pad=0.4, h_pad=0.9, w_pad=0.5)
    fig.savefig(FIG, bbox_inches="tight")
    print("wrote", FIG)


if __name__ == "__main__":
    main()
