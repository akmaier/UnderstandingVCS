#!/usr/bin/env python3
"""Animated GIFs of the two relaxation knobs (supplementary media).

These animate the underlying soft primitives directly (no emulator run needed),
each frame INDIVIDUALLY autoscaled so the shape -- not just the magnitude -- is
visible at every setting:

  fig_alpha_anim.gif : sweep the soft-branch sharpness alpha. Left: the gate
      g = sigma(alpha z); right: its gradient dg/dz (autoscaled per frame).
      Reads off the message: too-small alpha = a shallow gate whose gradient
      barely varies with z; too-large alpha = a saturated gate with ~zero
      gradient; an intermediate alpha gives a sharp, informative gradient.

  fig_temp_anim.gif : sweep the soft-select temperature T. The softmax weight
      distribution over candidate cannon columns (autoscaled per frame): a hard,
      one-hot pick at low T spreads into a distribution as T rises.

    python3 make_alpha_temp_anim.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                      "jutari_paper", "paper", "figures"))
plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
INK, STEEL, ORANGE = "#1a1a1a", "#35618f", "#c8641a"
Z0 = 0.25


def sigma(x):
    return 1.0 / (1.0 + np.exp(-x))


def alpha_gif():
    z = np.linspace(-3, 3, 601)
    alphas = np.concatenate([np.linspace(1, 20, 42), np.linspace(20, 1, 14)])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(5.4, 2.4))

    def draw(a):
        for ax in (axL, axR):
            ax.clear()
            ax.axvline(Z0, color=INK, lw=0.5, ls=(0, (2, 2)), alpha=0.7)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axL.plot(z, sigma(a * z), color=ORANGE, lw=1.8)
        axL.set_xlim(-3, 3); axL.set_ylim(-0.03, 1.03); axL.set_yticks([0, 0.5, 1])
        axL.set_title(r"gate $g=\sigma(\alpha z)$", fontsize=8.5)
        axL.set_xlabel("branch offset $z$", fontsize=8); axL.set_ylabel("$g(z)$", fontsize=8)
        d = a * sigma(a * z) * (1 - sigma(a * z))
        axR.plot(z, d, color=ORANGE, lw=1.8)
        axR.plot([Z0], [a * sigma(a * Z0) * (1 - sigma(a * Z0))], "o",
                 color=ORANGE, ms=4, clip_on=False, zorder=5)
        axR.set_xlim(-3, 3); axR.set_ylim(0, max(d.max() * 1.15, 1e-3))  # autoscale
        axR.set_title(r"gradient $\mathrm{d}g/\mathrm{d}z$ (autoscaled)", fontsize=8.5)
        axR.set_xlabel("branch offset $z$", fontsize=8)
        fig.suptitle(rf"soft branch sharpness  $\alpha = {a:.1f}$", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.90], w_pad=1.4)

    anim = FuncAnimation(fig, draw, frames=alphas, interval=120)
    out = os.path.join(FIGDIR, "fig_alpha_anim.gif")
    anim.save(out, writer=PillowWriter(fps=8))
    plt.close(fig)
    print("wrote", out)


# Same invader sprite as alpha_temp_demo.jl / make_temp_heatmap_fig.py.
_INVADER = [0b00011000, 0b00111100, 0b01111110, 0b11011011,
            0b11111111, 0b10100101, 0b00100100, 0b01000010]
_H, _W, _VS, _YC = 80, 140, 2, 30
_COLS = np.arange(30, 111, dtype=int)
_TARGET, _SIGMA = 70.0, 6.0
_CR = (slice(22, 54), slice(44, 100))


def _sprite_occ(x):
    occ = np.zeros((_H, _W), np.float32)
    for rr, byte in enumerate(_INVADER):
        for b in range(8):
            if (byte >> (7 - b)) & 1:
                occ[_YC + rr * _VS: _YC + rr * _VS + _VS, x + b] = 1.0
    return occ


def temp_gif():
    """Pixel-space sampled occupancy as the soft-select temperature T sweeps:
    100 draws ~ softmax(l/T) at pixel resolution, rendered + averaged each frame,
    so the occupancy cloud broadens with T and tightens to a sharp sprite."""
    occ = np.stack([_sprite_occ(int(x)) for x in _COLS])

    def softmax(T):
        l = -((_COLS - _TARGET) / _SIGMA) ** 2
        e = np.exp((l - l.max()) / T)
        return e / e.sum()

    Ts = np.concatenate([np.linspace(3.0, 0.1, 42), np.linspace(0.1, 3.0, 14)])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    rng = np.random.default_rng(0)

    def draw(T):
        ax.clear()
        w = softmax(T)
        idx = rng.choice(len(_COLS), size=100, p=w)
        heat = occ[idx].mean(axis=0)[_CR]
        std = float(np.sqrt(np.sum(w * (_COLS - np.sum(w * _COLS)) ** 2)))
        ax.imshow(heat, cmap="magma", vmin=0.0, vmax=1.0, aspect="equal",
                  interpolation="bilinear")
        ax.set_title(rf"soft select  $T={T:.2f}$   ($\sigma\approx{std:.1f}$ px)",
                     fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()

    anim = FuncAnimation(fig, draw, frames=Ts, interval=120)
    out = os.path.join(FIGDIR, "fig_temp_anim.gif")
    anim.save(out, writer=PillowWriter(fps=8))
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    alpha_gif()
    temp_gif()
