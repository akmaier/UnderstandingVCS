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


def temp_gif():
    cand = np.arange(30, 111, 8.0)
    target = 70.0
    logits = -((cand - target) / 6.0) ** 2
    Ts = np.concatenate([np.linspace(0.08, 3.0, 42), np.linspace(3.0, 0.08, 14)])
    fig, ax = plt.subplots(figsize=(4.6, 2.8))

    def softmax(T):
        e = np.exp((logits - logits.max()) / T)
        return e / e.sum()

    def draw(T):
        ax.clear()
        w = softmax(T)
        ax.bar(cand, w, width=6.0, color=ORANGE, edgecolor=INK, lw=0.4)
        ax.axvline(target, color=STEEL, lw=0.9, ls="--")
        ax.set_ylim(0, max(w.max() * 1.15, 1e-3))                    # autoscale
        ax.set_xlim(cand[0] - 5, cand[-1] + 5)
        ax.set_xlabel("candidate cannon column", fontsize=8)
        ax.set_ylabel("softmax weight $w_k$ (autoscaled)", fontsize=8)
        neff = 1.0 / np.sum(w ** 2)
        ax.set_title(rf"soft select temperature  $T={T:.2f}$"
                     rf"   ($N_{{\mathrm{{eff}}}}={neff:.2f}$ columns)", fontsize=9.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()

    anim = FuncAnimation(fig, draw, frames=Ts, interval=120)
    out = os.path.join(FIGDIR, "fig_temp_anim.gif")
    anim.save(out, writer=PillowWriter(fps=8))
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    alpha_gif()
    temp_gif()
