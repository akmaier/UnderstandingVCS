#!/usr/bin/env python3
"""Two-panel line figure illustrating how the soft-branch sharpness alpha
controls the gate and its gradient, for alpha in {2, 6, 20} (the three values
used elsewhere in the paper).

The branch gate is jutari's `soft_branch` primitive:

    g(z) = sigmoid(alpha * z) = 1 / (1 + exp(-alpha * z))

with derivative

    dg/dz = alpha * sigma(alpha z) * (1 - sigma(alpha z)).

LEFT  : the gate g(z) vs z for the three alpha.
RIGHT : the gradient dg/dz vs z for the three alpha.

Operating point z = 0.25 (used in the paper) is marked on both panels.
Key message: alpha=20 saturates (no gradient away from 0); alpha=2 is shallow
(weak, broad gradient); alpha=6 gives the largest gradient at z=0.25.

  python3 make_sigmoid_fig.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIG = os.path.abspath(os.path.join(HERE, "..", "..", "jutari_paper", "paper",
                                   "figures", "fig_sigmoid_alpha.pdf"))

plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

INK = "#1a1a1a"
STEEL = "#35618f"
ORANGE = "#c8641a"
# Three distinguishable styles for alpha = 2 / 6 / 20. The intermediate alpha=6
# (the "good" choice) gets the accent orange + solid line so it reads as the
# protagonist; the two extremes get steel/ink with dashed/dotted strokes.
STYLES = {
    2:  dict(color=STEEL,  ls="--", lw=1.4, label=r"$\alpha=2$"),
    6:  dict(color=ORANGE, ls="-",  lw=1.6, label=r"$\alpha=6$"),
    20: dict(color=INK,    ls=":",  lw=1.4, label=r"$\alpha=20$"),
}
ALPHAS = (2, 6, 20)
Z0 = 0.25  # operating point used in the paper


def sigma(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    z = np.linspace(-3, 3, 1201)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(5.0, 2.2))

    # ---- LEFT: gate g = sigma(alpha z) ------------------------------------
    for a in ALPHAS:
        axL.plot(z, sigma(a * z), **STYLES[a])
    axL.axvline(Z0, color=INK, ls=(0, (2, 2)), lw=0.6, alpha=0.7)
    axL.set_title(r"gate $\;g=\sigma(\alpha z)$", fontsize=8)
    axL.set_xlabel(r"branch offset $z$", fontsize=7.5)
    axL.set_ylabel(r"$g(z)$", fontsize=7.5)
    axL.set_xlim(-3, 3)
    axL.set_ylim(-0.03, 1.03)
    axL.set_yticks([0, 0.5, 1])
    axL.legend(loc="upper left", fontsize=6.5, frameon=False,
               handlelength=1.8, labelspacing=0.25, borderaxespad=0.2)

    # ---- RIGHT: gradient dg/dz = alpha sigma (1 - sigma) ------------------
    deriv = {}
    for a in ALPHAS:
        s = sigma(a * z)
        d = a * s * (1.0 - s)
        deriv[a] = d
        axR.plot(z, d, **STYLES[a])
    axR.axvline(Z0, color=INK, ls=(0, (2, 2)), lw=0.6, alpha=0.7)
    axR.set_title(r"gradient $\;\mathrm{d}g/\mathrm{d}z$", fontsize=8)
    axR.set_xlabel(r"branch offset $z$", fontsize=7.5)
    axR.set_ylabel(r"$\mathrm{d}g/\mathrm{d}z$", fontsize=7.5)
    axR.set_xlim(-3, 3)
    ymax = max(d.max() for d in deriv.values())
    axR.set_ylim(0, ymax * 1.18)

    # Derivative value at the operating point z = Z0 for each alpha; the
    # intermediate alpha=6 wins there, which is the figure's punchline.
    dvals = {a: float(av * sigma(av * Z0) * (1 - sigma(av * Z0)))
             for a, av in zip(ALPHAS, ALPHAS)}
    best = max(dvals, key=dvals.get)

    # Mark the operating-point gradient for the winning alpha with a dot.
    axR.plot([Z0], [dvals[best]], "o", color=ORANGE, ms=3.5,
             zorder=5, clip_on=False)

    # ---- annotations: the key message, drawn directly on the panels ------
    # alpha=20 saturates -> ~0 gradient away from z=0
    axR.annotate(r"$\alpha=20$: saturated," "\n" r"no gradient",
                 xy=(1.55, deriv[20][np.searchsorted(z, 1.55)] + 0.05),
                 xytext=(1.05, ymax * 0.74), fontsize=6.0, color=INK,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=INK, lw=0.5))
    # alpha=2 shallow -> weak, broad gradient
    axR.annotate(r"$\alpha=2$: shallow," "\n" r"weak gradient",
                 xy=(-1.8, deriv[2][np.searchsorted(z, -1.8)] + 0.02),
                 xytext=(-2.85, ymax * 0.62), fontsize=6.0, color=STEEL,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=STEEL, lw=0.5))
    # alpha=6 best at z=0.25
    axR.annotate(r"$\alpha=6$: largest" "\n" r"gradient at $z=0.25$",
                 xy=(Z0, dvals[best]),
                 xytext=(0.55, ymax * 1.05), fontsize=6.0, color=ORANGE,
                 ha="left", va="top",
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.6))

    # Label the operating point once, on the left panel near the axis.
    axL.text(Z0 + 0.12, 0.04, r"$z=0.25$", fontsize=6.0, color=INK,
             ha="left", va="bottom")

    for ax in (axL, axR):
        ax.tick_params(labelsize=6.5, length=2.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)

    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(FIG, bbox_inches="tight", dpi=300)
    print("wrote", FIG)
    print("dg/dz at z=0.25:",
          {a: round(dvals[a], 4) for a in ALPHAS}, " best:", best)


if __name__ == "__main__":
    main()
