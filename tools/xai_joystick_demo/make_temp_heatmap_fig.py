#!/usr/bin/env python3
"""Pixel-space sprite-occupancy heatmap of the soft-select temperature T.

For each T in {2.0, 0.5, 0.1} we sample the target sprite's column 100 times from
w = softmax(logits / T) over a PIXEL-RESOLUTION grid of candidate columns, render
the (hard) invader sprite at each sampled column, sum the 100 images, and divide
by 100 --- a screen-domain Monte-Carlo estimate of the expected sprite occupancy.

Sampling at 1-pixel column resolution (not the coarse 8-px candidate spacing used
for the soft_select gradient demo) makes neighbouring draws overlap, so the
average is a smooth occupancy cloud that broadens with T, rather than a row of
discrete sprite copies.

    python3 make_temp_heatmap_fig.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                      "jutari_paper", "paper", "figures", "fig_temp_heatmap.pdf"))
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# Target sprite: the same 8-wide invader bitmap as alpha_temp_demo.jl.
INVADER = [0b00011000, 0b00111100, 0b01111110, 0b11011011,
           0b11111111, 0b10100101, 0b00100100, 0b01000010]
H, W, VS, YC = 80, 140, 2, 30
TARGET = 70.0
SIGMA = 6.0                                   # logits = -((x-target)/SIGMA)^2
COLS = np.arange(30, 111, dtype=int)          # PIXEL-resolution candidate columns
N_SAMPLES = 100
TS = (2.0, 0.5, 0.1)
CR = (slice(27, 49), slice(44, 100))          # tight zoom around the sprite


def sprite_occ(x):
    occ = np.zeros((H, W), dtype=np.float32)
    for rr, byte in enumerate(INVADER):
        for b in range(8):
            if (byte >> (7 - b)) & 1:
                r0 = YC + rr * VS
                occ[r0:r0 + VS, x + b] = 1.0
    return occ


def softmax(T):
    logits = -((COLS - TARGET) / SIGMA) ** 2
    e = np.exp((logits - logits.max()) / T)
    return e / e.sum()


def main():
    occ = np.stack([sprite_occ(int(x)) for x in COLS])      # (n_cols, H, W)
    rng = np.random.default_rng(0)                          # reproducible

    # Single-column size (the LaTeX caption carries the description).
    fig, axes = plt.subplots(1, 3, figsize=(3.35, 1.45))
    fig.subplots_adjust(left=0.02, right=0.84, top=0.88, bottom=0.16, wspace=0.06)
    im = None
    for ax, T in zip(axes, TS):
        w = softmax(T)
        idx = rng.choice(len(COLS), size=N_SAMPLES, p=w)     # 100 sampled columns
        heat = occ[idx].mean(axis=0)[CR]                     # sum images / 100
        std = float(np.sqrt(np.sum(w * (COLS - np.sum(w * COLS)) ** 2)))
        im = ax.imshow(heat, cmap="magma", vmin=0.0, vmax=1.0, aspect="equal",
                       interpolation="bilinear")
        ax.set_title(rf"$T = {T}$", fontsize=7.5)
        ax.set_xlabel(rf"$\sigma \approx {std:.1f}$ px", fontsize=6.0)
        ax.set_xticks([]); ax.set_yticks([])

    cax = fig.add_axes([0.855, 0.18, 0.022, 0.66])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("occupancy", fontsize=6.0)
    cb.ax.tick_params(labelsize=5.0, length=1.5)
    cb.outline.set_linewidth(0.4)

    fig.savefig(FIG, bbox_inches="tight", dpi=300)
    print("wrote", FIG)
    for T in TS:
        w = softmax(T)
        std = np.sqrt(np.sum(w * (COLS - np.sum(w * COLS)) ** 2))
        print(f"  T={T:4.2f}  sigma={std:.2f}px  max w={w.max():.3f}")


if __name__ == "__main__":
    main()
