#!/usr/bin/env python3
"""Pixel-space sprite-occupancy heatmap of the soft-select temperature T.

For each T in {2.0, 0.5, 0.1} we SAMPLE the cannon's column 100 times from the
soft-select distribution w = softmax(logits / T) over the candidate columns,
render the (hard) cannon sprite at each sampled column, sum the 100 rendered
images, and divide by 100. The result is a screen-domain heatmap whose value at
each pixel is the fraction of samples in which the sprite covered that pixel ---
i.e. a Monte-Carlo estimate of the expected sprite occupancy E[occupancy].

  high T (2.0): the draws spread over several neighbouring columns -> several
                cannon copies of graded brightness (a blurry, spread sprite);
  low T  (0.1): the draws collapse onto the target column -> a single sharp
                cannon at full occupancy.

The per-column hard sprites come from alpha_temp_demo.jl (at_cand_occ*). Sampling
uses a fixed seed so the figure is reproducible.

    python3 make_temp_heatmap_fig.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
FIG = os.path.abspath(os.path.join(HERE, "..", "..", "jutari_paper", "paper",
                                   "figures", "fig_temp_heatmap.pdf"))
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# Same setup as alpha_temp_demo.jl.
CAND = np.arange(30, 111, 8, dtype=float)          # 11 candidate columns
TARGET = 70.0
LOGITS = -((CAND - TARGET) / 6.0) ** 2
K = len(CAND)
N_SAMPLES = 100
TS = (2.0, 0.5, 0.1)
CR = (slice(24, 54), slice(26, 122))               # zoom window around the cannon


def softmax(T):
    e = np.exp((LOGITS - LOGITS.max()) / T)
    return e / e.sum()


def load_sprites():
    shape = {}
    with open(os.path.join(OUT, "alpha_temp_manifest.txt")) as f:
        for line in f:
            name, r, c = line.split()
            shape[name] = (int(r), int(c))
    occ = []
    for k in range(1, K + 1):
        name = f"at_cand_occ{k}"
        a = np.fromfile(os.path.join(OUT, name + ".bin"), dtype="<f4")
        occ.append(a.reshape(shape[name]))
    return np.stack(occ)                            # (K, H, W)


def main():
    occ = load_sprites()                            # (K, H, W), values in {0,1}
    rng = np.random.default_rng(0)                  # reproducible sampling

    fig, axes = plt.subplots(1, 3, figsize=(5.6, 2.1))
    im = None
    for ax, T in zip(axes, TS):
        w = softmax(T)
        idx = rng.choice(K, size=N_SAMPLES, p=w)    # 100 sampled columns ~ w(T)
        heat = occ[idx].mean(axis=0)[CR]            # sum images / 100 = E[occ]
        neff = 1.0 / np.sum(w ** 2)
        im = ax.imshow(heat, cmap="magma", vmin=0.0, vmax=1.0,
                       aspect="equal", interpolation="nearest")
        ax.set_title(rf"$T = {T}$", fontsize=8.5)
        ax.set_xlabel(rf"$N_{{\mathrm{{eff}}}} = {neff:.1f}$ cols", fontsize=6.8)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(r"Sampled sprite occupancy (100 draws from "
                 r"$\mathrm{softmax}(\ell/T)$, averaged in pixel space)",
                 fontsize=8.5, y=1.02)
    fig.subplots_adjust(left=0.02, right=0.88, top=0.84, bottom=0.10, wspace=0.10)
    cax = fig.add_axes([0.90, 0.12, 0.018, 0.70])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("occupancy fraction", fontsize=7)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.5)

    fig.savefig(FIG, bbox_inches="tight", dpi=300)
    print("wrote", FIG)
    for T in TS:
        w = softmax(T)
        print(f"  T={T:4.2f}  N_eff={1/np.sum(w**2):.2f}  max w={w.max():.3f}")


if __name__ == "__main__":
    main()
