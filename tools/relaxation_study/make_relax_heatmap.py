#!/usr/bin/env python3
"""Overview heatmap of the per-step bit-exactness likelihood P_step over the
(alpha, T) plane for the fully-relaxed soft pass on the full simulator.

P_step is separable, P_step = p_read(T)^rho * p_branch(alpha)^f_b, so the 2D
field is the outer combination of the two 1D profiles dumped by
dump_profiles.jl. The bit-exact region (P_step = 1) is outlined; the colour
shows how the likelihood falls off across the two boundaries.

    python3 make_relax_heatmap.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
PROF = os.path.join(HERE, "relax_profiles.txt")
FIG = os.path.abspath(os.path.join(HERE, "..", "..", "jutari_paper", "paper",
                      "figures", "fig_relax_heatmap.pdf"))
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def load():
    d = {}
    with open(PROF) as f:
        for line in f:
            k, *rest = line.split()
            d[k] = np.array([float(x) for x in rest]) if len(rest) > 1 else float(rest[0])
    return d


def main():
    d = load()
    alpha, pb = d["ALPHA"], d["PBRANCH"]
    T, pr = d["TEMP"], d["PREAD"]
    rho, fb = d["RHO"], d["FB"]
    # separable outer combination -> P_step[i (alpha), j (T)]
    P = (pr[None, :] ** rho) * (pb[:, None] ** fb)

    fig, ax = plt.subplots(figsize=(3.35, 2.65))
    fig.subplots_adjust(left=0.16, right=0.86, top=0.99, bottom=0.14)
    im = ax.pcolormesh(T, alpha, P, cmap="magma", vmin=0.0, vmax=1.0,
                       shading="gouraud", rasterized=True)
    # outline the bit-exact region (P_step -> 1)
    ax.contour(T, alpha, P, levels=[0.5], colors="white", linewidths=0.8,
               linestyles="--")
    ax.contour(T, alpha, P, levels=[0.999], colors="#39ff14", linewidths=1.1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"softmax-read temperature $T$")
    ax.set_ylabel(r"branch sharpness $\alpha$")
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0])
    ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1.0"])
    ax.set_yticks([1, 2, 5, 10, 20]); ax.set_yticklabels(["1", "2", "5", "10", "20"])
    ax.tick_params(labelsize=6.5)
    # annotate regions
    ax.text(0.072, 16, "bit-exact", color="white", fontsize=7.5, ha="center",
            va="center", weight="bold")
    ax.text(0.55, 1.5, "diverges", color="white", fontsize=7.5, ha="center",
            va="center")
    # recommended operating point: corner of the exact region (smallest alpha,
    # largest T that stay exact) -- where the two boundary sweeps cross.
    ax.plot([0.12], [6], marker="*", markersize=12, color="#39ff14",
            markeredgecolor="black", markeredgewidth=0.6, zorder=5)
    ax.annotate(r"set here: $\alpha{=}6,\,T{=}0.12$", xy=(0.12, 6),
                xytext=(0.13, 2.4), color="white", fontsize=6.5,
                arrowprops=dict(arrowstyle="->", color="white", lw=0.6))
    cax = fig.add_axes([0.875, 0.14, 0.03, 0.85])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"per-step likelihood $P_{\mathrm{step}}$", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_linewidth(0.4)

    fig.savefig(FIG, bbox_inches="tight", dpi=300)
    print("wrote", FIG)
    print(f"  rho={rho:.3f} f_b={fb:.3f}; exact corner ~ alpha>=6, T<=0.12")


if __name__ == "__main__":
    main()
