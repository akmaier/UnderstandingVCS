#!/usr/bin/env python3
"""Plot jaxtari GPU batched-throughput scaling vs batch size N.

Reads every results/gpu/*.json produced by tools/bench_jaxtari_gpu.py and draws
aggregate env-steps/s vs N (log-log), one line per (GPU, mode), with the CPU
reference lines from the supplement. Saves to jutari_paper/paper/figures/.

  python tools/plot_gpu_throughput.py
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "results" / "gpu"
OUT = REPO / "jutari_paper" / "paper" / "figures"

# CPU baselines from supplementary.tex §Throughput (M1 Max, Pong, soft mode).
CPU_VMAP_ASYMPTOTE = 60_000      # batched lax.scan vmap aggregate ceiling on CPU
CPU_SINGLE_STEP = 1_178          # single-step jaxtari on CPU


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(glob.glob(str(SRC / "*.json")))
    if not files:
        print(f"no results in {SRC}", file=sys.stderr); return 1

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    styles = {"fwd": dict(ls="-", marker="o"), "fwdbwd": dict(ls="--", marker="s")}
    colors = plt.cm.viridis([0.15, 0.55, 0.8, 0.95])
    label_for = {"fwd": "forward", "fwdbwd": "forward+grad"}

    for ci, f in enumerate(files):
        d = json.load(open(f))
        gpu = d["env"].get("device_kind", Path(f).stem).replace("NVIDIA ", "").replace("GeForce ", "")
        for mode in ("fwd", "fwdbwd"):
            pts = [(r["N"], r["env_steps_per_s"]) for r in d["results"]
                   if r["mode"] == mode and r.get("ok")]
            if not pts:
                continue
            xs, ys = zip(*sorted(pts))
            ax.plot(xs, ys, color=colors[ci % len(colors)], **styles[mode],
                    label=f"{gpu} ({label_for[mode]})", markersize=4, lw=1.5)

    ax.axhline(CPU_VMAP_ASYMPTOTE, color="0.5", ls=":", lw=1)
    ax.text(1.2, CPU_VMAP_ASYMPTOTE * 1.15, "CPU vmap asymptote (~60k)",
            color="0.4", fontsize=7.5)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("batch size $N$ (parallel environments)")
    ax.set_ylabel("aggregate throughput (env-steps/s)")
    ax.set_title("jaxtari soft-mode GPU throughput vs batch size (Pong, 3000 steps)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUT / f"gpu_throughput.{ext}"
        fig.savefig(p, dpi=150)
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
