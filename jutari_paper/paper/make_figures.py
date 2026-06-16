#!/usr/bin/env python3
"""Generate the vector figures for the AAAI-27 differentiable-VCS paper.

Outputs (PDF, grayscale-safe, distinct line styles + hatching so the
figures remain decipherable without colour, per the AAAI WCAG rule):

  figures/fig_architecture.pdf  VCS hardware block diagram (Figure 1)
  figures/fig_pipeline.pdf      hard/soft dual-path + straight-through (Figure 2)
  figures/fig_timeline.pdf      effort timeline from real git history (Figure 3)

Run from the paper/ directory:  python3 make_figures.py
The timeline reads the repository git log directly, so the effort plot is
reproducible from the commit history (the project's developer log).
"""
import subprocess, datetime as dt, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,   # embed TrueType (no Type-3 fonts)
    "ps.fonttype": 42,
})
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

INK = "#1a1a1a"
STEEL = "#35618f"
ORANGE = "#c8641a"
GREY = "#8a8a8a"
LIGHT = "#e9eef4"


# --------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc="white", ec=INK, lw=1.0, fs=8.5, style="round"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"{style},pad=0.0,rounding_size=0.05",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3, color=INK)
    return (x + w / 2, y + h / 2)


def arrow(ax, p1, p2, style="-|>", lw=1.0, ls="-", color=INK, rad=0.0):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=10,
                        lw=lw, ls=ls, color=color,
                        connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(a)


# --------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9.6); ax.axis("off")

    # top row: CPU | TIA | RIOT
    cpu = box(ax, 0.3, 6.7, 3.2, 1.5, "6507 CPU\n(6502 core)", fs=7.6)
    tia = box(ax, 4.4, 6.7, 3.2, 1.5, "TIA\nvideo + audio", fs=7.6)
    riot = box(ax, 8.5, 6.7, 3.2, 1.5, "RIOT / M6532\nRAM · timer · I/O", fs=7.2)
    # central bus
    box(ax, 3.0, 3.9, 6.0, 0.8, "address + data bus", fc=LIGHT, fs=7.8)
    # bottom row: controllers | cartridge | screen
    ctrl = box(ax, 0.3, 0.5, 3.2, 1.4, "joystick · paddles\n· switches", fs=7.0)
    cart = box(ax, 4.4, 0.5, 3.2, 1.4, "cartridge ROM\n(bank-switched)", fs=7.2)
    screen = box(ax, 8.5, 0.5, 3.2, 1.4, "TV picture\n+ audio", fs=7.4)

    for blk in (cpu, tia, riot):
        arrow(ax, (blk[0], 6.7), (blk[0], 4.7), style="<|-|>", lw=1.0)
    arrow(ax, (cart[0], 1.9), (cart[0], 3.9), style="-|>", lw=1.0)
    arrow(ax, (ctrl[0] + 1.0, 1.9), (riot[0], 6.7), style="-|>",
          ls=(0, (4, 2)), color=STEEL, rad=-0.32)
    arrow(ax, (tia[0] + 0.6, 6.7), (screen[0], 1.9), style="-|>",
          lw=1.4, color=ORANGE, rad=-0.22)

    ax.text(6.0, 9.2, "Atari 2600 VCS", ha="center", fontsize=9.5,
            fontweight="bold", color=INK)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, "fig_architecture.pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    ax.set_xlim(0, 20); ax.set_ylim(0, 7); ax.axis("off")

    # Shared state inputs
    box(ax, 0.3, 4.7, 3.1, 1.6,
        "ROM  →  weight tensor\nRAM  →  soft tape\nflags/branch → gates",
        fc=LIGHT, fs=7.6)
    box(ax, 0.3, 0.5, 3.1, 1.4, "VCS state\n(regs, RAM, TIA)", fc=LIGHT, fs=7.8)

    # HARD path
    hard = box(ax, 4.7, 4.8, 4.6, 1.5,
               "HARD step\nuint8 dispatch, integer ALU,\nexact bit logic",
               fc="white", fs=7.6)
    # SOFT path
    soft = box(ax, 4.7, 0.5, 4.6, 1.7,
               "SOFT step (float32)\none-hot ROM/RAM reads,\nsigmoid branch gate $g{=}\\sigma(\\alpha z)$",
               fc="white", ec=STEEL, fs=7.4)

    # STE join
    ste = box(ax, 10.6, 2.6, 5.2, 1.7,
              "Straight-through estimator\n"
              r"$\mathrm{STE}=\mathrm{soft}+\mathrm{sg}(\mathrm{hard}-\mathrm{soft})$"
              "\nforward = hard · backward = soft",
              fc="#fbeede", ec=ORANGE, fs=7.2)

    out_fwd = box(ax, 16.9, 4.5, 2.8, 1.4,
                  "bit-exact\nframe / RAM", fc="white", fs=7.4)
    out_bwd = box(ax, 16.9, 0.7, 2.8, 1.4,
                  r"gradients $\partial$pixel$/\partial$ROM",
                  fc="white", ec=STEEL, fs=7.0)

    arrow(ax, (3.4, 5.5), (4.7, 5.55), style="-|>")
    arrow(ax, (3.4, 1.2), (4.7, 1.35), style="-|>", color=STEEL)
    arrow(ax, (hard[0] + 2.3, hard[1]), (10.6, 3.7), style="-|>", rad=-0.15)
    arrow(ax, (soft[0] + 2.3, soft[1]), (10.6, 3.2), style="-|>",
          color=STEEL, rad=0.15)
    arrow(ax, (ste[0] + 2.6, 3.7), (16.9, 5.1), style="-|>", lw=1.3,
          color=INK, rad=-0.1)
    arrow(ax, (ste[0] + 2.6, 3.1), (16.9, 1.4), style="-|>", lw=1.3,
          ls=(0, (4, 2)), color=STEEL, rad=0.1)

    ax.text(2.0, 6.7, "known mechanism", ha="center", fontsize=8,
            style="italic", color=GREY)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, "fig_pipeline.pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def git_commit_times():
    repo = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True).stdout.strip()
    out = subprocess.run(["git", "-C", repo, "log", "--format=%cI"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    times = sorted(dt.datetime.fromisoformat(t) for t in out)
    return times


def fig_timeline():
    times = git_commit_times()
    t0 = times[0]
    hours = [(t - t0).total_seconds() / 3600.0 for t in times]
    n = len(times)

    # active sessions: gap > 3 h is a boundary
    GAP = 3.0
    active = 0.0
    sessions = []
    s_start = 0
    for i in range(1, n):
        if hours[i] - hours[i - 1] > GAP:
            sessions.append((s_start, i - 1))
            s_start = i
    sessions.append((s_start, n - 1))
    for a, b in sessions:
        active += hours[b] - hours[a]

    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    days = [h / 24.0 for h in hours]
    # shade active sessions
    for a, b in sessions:
        ax.axvspan(days[a], days[b], color=LIGHT, zorder=0)
    ax.step(days, range(1, n + 1), where="post", color=STEEL, lw=1.3,
            zorder=3)
    ax.set_xlabel("calendar time since first commit (days)")
    ax.set_ylabel("cumulative commits")
    ax.set_xlim(-0.5, days[-1] + 0.5)
    ax.set_ylim(0, n + 12)
    ax.grid(True, axis="y", ls=":", lw=0.5, color=GREY, alpha=0.6)

    cal_days = days[-1]
    txt = (f"{n} commits · {len(sessions)} active sessions\n"
           f"{active:.0f} active h ($\\approx${active/24:.1f} working days)\n"
           f"compressed into {cal_days:.0f} calendar days")
    ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GREY, lw=0.6))

    # mark longest idle gaps
    gaps = sorted(((hours[i] - hours[i - 1], i) for i in range(1, n)),
                  reverse=True)[:3]
    for g, i in gaps:
        xm = (days[i] + days[i - 1]) / 2
        ax.annotate(f"{g/24:.1f} d idle", xy=(xm, i), xytext=(xm, i + 8),
                    ha="center", fontsize=6.8, color=ORANGE,
                    arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.7))

    handles = [Line2D([0], [0], color=STEEL, lw=1.3, label="cumulative commits"),
               matplotlib.patches.Patch(fc=LIGHT, ec=GREY, lw=0.5,
                                         label="active session (gap $\\leq$ 3 h)")]
    ax.legend(handles=handles, loc="upper left", fontsize=7.2, frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_timeline.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"timeline: {n} commits, {len(sessions)} sessions, "
          f"{active:.1f} active h, {cal_days:.1f} calendar days")


if __name__ == "__main__":
    fig_architecture()
    fig_pipeline()
    fig_timeline()
    print("figures written to", OUT)
