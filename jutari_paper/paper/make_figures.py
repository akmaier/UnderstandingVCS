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
    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    # Tight y-range around the drawn content (lowest box bottom 2.4, title 9.55)
    # so the saved PDF has no empty band that would gap it from the caption.
    ax.set_xlim(0, 14); ax.set_ylim(2.15, 10.0); ax.axis("off")

    # The address/data bus runs across the top, spanning the full device row.
    box(ax, 0.5, 8.0, 13.0, 0.85, "address + data bus", fc=LIGHT, fs=8)

    # Bus devices, left to right: RIOT, CPU, cartridge, TIA.
    riot = box(ax, 0.4, 5.5, 3.3, 1.8, "RIOT / M6532\nRAM, timer, I/O", fs=6.7)
    cpu  = box(ax, 4.2, 5.5, 2.8, 1.8, "6507 CPU\n(6502 core)", fs=7.2)
    cart = box(ax, 7.5, 5.5, 2.8, 1.8, "cartridge\nROM", fs=7.4)
    tia  = box(ax, 10.8, 5.5, 2.8, 1.8, "TIA\nvideo+audio", fs=6.9)

    # Peripherals below: input under RIOT (left), TV under TIA (right).
    inp = box(ax, 0.4, 2.4, 3.3, 1.6, "joystick,\nswitches", fs=7.2)
    tv  = box(ax, 10.8, 2.4, 2.8, 1.6, "TV picture\n+ audio", fs=7.2)

    for b in (riot, cpu, cart, tia):
        arrow(ax, (b[0], 7.3), (b[0], 8.0), style="<|-|>", lw=1.0)
    arrow(ax, (inp[0], 4.0), (riot[0], 5.5), style="-|>", lw=1.1, color=STEEL)
    arrow(ax, (tia[0], 5.5), (tv[0], 4.0), style="-|>", lw=1.3, color=ORANGE)

    ax.text(7.0, 9.55, "Atari 2600 VCS", ha="center", fontsize=9.5,
            fontweight="bold", color=INK)
    fig.tight_layout(pad=0.1)
    fig.savefig(os.path.join(OUT, "fig_architecture.pdf"),
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.set_xlim(0, 21); ax.set_ylim(0, 8.4); ax.axis("off")

    # Shared inputs (left).
    box(ax, 0.3, 5.5, 4.0, 2.2,
        "ROM = weights\nRAM = soft tape\nbranches = gates", fc=LIGHT, fs=8.5)
    box(ax, 0.3, 0.6, 4.0, 1.9, "VCS state\n(regs, RAM, TIA)", fc=LIGHT, fs=8.5)

    # The two execution paths.
    hard = box(ax, 5.3, 5.5, 5.4, 2.2,
               "HARD step\nuint8 dispatch\ninteger ALU", fs=8.5)
    soft = box(ax, 5.3, 0.6, 5.4, 2.2,
               "SOFT step (float32)\none-hot reads\nsigmoid gate",
               ec=STEEL, fs=8.5)

    # Straight-through join.
    ste = box(ax, 12.0, 3.0, 5.6, 2.3,
              "straight-through\nforward = hard\nbackward = soft",
              fc="#fbeede", ec=ORANGE, fs=8.5)

    box(ax, 18.2, 5.4, 2.6, 1.9, "bit-exact\nframe / RAM", fs=8)
    box(ax, 18.2, 0.7, 2.6, 1.9, "gradients\n" r"$\partial$pixel$/\partial$ROM",
        ec=STEEL, fs=8)

    arrow(ax, (4.3, 6.6), (5.3, 6.6), style="-|>")
    arrow(ax, (4.3, 1.55), (5.3, 1.55), style="-|>", color=STEEL)
    arrow(ax, (10.7, 6.4), (12.0, 4.6), style="-|>", rad=-0.12)
    arrow(ax, (10.7, 1.8), (12.0, 3.6), style="-|>", color=STEEL, rad=0.12)
    arrow(ax, (17.6, 4.6), (18.2, 6.0), style="-|>", lw=1.3, rad=-0.1)
    arrow(ax, (17.6, 3.6), (18.2, 1.7), style="-|>", lw=1.3,
          ls=(0, (4, 2)), color=STEEL, rad=0.1)

    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, "fig_pipeline.pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Freeze the effort analysis at the implementation cutoff: paper-writing
# commits (which begin here) are not implementation effort, and excluding them
# keeps the figure stable as the paper repo keeps moving.
IMPL_CUTOFF = dt.datetime.fromisoformat("2026-06-16T17:00:00+02:00")


def git_commit_times():
    repo = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True).stdout.strip()
    out = subprocess.run(["git", "-C", repo, "log", "--format=%cI"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    times = sorted(dt.datetime.fromisoformat(t) for t in out)
    return [t for t in times if t <= IMPL_CUTOFF]


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
