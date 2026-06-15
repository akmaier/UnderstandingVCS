"""Console — the assembled VCS (CPU + Bus).

A `Console` is just a tagged pair `(cpu_state, bus)` plus the standard
operations games need: reset (load PC from the cart's reset vector),
single-instruction step, and "run until the next frame ends".

This module is the natural entry point for tests, front-ends, and the
StellaEnvironment wrapper — it hides the explicit `step(state, bus)`
threading.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from jaxtari.bus import Bus, initial_bus, peek
from jaxtari.cpu.m6502 import step as cpu_step
from jaxtari.types import CPUState, initial_cpu_state


# Task #106 (partial-frame model): xitari runs at most this many
# INSTRUCTIONS per mediaSource().update() — `m6502().execute(25000)` in
# TIA::update (M6502Low.cxx:65 decrements once per instruction, NOT per
# cycle). When the budget is exhausted WITHOUT a frame end, xitari returns
# a "grey" (incomplete) frame: the frame counter is NOT advanced and the
# beam/cycle state is preserved, so the next update() continues the same
# frame. `run_until_frame` mirrors this exactly — which is what reproduces
# Q*Bert's boot→step "sliver" frame (its RESET-boot self-test, task #52,
# has no TIA pokes for thousands of scanlines, so the frame can only be
# sliced by this budget, exactly as in xitari).
_UPDATE_INSTRUCTION_BUDGET = 25_000


class Console(NamedTuple):
    """Top-level VCS state."""
    cpu: CPUState
    bus: Bus


def initial_console(rom=None) -> Console:
    """Build a console for the given ROM. `rom` may be a 2K/4K/8K/16K/32K
    `jnp.ndarray`; the cart kind is auto-detected by `initial_bus`."""
    return Console(cpu=initial_cpu_state(), bus=initial_bus(rom))


def console_reset(console: Console) -> Console:
    """Reset the console: fresh CPU state, fresh TIA/RIOT (RAM zeroed),
    and load PC from the cart's reset vector at \$FFFC/\$FFFD.

    Cart state itself (current bank) is NOT reset — real hardware doesn't
    have a bank-reset line; the cart's reset code is responsible for any
    initial bank selection.
    """
    fresh_bus = initial_bus()
    new_bus = console.bus._replace(
        ram=fresh_bus.ram,
        tia=fresh_bus.tia,
        riot=fresh_bus.riot,
    )
    fresh_cpu = initial_cpu_state()
    # P4d: peek returns `(value, new_bus)`. Vector reads don't have
    # side effects on the cart-mirror addresses ($FFFC/$FFFD) but we
    # thread the bus through for consistency.
    lo, new_bus = peek(new_bus, 0xFFFC)
    hi, new_bus = peek(new_bus, 0xFFFD)
    reset_pc = lo | (hi << 8)
    new_cpu = fresh_cpu._replace(PC=jnp.uint16(reset_pc))
    return Console(cpu=new_cpu, bus=new_bus)


def console_step(console: Console) -> Console:
    """Execute one CPU instruction (with the usual TIA/RIOT post-step)."""
    new_cpu, new_bus = cpu_step(console.cpu, console.bus)
    return Console(cpu=new_cpu, bus=new_bus)


def run_until_frame(console: Console) -> Console:
    """Run one xitari `mediaSource().update()`: step the CPU until the TIA's
    frame counter advances by one (a VSYNC-clear hold-gate or the poke-time
    max-scanlines cutoff — see `tia_poke`), OR until the 25000-instruction
    budget is exhausted.

    Task #106 (partial-frame model): the budget is xitari's
    `m6502().execute(25000)`. When it runs out WITHOUT a frame boundary, this
    returns a *grey* frame — the frame counter has NOT advanced and the TIA's
    beam/scanline/cycle state is preserved, so the next `run_until_frame`
    continues the same TIA frame (xitari leaves `myPartialFrameFlag` true and
    skips `startFrame()`). This reproduces qbert's boot→step "sliver" frame.
    Mirror of jutari `run_until_frame!`.
    """
    start_frame = int(console.bus.tia.frame)
    cpu, bus = console.cpu, console.bus
    for _ in range(_UPDATE_INSTRUCTION_BUDGET):
        cpu, bus = cpu_step(cpu, bus)
        if int(bus.tia.frame) != start_frame:
            return Console(cpu=cpu, bus=bus)   # frame completed (endFrame)
    return Console(cpu=cpu, bus=bus)           # grey frame: execute(25000) budget
