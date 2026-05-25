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


# Hard ceiling on the number of CPU instructions allowed inside one
# `run_until_frame`. ~30k is well above the natural 19,912 CPU cycles per
# NTSC frame; a ROM that exceeds it has either crashed or is in a tight
# loop ignoring VSYNC. (Tried bumping to 1M during the Q*Bert
# investigation — task #52 — but Q*Bert's RESET-pressed self-test
# doesn't toggle VSYNC for arbitrarily long, so a higher limit just
# delays the same failure. The real fix is in env_reset's RESET
# handling for games that treat console RESET as "enter test mode",
# not the per-frame budget.)
_FRAME_INSTRUCTION_LIMIT = 100_000


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
    """Step the CPU until the TIA's frame counter advances by one.

    The frame counter increments either on a software VSYNC falling edge
    (the standard Atari pattern — see P3f) or, as a safety fallback,
    when the scanline counter wraps past 262.
    """
    start_frame = int(console.bus.tia.frame)
    cpu, bus = console.cpu, console.bus
    for _ in range(_FRAME_INSTRUCTION_LIMIT):
        cpu, bus = cpu_step(cpu, bus)
        if int(bus.tia.frame) != start_frame:
            return Console(cpu=cpu, bus=bus)
    raise RuntimeError(
        f"run_until_frame exceeded {_FRAME_INSTRUCTION_LIMIT} instructions "
        f"without a frame boundary (start_frame={start_frame})."
    )
