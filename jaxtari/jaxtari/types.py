"""Shared state types for jaxtari.

P0 defines only `CPUState`. `BusState`, `TIAState`, `RIOTState`, and
`ConsoleState` will be added in their respective phases (see PORTING_PLAN.md §3.1).

CPUState is a NamedTuple so it is a JAX PyTree by default and works inside
`jax.lax.scan` / `jax.jit` without extra registration.
"""

from typing import NamedTuple

import jax.numpy as jnp


class CPUState(NamedTuple):
    A: jnp.uint8       # accumulator
    X: jnp.uint8       # X index register
    Y: jnp.uint8       # Y index register
    SP: jnp.uint8      # stack pointer (low byte; stack lives at 0x0100 + SP)
    PC: jnp.uint16     # program counter
    P: jnp.uint8       # processor status register (NV-BDIZC); bit 5 always 1
    cycles: jnp.uint64 # cumulative cycle counter (wide to avoid wrap in long runs)


def initial_cpu_state() -> CPUState:
    """6502 state immediately after RESET (PC will be loaded from $FFFC/$FFFD by the bus)."""
    return CPUState(
        A=jnp.uint8(0),
        X=jnp.uint8(0),
        Y=jnp.uint8(0),
        SP=jnp.uint8(0xFD),
        PC=jnp.uint16(0),
        P=jnp.uint8(0x34),  # I=1, B=1, unused=1
        cycles=jnp.uint64(0),
    )
