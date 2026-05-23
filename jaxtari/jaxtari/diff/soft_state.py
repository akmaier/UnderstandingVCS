"""SoftCPUState + SoftBus — parallel float-valued state for SOFT-mode execution.

The HARD-mode `CPUState` and `Bus` are kept untouched; SOFT mode runs
against these parallel types so neither the HARD test suite nor the
existing CPU dispatch needs to change. The two paths share nothing
runtime-wise — only the *primitives* in `jaxtari.diff` and the address
constants from the HARD modules.

`SoftCPUState` carries `A, X, Y, SP, PC, P, cycles` as `float32`
scalars so `jax.grad` can flow through arithmetic on them. The status
flags are packed in `P` as a single float (8-bit semantics, but value
is computed via float arithmetic for differentiability).

`SoftBus` carries the 128 B RIOT RAM as `(128,)` float32 and the ROM
as an `(N,)` array of any numeric dtype — peeks convert to float32 on
the fly via the `RomTensor`-style one-hot dot product. The cart is
**not** wrapped as a `RomTensor` object here because that's a Python
class, not a PyTree; carrying the raw `jnp.ndarray` keeps `SoftBus`
a clean NamedTuple that `jax.grad` can traverse.

TIA / RIOT region writes are out of scope for P7b — soft_step
currently only handles instructions whose visible effect is on CPU
registers and/or RAM.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class SoftCPUState(NamedTuple):
    """CPU registers as float32 scalars.

    The status flag byte `P` keeps the canonical 8-bit layout
    (NV-BDIZC) so PHP / PLP / RTI continue to push and pull the same
    byte the HARD CPU does. **P7c-dx** adds four parallel scalar
    fields `P_N` / `P_Z` / `P_C` / `P_V` that mirror the same flag
    bits as floats — that's the differentiability hook for control
    flow: `_do_branch` reads its predicate out of the matching float
    field and feeds it to `soft_branch` so the branch's PC selection
    becomes a sigmoid blend (with `straight_through` keeping the
    forward result bit-exact). All four default to 0.0 so older
    constructions that built `SoftCPUState(..., P=0x34)` keep working
    — 0x34 has N=Z=C=V=0 anyway, and any flag-touching opcode
    immediately re-syncs both the packed `P` and the floats via
    `_with_p` in `soft_step`.
    """
    A: jnp.ndarray
    X: jnp.ndarray
    Y: jnp.ndarray
    SP: jnp.ndarray
    PC: jnp.ndarray
    P: jnp.ndarray
    cycles: jnp.ndarray
    P_N: jnp.ndarray = jnp.float32(0.0)
    P_Z: jnp.ndarray = jnp.float32(0.0)
    P_C: jnp.ndarray = jnp.float32(0.0)
    P_V: jnp.ndarray = jnp.float32(0.0)


def initial_soft_cpu_state(pc: float = 0xF000) -> SoftCPUState:
    """SOFT-mode CPU state after RESET. PC defaults to \$F000 — the
    standard cart-mapped reset entry point — so a fresh ROM at offset 0
    starts executing immediately. P=0x34 (I=1, B=1, U=1, N=Z=C=V=0)
    matches HARD reset; the float-flag mirrors are zeroed accordingly.
    """
    return SoftCPUState(
        A=jnp.float32(0.0),
        X=jnp.float32(0.0),
        Y=jnp.float32(0.0),
        SP=jnp.float32(0xFD),
        PC=jnp.float32(pc),
        P=jnp.float32(0x34),                # I=1, B=1, U=1 — matches HARD reset
        cycles=jnp.float32(0.0),
        P_N=jnp.float32(0.0),
        P_Z=jnp.float32(0.0),
        P_C=jnp.float32(0.0),
        P_V=jnp.float32(0.0),
    )


class SoftBus(NamedTuple):
    """SOFT-mode bus: float-valued RAM + raw ROM array, plus the
    minimal RIOT timer state introduced in **P8-cx**.

    The ROM is the differentiability target: `jax.grad` of any output
    that depends on a `soft_rom_peek(bus.rom, addr)` call flows back
    here, one-hot at the accessed address.

    P8-cx adds four scalar fields modelling the RIOT M6532 interval
    timer — enough to get past the standard "load TIM*T → poll INTIM"
    boot pattern that stalled the SOFT execution of real ROMs (Pong's
    init being the headline case). All four default to inert values so
    existing tests / SoftBus constructions are not affected — the
    timer only matters once a program writes TIM1T/TIM8T/TIM64T/TIM1024T.
    """
    ram: jnp.ndarray    # (128,) float32
    rom: jnp.ndarray    # (N,)   any numeric dtype — float32 conversion at peek
    # P8-cx RIOT timer (defaults are inert; first TIM*T write activates):
    riot_intim: jnp.ndarray = jnp.float32(0.0)
    riot_prescaler_shift: jnp.ndarray = jnp.float32(0.0)   # 0/3/6/10 → 1/8/64/1024×
    riot_residual_cycles: jnp.ndarray = jnp.float32(0.0)
    riot_expired: jnp.ndarray = jnp.float32(0.0)           # latch, 1.0 = INTIM reached 0


def initial_soft_bus(rom: jnp.ndarray) -> SoftBus:
    """Build a `SoftBus` with all-zero RAM and the given ROM. The ROM
    may be any byte length the cart formats support (2K/4K/8K/16K/32K);
    `soft_rom_peek` handles the 13-bit address mirror. RIOT timer
    fields default to inert; they activate the moment a program writes
    TIM*T (see P8-cx in `soft_step`).
    """
    return SoftBus(
        ram=jnp.zeros((128,), dtype=jnp.float32),
        rom=jnp.asarray(rom),
    )
