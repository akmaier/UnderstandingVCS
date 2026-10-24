"""Full-system SOFT step — the one-step map Phi_S of Theorem 1 on the WHOLE
machine state.

`soft_step` alone is a differentiable CPU core over a simplified bus (one ROM
window, TIA / RIOT folded into RAM); on a real ROM it drifts from the hard
machine within ~1k instructions. `soft_system_step` joins it to the bit-exact
hard console with the straight-through estimator (paper Eq. "ste"), exactly as
Fig. "pipeline" draws it:

  1. the HARD console executes the instruction (CPU, bus, TIA, RIOT, cart) —
     this is the forward value, bit-exact by construction;
  2. the SOFT handlers re-execute it against the same machine (the
     full-system context fields on `SoftBus`): ROM bytes through the hard
     cart's current bank / slice mapping, TIA / RIOT / Superchip reads
     replayed from the hard bus as constants;
  3. every CPU register and RAM cell is returned as STE(soft, hard) =
     soft + sg(hard - soft): forward = hard, gradient = d(soft)/d(inputs).

TIA, RIOT, cartridge and frame-buffer state live in the hard console and
carry no gradient. Gradients reach registers, RAM and ROM bytes — the
quantities the paper differentiates.

jaxtari's hard `Console` is a functional NamedTuple (the cart object is the
one mutable piece), so unlike jutari's mutating `soft_system_step(state, bus,
console)` the jaxtari functions RETURN the advanced console. The hard step is
eager Python (not traceable); it is not a function of the traced soft inputs,
so `jax.grad` of a Python loop of `soft_system_step` works. Mirror of jutari
`src/diff/SoftSystem.jl`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxtari.bus.system import read_log_enable, read_log_take
from jaxtari.cart.system import cart_rom_index
from jaxtari.console import Console, console_step
from jaxtari.diff.soft_state import IO_LOG_SLOTS, SoftBus, SoftCPUState
from jaxtari.diff.soft_step import soft_step

_CYCLE_MASK = 0xFFFFFF       # cycles kept mod 2^24 — exact in float32

# One compiled soft step serves every call (the context lives in SoftBus
# array fields, so changing bank / IO log does not retrace). Differentiable:
# `jax.grad` goes straight through `jax.jit`.
_soft_step_jit = jax.jit(soft_step)


def _is_ram_addr(a: int) -> bool:
    return (a & 0x1000) == 0 and (a & 0x80) != 0 and (a & 0x200) == 0


def _hard_snapshot(con: Console) -> dict:
    """Hard values as Python ints / a numpy RAM vector."""
    c = con.cpu
    return dict(A=int(c.A), X=int(c.X), Y=int(c.Y), SP=int(c.SP),
                PC=int(c.PC), P=int(c.P), cycles=int(c.cycles) & _CYCLE_MASK,
                ram=np.asarray(con.bus.ram, dtype=np.float32))


def _hard_step_with_io_log(con: Console):
    """Run the hard instruction, recording the values of its non-RAM,
    non-ROM reads. Returns `(new_console, ctx, hard_snapshot)` where `ctx`
    is the dict of SoftBus context fields for the soft re-execution."""
    cart = con.bus.cart
    kind, bank = int(cart.kind), int(cart.current_bank)
    slices = [int(s) for s in cart.slice_slots] or [0, 0, 0]
    prev = read_log_enable()
    try:
        con = console_step(con)
    finally:
        events = read_log_take(prev)
    io = [(a, v) for (a, v) in events
          if not _is_ram_addr(a)
          and ((a & 0x1000) == 0 or cart_rom_index(kind, bank, slices, a) < 0)]
    if len(io) > IO_LOG_SLOTS:
        raise RuntimeError(f"{len(io)} I/O reads in one instruction "
                           f"exceed IO_LOG_SLOTS={IO_LOG_SLOTS}")
    addr = np.zeros(IO_LOG_SLOTS, dtype=np.float32)
    val = np.zeros(IO_LOG_SLOTS, dtype=np.float32)
    valid = np.zeros(IO_LOG_SLOTS, dtype=np.float32)
    for i, (a, v) in enumerate(io):
        addr[i], val[i], valid[i] = a, v & 0xFF, 1.0
    ctx = dict(sys_on=jnp.float32(1.0), sys_kind=jnp.float32(kind),
               sys_bank=jnp.float32(bank),
               sys_slices=jnp.asarray(slices, dtype=jnp.float32),
               io_addr=jnp.asarray(addr), io_val=jnp.asarray(val),
               io_valid=jnp.asarray(valid))
    return con, ctx, _hard_snapshot(con)


# Inert context values (SoftBus defaults) restored on the returned bus.
_CTX_OFF = dict(
    sys_on=SoftBus._field_defaults["sys_on"],
    sys_kind=SoftBus._field_defaults["sys_kind"],
    sys_bank=SoftBus._field_defaults["sys_bank"],
    sys_slices=SoftBus._field_defaults["sys_slices"],
    io_addr=SoftBus._field_defaults["io_addr"],
    io_val=SoftBus._field_defaults["io_val"],
    io_valid=SoftBus._field_defaults["io_valid"],
)


def _ste(soft, hard):
    return soft + jax.lax.stop_gradient(jnp.float32(hard) - soft)


def _soft_and_hard(state: SoftCPUState, bus: SoftBus, con: Console):
    con, ctx, h = _hard_step_with_io_log(con)
    s, b = _soft_step_jit(state, bus._replace(**ctx))
    return s, b._replace(**_CTX_OFF), con, h


def _join(s: SoftCPUState, b: SoftBus, h: dict):
    p = h["P"]
    new_state = SoftCPUState(
        A=_ste(s.A, h["A"]), X=_ste(s.X, h["X"]), Y=_ste(s.Y, h["Y"]),
        SP=_ste(s.SP, h["SP"]), PC=_ste(s.PC, h["PC"]), P=_ste(s.P, p),
        cycles=_ste(s.cycles, h["cycles"]),
        P_N=_ste(s.P_N, (p >> 7) & 1), P_Z=_ste(s.P_Z, (p >> 1) & 1),
        P_C=_ste(s.P_C, p & 1), P_V=_ste(s.P_V, (p >> 6) & 1))
    hard_ram = jnp.asarray(h["ram"])
    new_ram = b.ram + jax.lax.stop_gradient(hard_ram - b.ram)
    return new_state, b._replace(ram=new_ram)


def soft_system_init(con: Console):
    """`(SoftCPUState, SoftBus)` mirroring a hard console at any point (e.g.
    a live game state after `StellaEnvironment.reset` / `step`). The bus
    carries the full ROM image as float32, so gradients reach every bank."""
    h = _hard_snapshot(con)
    p = h["P"]
    state = SoftCPUState(
        A=jnp.float32(h["A"]), X=jnp.float32(h["X"]), Y=jnp.float32(h["Y"]),
        SP=jnp.float32(h["SP"]), PC=jnp.float32(h["PC"]), P=jnp.float32(p),
        cycles=jnp.float32(h["cycles"]),
        P_N=jnp.float32((p >> 7) & 1), P_Z=jnp.float32((p >> 1) & 1),
        P_C=jnp.float32(p & 1), P_V=jnp.float32((p >> 6) & 1))
    bus = SoftBus(ram=jnp.asarray(h["ram"]),
                  rom=jnp.asarray(np.asarray(con.bus.cart.rom), dtype=jnp.float32))
    return state, bus


def soft_system_step(state: SoftCPUState, bus: SoftBus, con: Console):
    """One instruction of the full machine in SOFT mode.

    Returns `(new_state, new_bus, new_console)`: the hard console advanced
    one instruction, and the soft state whose forward value equals the hard
    registers + RAM bit-for-bit (Theorem 1) and whose gradient is the soft
    handlers' surrogate (Corollary 1). Differentiable w.r.t. `state` and
    `bus` (ROM and RAM). Note: the cart object inside `con` is mutable and
    shared with the returned console (as everywhere in jaxtari), so `con`
    must not be stepped again after this call.
    """
    s, b, con, h = _soft_and_hard(state, bus, con)
    new_state, new_bus = _join(s, b, h)
    return new_state, new_bus, con


def soft_system_run(state: SoftCPUState, bus: SoftBus, con: Console, n_steps: int):
    """`n_steps` full-system soft instructions (Python loop — the hard step
    is eager). Returns `(state, bus, console)`."""
    for _ in range(n_steps):
        state, bus, con = soft_system_step(state, bus, con)
    return state, bus, con


def soft_handler_agreement(state: SoftCPUState, bus: SoftBus, con: Console):
    """Diagnostic: run one full-system step and report whether the SOFT
    handlers alone (before the STE join) already produced the hard registers
    (A, X, Y, SP, PC, P) and RAM. When True the STE correction is the
    identity in value and the gradient is taken through a computation that
    reproduces the hard step exactly.

    Returns `(agree, new_state, new_bus, new_console)` — the last three are
    exactly `soft_system_step`'s outputs, so a caller can count
    disagreements while advancing the run.
    """
    s, b, con, h = _soft_and_hard(state, bus, con)
    regs = all(float(getattr(s, k)) == float(h[k])
               for k in ("A", "X", "Y", "SP", "PC", "P"))
    agree = bool(regs and np.array_equal(np.asarray(b.ram), h["ram"]))
    new_state, new_bus = _join(s, b, h)
    return agree, new_state, new_bus, con
