"""Record / replay full-system SOFT step — `jit`, `vmap` and `grad` on device.

`soft_system_step` (jaxtari.diff.soft_system) runs the hard console on every
instruction. The hard console (bank switching, TIA, RIOT) is eager Python and
cannot be traced, so a loop of `soft_system_step` cannot be `jit`-compiled or
`vmap`ped and returns to the host on every instruction; only the small soft
step inside it is compiled.

But the hard machine reaches the soft handlers only through recorded data:
the cart mapping in force (kind / bank / E0 slices), the values of the
non-RAM, non-ROM reads (TIA / RIOT / Superchip), and the hard registers + RAM
used in the straight-through join. None of it depends on the traced soft
inputs (the STE forward value is the hard value whatever the ROM / RAM
perturbation). So the full-system step splits into

  1. `record_soft_trace` — run the hard console once (eager, CPU) and store
     that data per instruction in a `SoftTrace` (arrays with a leading time
     axis);
  2. `soft_system_replay` — re-run the soft handlers + STE join over the
     trace with `jax.lax.scan`. This is a pure array function of
     `(state, bus, trace)`, so `jax.jit`, `jax.vmap` and `jax.grad` all apply,
     on CPU or GPU, and long windows can be rematerialised
     (`checkpoint_every`).

Replaying a trace computes the same values and the same gradient as the eager
`soft_system_run` over the same instructions (tests/test_soft_replay.py).
A trace is tied to the hard trajectory it was recorded on; re-record when the
trajectory itself must change (new start state, actions, or a different
program run on the hard machine).

Placement and speed (measured, tools/bench_soft_system_replay.py): one trajectory
replays fastest on the CPU (a sequential scan is launch-latency bound on a GPU);
large batches (`vmap`) belong on the GPU. On a GPU host, boot the environment and
record under `with jax.default_device(jax.devices("cpu")[0]):` — the eager hard
console issues many tiny array ops, and dispatching them to the GPU made recording
~40x slower (53-59 instr/s on V100 / Q5000 / RTX A6000 nodes vs 2,306 instr/s on an
M1 Max CPU). The replay `jit` then runs on the default device.

Batching across games: SoftBus shapes must agree under `vmap`; the RAM is
always 128 B, ROMs differ in size. With the full-system context on, every ROM
read uses the context's explicit index (`bank * 0x1000 + ...`, E0 / FE
decodes), always inside the true ROM, so zero-padding ROMs to a common length
(`pad_rom`) changes no value; the padded tail receives zero gradient.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxtari.console import Console
from jaxtari.diff.soft_state import IO_LOG_SLOTS, SoftBus, SoftCPUState
from jaxtari.diff.soft_step import soft_step
from jaxtari.diff.soft_system import _CTX_OFF, _hard_step_record


class SoftTrace(NamedTuple):
    """Per-instruction hard-machine data for `soft_system_replay`. Every field
    has a leading time axis T (a batch axis in front under `vmap`); all are
    float32 holding exact small integers."""
    # Context the soft handlers run against (SoftBus `sys_*` / `io_*` fields).
    kind: jnp.ndarray       # (T,)
    bank: jnp.ndarray       # (T,)
    slices: jnp.ndarray     # (T, 3)
    io_addr: jnp.ndarray    # (T, IO_LOG_SLOTS)
    io_val: jnp.ndarray     # (T, IO_LOG_SLOTS)
    io_valid: jnp.ndarray   # (T, IO_LOG_SLOTS)
    # Hard values after the instruction (the STE forward values).
    A: jnp.ndarray          # (T,)
    X: jnp.ndarray
    Y: jnp.ndarray
    SP: jnp.ndarray
    PC: jnp.ndarray
    P: jnp.ndarray
    cycles: jnp.ndarray
    P_N: jnp.ndarray
    P_Z: jnp.ndarray
    P_C: jnp.ndarray
    P_V: jnp.ndarray
    ram: jnp.ndarray        # (T, 128)


def record_soft_trace(con: Console, n_steps: int):
    """Run the hard console `n_steps` instructions and record what the soft
    handlers need. Returns `(trace, new_console)`.

    Same caveat as `soft_system_step`: the cart object inside `con` is
    mutable and shared with the returned console, so do not step `con` again
    afterwards. Pair the trace with the soft state taken BEFORE recording,
    e.g. `state, bus = soft_system_init(con); trace, con = record_soft_trace(con, n)`.
    """
    n = int(n_steps)
    kind = np.zeros(n, np.float32)
    bank = np.zeros(n, np.float32)
    slices = np.zeros((n, 3), np.float32)
    io_addr = np.zeros((n, IO_LOG_SLOTS), np.float32)
    io_val = np.zeros((n, IO_LOG_SLOTS), np.float32)
    io_valid = np.zeros((n, IO_LOG_SLOTS), np.float32)
    regs = {k: np.zeros(n, np.float32) for k in ("A", "X", "Y", "SP", "PC", "P", "cycles")}
    ram = np.zeros((n, 128), np.float32)
    for t in range(n):
        con, r = _hard_step_record(con)
        kind[t], bank[t] = r["kind"], r["bank"]
        slices[t] = r["slices"]
        io_addr[t], io_val[t], io_valid[t] = r["io_addr"], r["io_val"], r["io_valid"]
        h = r["hard"]
        for k in regs:
            regs[k][t] = h[k]
        ram[t] = h["ram"]
    p = regs["P"].astype(np.int64)
    trace = SoftTrace(
        kind=kind, bank=bank, slices=slices,
        io_addr=io_addr, io_val=io_val, io_valid=io_valid,
        **regs,
        P_N=((p >> 7) & 1).astype(np.float32), P_Z=((p >> 1) & 1).astype(np.float32),
        P_C=(p & 1).astype(np.float32), P_V=((p >> 6) & 1).astype(np.float32),
        ram=ram)
    return jax.tree.map(jnp.asarray, trace), con


def _ste(soft, hard):
    return soft + jax.lax.stop_gradient(hard - soft)


def _replay_step(state: SoftCPUState, bus: SoftBus, x: SoftTrace):
    """One full-system soft instruction from recorded data — the traced twin
    of `soft_system_step` (`_soft_and_hard` + `_join`)."""
    ctx = dict(sys_on=jnp.float32(1.0), sys_kind=x.kind, sys_bank=x.bank,
               sys_slices=x.slices, io_addr=x.io_addr, io_val=x.io_val,
               io_valid=x.io_valid)
    s, b = soft_step(state, bus._replace(**ctx))
    b = b._replace(**_CTX_OFF)
    new_state = SoftCPUState(
        A=_ste(s.A, x.A), X=_ste(s.X, x.X), Y=_ste(s.Y, x.Y),
        SP=_ste(s.SP, x.SP), PC=_ste(s.PC, x.PC), P=_ste(s.P, x.P),
        cycles=_ste(s.cycles, x.cycles),
        P_N=_ste(s.P_N, x.P_N), P_Z=_ste(s.P_Z, x.P_Z),
        P_C=_ste(s.P_C, x.P_C), P_V=_ste(s.P_V, x.P_V))
    new_ram = b.ram + jax.lax.stop_gradient(x.ram - b.ram)
    return new_state, b._replace(ram=new_ram)


def _normalize(state: SoftCPUState, bus: SoftBus):
    """Cast every leaf to float32 so the scan carry has a fixed type (the
    SoftBus / SoftCPUState defaults are float32 already; this guards against
    caller-built states with Python floats or integer ROM dtypes)."""
    f = lambda v: jnp.asarray(v, dtype=jnp.float32)
    return jax.tree.map(f, state), jax.tree.map(f, bus)


def soft_system_replay(state: SoftCPUState, bus: SoftBus, trace: SoftTrace, *,
                       collect: Callable | None = None,
                       checkpoint_every: int | None = None):
    """Replay the full-system SOFT step over a recorded `trace`.

    Pure function of `(state, bus, trace)`: wrap it in `jax.jit`, `jax.vmap`
    (batch axis on all three) and `jax.grad` freely. Forward value equals the
    hard console after every instruction (Theorem 1); the gradient is the
    soft handlers' surrogate (Corollary 1), identical to the eager
    `soft_system_run` over the same instructions.

    `collect(state, bus) -> pytree`, if given, is evaluated after every
    instruction and the results are returned stacked along a leading time
    axis (for losses over a whole trajectory); the return value is then
    `(state, bus, collected)`, otherwise `(state, bus)`.

    `checkpoint_every=k` rematerialises the scan in segments of `k`
    instructions (`jax.checkpoint`): reverse-mode memory drops from O(T) step
    residuals to O(T/k + k) at the cost of one extra forward pass. The
    returned values and gradients are unchanged.
    """
    state, bus = _normalize(state, bus)

    def body(carry, x):
        s, b = _replay_step(*carry, x)
        y = collect(s, b) if collect is not None else None
        return (s, b), y

    T = trace.kind.shape[0]
    if not checkpoint_every or checkpoint_every >= T:
        (state, bus), ys = jax.lax.scan(body, (state, bus), trace)
    else:
        k = int(checkpoint_every)
        n_seg, tail = divmod(T, k)
        head = jax.tree.map(lambda a: a[:n_seg * k].reshape((n_seg, k) + a.shape[1:]), trace)

        @jax.checkpoint
        def segment(carry, xs):
            return jax.lax.scan(body, carry, xs)

        (state, bus), ys = jax.lax.scan(segment, (state, bus), head)
        if ys is not None:
            ys = jax.tree.map(lambda a: a.reshape((n_seg * k,) + a.shape[2:]), ys)
        if tail:
            rest = jax.tree.map(lambda a: a[n_seg * k:], trace)
            (state, bus), ys_tail = jax.lax.scan(body, (state, bus), rest)
            if ys is not None:
                ys = jax.tree.map(lambda a, b: jnp.concatenate([a, b]), ys, ys_tail)
    if collect is not None:
        return state, bus, ys
    return state, bus


def pad_rom(bus: SoftBus, n: int) -> SoftBus:
    """Zero-pad `bus.rom` to length `n` so buses of different games can be
    stacked for `vmap`. Value-exact for the full-system replay (every ROM read
    uses the context's explicit index inside the true ROM); the padded tail
    receives zero gradient — slice `grad[:original_len]` to drop it."""
    rom = jnp.asarray(bus.rom, dtype=jnp.float32)
    if rom.shape[0] > n:
        raise ValueError(f"ROM of {rom.shape[0]} bytes does not fit in {n}")
    return bus._replace(rom=jnp.pad(rom, (0, n - rom.shape[0])))


def stack_trees(items):
    """Stack a list of equally-shaped pytrees (states, buses or traces) along a
    new leading batch axis, e.g. for `jax.vmap(soft_system_replay)`."""
    return jax.tree.map(lambda *xs: jnp.stack(xs), *items)
