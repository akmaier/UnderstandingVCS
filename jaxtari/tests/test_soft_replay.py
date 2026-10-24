"""Record / replay full-system SOFT step (`jaxtari.diff.soft_replay`).

`record_soft_trace` runs the hard console once and stores what the soft
handlers need; `soft_system_replay` re-runs them over the trace with
`lax.scan` (pure, so `jit` / `vmap` / `grad` apply). These tests check that
the replay is the eager `soft_system_run` in another form:

  1. forward: identical to the eager path and to the hard console, on a 2K
     ROM, a bank-switched one (E0 slices) and an F8SC one whose program reads
     the Superchip RAM uninitialised (elevator_action: the replay must carry
     the hard cart's LCG power-on bytes, bug_fix_log #123 / bef47885, through
     the recorded reads);
  2. gradient w.r.t. the ROM: equal to the eager gradient. The loss is summed
     over the whole trajectory (`collect`): under the STE the ROM gradient
     enters only where a soft handler computes with a ROM byte, so a
     final-state loss right after boot is near-empty (0-8 non-zeros) and would
     make an equality check vacuous; the trajectory loss has >= 10;
  3. `checkpoint_every` (segmented remat) changes neither values nor grads;
  4. `collect` returns the per-instruction forward, which is the hard trace;
  5. `vmap` across two games of different ROM size (`pad_rom`) equals the
     single-game replays, and the padded ROM tail gets zero gradient.

Each ROM is booted once per module (`_LIGHT_BOOT_FRAMES` NOOP frames; jaxtari
boots at seconds per frame) and the hard console is deep-copied so eager and
replay start from the same state.
"""

from __future__ import annotations

import copy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxtari.diff import (
    pad_rom,
    record_soft_trace,
    soft_system_init,
    soft_system_replay,
    soft_system_run,
    soft_system_step,
    stack_trees,
)
from jaxtari.env.stella_environment import StellaEnvironment
from jaxtari.games import (
    ElevatorActionRomSettings,
    MontezumaRevengeRomSettings,
    PongRomSettings,
)

_ROMS = Path(__file__).resolve().parents[2] / "tools" / "rom_sweep" / "roms"
_LIGHT_BOOT_FRAMES = 3
_N = 120
_REGS = ("A", "X", "Y", "SP", "PC", "P", "cycles", "P_N", "P_Z", "P_C", "P_V")


def _boot(name, settings_cls):
    path = _ROMS / f"{name}.bin"
    if not path.exists():
        pytest.skip(f"ROM {path} not available")
    rom = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    env = StellaEnvironment(rom, settings_cls())
    env.reset(boot_noop_steps=_LIGHT_BOOT_FRAMES)
    return env.console


@pytest.fixture(scope="module")
def pong():
    return _boot("pong", PongRomSettings)


@pytest.fixture(scope="module")
def montezuma():
    return _boot("montezuma_revenge", MontezumaRevengeRomSettings)


@pytest.fixture(scope="module")
def elevator():
    return _boot("elevator_action", ElevatorActionRomSettings)


def _record(con0, n=_N):
    state, bus = soft_system_init(con0)
    trace, con = record_soft_trace(copy.deepcopy(con0), n)
    return state, bus, trace, con


def _assert_same_forward(sa, ba, sb, bb):
    for k in _REGS:
        assert float(getattr(sa, k)) == float(getattr(sb, k)), k
    np.testing.assert_array_equal(np.asarray(ba.ram), np.asarray(bb.ram))


_MIN_NNZ = 10


def _per_step(s, b):
    return jnp.sum(b.ram) + s.A + s.X + s.Y + s.PC


def _loss_replay(state, bus, trace, **kw):
    """Trajectory loss: per-step `_per_step` summed over all instructions."""
    def loss(rom):
        _, _, ys = soft_system_replay(state, bus._replace(rom=rom), trace,
                                      collect=_per_step, **kw)
        return jnp.sum(ys)
    return loss


@pytest.mark.parametrize("which", ["pong", "montezuma", "elevator"])
def test_replay_forward_equals_eager_and_hard(which, request):
    con0 = request.getfixturevalue(which)
    state, bus, trace, con_hard = _record(con0)
    sR, bR = jax.jit(soft_system_replay)(state, bus, trace)
    sE, bE, con_eager = soft_system_run(state, bus, copy.deepcopy(con0), _N)
    _assert_same_forward(sR, bR, sE, bE)
    c = con_hard.cpu
    assert [int(float(getattr(sR, k))) for k in ("A", "X", "Y", "SP", "PC", "P")] == \
           [int(c.A), int(c.X), int(c.Y), int(c.SP), int(c.PC), int(c.P)]
    np.testing.assert_array_equal(np.asarray(bR.ram),
                                  np.asarray(con_hard.bus.ram, dtype=np.float32))
    assert float(bR.sys_on) == 0.0          # context cleared, as in the eager path


def test_replay_grad_equals_eager(pong):
    state, bus, trace, _ = _record(pong)
    gR = np.asarray(jax.jit(jax.grad(_loss_replay(state, bus, trace)))(bus.rom))
    con = copy.deepcopy(pong)

    def loss_eager(rom):
        s, b, c, total = state, bus._replace(rom=rom), con, 0.0
        for _ in range(_N):
            s, b, c = soft_system_step(s, b, c)
            total = total + _per_step(s, b)
        return total

    gE = np.asarray(jax.grad(loss_eager)(bus.rom))
    assert np.count_nonzero(gR) >= _MIN_NNZ and np.all(np.isfinite(gR))
    np.testing.assert_allclose(gR, gE, rtol=1e-6, atol=1e-6)


def test_checkpointing_changes_nothing(pong):
    state, bus, trace, _ = _record(pong)
    s0, b0 = soft_system_replay(state, bus, trace)
    for k in (7, 40, _N + 5):               # remainder, even split, no split
        s1, b1 = soft_system_replay(state, bus, trace, checkpoint_every=k)
        _assert_same_forward(s0, b0, s1, b1)
    g0 = jax.grad(_loss_replay(state, bus, trace))(bus.rom)
    g1 = jax.grad(_loss_replay(state, bus, trace, checkpoint_every=7))(bus.rom)
    assert np.count_nonzero(np.asarray(g0)) >= _MIN_NNZ
    np.testing.assert_allclose(np.asarray(g1), np.asarray(g0), rtol=1e-6, atol=1e-6)


def test_collect_returns_the_hard_trajectory(pong):
    state, bus, trace, _ = _record(pong)
    collect = lambda s, b: (s.PC, s.A, b.ram)
    _, _, (pc, a, ram) = soft_system_replay(state, bus, trace, collect=collect,
                                            checkpoint_every=50)
    np.testing.assert_array_equal(np.asarray(pc), np.asarray(trace.PC))
    np.testing.assert_array_equal(np.asarray(a), np.asarray(trace.A))
    np.testing.assert_array_equal(np.asarray(ram), np.asarray(trace.ram))


def test_vmap_across_games_with_padded_roms(pong, montezuma):
    n = 80
    runs = [_record(c, n) for c in (pong, montezuma)]
    width = max(int(r[1].rom.shape[0]) for r in runs)
    assert width > int(runs[0][1].rom.shape[0])   # really mixes ROM sizes
    states = stack_trees([r[0] for r in runs])
    buses = stack_trees([pad_rom(r[1], width) for r in runs])
    traces = stack_trees([r[2] for r in runs])

    sB, bB = jax.jit(jax.vmap(soft_system_replay))(states, buses, traces)
    for i, (state, bus, trace, _) in enumerate(runs):
        s1, b1 = soft_system_replay(state, bus, trace)
        _assert_same_forward(jax.tree.map(lambda v: v[i], sB),
                             jax.tree.map(lambda v: v[i], bB), s1, b1)

    def batch_loss(roms):
        one = lambda s, b, x: jnp.sum(soft_system_replay(s, b, x, collect=_per_step)[2])
        return jnp.sum(jax.vmap(one)(states, buses._replace(rom=roms), traces))

    gB = np.asarray(jax.jit(jax.grad(batch_loss))(buses.rom))
    for i, (state, bus, trace, _) in enumerate(runs):
        L = int(bus.rom.shape[0])
        g1 = np.asarray(jax.grad(_loss_replay(state, bus, trace))(bus.rom))
        assert np.count_nonzero(g1) > 0
        np.testing.assert_allclose(gB[i, :L], g1, rtol=1e-6, atol=1e-6)
        assert not np.any(gB[i, L:])        # padded tail: zero gradient
