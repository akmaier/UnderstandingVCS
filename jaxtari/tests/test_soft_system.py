"""Full-system SOFT step (`jaxtari.diff.soft_system`) — Theorem 1 on the
whole machine state.

`soft_system_step` runs the hard instruction (bit-exact forward) and
re-executes it with the soft handlers against the same machine (banked ROM
via the hard cart mapping, TIA / RIOT / Superchip reads replayed from the
hard bus), then joins the two with the straight-through estimator. These
tests check, on real ROMs covering every cart mapper the context decodes
(2K, 4K, F8SC, E0, FE, F6/F8 bank switching, code executing from RAM):

  1. forward == hard console on every step (all registers + 128 B RAM);
  2. the soft handlers ALONE already agree with the hard step (the STE
     correction is the identity), i.e. the gradient is taken through a
     computation that reproduces the hard machine;
  3. `jax.grad` of sum(RAM) + A after ~100 full-system steps on Pong w.r.t.
     the ROM is finite and non-zero.

jaxtari boots at ~7 s per frame, so by default each ROM gets a short boot
(`_LIGHT_BOOT_FRAMES` NOOP frames) and `_LIGHT_INSTR` instructions. Set
`JAXTARI_SOFT_SYSTEM_FULL=1` for the paper-scale run: standard env reset
(60 NOOP + 4 RESET frames), 20 NOOP steps, 1000 instructions (~10 min/ROM).
Mirror of jutari's soft-system tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxtari.diff import (
    soft_handler_agreement,
    soft_system_init,
    soft_system_run,
    soft_system_step,
)
from jaxtari.env.stella_environment import StellaEnvironment
from jaxtari.games import (
    BreakoutRomSettings,
    ElevatorActionRomSettings,
    GenericRomSettings,
    MontezumaRevengeRomSettings,
    PongRomSettings,
    UpNDownRomSettings,
)

_ROMS = Path(__file__).resolve().parents[2] / "tools" / "rom_sweep" / "roms"
_FULL = os.environ.get("JAXTARI_SOFT_SYSTEM_FULL", "") not in ("", "0")
_LIGHT_BOOT_FRAMES = 3
_LIGHT_INSTR = 400

# (rom basename, RomSettings) — settings match tools/jaxtari_dump.py.
_CASES = [
    ("pong", PongRomSettings),                       # 2K
    ("breakout", BreakoutRomSettings),               # 2K, paddles
    ("montezuma_revenge", MontezumaRevengeRomSettings),  # E0 slices
    ("elevator_action", ElevatorActionRomSettings),  # F8SC (Superchip RAM)
    ("robotank", GenericRomSettings),                # FE (A13 banking)
    ("up_n_down", UpNDownRomSettings),               # F6/F8SC-class, RAM code
]


def _boot(name: str, settings_cls):
    path = _ROMS / f"{name}.bin"
    if not path.exists():
        pytest.skip(f"ROM {path} not available")
    rom = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    env = StellaEnvironment(rom, settings_cls())
    if _FULL:
        env.reset(boot_noop_steps=60, boot_reset_steps=4)
        for _ in range(20):
            env.step(0)
    else:
        env.reset(boot_noop_steps=_LIGHT_BOOT_FRAMES)
    return env.console


def _hard_regs(con):
    c = con.cpu
    return [int(c.A), int(c.X), int(c.Y), int(c.SP), int(c.PC), int(c.P),
            int(c.cycles) & 0xFFFFFF]


def _soft_regs(st):
    return [int(float(v)) for v in (st.A, st.X, st.Y, st.SP, st.PC, st.P, st.cycles)]


@pytest.mark.parametrize("name,settings_cls", _CASES, ids=[c[0] for c in _CASES])
def test_forward_equals_hard_and_handlers_agree(name, settings_cls):
    con = _boot(name, settings_cls)
    state, bus = soft_system_init(con)
    n = 1000 if _FULL else _LIGHT_INSTR
    disagreements = []
    for i in range(n):
        pc = int(con.cpu.PC)
        agree, state, bus, con = soft_handler_agreement(state, bus, con)
        if not agree:
            disagreements.append((i, hex(pc)))
        assert _soft_regs(state) == _hard_regs(con), f"{name}: registers diverge at step {i}"
        np.testing.assert_array_equal(
            np.asarray(bus.ram), np.asarray(con.bus.ram, dtype=np.float32),
            err_msg=f"{name}: RAM diverges at step {i}")
    assert disagreements == [], f"{name}: soft handlers disagree at {disagreements[:10]}"


def test_soft_system_step_matches_agreement_path():
    """`soft_system_step` / `soft_system_run` advance exactly like the
    agreement diagnostic (same forward state, same hard console)."""
    con = _boot("pong", PongRomSettings)
    state, bus = soft_system_init(con)
    state, bus, con = soft_system_step(state, bus, con)
    state, bus, con = soft_system_run(state, bus, con, 20)
    assert _soft_regs(state) == _hard_regs(con)
    np.testing.assert_array_equal(np.asarray(bus.ram),
                                  np.asarray(con.bus.ram, dtype=np.float32))
    # The context is cleared on the returned bus (inert defaults).
    assert float(bus.sys_on) == 0.0


def test_grad_through_full_system_steps_reaches_rom():
    con0 = _boot("pong", PongRomSettings)
    state0, bus0 = soft_system_init(con0)
    # The hard console is eager Python and not a function of the traced ROM;
    # `jax.grad` evaluates `loss` once, so stepping `con0` inside it is fine.

    def loss(rom):
        s, b, _ = soft_system_run(state0, bus0._replace(rom=rom), con0, 100)
        return jnp.sum(b.ram) + s.A

    g = jax.grad(loss)(bus0.rom)
    g = np.asarray(g)
    assert np.all(np.isfinite(g))
    assert np.count_nonzero(g) > 0
