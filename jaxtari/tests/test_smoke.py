"""P0 smoke tests — verify the scaffolding imports and the step stub advances PC."""

import jax.numpy as jnp

import jaxtari
from jaxtari.cpu.m6502 import step
from jaxtari.cpu.tables import ADDRESSING_MODE_TABLE, CYCLE_TABLE, FLAG_I, FLAG_B
from jaxtari.diff.modes import Mode, current_mode, set_mode, using_mode
from jaxtari.types import CPUState, initial_cpu_state


def test_package_version():
    assert jaxtari.__version__ == "0.0.1"


def test_opcode_tables_are_256_entries():
    assert ADDRESSING_MODE_TABLE.shape == (256,)
    assert CYCLE_TABLE.shape == (256,)


def test_initial_cpu_state_matches_reset_convention():
    s = initial_cpu_state()
    assert int(s.SP) == 0xFD
    assert (int(s.P) & FLAG_I) != 0
    assert (int(s.P) & FLAG_B) != 0


def test_step_stub_advances_pc_and_cycles():
    memory = jnp.zeros((1 << 16,), dtype=jnp.uint8).at[0].set(0xEA)  # NOP at $0000
    state = initial_cpu_state()
    state2, memory2 = step(state, memory)
    assert int(state2.PC) == 1
    assert int(state2.cycles) == int(CYCLE_TABLE[0xEA])
    assert memory2 is memory or jnp.array_equal(memory2, memory)


def test_mode_default_is_hard_and_context_manager_restores():
    assert current_mode() is Mode.HARD
    with using_mode(Mode.SOFT):
        assert current_mode() is Mode.SOFT
    assert current_mode() is Mode.HARD


def test_set_mode_persists():
    set_mode(Mode.SOFT)
    try:
        assert current_mode() is Mode.SOFT
    finally:
        set_mode(Mode.HARD)
