"""P7d tests — RomTensor as a JAX PyTree.

Before P7d the SoftBus had to carry a bare `jnp.ndarray` in its `rom`
slot: a plain Python class is opaque to `jax.grad`, so a `RomTensor`
wrapper could not be differentiated through. P7d registers `RomTensor`
as a JAX PyTree node — its single child is the `rom` array — so it can
sit directly in `SoftBus.rom` and `jax.grad` threads the cotangent
back out as a `RomTensor`.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxtari.diff import (
    RomTensor,
    SoftBus,
    initial_soft_cpu_state,
    soft_run,
    soft_step,
    soft_rom_peek,
)


def _rom_with(bytes_: list[int], size: int = 256) -> jnp.ndarray:
    rom = jnp.zeros((size,), dtype=jnp.float32)
    for i, b in enumerate(bytes_):
        rom = rom.at[i].set(jnp.float32(b))
    return rom


# --------------------------------------------------------------------------- #
# PyTree registration
# --------------------------------------------------------------------------- #

def test_rom_tensor_is_a_registered_pytree():
    rt = RomTensor(jnp.arange(8, dtype=jnp.float32))
    leaves = jax.tree_util.tree_leaves(rt)
    # The single leaf is the rom array.
    assert len(leaves) == 1
    assert jnp.array_equal(leaves[0], jnp.arange(8, dtype=jnp.float32))


def test_rom_tensor_flatten_unflatten_round_trips():
    rt = RomTensor(jnp.arange(16, dtype=jnp.float32))
    children, aux = rt.tree_flatten()
    rebuilt = RomTensor.tree_unflatten(aux, children)
    assert isinstance(rebuilt, RomTensor)
    assert jnp.array_equal(rebuilt.rom, rt.rom)


def test_rom_tensor_tree_map_applies_to_the_array():
    rt = RomTensor(jnp.ones((4,), dtype=jnp.float32))
    doubled = jax.tree_util.tree_map(lambda x: x * 2.0, rt)
    assert isinstance(doubled, RomTensor)
    assert jnp.array_equal(doubled.rom, jnp.full((4,), 2.0, dtype=jnp.float32))


def test_unflatten_bypasses_init_dtype_coercion():
    """tree_unflatten must store the child verbatim — no jnp.asarray /
    float32 cast — so a tracer or cotangent passes through untouched."""
    sentinel = jnp.arange(4, dtype=jnp.int32)   # deliberately not float32
    rebuilt = RomTensor.tree_unflatten(None, (sentinel,))
    assert rebuilt.rom.dtype == jnp.int32       # untouched by __init__


# --------------------------------------------------------------------------- #
# soft_rom_peek accepts a RomTensor
# --------------------------------------------------------------------------- #

def test_soft_rom_peek_accepts_rom_tensor():
    arr = jnp.array([10, 20, 30, 40], dtype=jnp.float32)
    rt = RomTensor(arr)
    # Same answer whether we pass the raw array or the wrapper.
    for addr in range(4):
        assert float(soft_rom_peek(arr, addr)) == float(soft_rom_peek(rt, addr))


# --------------------------------------------------------------------------- #
# A SoftBus carrying a RomTensor executes
# --------------------------------------------------------------------------- #

def test_soft_step_runs_with_rom_tensor_backed_bus():
    rt = RomTensor(_rom_with([0xA9, 0x42]))     # LDA #$42
    bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rt)
    state = initial_soft_cpu_state()
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x42


def test_soft_run_two_instructions_with_rom_tensor():
    rt = RomTensor(_rom_with([0xA9, 0x42, 0x85, 0x00]))   # LDA #$42 / STA $00
    bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rt)
    state = initial_soft_cpu_state()
    state, bus = soft_run(state, bus, n_steps=2)
    assert float(bus.ram[0]) == 0x42


# --------------------------------------------------------------------------- #
# jax.grad threads the cotangent back as a RomTensor
# --------------------------------------------------------------------------- #

def test_grad_through_rom_tensor_returns_rom_tensor():
    """The headline XAI demo, but with the ROM wrapped in a RomTensor:
    `jax.grad` of RAM[0] w.r.t. the RomTensor is itself a RomTensor,
    one-hot at the immediate-operand byte."""
    rom_tensor = RomTensor(_rom_with([0xA9, 0x42, 0x85, 0x00]))

    def simulator(rt):
        bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rt)
        state = initial_soft_cpu_state()
        state, bus = soft_run(state, bus, n_steps=2)
        return bus.ram[0]

    assert float(simulator(rom_tensor)) == 0x42

    grad = jax.grad(simulator)(rom_tensor)
    # The cotangent comes back as a RomTensor — the PyTree structure is
    # preserved through the transform.
    assert isinstance(grad, RomTensor)
    assert float(grad.rom[1]) == pytest.approx(1.0)
    other = float(jnp.sum(jnp.abs(grad.rom))) - float(jnp.abs(grad.rom[1]))
    assert other == pytest.approx(0.0)


def test_grad_through_soft_bus_with_rom_tensor():
    """Differentiate w.r.t. the whole SoftBus — the gradient is a SoftBus
    PyTree whose `.rom` slot is a RomTensor cotangent."""
    rom_tensor = RomTensor(_rom_with([0xA9, 0x7F]))

    def simulator(bus):
        state = initial_soft_cpu_state()
        state, _ = soft_step(state, bus)
        return state.A

    bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom_tensor)
    grad = jax.grad(simulator)(bus)
    assert isinstance(grad.rom, RomTensor)
    assert float(grad.rom.rom[1]) == pytest.approx(1.0)
