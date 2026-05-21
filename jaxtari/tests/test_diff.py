"""P7 tests — differentiability primitives.

Forward-behaviour tests assert each primitive returns the right scalar
or array. Gradient tests assert each primitive provides useful (i.e.
non-zero where appropriate, zero-elsewhere when one-hot) backward
signal. The end-to-end test in `test_xai_rom_byte_attribution_demo`
ties them together: a tiny "simulator" that reads a byte from a
RomTensor and squares it; the gradient of that scalar w.r.t. the ROM
should be one-hot-ish at the accessed address.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxtari.diff import (
    RomTensor,
    soft_branch,
    soft_memory_read,
    soft_select,
    straight_through_clamp,
    straight_through_round,
)


# --------------------------------------------------------------------------- #
# RomTensor — forward + gradient
# --------------------------------------------------------------------------- #

def test_rom_tensor_peek_returns_correct_byte_value():
    rom_bytes = jnp.arange(16, dtype=jnp.uint8)
    rom = RomTensor(rom_bytes)
    for addr in (0, 1, 7, 15):
        assert float(rom.peek(addr)) == float(addr)


def test_rom_tensor_size():
    assert RomTensor(jnp.zeros((1024,), dtype=jnp.uint8)).size == 1024


def test_rom_tensor_peek_many_returns_vector():
    rom_bytes = jnp.arange(16, dtype=jnp.uint8)
    rom = RomTensor(rom_bytes)
    out = rom.peek_many(jnp.array([0, 5, 10]))
    assert out.shape == (3,)
    assert jnp.allclose(out, jnp.array([0.0, 5.0, 10.0]))


def test_rom_tensor_gradient_is_one_hot_at_address():
    rom_bytes = jnp.full((32,), 7.0)

    def output(rom_arr):
        rom = RomTensor(rom_arr)
        return rom.peek(11)

    grad = jax.grad(output)(rom_bytes)
    # Exactly one entry should be non-zero: position 11.
    assert float(grad[11]) == pytest.approx(1.0)
    other_sum = float(jnp.sum(jnp.abs(grad))) - float(jnp.abs(grad[11]))
    assert other_sum == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# soft_select — saturation + gradient
# --------------------------------------------------------------------------- #

def test_soft_select_saturates_at_large_logit():
    logits = jnp.array([0.0, 100.0, 0.0])           # entry 1 dominates
    values = jnp.array([1.0, 2.0, 3.0])
    out = soft_select(logits, values, temperature=1.0)
    assert float(out) == pytest.approx(2.0, abs=1e-5)


def test_soft_select_uniform_at_high_temperature():
    logits = jnp.array([0.0, 100.0, 0.0])
    values = jnp.array([1.0, 2.0, 3.0])
    out = soft_select(logits, values, temperature=1000.0)
    # softmax with huge T → uniform → average of values = 2.0
    assert float(out) == pytest.approx(2.0, abs=1e-2)


def test_soft_select_keeps_trailing_dims():
    """If `values` has shape (N, K), result should have shape (K,)."""
    logits = jnp.zeros(3)
    values = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
    out = soft_select(logits, values)
    assert out.shape == (2,)
    # Uniform softmax → average of rows.
    assert jnp.allclose(out, jnp.array([2.0, 3.0]))


def test_soft_select_gradient_distributes_per_logit_softmax_jacobian():
    """∂(softmax · values)/∂logits ≠ 0 for every entry — useful for
    learning which opcode the dispatcher should fire."""
    logits = jnp.array([0.0, 0.0, 0.0])
    values = jnp.array([1.0, 2.0, 3.0])

    def loss(l):
        return soft_select(l, values)

    grad = jax.grad(loss)(logits)
    # Uniform softmax → ∂/∂l_i = (values[i] - mean(values)) / 3
    expected_mean = 2.0
    for i in range(3):
        assert float(grad[i]) == pytest.approx((values[i] - expected_mean) / 3.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# soft_memory_read — NTM-style positional read
# --------------------------------------------------------------------------- #

def test_soft_memory_read_at_integer_addr_low_temperature():
    mem = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = soft_memory_read(mem, jnp.array(2.0), temperature=0.01)
    assert float(out) == pytest.approx(3.0, abs=1e-3)


def test_soft_memory_read_between_two_addresses_blends():
    mem = jnp.array([10.0, 20.0, 30.0])
    # At addr=0.5 with τ=0.1 the read should be close to (10+20)/2 = 15.
    out = soft_memory_read(mem, jnp.array(0.5), temperature=0.1)
    assert float(out) == pytest.approx(15.0, abs=1.0)


def test_soft_memory_read_gradient_flows_to_address():
    """Reading at a non-integer address should give a meaningful gradient
    to the address — that's how an XAI run could back-propagate "which
    RAM cell mattered" through indirect addressing.

    Sit at addr = 0.5 (between cells 0 and 1, both equidistant) — well
    away from any `|x|` kink point. Cell 1 holds the spike value 100,
    so moving the address toward 1 raises the read and the gradient
    should be strictly positive.
    """
    mem = jnp.array([0.0, 100.0, 0.0])

    def loss(addr):
        return soft_memory_read(mem, addr, temperature=1.0)

    grad = jax.grad(loss)(jnp.array(0.5))
    assert float(grad) > 0.0


def test_soft_memory_read_gradient_flows_to_memory():
    mem = jnp.array([0.0, 100.0, 0.0])

    def loss(m):
        return soft_memory_read(m, jnp.array(1.0), temperature=0.01)

    grad = jax.grad(loss)(mem)
    # With τ → 0 the weight is one-hot at index 1.
    assert float(grad[1]) == pytest.approx(1.0, abs=1e-2)
    assert float(grad[0]) == pytest.approx(0.0, abs=1e-2)


# --------------------------------------------------------------------------- #
# soft_branch — saturation + smoothness
# --------------------------------------------------------------------------- #

def test_soft_branch_saturated_to_branch_when_flag_high():
    out = soft_branch(jnp.array(1.0), jnp.array(0x100), jnp.array(0x200), alpha=100.0)
    assert float(out) == pytest.approx(0x200, abs=1e-3)


def test_soft_branch_saturated_to_no_branch_when_flag_low():
    out = soft_branch(jnp.array(-1.0), jnp.array(0x100), jnp.array(0x200), alpha=100.0)
    assert float(out) == pytest.approx(0x100, abs=1e-3)


def test_soft_branch_blends_at_low_alpha():
    """At flag = 0 and low α the gate is exactly 0.5 and the result is
    the arithmetic mean of the two PCs."""
    out = soft_branch(jnp.array(0.0), jnp.array(100.0), jnp.array(200.0), alpha=1.0)
    assert float(out) == pytest.approx(150.0, abs=1e-3)


def test_soft_branch_gradient_to_flag_is_nonzero_near_threshold():
    def loss(flag):
        return soft_branch(flag, jnp.array(0.0), jnp.array(1.0), alpha=1.0)

    grad = jax.grad(loss)(jnp.array(0.0))
    # ∂sigmoid(αx)/∂x at x=0 is α * 0.25 = 0.25 here; PC difference is 1.0,
    # so loss derivative is 0.25.
    assert float(grad) == pytest.approx(0.25, abs=1e-3)


# --------------------------------------------------------------------------- #
# Straight-through estimators
# --------------------------------------------------------------------------- #

def test_ste_round_forward_rounds_to_nearest():
    assert float(straight_through_round(jnp.array(2.7))) == 3.0
    assert float(straight_through_round(jnp.array(2.3))) == 2.0
    assert float(straight_through_round(jnp.array(-1.7))) == -2.0


def test_ste_round_gradient_is_identity():
    """Even though jnp.round is piecewise-constant (true gradient = 0
    almost everywhere), the STE forces a gradient of 1 through."""
    def loss(x):
        return straight_through_round(x) ** 2

    g = jax.grad(loss)(jnp.array(2.7))
    # STE: ∂loss/∂x = ∂(round(x)²)/∂x with d(round(x))/dx ≡ 1
    #               = 2 * round(2.7) * 1 = 6.0
    assert float(g) == pytest.approx(6.0, abs=1e-5)


def test_ste_clamp_forward_clips():
    assert float(straight_through_clamp(jnp.array(0.5), 0.0, 1.0)) == 0.5
    assert float(straight_through_clamp(jnp.array(2.0), 0.0, 1.0)) == 1.0
    assert float(straight_through_clamp(jnp.array(-0.5), 0.0, 1.0)) == 0.0


def test_ste_clamp_gradient_inside_interval_passes_through():
    def loss(x):
        return straight_through_clamp(x, 0.0, 1.0)

    g = jax.grad(loss)(jnp.array(0.5))
    assert float(g) == pytest.approx(1.0)


def test_ste_clamp_gradient_outside_interval_is_zero():
    def loss(x):
        return straight_through_clamp(x, 0.0, 1.0) ** 2

    g_above = jax.grad(loss)(jnp.array(2.0))
    g_below = jax.grad(loss)(jnp.array(-1.0))
    assert float(g_above) == pytest.approx(0.0)
    assert float(g_below) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# End-to-end demo — "which ROM byte explains this output?"
# --------------------------------------------------------------------------- #

def test_xai_rom_byte_attribution_demo():
    """A toy simulator that reads ROM[$42] and squares it. The gradient
    of that output w.r.t. the ROM array should be non-zero at exactly
    one position — position $42 — and zero everywhere else. This is the
    "ROM-as-weights" XAI use case spelled out in PORTING_PLAN.md §6.2."""
    rom_bytes = jnp.full((256,), 5.0)

    def simulator(rom_arr):
        rom = RomTensor(rom_arr)
        return rom.peek(0x42) ** 2

    grad = jax.grad(simulator)(rom_bytes)
    assert float(grad[0x42]) == pytest.approx(2.0 * 5.0, abs=1e-5)
    # Everything else is zero.
    other = jnp.sum(jnp.abs(grad)) - jnp.abs(grad[0x42])
    assert float(other) == pytest.approx(0.0, abs=1e-5)


def test_xai_demo_composes_soft_select_with_rom_tensor():
    """ROM peeks at multiple addresses combined via soft_select — a
    simplified model of "the dispatcher chose between two ROM bytes
    based on a learnable logit". The gradient should split between the
    two ROM positions according to the softmax weight."""
    rom_bytes = jnp.full((16,), 10.0)
    logits = jnp.array([0.0, 0.0])    # uniform → equal weights

    def simulator(rom_arr):
        rom = RomTensor(rom_arr)
        byte_a = rom.peek(3)
        byte_b = rom.peek(7)
        # Combine via soft_select(logits, [byte_a, byte_b]).
        return soft_select(logits, jnp.stack([byte_a, byte_b]))

    grad = jax.grad(simulator)(rom_bytes)
    # Each of positions 3 and 7 gets weight 0.5; everything else 0.
    assert float(grad[3]) == pytest.approx(0.5)
    assert float(grad[7]) == pytest.approx(0.5)
    total = float(jnp.sum(jnp.abs(grad)))
    assert total == pytest.approx(1.0, abs=1e-5)
