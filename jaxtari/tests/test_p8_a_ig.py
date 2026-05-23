"""P8-a tests — Integrated Gradients primitive.

Verifies the IG implementation against four hand-crafted functions
where the attribution is known analytically:

  1. Linear         f(x) = w · x                    →  IG_i = w_i * x_i
  2. Quadratic      f(x) = (x[k])^2                 →  IG_k = x_k^2, else 0
  3. Sum-of-squares f(x) = sum_i (x_i)^2            →  IG_i = x_i^2
  4. Two-variable   f(x,y) = a*x + b*y              →  IG_x = a*x, IG_y = b*y
                                                     (PyTree input)

Plus the completeness axiom (sum(IG) ≈ f(target) - f(baseline)) and
the headline integration test: the same XAI demo as
`test_diff.test_xai_rom_byte_attribution_demo`, but with IG instead of
raw `jax.grad` — the attribution localises one-hot at the addressed
ROM byte AND its magnitude equals `rom[addr]^2` (the function's value),
which a plain gradient does not give you.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxtari.diff import soft_rom_peek
from jaxtari.xai import assert_completeness, integrated_gradients


# --------------------------------------------------------------------------- #
# 1. Linear function — IG = w * x exactly (with infinite-step limit)
# --------------------------------------------------------------------------- #

def test_ig_linear_function_attributes_w_times_x():
    """f(x) = w · x → IG_i = w_i * x_i for any number of steps (the
    integrand is constant in alpha, so the Riemann sum is exact)."""
    w = jnp.array([1.0, -2.0, 3.0, 0.5], dtype=jnp.float32)
    x = jnp.array([5.0, 4.0, -1.0, 2.0], dtype=jnp.float32)
    f = lambda v: jnp.dot(w, v)
    ig = integrated_gradients(f, x, steps=10)
    expected = w * x
    assert jnp.allclose(ig, expected, atol=1e-5)


# --------------------------------------------------------------------------- #
# 2. Quadratic on one variable — localisation
# --------------------------------------------------------------------------- #

def test_ig_quadratic_localises_to_indexed_byte():
    """f(x) = x[k]^2 → IG_k = x_k^2, IG_i = 0 elsewhere."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
    k = 2
    f = lambda v: v[k] ** 2
    ig = integrated_gradients(f, x, steps=50)
    assert float(ig[k]) == pytest.approx(x[k] ** 2, abs=1e-4)
    for i in range(len(x)):
        if i != k:
            assert float(ig[i]) == pytest.approx(0.0, abs=1e-5)


def test_ig_quadratic_completeness():
    """sum(IG) ≈ f(target) - f(baseline) — the completeness axiom."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
    f = lambda v: v[2] ** 2
    ig = integrated_gradients(f, x, steps=100)
    assert_completeness(f, x, ig)


# --------------------------------------------------------------------------- #
# 3. Sum of squares — every byte contributes its own square
# --------------------------------------------------------------------------- #

def test_ig_sum_of_squares_per_element():
    """f(x) = sum_i x_i^2 → IG_i = x_i^2."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    f = lambda v: jnp.sum(v ** 2)
    ig = integrated_gradients(f, x, steps=50)
    assert jnp.allclose(ig, x ** 2, atol=1e-3)
    # Completeness: sum(x^2) = f(x) - 0
    assert_completeness(f, x, ig, atol=1e-3)


# --------------------------------------------------------------------------- #
# 4. PyTree input — two named inputs combine linearly
# --------------------------------------------------------------------------- #

def test_ig_pytree_input_attributes_each_branch():
    """f({'x': x, 'y': y}) = a·x + b·y → IG_x = a*x, IG_y = b*y."""
    a, b = 3.0, -2.0
    x = jnp.array([1.0, 2.0], dtype=jnp.float32)
    y = jnp.array([10.0, 5.0], dtype=jnp.float32)
    f = lambda d: a * jnp.sum(d["x"]) + b * jnp.sum(d["y"])
    target = {"x": x, "y": y}
    ig = integrated_gradients(f, target, steps=10)
    assert jnp.allclose(ig["x"], a * x, atol=1e-5)
    assert jnp.allclose(ig["y"], b * y, atol=1e-5)


# --------------------------------------------------------------------------- #
# 5. Custom baseline
# --------------------------------------------------------------------------- #

def test_ig_with_custom_baseline_changes_attribution():
    """IG depends on the baseline. f(x) = x[0]^2 with baseline x'=[1,0]:
    completeness ⇒ sum(IG) = f(x) - f(x') = x[0]^2 - 1."""
    x = jnp.array([3.0, 0.0], dtype=jnp.float32)
    baseline = jnp.array([1.0, 0.0], dtype=jnp.float32)
    f = lambda v: v[0] ** 2
    ig = integrated_gradients(f, x, baseline=baseline, steps=100)
    expected_sum = float(x[0] ** 2) - float(baseline[0] ** 2)  # = 9 - 1 = 8
    assert float(jnp.sum(ig)) == pytest.approx(expected_sum, abs=1e-3)


# --------------------------------------------------------------------------- #
# 6. Headline — IG of the project's XAI demo
# --------------------------------------------------------------------------- #

def test_ig_xai_rom_byte_attribution_demo_localises_and_quantifies():
    """The headline `rom -> soft_rom_peek(rom, $42)^2` from
    `test_diff.test_xai_rom_byte_attribution_demo`, but with IG.

    Plain `jax.grad` gives `2*rom[addr] * one_hot(addr)` — the right
    *direction* but the magnitude is the derivative, not the
    contribution. IG instead gives `rom[addr]^2 * one_hot(addr)`: the
    actual amount that byte contributes to f. Completeness ties it
    together: sum(IG) == rom[addr]^2 == f(rom)."""
    addr = 0x42
    rom_size = 256
    rom = jnp.arange(rom_size, dtype=jnp.float32)

    f = lambda r: soft_rom_peek(r, addr) ** 2
    ig = integrated_gradients(f, rom, steps=64)

    # Localisation: all attribution at rom[addr], zero elsewhere.
    assert float(ig[addr]) == pytest.approx(float(rom[addr]) ** 2, abs=1e-2)
    other = float(jnp.sum(jnp.abs(ig))) - abs(float(ig[addr]))
    assert other == pytest.approx(0.0, abs=1e-3)

    # Completeness: sum(IG) = f(rom) - f(0) = rom[addr]^2 - 0 = rom[addr]^2.
    assert_completeness(f, rom, ig, atol=1e-2)
