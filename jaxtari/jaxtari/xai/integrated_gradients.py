"""Integrated Gradients (Sundararajan, Taly, Yan 2017) on top of `jax.grad`.

P8-a: the first XAI primitive built on the differentiable VCS. Given a
scalar-valued function `f` of an input PyTree, `integrated_gradients`
attributes the difference `f(target) - f(baseline)` back to each
element of `target` by integrating `∂f/∂x` along the straight-line
path from `baseline` to `target`.

The attribution satisfies the **completeness axiom**:

    sum(IG) ≈ f(target) - f(baseline)

up to the Riemann discretisation error (controlled by `steps`).

Why it matters here
-------------------

Inside this project the differentiable VCS lets you compute
`jax.grad(framebuffer_pixel)(rom_array)` — but that gradient is the
*local* slope at one specific ROM, which can be sparse / saturated /
misleading when the simulator passes through hard quantizations
(int-cast flag bits, one-hot opcode dispatch, etc.). IG instead
averages the gradient along the whole path from a zero-baseline ROM
to the actual ROM, which is a *much* better attribution signal — it
sees every byte that participated in producing the output, weighted
by how much it changed the output along the way.

This is the foundation the P8-b/c attribution-demo experiments stand
on.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def integrated_gradients(f, target, baseline=None, steps: int = 50):
    """Integrated Gradients attribution.

    Parameters
    ----------
    f
        Callable taking the input PyTree (same structure as `target`)
        and returning a scalar.
    target
        The point at which to attribute — a `jnp.ndarray` or any
        PyTree of arrays.
    baseline
        The path's starting point. Defaults to `zeros_like(target)`
        (the standard choice — produces attributions whose sum equals
        `f(target) - f(0)`).
    steps
        Number of Riemann subdivisions along the path. The estimate
        converges to the true integral as `steps → ∞`; 50 is a typical
        default that's usually within a fraction of a percent of the
        true value for smooth `f`.

    Returns
    -------
    PyTree matching `target`'s structure, where each leaf holds the
    per-element attribution. The total `sum(IG) ≈ f(target) - f(baseline)`
    is the completeness axiom — a property `assert_completeness` checks
    if you want it as a regression.
    """
    if baseline is None:
        baseline = jax.tree_util.tree_map(jnp.zeros_like, target)

    diff = jax.tree_util.tree_map(lambda t, b: t - b, target, baseline)
    grad_f = jax.grad(f)

    # Midpoint Riemann rule (alpha = 0.5/N, 1.5/N, ..., (N-0.5)/N).
    # Midpoint is **exact** for linear integrands — which is the case
    # when `f` is quadratic in `x` along the path — so a quadratic head
    # function (e.g. the project's headline `soft_rom_peek(rom, addr)²`
    # demo) gives the exact analytic attribution with any `steps ≥ 1`,
    # not the ~`(N+1)/N` bias the right-endpoint rule has. Sundararajan
    # et al. use left/right endpoints; midpoint is the standard tightened
    # variant used by careful IG implementations (e.g. Captum's smooth-
    # Riemann option).
    alphas = (jnp.arange(steps, dtype=jnp.float32) + 0.5) / float(steps)

    # Sum the gradient at each alpha along the path. `jax.lax.scan` lets
    # us do this without materialising the full (steps, *target_shape)
    # gradient array — useful when `target` is a large ROM.
    def _accumulate(carry_grad, alpha):
        x_alpha = jax.tree_util.tree_map(
            lambda b, d: b + alpha * d, baseline, diff)
        g = grad_f(x_alpha)
        new_carry = jax.tree_util.tree_map(lambda c, gv: c + gv, carry_grad, g)
        return new_carry, None

    zero_grad = jax.tree_util.tree_map(jnp.zeros_like, target)
    summed_grad, _ = jax.lax.scan(_accumulate, zero_grad, alphas)
    avg_grad = jax.tree_util.tree_map(lambda s: s / steps, summed_grad)

    # IG_i = (target_i - baseline_i) * mean_alpha[ ∂f/∂x_i (x(alpha)) ]
    return jax.tree_util.tree_map(lambda d, g: d * g, diff, avg_grad)


def assert_completeness(f, target, attribution, baseline=None,
                        atol: float = 1e-3) -> None:
    """Verify that `sum(attribution) ≈ f(target) - f(baseline)`.

    The completeness axiom is the canonical sanity check for an IG
    implementation; it must hold up to the discretisation error. Raises
    `AssertionError` on a violation larger than `atol`.
    """
    if baseline is None:
        baseline = jax.tree_util.tree_map(jnp.zeros_like, target)
    total_attr = sum(
        float(jnp.sum(a)) for a in jax.tree_util.tree_leaves(attribution)
    )
    delta_f = float(f(target)) - float(f(baseline))
    err = abs(total_attr - delta_f)
    if err > atol:
        raise AssertionError(
            f"IG completeness violated: sum(IG)={total_attr:.6g}, "
            f"f(target)-f(baseline)={delta_f:.6g}, error={err:.3e} > "
            f"atol={atol}"
        )
