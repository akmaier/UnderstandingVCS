"""Straight-through estimator helpers — hard forward, soft backward.

Paper reference: the straight-through estimator (STE) of the "Hard and
Soft Execution" section, STE(soft, hard) = soft + sg(hard - soft) with
sg the stop-gradient operator (Eq. "ste"; supplementary fourth
primitive). The round and clamp operations here are the "round and clamp
... treated the same way (forward exact, backward identity inside the
valid range)" the paper calls out. This is the device that makes the
soft forward pass bit-exact to the hard one at any finite temperature
(Theorem 1, "Exact forward equivalence") while exposing a *surrogate*
gradient (Corollary 1: STE is a surrogate-gradient estimator, not the
limiting hard derivative), the same device that makes max-pooling
differentiable; cf. Bengio et al. 2013.

PORTING_PLAN.md §6.3: for opcodes where the soft relaxation is too
expensive (or where the discrete behaviour is genuinely correct in the
forward pass), we want the forward computation to give the hard
integer answer while the backward pass treats the operation as the
identity, so gradients can still flow.

Two helpers are provided:

  * `straight_through_round(x)` — forward: round-to-nearest-even
    (matching JAX's default `jnp.round`); backward: identity.
  * `straight_through_clamp(x, lo, hi)` — forward: clip to [lo, hi];
    backward: identity inside the interval, zero outside.

Both use `jax.custom_vjp` so they compose with `jax.grad`,
`jax.value_and_grad`, etc.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# --------------------------------------------------------------------------- #
# straight_through_round
# --------------------------------------------------------------------------- #

@jax.custom_vjp
def straight_through_round(x: jnp.ndarray) -> jnp.ndarray:
    """Forward: `jnp.round(x)`; backward: identity."""
    return jnp.round(x)


def _ste_round_fwd(x):
    return jnp.round(x), ()


def _ste_round_bwd(_residuals, g):
    return (g,)


straight_through_round.defvjp(_ste_round_fwd, _ste_round_bwd)


# --------------------------------------------------------------------------- #
# straight_through_clamp
# --------------------------------------------------------------------------- #

@jax.custom_vjp
def straight_through_clamp(x: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray:
    """Forward: `jnp.clip(x, lo, hi)`; backward: 1 inside [lo, hi], 0 outside.

    Outside-the-interval zero is the standard STE convention — values
    saturated against the bound have no influence on the output and so
    should get no gradient.
    """
    return jnp.clip(x, lo, hi)


def _ste_clamp_fwd(x, lo, hi):
    inside = jnp.logical_and(x >= lo, x <= hi).astype(jnp.float32)
    return jnp.clip(x, lo, hi), inside


def _ste_clamp_bwd(inside, g):
    return (g * inside, None, None)


straight_through_clamp.defvjp(_ste_clamp_fwd, _ste_clamp_bwd)
