"""Soft-selection primitive — softmax-weighted mixture over choices.

Paper reference: the second of the three relaxations in "Hard and Soft
Execution" — opcode dispatch as a convex combination over the handler
outputs, select(l, V; T) = w^T V with w = softmax(l / T) (Eq. "select";
supplementary "Setup and Notation", second primitive). The same softmax
select governs *any* discrete choice, with T setting how widely the
soft pick spreads (the "capture range" of Fig. "Soft-select temperature
T in pixel space"). The executed step uses the *saturated* (hard) form
of this select on the forward path, so dispatch is forward-exact
(Theorem 1); the relaxed form here is what the fully relaxed variant of
Theorem 2 ("Temperature-limit bound") uses. The temperature-limit proof
bounds the non-winning softmax mass by (K-1) exp(-Delta_min / T), so the
mixture collapses to the hard pick as T -> 0.

Mirrors the hard opcode dispatch of xitari M6502Low::execute (the
`switch(peekWithPC())` over the decoded opcode byte): here that switch
is expressed as a softmax-weighted mixture so gradients can flow into
the non-selected handlers.

PORTING_PLAN.md §6.2 calls this out for opcode dispatch ("a 256-way
softmax over the decoded opcode logits, with each branch's effect
weighted by its activation"). It's also the natural building block for
*any* if/else relaxation where the discrete pick can be expressed as
"which row of `values` should this be?"

Saturation: with very-large logits (one entry ≫ others) the softmax
collapses to one-hot and the result is bit-exact equal to the
hard-pick value. With moderate logits, gradient flows back through the
mix into the alternative branches.

A temperature argument lets the caller dial between "almost hard"
(small T) and "softer" (larger T). T → 0 forces argmax; T → ∞ forces
uniform.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def soft_select(
    logits: jnp.ndarray,
    values: jnp.ndarray,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """Compute `softmax(logits / T) · values`.

    `logits` and `values` must have the same leading dimension. `values`
    may have additional trailing dimensions; the result keeps them.
    """
    weights = jax.nn.softmax(logits / temperature)
    return jnp.tensordot(weights, values, axes=1)
