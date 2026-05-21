"""Soft-selection primitive — softmax-weighted mixture over choices.

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
