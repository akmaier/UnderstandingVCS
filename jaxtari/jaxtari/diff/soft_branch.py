"""Soft branch gate — sigmoid-relaxed conditional PC update.

Paper reference: the third of the three relaxations in "Hard and Soft
Execution" — a conditional branch relaxed with a sigmoid gate, g =
sigmoid(alpha * z) and pc_soft = (1-g) pc_notaken + g pc_taken =
pc_notaken + g * delta (Eq. "branch"; supplementary "Setup and
Notation", third primitive). The flag logit is z = s (2f - 1) with the
sign s selecting branch-when-set / branch-when-clear and sharpness
alpha. As alpha -> infinity the gate saturates to the hard branch; the
temperature-limit proof of Theorem 2 ("Temperature-limit bound") bounds
the gate error |sigmoid(alpha z) - [z>0]| by exp(-alpha * m_min). The
gate gradient dg/dz = alpha * sigmoid(alpha z)(1 - sigmoid(alpha z)) is
the dipole studied in Fig. "soft-branch gate" / supplementary "Effect
of the Relaxation Parameters".

In the executed (SOFT-STE) step this soft gate is wrapped by a
straight-through estimator (see soft_step._do_branch), so the forward
PC is the exact hard branch (Theorem 1) and only the backward pass uses
this sigmoid blend.

Mirrors the hard relative-branch of xitari (M6502Hi.m4 branch macro,
`address = PC + (Int8)operand`): there the next PC is a discrete pick on
the status-flag bit; here it is the sigmoid blend of the two PC values.

PORTING_PLAN.md §6.2: a conditional branch like `BNE` normally picks
between two next-PC values based on a flag. The hard form is
discontinuous and provides no gradient. The soft form is

    g       = sigmoid(α * flag_logit)
    pc_next = (1 - g) * pc_no_branch + g * pc_branch

With large α the gate saturates to either 0 or 1 and the result is
bit-exact identical to the hard branch. With moderate α gradients flow
through both branches' destinations and through the flag itself.

`flag_logit` is the SOFT representation of the flag — for a hard flag
in {0, 1}, pass `2 * flag - 1` (so that g = 0 when flag = 0 and g = 1
when flag = 1, modulo α).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def soft_branch(
    flag_logit: jnp.ndarray,
    pc_no_branch: jnp.ndarray,
    pc_branch: jnp.ndarray,
    alpha: float = 10.0,
) -> jnp.ndarray:
    """Return the soft-gated PC.

    `flag_logit` is a 0-d (or matching-shape) signal — positive ⇒ take
    branch, negative ⇒ do not. `pc_no_branch` and `pc_branch` may be
    int-like or float; the result is float and may need rounding for
    use as an actual PC in HARD mode (see straight_through.py)."""
    g = jax.nn.sigmoid(alpha * flag_logit)
    return (1.0 - g) * pc_no_branch.astype(jnp.float32) + g * pc_branch.astype(jnp.float32)
