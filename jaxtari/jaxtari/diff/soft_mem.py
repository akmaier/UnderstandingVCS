"""NTM-style soft memory read — differentiable access to RAM by a
continuous-valued address.

Paper reference: this is the *relaxed read* primitive — the
distance-softmax (temperature-T) read peek_R(r, a; T) = sum_k w_k
r_{a+k} with w_k proportional to exp(-|k| / T) (supplementary "Setup
and Notation", fifth primitive, Eq. "s-read"). It is the differentiable
counterpart of the one-hot exact read (rom_as_weights.peek): peek_R ->
r_a as T -> 0 while the address carries a nonzero gradient for T > 0,
the discrete limit of the soft attention-style addressing of a Neural
Turing Machine (Graves et al. 2014). The executed (SOFT-STE) path uses
the *exact* one-hot read instead; only the fully relaxed variant of
Theorem 2 ("Temperature-limit bound") uses this read, whose off-target
weights decay unconditionally as exp(-1/T) (the proof's "Read term").

Mirrors the hard read xitari M6502Low::peek / System::peek (a discrete
`mem[address]`); here a continuous address selects a temperature-blended
neighbourhood so gradients reach the address itself.

PORTING_PLAN.md §6.2: indirect / indexed RAM addressing becomes a
positional-attention read,

    weights = softmax(-|positions - addr| / τ)
    value   = sum_i weights[i] * RAM[i]

so that an address `addr` placed between two cells reads a blend of
their values, and gradients can flow back to both the address itself
and to the memory contents. As τ → 0 the read collapses to ordinary
indexing.

The hard / direct case (LDA \$80 with the literal `$80` baked into the
opcode stream) stays as a normal index; this primitive is only used
for the indirect / indexed modes the plan calls out.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def soft_memory_read(
    memory: jnp.ndarray,
    addr_continuous: jnp.ndarray,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """Differentiable read of `memory` at a continuous-valued `addr`.

    `memory` is a 1-d float array of length N. `addr_continuous` is a
    scalar in [0, N). Returns a 0-d float array.

    With `temperature → 0` the result equals `memory[round(addr)]` (up
    to numerical precision). With larger temperatures the weights
    spread, returning a smooth blend that gives non-trivial gradients
    to surrounding cells.
    """
    n = memory.shape[0]
    positions = jnp.arange(n, dtype=jnp.float32)
    distances = jnp.abs(positions - addr_continuous)
    logits = -distances / temperature
    weights = jax.nn.softmax(logits)
    return jnp.dot(weights, memory.astype(jnp.float32))
