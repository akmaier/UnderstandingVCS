"""Occlusion attribution — per-element sensitivity by ablation.

For a function `f` of a vector input `x`, occlusion attribution
replaces each element of `x` in turn with a baseline value (typically
0) and measures how much `f` changes:

    attribution[i] = f(x) - f(x_with_x[i]=baseline)

Where Integrated Gradients answers *"how much does each element's
value contribute to the output, weighted by the gradient along the
path?"*, occlusion answers a different question: *"is each element
**necessary** for the output?"*. An element with high occlusion
attribution is one whose removal (zeroing) measurably degrades `f`.

Why ship it for the SOFT-mode VCS
---------------------------------

On a SOFT-mode emulator the ROM is a discrete-input function in
disguise: even though `soft_step` runs on Float32, each byte's
*meaning* is integer-quantised by the opcode-dispatch one-hot. That
makes naive zero-baseline IG misleading — interpolating an opcode
byte along the path `0 → 0xA9` produces a stream of *different,
mostly unhandled* opcodes, so the gradient along the path is
essentially zero almost everywhere and IG returns 0 even though the
byte is critical.

Occlusion has no such pathology: at every probe point the ROM is the
real program with one byte zeroed, the simulator runs a real (if
slightly modified) program, and the output difference is meaningful.
For discrete-input simulators it's frequently the *more honest*
attribution to start with.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def occlusion_attribution(f, x, baseline_value: float = 0.0) -> jnp.ndarray:
    """Per-element occlusion sensitivity of `f` at `x`.

    Parameters
    ----------
    f
        Callable taking a 1-D `jnp.ndarray` of the same length as `x`
        and returning a scalar.
    x
        The 1-D input — typically a ROM array — at which to measure
        per-byte sensitivity.
    baseline_value
        The value substituted in for each occluded byte. Default 0.0
        (the standard ablation baseline); use any sentinel that the
        downstream simulator treats as "no operation" if 0 doesn't
        give that semantically.

    Returns
    -------
    `(len(x),)` float32 array where `attribution[i] = f(x) - f(x')`,
    `x'` being `x` with `x[i]` set to `baseline_value`. A large
    absolute value at index `i` means byte `i` materially affects `f`.
    """
    base = f(x)

    def _delta_at(i):
        x_occluded = x.at[i].set(jnp.float32(baseline_value))
        return base - f(x_occluded)

    return jax.vmap(_delta_at)(jnp.arange(x.shape[0]))
