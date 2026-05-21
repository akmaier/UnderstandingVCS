"""ROM-as-weights — make cartridge bytes differentiable parameters.

The XAI use case this enables: "which ROM byte explains this output?"
A normal `int(rom[addr])` indexing has zero gradient w.r.t. the ROM —
JAX can't backprop through an integer index. By replacing the index
with a one-hot dot product, the same value comes out forward but the
gradient is a clean one-hot at the accessed address.

`RomTensor` wraps a float32 view of the ROM bytes; its `peek(addr)`
method computes `one_hot(addr) · rom`. For batched / multi-address use,
`peek_many(addrs)` is provided.

This is the PORTING_PLAN.md §6.2 "ROM-as-weights" primitive — the
single most important building block for the SOFT execution mode and
the eventual XAI attribution work.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


class RomTensor:
    """A cartridge ROM image with differentiable peek.

    Construct with any `jnp.ndarray` (typically uint8); the bytes are
    cast to float32 internally. Read via `peek(addr)`; the returned
    scalar's gradient w.r.t. `rom_tensor.rom` is one-hot at `addr`.
    """

    __slots__ = ("rom",)

    def __init__(self, rom) -> None:
        self.rom = jnp.asarray(rom, dtype=jnp.float32)

    @property
    def size(self) -> int:
        return int(self.rom.shape[0])

    def peek(self, addr) -> jnp.ndarray:
        """Differentiable read of one byte. `addr` is an integer or int-like
        scalar. Returns a 0-d float32 array."""
        one_hot = jax.nn.one_hot(addr, self.size, dtype=jnp.float32)
        return jnp.dot(one_hot, self.rom)

    def peek_many(self, addrs) -> jnp.ndarray:
        """Differentiable batched read. `addrs` is a 1-d int array of
        length N; returns a (N,) float32 vector."""
        one_hot = jax.nn.one_hot(addrs, self.size, dtype=jnp.float32)
        return one_hot @ self.rom
