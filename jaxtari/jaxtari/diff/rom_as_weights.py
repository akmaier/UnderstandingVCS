"""ROM-as-weights — make cartridge bytes differentiable parameters.

Paper reference: this is the first of the three relaxations in
"Building Two Differentiable VCS Ports / Hard and Soft Execution" —
"every ROM and RAM read is written as a dot product against a one-hot
address vector", i.e. the memory-read primitive peek(r, a) = 1_a^T r =
r_a (Eq. "peek"; supplementary "Setup and Notation", first primitive).
The forward value is exactly the addressed byte r_a so the soft read is
bit-exact to the hard one (Theorem 1, "Exact forward equivalence"),
while the gradient d peek / d r = 1_a is now defined and one-hot — the
construction that turns "which ROM byte explains this pixel?" into a
gradient question and the discrete limit of the Neural-Turing-Machine
soft addressing (Graves et al. 2014).

Mirrors the hard read xitari M6502Low::peek / System::peek (a plain
`mem[address]` array index); here the same value is produced by the
one-hot dot product so autodiff can flow through it.

The XAI use case this enables: "which ROM byte explains this output?"
A normal `int(rom[addr])` indexing has zero gradient w.r.t. the ROM —
JAX can't backprop through an integer index. By replacing the index
with a one-hot dot product, the same value comes out forward but the
gradient is a clean one-hot at the accessed address.

`RomTensor` wraps a float32 view of the ROM bytes; its `peek(addr)`
method computes `one_hot(addr) · rom`. For batched / multi-address use,
`peek_many(addrs)` is provided.

**P7d** registers `RomTensor` as a JAX PyTree node. That lets it sit
directly in the `SoftBus.rom` slot — `jax.grad` traverses into the
wrapper, treats `rom` as the single differentiable leaf, and threads
the cotangent back out as a `RomTensor`. Before P7d the SoftBus had to
carry a bare `jnp.ndarray` because a plain Python class is opaque to
`jax.grad`; now either form works.

This is the PORTING_PLAN.md §6.2 "ROM-as-weights" primitive — the
single most important building block for the SOFT execution mode and
the eventual XAI attribution work.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class RomTensor:
    """A cartridge ROM image with differentiable peek.

    Construct with any `jnp.ndarray` (typically uint8); the bytes are
    cast to float32 internally. Read via `peek(addr)`; the returned
    scalar's gradient w.r.t. `rom_tensor.rom` is one-hot at `addr`.

    Registered as a JAX PyTree (P7d): the single child is the `rom`
    array, there is no static aux data. This means a `RomTensor` can be
    a leaf-container inside any PyTree `jax.grad` differentiates —
    notably `SoftBus.rom`.
    """

    __slots__ = ("rom",)

    def __init__(self, rom) -> None:
        self.rom = jnp.asarray(rom, dtype=jnp.float32)

    @property
    def size(self) -> int:
        return int(self.rom.shape[0])

    def peek(self, addr) -> jnp.ndarray:
        """Differentiable read of one byte. `addr` is an integer or int-like
        scalar. Returns a 0-d float32 array.

        Implements peek(r, a) = one_hot(a) . r = r_a (paper Eq. "peek").
        Forward value equals the array element r_a (so it is bit-exact to
        a hard `rom[a]` read, Theorem 1), and the gradient w.r.t. `rom`
        is the one-hot vector 1_a."""
        one_hot = jax.nn.one_hot(addr, self.size, dtype=jnp.float32)
        return jnp.dot(one_hot, self.rom)

    def peek_many(self, addrs) -> jnp.ndarray:
        """Differentiable batched read. `addrs` is a 1-d int array of
        length N; returns a (N,) float32 vector."""
        one_hot = jax.nn.one_hot(addrs, self.size, dtype=jnp.float32)
        return one_hot @ self.rom

    # --- PyTree protocol (P7d) --------------------------------------------- #

    def tree_flatten(self):
        """Children = (rom array,); no static aux data."""
        return (self.rom,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuild from a flattened representation. Bypasses `__init__`
        so the child (which may be a JAX tracer or a cotangent during a
        transform) is stored verbatim, with no dtype coercion."""
        obj = object.__new__(cls)
        obj.rom = children[0]
        return obj

    def __repr__(self) -> str:
        return f"RomTensor(size={self.rom.shape[0]})"
