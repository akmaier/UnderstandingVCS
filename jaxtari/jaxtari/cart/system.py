"""Cartridge ROM + bank-switching.

The Atari 2600's CPU has a 4 KB cartridge window (\$1000–\$1FFF after the
6507's 13-bit mirror). Cartridges larger than that use one of several
bank-switching schemes to expose a 4 KB slice of a larger ROM via
"hotspot" memory accesses that have side-effects on the bank state.

Supported in P5:

  2K       2 KB ROM, mirrored across the 4 KB cart window. No banks.
  4K       4 KB ROM, plain mapping. No banks.
  F8       8 KB ROM, 2 banks of 4 KB. Hotspots \$1FF8 (bank 0) /
           \$1FF9 (bank 1). Initial bank = 1 (the typical reset vector).
  F6      16 KB ROM, 4 banks. Hotspots \$1FF6..\$1FF9. Initial bank = 3.
  F4      32 KB ROM, 8 banks. Hotspots \$1FF4..\$1FFB. Initial bank = 7.

Deferred to a P5 follow-up:
  * SC variants of F8/F6/F4 (add 128 B of on-cart RAM at \$1000–\$10FF).
  * E0  (8K, 4 × 1KB-slice banking).
  * FE  (8K, JSR-driven banking).
  * 3F / 3E (Tigervision / Boulder Dash with different write-driven
    bank-switching at \$3F-area addresses).
  * MB / MC / AR / DPC (Mega Boy, MegaCart, Atari Supercharger,
    Pitfall II's coprocessor).

Important: bank-switching hotspots fire on ANY access — read or write —
not just writes. So `cart_peek` may have a side-effect on cart state.
Because of that, `Cart` is implemented as a mutable Python class rather
than a NamedTuple; the surrounding `Bus` still uses NamedTuple semantics
but holds a reference to the live Cart object.
"""

from __future__ import annotations

import jax.numpy as jnp

# --------------------------------------------------------------------------- #
# Kind tags
# --------------------------------------------------------------------------- #

KIND_2K = 0
KIND_4K = 1
KIND_F8 = 2
KIND_F6 = 3
KIND_F4 = 4

_SIZE_TO_KIND = {
    2048:  KIND_2K,
    4096:  KIND_4K,
    8192:  KIND_F8,
    16384: KIND_F6,
    32768: KIND_F4,
}

# Bank-switch hotspot addresses within the 13-bit-masked cart window.
F8_HOTSPOTS = {0x1FF8: 0, 0x1FF9: 1}
F6_HOTSPOTS = {0x1FF6: 0, 0x1FF7: 1, 0x1FF8: 2, 0x1FF9: 3}
F4_HOTSPOTS = {
    0x1FF4: 0, 0x1FF5: 1, 0x1FF6: 2, 0x1FF7: 3,
    0x1FF8: 4, 0x1FF9: 5, 0x1FFA: 6, 0x1FFB: 7,
}

# Bank typically containing the reset vector at power-on — the highest bank.
_DEFAULT_BANK = {
    KIND_2K: 0,
    KIND_4K: 0,
    KIND_F8: 1,
    KIND_F6: 3,
    KIND_F4: 7,
}


# --------------------------------------------------------------------------- #
# Cart state
# --------------------------------------------------------------------------- #

class Cart:
    """Mutable cart state: ROM image, kind tag, and (for bank-switched
    carts) the currently-mapped bank index.

    Mutable because hotspot reads in real hardware change bank state as a
    side-effect of the read itself; modelling that in a pure-functional
    NamedTuple would require returning a new world from `peek`, which
    would ripple through every memory-access call site in the CPU.
    """

    __slots__ = ("kind", "rom", "current_bank")

    def __init__(self, kind: int, rom: jnp.ndarray) -> None:
        self.kind = kind
        self.rom = rom
        self.current_bank = _DEFAULT_BANK[kind]

    def __repr__(self) -> str:
        names = {KIND_2K: "2K", KIND_4K: "4K", KIND_F8: "F8",
                 KIND_F6: "F6", KIND_F4: "F4"}
        return (f"Cart({names[self.kind]}, "
                f"rom_size={len(self.rom)}, bank={self.current_bank})")


def make_cart(rom: jnp.ndarray) -> Cart:
    """Build a `Cart` by auto-detecting the kind from the ROM size.

    Signature-based detection (some carts have ambiguous size — e.g. 8K
    could be F8 or E0) is a deferred follow-up; for now we assume the
    canonical mapping in `_SIZE_TO_KIND`.
    """
    n = len(rom)
    if n not in _SIZE_TO_KIND:
        raise ValueError(
            f"unrecognised ROM size {n} bytes. "
            f"P5 supports sizes {sorted(_SIZE_TO_KIND.keys())}."
        )
    if rom.dtype != jnp.uint8:
        rom = rom.astype(jnp.uint8)
    return Cart(kind=_SIZE_TO_KIND[n], rom=rom)


# --------------------------------------------------------------------------- #
# Peek / poke
# --------------------------------------------------------------------------- #

def cart_peek(cart: Cart, addr: int) -> int:
    """Read a byte from the cart window. Any access to a hotspot address
    triggers a bank switch as a side-effect AFTER the byte has been read
    (so the value returned is from the bank that was current at the time
    of the access)."""
    a = addr & 0x1FFF
    if cart.kind == KIND_2K:
        return int(cart.rom[a & 0x07FF])
    if cart.kind == KIND_4K:
        return int(cart.rom[a & 0x0FFF])
    bank_offset = cart.current_bank * 0x1000
    value = int(cart.rom[bank_offset + (a & 0x0FFF)])
    _maybe_switch_bank(cart, a)
    return value


def cart_poke(cart: Cart, addr: int, value: int) -> None:
    """ROM is read-only, so the value is discarded. But hotspot accesses
    still fire on writes, switching the bank."""
    a = addr & 0x1FFF
    _maybe_switch_bank(cart, a)


# --------------------------------------------------------------------------- #
# Internal bank-switch logic
# --------------------------------------------------------------------------- #

def _maybe_switch_bank(cart: Cart, masked_addr: int) -> None:
    if cart.kind == KIND_F8:
        if masked_addr in F8_HOTSPOTS:
            cart.current_bank = F8_HOTSPOTS[masked_addr]
    elif cart.kind == KIND_F6:
        if masked_addr in F6_HOTSPOTS:
            cart.current_bank = F6_HOTSPOTS[masked_addr]
    elif cart.kind == KIND_F4:
        if masked_addr in F4_HOTSPOTS:
            cart.current_bank = F4_HOTSPOTS[masked_addr]
