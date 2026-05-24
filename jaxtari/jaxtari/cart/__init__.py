"""Cartridge — ROM image + bank-switching state.

See `jaxtari.cart.system` for the supported cart types and the auto-detect
helper.
"""

from jaxtari.cart.system import (
    KIND_2K,
    KIND_4K,
    KIND_E0,
    KIND_F4,
    KIND_F4SC,
    KIND_F6,
    KIND_F6SC,
    KIND_F8,
    KIND_F8SC,
    Cart,
    cart_peek,
    cart_poke,
    make_cart,
)

__all__ = [
    "KIND_2K",
    "KIND_4K",
    "KIND_E0",
    "KIND_F4",
    "KIND_F4SC",
    "KIND_F6",
    "KIND_F6SC",
    "KIND_F8",
    "KIND_F8SC",
    "Cart",
    "cart_peek",
    "cart_poke",
    "make_cart",
]
