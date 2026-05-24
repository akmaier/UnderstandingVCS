"""6507 address bus.

The Atari 2600's CPU is a 6507 — a 6502 variant with only 13 address pins
(A0–A12). Any 16-bit address from the CPU is masked to 13 bits before
decode. Within the 8K window the decode is:

    A12=1                              → Cartridge ROM (\$1000–\$1FFF)
    A12=0, A7=0                        → TIA registers (\$0000–\$007F, with mirrors)
    A12=0, A7=1, A9=0                  → RIOT RAM (\$0080–\$00FF, 128 B, with mirrors)
    A12=0, A7=1, A9=1                  → RIOT I/O (\$0280–\$029F, with mirrors)

A real-world consequence: the 6502 stack lives at \$0100–\$01FF, but on the
6507 that range is split — \$0100–\$017F is a TIA mirror (writes go nowhere
useful) while \$0180–\$01FF mirrors RIOT RAM. Programs running on this
hardware therefore have an effective stack of 128 bytes shared with the
zero-page-relative RAM, and they keep SP in the upper half (initial
SP=\$FD → first push lands at RAM \$7D).

Status as of P5: address decode, RAM peek/poke, full TIA register file +
rendering (P3), RIOT timer + I/O ports (P4), and cartridge peek/poke
with 2K/4K/F8/F6/F4 bank-switching (P5). SC variants and the more
exotic cart formats (E0, FE, 3F, 3E, MB, MC, AR, DPC) are deferred to a
P5 follow-up.

A second, simpler memory model — a flat 65,536-byte `jnp.ndarray` — is
supported too, so the P1 unit tests can keep building tiny programs at
arbitrary addresses without needing a Bus. `peek` / `poke` dispatch on
the world type.
"""

from __future__ import annotations

from typing import NamedTuple, Union

import jax.numpy as jnp

from jaxtari.cart.system import Cart, cart_peek, cart_poke, make_cart
from jaxtari.riot.system import RIOTState, initial_riot_state, riot_peek, riot_poke
from jaxtari.tia.system import TIAState, initial_tia_state, tia_peek, tia_poke


# --------------------------------------------------------------------------- #
# Bus type
# --------------------------------------------------------------------------- #

class Bus(NamedTuple):
    """6507 system bus state.

    Attributes
    ----------
    ram : jnp.ndarray (128,) uint8
        128 bytes of RIOT internal RAM.
    cart : Cart
        Cartridge ROM image + (for bank-switched carts) the current bank
        index. As of P5 supports 2K, 4K, F8, F6, F4. `Cart` is mutable
        because hotspot reads change bank state.
    tia : TIAState
        TIA register file + scanline/frame timing state.
    riot : RIOTState
        RIOT timer + I/O port state (the 128 B RAM is in `ram` above).
    """
    ram: jnp.ndarray
    cart: Cart
    tia: TIAState
    riot: RIOTState


def initial_bus(rom: Union[jnp.ndarray, None] = None,
                *, cart_kind: int | None = None) -> Bus:
    """Build a `Bus` with all-zero RAM, an auto-detected `Cart` built from
    `rom` (default: all-zero 4 KB), and fresh TIA / RIOT states.

    `cart_kind` overrides cart-format auto-detection — required for
    F8SC (8K cart with on-cart RAM), which is size-indistinguishable
    from plain F8.
    """
    if rom is None:
        rom = jnp.zeros((4096,), dtype=jnp.uint8)
    return Bus(
        ram=jnp.zeros((128,), dtype=jnp.uint8),
        cart=make_cart(rom, kind=cart_kind),
        tia=initial_tia_state(),
        riot=initial_riot_state(),
    )


# --------------------------------------------------------------------------- #
# peek / poke — type-dispatched between Bus and flat memory
# --------------------------------------------------------------------------- #

# A "world" is whatever the CPU steps against. Either a real Bus or — for
# the P1 unit tests — a flat 65,536-byte jnp.ndarray.
World = Union[Bus, jnp.ndarray]


def peek(world: World, addr: int):
    """Read a byte from `world` at the 16-bit address `addr`.

    **Returns** `(value, new_world)` — the byte read plus a possibly-
    updated world. The 6507 has read-with-side-effects on some RIOT
    addresses (P4d: INTIM read clears the timer-expired latch), so a
    pure-functional bus model needs the caller to thread the returned
    world forward. Most reads return the same world unchanged; only
    RIOT reads with side effects construct a new bus.

    For a flat `jnp.ndarray` (the P1 unit-test scratch memory), the
    array is always returned unchanged.

    The pre-P4d signature was `peek(world, addr) -> int` — the
    sweeping change to a tuple return is the cost of getting
    bit-exact PXC1-x semantics; see the P4d note in STATUS.md for
    rationale.
    """
    if isinstance(world, Bus):
        return _bus_peek(world, addr)
    return int(world[addr & 0xFFFF]), world


def poke(world: World, addr: int, value: int) -> World:
    """Write `value` (low 8 bits) to `world` at `addr`. Returns the new world.

    For a Bus: writes to RAM land in RAM; writes to ROM / TIA / RIOT are
    silently dropped (TIA/RIOT will land in P3/P4).
    For a flat array: the byte at `addr & 0xFFFF` is updated.
    """
    if isinstance(world, Bus):
        return _bus_poke(world, addr, value)
    return world.at[addr & 0xFFFF].set(jnp.uint8(value & 0xFF))


# --------------------------------------------------------------------------- #
# Bus internals
# --------------------------------------------------------------------------- #

def _bus_peek(bus: Bus, addr: int):
    """Internal bus read — returns `(value, new_bus)`."""
    addr = addr & 0x1FFF  # 6507 13-bit mirror
    if addr & 0x1000:
        # Cartridge — delegate to the cart, which handles bank switching
        # for F8/F6/F4 on hotspot access (read or write). Cart mutates
        # in place (it's a mutable class), so bus identity preserved.
        return cart_peek(bus.cart, addr), bus
    if not (addr & 0x80):
        # TIA region (A7=0). Currently side-effect free.
        return tia_peek(bus.tia, addr), bus
    if addr & 0x200:
        # RIOT I/O (A9=1). P4d: INTIM read clears timer_expired —
        # riot_peek now returns (value, new_riot). Thread the new
        # RIOT into a new Bus if it changed.
        value, new_riot = riot_peek(bus.riot, addr)
        if new_riot is bus.riot:
            return value, bus
        return value, bus._replace(riot=new_riot)
    # RIOT RAM (A7=1, A9=0). 128 bytes, mirrored at offset addr & 0x7F.
    return int(bus.ram[addr & 0x7F]), bus


def _bus_poke(bus: Bus, addr: int, value: int) -> Bus:
    addr = addr & 0x1FFF
    if addr & 0x1000:
        # Cart writes don't store anything (ROM is read-only) but they
        # CAN trigger a bank switch when they hit a hotspot. The cart
        # mutates in place; bus identity is preserved.
        cart_poke(bus.cart, addr, value)
        return bus
    if not (addr & 0x80):
        # TIA write — record the byte and apply P3a side-effects (WSYNC).
        return bus._replace(tia=tia_poke(bus.tia, addr, value))
    if addr & 0x200:
        # RIOT I/O write (P4): SWCHA/SWACNT/SWCHB/SWBCNT or TIM*T.
        return bus._replace(riot=riot_poke(bus.riot, addr, value))
    return bus._replace(
        ram=bus.ram.at[addr & 0x7F].set(jnp.uint8(value & 0xFF))
    )
