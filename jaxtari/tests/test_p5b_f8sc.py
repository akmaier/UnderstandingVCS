"""P5b tests — F8SC cart variant (F8 + 128 B on-cart RAM).

F8SC is an 8 KB F8 cart with extra 128 bytes of RAM at $1000-$10FF:
writes to $1000-$107F land in the RAM, reads from $1080-$10FF read it
back (mirror of the same buffer). Bank-switching otherwise identical
to F8 (hotspots at $1FF8 / $1FF9).

Size alone can't disambiguate F8 vs F8SC, so `make_cart` /
`initial_bus` take an explicit `kind=KIND_F8SC` override.
"""

import jax.numpy as jnp
import pytest

from jaxtari.bus.system import initial_bus, peek, poke
from jaxtari.cart import KIND_F8, KIND_F8SC, Cart, cart_peek, cart_poke, make_cart


# --------------------------------------------------------------------------- #
# Detection / construction
# --------------------------------------------------------------------------- #

def test_autodetect_picks_f8_for_8k_rom():
    """No `kind` override → 8K ROM defaults to F8 (not F8SC)."""
    cart = make_cart(jnp.zeros((8192,), dtype=jnp.uint8))
    assert cart.kind == KIND_F8


def test_kind_override_forces_f8sc():
    cart = make_cart(jnp.zeros((8192,), dtype=jnp.uint8), kind=KIND_F8SC)
    assert cart.kind == KIND_F8SC
    assert len(cart.ram) == 128


def test_f8sc_requires_8k_rom():
    with pytest.raises(ValueError):
        make_cart(jnp.zeros((4096,), dtype=jnp.uint8), kind=KIND_F8SC)


def test_initial_bus_threads_kind_through():
    bus = initial_bus(jnp.zeros((8192,), dtype=jnp.uint8), cart_kind=KIND_F8SC)
    assert bus.cart.kind == KIND_F8SC


def test_f8_initial_bank_is_1():
    """The boot-bank invariant must hold for F8SC too — programs put
    their reset vector in the high bank."""
    cart = make_cart(jnp.zeros((8192,), dtype=jnp.uint8), kind=KIND_F8SC)
    assert cart.current_bank == 1


def test_plain_f8_has_no_on_cart_ram():
    """Regression guard — plain F8 carts must not allocate an SC RAM
    buffer (avoids hidden state showing up in non-SC tests)."""
    cart = make_cart(jnp.zeros((8192,), dtype=jnp.uint8))
    assert len(cart.ram) == 0


# --------------------------------------------------------------------------- #
# RAM read / write
# --------------------------------------------------------------------------- #

def _f8sc_cart() -> Cart:
    return make_cart(jnp.zeros((8192,), dtype=jnp.uint8), kind=KIND_F8SC)


def test_write_to_1000_lands_in_ram():
    cart = _f8sc_cart()
    cart_poke(cart, 0x1000, 0x42)
    # Read back from the read window — should see the same byte.
    assert cart_peek(cart, 0x1080) == 0x42


def test_writes_at_offsets_0_to_7f_independent():
    cart = _f8sc_cart()
    for off in (0, 1, 0x42, 0x7F):
        cart_poke(cart, 0x1000 + off, off ^ 0x55)
    for off in (0, 1, 0x42, 0x7F):
        assert cart_peek(cart, 0x1080 + off) == (off ^ 0x55)


def test_read_at_1080_mirrors_write_at_1000():
    cart = _f8sc_cart()
    cart_poke(cart, 0x1000 + 0x30, 0xAB)
    assert cart_peek(cart, 0x1080 + 0x30) == 0xAB


def test_read_at_1100_falls_through_to_rom():
    """Outside the RAM window, reads come from ROM at the current
    bank — same as plain F8."""
    rom = jnp.zeros((8192,), dtype=jnp.uint8).at[0x1F00].set(0xCC)   # bank 1
    cart = make_cart(rom, kind=KIND_F8SC)
    # Bank 1, offset 0xF00 → ROM byte 0x1F00 = 0xCC.
    assert cart_peek(cart, 0x1F00) == 0xCC


def test_write_at_1100_does_not_touch_ram():
    """Writes outside the SC write area ($1000-$107F) must not land in
    the RAM buffer — they fall through to the normal "drop / maybe
    bank-switch" path."""
    cart = _f8sc_cart()
    cart_poke(cart, 0x1080, 0xFF)                    # write to RAM read window
    cart_poke(cart, 0x1100, 0xFF)                    # write outside SC area
    # The RAM buffer at offset 0x00 must still be 0 (initial value).
    assert cart_peek(cart, 0x1080) == 0
    # And offset 0 of the buffer wasn't touched by either write.


# --------------------------------------------------------------------------- #
# Bank switching still works
# --------------------------------------------------------------------------- #

def test_f8sc_hotspot_switches_bank():
    rom = jnp.zeros((8192,), dtype=jnp.uint8)
    rom = rom.at[0x0500].set(0xAA)     # bank 0 cell
    rom = rom.at[0x1500].set(0xBB)     # bank 1 cell
    cart = make_cart(rom, kind=KIND_F8SC)
    assert cart.current_bank == 1
    assert cart_peek(cart, 0x1500) == 0xBB

    cart_peek(cart, 0x1FF8)            # hotspot read → switch to bank 0
    assert cart.current_bank == 0
    assert cart_peek(cart, 0x1500) == 0xAA


def test_f8sc_hotspot_via_write_also_switches():
    cart = _f8sc_cart()
    assert cart.current_bank == 1
    cart_poke(cart, 0x1FF8, 0x00)
    assert cart.current_bank == 0
    cart_poke(cart, 0x1FF9, 0x00)
    assert cart.current_bank == 1


def test_f8sc_ram_persists_across_bank_switches():
    """Bank switching changes the ROM window but the on-cart RAM is
    independent — it must keep its contents across switches."""
    cart = _f8sc_cart()
    cart_poke(cart, 0x1000 + 0x10, 0x55)
    cart_peek(cart, 0x1FF8)            # bank 0
    assert cart_peek(cart, 0x1080 + 0x10) == 0x55
    cart_peek(cart, 0x1FF9)            # bank 1
    assert cart_peek(cart, 0x1080 + 0x10) == 0x55


# --------------------------------------------------------------------------- #
# Bus-level (cart RAM is reachable through the normal CPU memory path)
# --------------------------------------------------------------------------- #

def test_bus_peek_poke_round_trip_through_f8sc_ram():
    """A CPU `poke` to $1000+offset followed by a `peek` of $1080+offset
    should round-trip the value — the bus's cart dispatch must hit our
    SC RAM, not fall into the ROM-read-only path."""
    bus = initial_bus(jnp.zeros((8192,), dtype=jnp.uint8), cart_kind=KIND_F8SC)
    bus = poke(bus, 0x1000 + 0x20, 0x99)
    assert peek(bus, 0x1080 + 0x20)[0] == 0x99


# --------------------------------------------------------------------------- #
# F6SC + F4SC — same SC RAM dispatch, more banks
# --------------------------------------------------------------------------- #

from jaxtari.cart import KIND_F4SC, KIND_F6SC


def test_f6sc_has_4_banks_and_boots_in_bank_3():
    cart = make_cart(jnp.zeros((16384,), dtype=jnp.uint8), kind=KIND_F6SC)
    assert cart.kind == KIND_F6SC
    assert cart.current_bank == 3
    assert len(cart.ram) == 128


def test_f4sc_has_8_banks_and_boots_in_bank_7():
    cart = make_cart(jnp.zeros((32768,), dtype=jnp.uint8), kind=KIND_F4SC)
    assert cart.kind == KIND_F4SC
    assert cart.current_bank == 7
    assert len(cart.ram) == 128


def test_f6sc_ram_round_trip():
    cart = make_cart(jnp.zeros((16384,), dtype=jnp.uint8), kind=KIND_F6SC)
    cart_poke(cart, 0x1000 + 0x40, 0x33)
    assert cart_peek(cart, 0x1080 + 0x40) == 0x33


def test_f4sc_ram_round_trip():
    cart = make_cart(jnp.zeros((32768,), dtype=jnp.uint8), kind=KIND_F4SC)
    cart_poke(cart, 0x1000 + 0x10, 0xEE)
    assert cart_peek(cart, 0x1080 + 0x10) == 0xEE


def test_f6sc_bank_switching_uses_f6_hotspots():
    cart = make_cart(jnp.zeros((16384,), dtype=jnp.uint8), kind=KIND_F6SC)
    assert cart.current_bank == 3
    cart_peek(cart, 0x1FF6)
    assert cart.current_bank == 0
    cart_peek(cart, 0x1FF9)
    assert cart.current_bank == 3


def test_f4sc_bank_switching_uses_f4_hotspots():
    cart = make_cart(jnp.zeros((32768,), dtype=jnp.uint8), kind=KIND_F4SC)
    assert cart.current_bank == 7
    cart_peek(cart, 0x1FF4)
    assert cart.current_bank == 0
    cart_peek(cart, 0x1FFB)
    assert cart.current_bank == 7


def test_f6sc_requires_16k_rom():
    with pytest.raises(ValueError):
        make_cart(jnp.zeros((8192,), dtype=jnp.uint8), kind=KIND_F6SC)


def test_f4sc_requires_32k_rom():
    with pytest.raises(ValueError):
        make_cart(jnp.zeros((16384,), dtype=jnp.uint8), kind=KIND_F4SC)


def test_f6sc_ram_persists_across_bank_switch():
    cart = make_cart(jnp.zeros((16384,), dtype=jnp.uint8), kind=KIND_F6SC)
    cart_poke(cart, 0x1000 + 0x05, 0x42)
    cart_peek(cart, 0x1FF7)             # bank 1
    assert cart_peek(cart, 0x1080 + 0x05) == 0x42
