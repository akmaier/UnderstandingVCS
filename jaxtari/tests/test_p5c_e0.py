"""P5c tests — E0 cart format (Parker Bros 8K, 4 × 1K slice slots).

E0 cart layout (xitari `CartE0.cxx`):
  $1000-$13FF  slot 0 — mutable, current slice from `slice_slots[0]`
  $1400-$17FF  slot 1 — mutable, current slice from `slice_slots[1]`
  $1800-$1BFF  slot 2 — mutable, current slice from `slice_slots[2]`
  $1C00-$1FFF  slot 3 — fixed at slice 7 (holds the reset vector)

Hotspots fire on any cart access (read or write):
  $1FE0..$1FE7  → set slot 0 to slice 0..7
  $1FE8..$1FEF  → set slot 1 to slice 0..7
  $1FF0..$1FF7  → set slot 2 to slice 0..7
"""

import jax.numpy as jnp
import pytest

from jaxtari.bus.system import initial_bus, peek
from jaxtari.cart import KIND_E0, Cart, cart_peek, cart_poke, make_cart


def _slice_rom() -> jnp.ndarray:
    """Build an 8K ROM where each slice's first byte equals its slice
    index — makes "which slice is in which slot" obvious from a peek."""
    rom = jnp.zeros((8192,), dtype=jnp.uint8)
    for s in range(8):
        rom = rom.at[s * 0x0400].set(jnp.uint8(s))
        # Mark middle and end of each slice too so the test can verify
        # the full 1K range maps correctly.
        rom = rom.at[s * 0x0400 + 0x200].set(jnp.uint8(0x40 | s))
        rom = rom.at[s * 0x0400 + 0x3FF].set(jnp.uint8(0x80 | s))
    return rom


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

def test_e0_kind_explicit_8k_rom():
    cart = make_cart(_slice_rom(), kind=KIND_E0)
    assert cart.kind == KIND_E0
    assert cart.slice_slots == [0, 0, 0]


def test_e0_requires_8k_rom():
    with pytest.raises(ValueError):
        make_cart(jnp.zeros((4096,), dtype=jnp.uint8), kind=KIND_E0)


def test_e0_autodetect_picks_f8_for_8k():
    """No `kind` override → 8K ROM defaults to F8 (not E0). Size alone
    can't disambiguate."""
    from jaxtari.cart import KIND_F8
    cart = make_cart(_slice_rom())
    assert cart.kind == KIND_F8


# --------------------------------------------------------------------------- #
# Slot 0 mapping
# --------------------------------------------------------------------------- #

def test_e0_slot0_default_is_slice_0():
    cart = make_cart(_slice_rom(), kind=KIND_E0)
    # Slot 0 = $1000-$13FF. First byte should be slice 0's first byte.
    assert cart_peek(cart, 0x1000) == 0


def test_e0_slot0_switches_to_slice_3_via_hotspot():
    cart = make_cart(_slice_rom(), kind=KIND_E0)
    cart_peek(cart, 0x1FE3)                        # hotspot — slot 0 ← slice 3
    assert cart.slice_slots[0] == 3
    assert cart_peek(cart, 0x1000) == 3


def test_e0_slot0_all_8_slices_selectable():
    cart = make_cart(_slice_rom(), kind=KIND_E0)
    for s in range(8):
        cart_peek(cart, 0x1FE0 + s)                 # slot 0 ← slice s
        assert cart_peek(cart, 0x1000) == s


# --------------------------------------------------------------------------- #
# Slot 1 / slot 2 mapping
# --------------------------------------------------------------------------- #

def test_e0_slot1_default_and_switch():
    cart = make_cart(_slice_rom(), kind=KIND_E0)
    assert cart_peek(cart, 0x1400) == 0
    cart_peek(cart, 0x1FEC)                        # slot 1 ← slice 4
    assert cart_peek(cart, 0x1400) == 4


def test_e0_slot2_default_and_switch():
    cart = make_cart(_slice_rom(), kind=KIND_E0)
    assert cart_peek(cart, 0x1800) == 0
    cart_peek(cart, 0x1FF5)                        # slot 2 ← slice 5
    assert cart_peek(cart, 0x1800) == 5


# --------------------------------------------------------------------------- #
# Slot 3 is hard-wired to slice 7
# --------------------------------------------------------------------------- #

def test_e0_slot3_is_fixed_slice_7():
    cart = make_cart(_slice_rom(), kind=KIND_E0)
    assert cart_peek(cart, 0x1C00) == 7


def test_e0_slot3_unaffected_by_hotspots():
    cart = make_cart(_slice_rom(), kind=KIND_E0)
    cart_peek(cart, 0x1FE3)                        # twiddle slot 0
    cart_peek(cart, 0x1FED)                        # twiddle slot 1
    cart_peek(cart, 0x1FF6)                        # twiddle slot 2
    assert cart_peek(cart, 0x1C00) == 7            # unchanged


# --------------------------------------------------------------------------- #
# Slice byte offsets cover the full 1K window
# --------------------------------------------------------------------------- #

def test_e0_full_1k_slice_visible_in_slot():
    """Mid-slice and end-of-slice bytes are addressable through the
    slot's 1K window — proves the in-slice offset math is right."""
    cart = make_cart(_slice_rom(), kind=KIND_E0)
    cart_peek(cart, 0x1FE2)                        # slot 0 ← slice 2
    assert cart_peek(cart, 0x1000)          == 2
    assert cart_peek(cart, 0x1000 + 0x200)  == 0x42
    assert cart_peek(cart, 0x13FF)          == 0x82


# --------------------------------------------------------------------------- #
# Hotspots fire on writes too
# --------------------------------------------------------------------------- #

def test_e0_hotspot_fires_on_write():
    cart = make_cart(_slice_rom(), kind=KIND_E0)
    cart_poke(cart, 0x1FE5, 0x00)                   # slot 0 ← slice 5 via write
    assert cart_peek(cart, 0x1000) == 5


def test_e0_non_hotspot_write_does_not_switch():
    cart = make_cart(_slice_rom(), kind=KIND_E0)
    cart_poke(cart, 0x1500, 0xAA)
    assert cart.slice_slots == [0, 0, 0]


# --------------------------------------------------------------------------- #
# Bus integration
# --------------------------------------------------------------------------- #

def test_e0_bus_peek_threads_through_cart():
    """A CPU peek of $1000 via the bus should land in E0 slot 0."""
    bus = initial_bus(_slice_rom(), cart_kind=KIND_E0)
    assert peek(bus, 0x1000)[0] == 0
    peek(bus, 0x1FE3)
    assert peek(bus, 0x1000)[0] == 3
