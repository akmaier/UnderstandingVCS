"""P5 tests — cartridge formats 2K / 4K / F8 / F6 / F4 with bank switching.

The bus-level integration (cart_peek / cart_poke routed through
`peek`/`poke` on a Bus) is what real CPU programs see. End-to-end tests
run a small program that triggers bank switches via `LDA` to a hotspot
and check that subsequent fetches return bank-1 bytes.
"""

import jax.numpy as jnp
import pytest

from jaxtari.bus import initial_bus, peek, poke
from jaxtari.cart import (
    KIND_2K,
    KIND_4K,
    KIND_F4,
    KIND_F6,
    KIND_F8,
    Cart,
    cart_peek,
    cart_poke,
    make_cart,
)
from jaxtari.cpu.m6502 import step
from jaxtari.types import CPUState, initial_cpu_state


def _state(**fields) -> CPUState:
    s = initial_cpu_state()
    return s._replace(**{
        k: jnp.asarray(v).astype(getattr(s, k).dtype) for k, v in fields.items()
    })


def _multi_bank_rom(bank_size: int, n_banks: int) -> jnp.ndarray:
    """Build an `n_banks × bank_size` ROM where each bank is filled with
    its bank index (bank 0 → all 0x00, bank 1 → all 0x01, etc.). Used by
    bank-switching tests so the per-bank content is trivially identifiable.
    """
    rom = jnp.zeros((n_banks * bank_size,), dtype=jnp.uint8)
    for b in range(n_banks):
        rom = rom.at[b * bank_size : (b + 1) * bank_size].set(jnp.uint8(b))
    return rom


# --------------------------------------------------------------------------- #
# make_cart — size → kind auto-detect
# --------------------------------------------------------------------------- #

def test_make_cart_detects_2k():
    cart = make_cart(jnp.zeros((2048,), dtype=jnp.uint8))
    assert cart.kind == KIND_2K
    assert cart.current_bank == 0


def test_make_cart_detects_4k():
    cart = make_cart(jnp.zeros((4096,), dtype=jnp.uint8))
    assert cart.kind == KIND_4K
    assert cart.current_bank == 0


def test_make_cart_detects_f8_and_starts_in_last_bank():
    cart = make_cart(jnp.zeros((8192,), dtype=jnp.uint8))
    assert cart.kind == KIND_F8
    assert cart.current_bank == 1


def test_make_cart_detects_f6_and_starts_in_last_bank():
    cart = make_cart(jnp.zeros((16384,), dtype=jnp.uint8))
    assert cart.kind == KIND_F6
    assert cart.current_bank == 3


def test_make_cart_detects_f4_and_starts_in_last_bank():
    cart = make_cart(jnp.zeros((32768,), dtype=jnp.uint8))
    assert cart.kind == KIND_F4
    assert cart.current_bank == 7


def test_make_cart_rejects_unknown_size():
    with pytest.raises(ValueError, match="unrecognised ROM size"):
        make_cart(jnp.zeros((1234,), dtype=jnp.uint8))


# --------------------------------------------------------------------------- #
# 2K — mirroring across the 4 KB cart window
# --------------------------------------------------------------------------- #

def test_2k_mirrors_across_4k_window():
    rom = jnp.zeros((2048,), dtype=jnp.uint8)
    rom = rom.at[0].set(jnp.uint8(0xAA))
    rom = rom.at[0x7FF].set(jnp.uint8(0xBB))
    cart = make_cart(rom)
    # First half: $1000..$17FF reads from the 2K ROM directly.
    assert cart_peek(cart, 0x1000) == 0xAA
    assert cart_peek(cart, 0x17FF) == 0xBB
    # Second half: $1800..$1FFF mirrors the first half.
    assert cart_peek(cart, 0x1800) == 0xAA
    assert cart_peek(cart, 0x1FFF) == 0xBB


# --------------------------------------------------------------------------- #
# 4K — no bank switching, hotspot accesses are inert
# --------------------------------------------------------------------------- #

def test_4k_peek_at_hotspot_does_not_change_anything():
    rom = jnp.full((4096,), 0x33, dtype=jnp.uint8)
    cart = make_cart(rom)
    initial_bank = cart.current_bank
    assert cart_peek(cart, 0x1FF8) == 0x33
    assert cart.current_bank == initial_bank          # 4K has no banks


# --------------------------------------------------------------------------- #
# F8 — 2 banks, hotspots $1FF8 / $1FF9
# --------------------------------------------------------------------------- #

def test_f8_initial_bank_is_one():
    rom = _multi_bank_rom(0x1000, 2)
    cart = make_cart(rom)
    assert cart.current_bank == 1
    # Reading any non-hotspot address returns the bank-1 byte (0x01).
    assert cart_peek(cart, 0x1000) == 0x01
    assert cart_peek(cart, 0x1500) == 0x01


def test_f8_hotspot_1ff8_switches_to_bank_0():
    cart = make_cart(_multi_bank_rom(0x1000, 2))
    # Hotspot access reads from CURRENT bank, then switches.
    val = cart_peek(cart, 0x1FF8)
    # The byte at $1FF8 in bank 1 (last byte block of bank 1) = 0x01.
    assert val == 0x01
    # After the read, bank is now 0.
    assert cart.current_bank == 0
    # Subsequent reads return bank-0 bytes.
    assert cart_peek(cart, 0x1000) == 0x00


def test_f8_hotspot_1ff9_switches_to_bank_1():
    cart = make_cart(_multi_bank_rom(0x1000, 2))
    # Start in bank 1; switch to bank 0 first.
    cart_peek(cart, 0x1FF8)
    assert cart.current_bank == 0
    # Now hit $1FF9 → switch to bank 1.
    cart_peek(cart, 0x1FF9)
    assert cart.current_bank == 1


def test_f8_write_to_hotspot_also_switches():
    """Writes to hotspots fire the bank switch even though the byte is dropped."""
    cart = make_cart(_multi_bank_rom(0x1000, 2))
    assert cart.current_bank == 1
    cart_poke(cart, 0x1FF8, 0x00)
    assert cart.current_bank == 0


# --------------------------------------------------------------------------- #
# F6 — 4 banks, hotspots $1FF6..$1FF9
# --------------------------------------------------------------------------- #

def test_f6_initial_bank_is_three():
    cart = make_cart(_multi_bank_rom(0x1000, 4))
    assert cart.current_bank == 3
    assert cart_peek(cart, 0x1000) == 0x03


def test_f6_all_four_hotspots():
    cart = make_cart(_multi_bank_rom(0x1000, 4))
    for hotspot, expected_bank in [(0x1FF6, 0), (0x1FF7, 1), (0x1FF8, 2), (0x1FF9, 3)]:
        cart_peek(cart, hotspot)
        assert cart.current_bank == expected_bank, f"hotspot ${hotspot:04X}"
        assert cart_peek(cart, 0x1500) == expected_bank


# --------------------------------------------------------------------------- #
# F4 — 8 banks, hotspots $1FF4..$1FFB
# --------------------------------------------------------------------------- #

def test_f4_initial_bank_is_seven():
    cart = make_cart(_multi_bank_rom(0x1000, 8))
    assert cart.current_bank == 7


def test_f4_all_eight_hotspots():
    cart = make_cart(_multi_bank_rom(0x1000, 8))
    for hotspot, expected_bank in [
        (0x1FF4, 0), (0x1FF5, 1), (0x1FF6, 2), (0x1FF7, 3),
        (0x1FF8, 4), (0x1FF9, 5), (0x1FFA, 6), (0x1FFB, 7),
    ]:
        cart_peek(cart, hotspot)
        assert cart.current_bank == expected_bank, f"hotspot ${hotspot:04X}"


# --------------------------------------------------------------------------- #
# Bus integration — hotspot fires via the bus's mirror addresses too
# --------------------------------------------------------------------------- #

def test_bus_peek_at_fff8_mirror_fires_f8_hotspot():
    """\$FFF8 on the CPU bus → 13-bit mirror → \$1FF8 → F8 hotspot."""
    bus = initial_bus(_multi_bank_rom(0x1000, 2))
    assert bus.cart.current_bank == 1
    peek(bus, 0xFFF8)
    assert bus.cart.current_bank == 0


def test_bus_peek_at_fff9_mirror_fires_f8_hotspot():
    bus = initial_bus(_multi_bank_rom(0x1000, 2))
    bus_val_before = peek(bus, 0xFFF8)   # → bank 0
    assert bus.cart.current_bank == 0
    peek(bus, 0xFFF9)
    assert bus.cart.current_bank == 1


# --------------------------------------------------------------------------- #
# End-to-end — CPU program with bank switching
# --------------------------------------------------------------------------- #

def test_f8_bank_switch_via_cpu_lda():
    """An F8 ROM that switches from bank 1 to bank 0 mid-program. The
    instruction stream must exist in BOTH banks at the same offsets,
    because after the switch the CPU's next fetch comes from bank 0.

    Identical instructions in both banks, but a data byte at \$F0FF
    differs (bank 0 → \$77, bank 1 → \$88). The test asserts the final
    LDA reads bank 0's value, proving the switch took effect.
    """
    rom = jnp.zeros((8192,), dtype=jnp.uint8)
    program = [
        0xA9, 0x22,                    # LDA #$22 at $F000
        0x2C, 0xF8, 0xFF,              # BIT $FFF8 at $F002 → hotspot switch
        0xAD, 0xFF, 0xF0,              # LDA $F0FF at $F005
    ]
    for i, b in enumerate(program):
        rom = rom.at[i].set(jnp.uint8(b))                  # bank 0 copy
        rom = rom.at[0x1000 + i].set(jnp.uint8(b))         # bank 1 copy
    rom = rom.at[0x0FF].set(jnp.uint8(0x77))               # bank 0: $F0FF = $77
    rom = rom.at[0x10FF].set(jnp.uint8(0x88))              # bank 1: $F0FF = $88

    bus = initial_bus(rom)
    assert bus.cart.current_bank == 1                      # F8 boots in bank 1
    s = _state(PC=0xF000)
    s, bus = step(s, bus)                                  # LDA #$22
    assert int(s.A) == 0x22
    s, bus = step(s, bus)                                  # BIT $FFF8 → bank 0
    assert bus.cart.current_bank == 0
    s, bus = step(s, bus)                                  # LDA $F0FF — bank 0 → $77
    assert int(s.A) == 0x77
