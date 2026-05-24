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
  F8SC     **P5b** — F8 with 128 B of on-cart RAM at \$1000–\$10FF:
           writes go to \$1000–\$107F, reads come from \$1080–\$10FF
           (mirror of the same 128 bytes). Bank-switching otherwise
           identical to F8. Size auto-detect picks plain F8 for an 8K
           ROM; pass `kind=KIND_F8SC` to `make_cart` / `initial_bus`
           to override when you know the cart has SC RAM.
  F6      16 KB ROM, 4 banks. Hotspots \$1FF6..\$1FF9. Initial bank = 3.
  F4      32 KB ROM, 8 banks. Hotspots \$1FF4..\$1FFB. Initial bank = 7.

Deferred to a P5 follow-up:
  * SC variants of F6/F4 (mirror P5b's F8SC for the larger banked
    carts — same shape, different bank count).
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

KIND_2K   = 0
KIND_4K   = 1
KIND_F8   = 2
KIND_F6   = 3
KIND_F4   = 4
KIND_F8SC = 5     # P5b — F8 + 128 B on-cart RAM
KIND_F6SC = 6     # P5b — F6 + 128 B on-cart RAM
KIND_F4SC = 7     # P5b — F4 + 128 B on-cart RAM
KIND_E0   = 8     # P5c — Parker Bros 8K, 4 × 1K slice slots

_SIZE_TO_KIND = {
    2048:  KIND_2K,
    4096:  KIND_4K,
    8192:  KIND_F8,         # F8SC is the same size; pass `kind=` to override
    16384: KIND_F6,         # F6SC same — pass `kind=KIND_F6SC` to override
    32768: KIND_F4,         # F4SC same — pass `kind=KIND_F4SC` to override
}

# F8SC / F6SC / F4SC all carry the 128 B on-cart RAM at $1000-$10FF.
_SC_KINDS = frozenset({KIND_F8SC, KIND_F6SC, KIND_F4SC})

# Map size-sensitive kinds back to their expected ROM size, used by
# `make_cart` to reject silly inputs early. Covers all formats that
# can be selected by an explicit `kind=` override.
_SC_EXPECTED_SIZE = {
    KIND_F8SC: 8192,
    KIND_F6SC: 16384,
    KIND_F4SC: 32768,
    KIND_E0:   8192,         # P5c
}

# Bank-switch hotspot addresses within the 13-bit-masked cart window.
F8_HOTSPOTS = {0x1FF8: 0, 0x1FF9: 1}
F6_HOTSPOTS = {0x1FF6: 0, 0x1FF7: 1, 0x1FF8: 2, 0x1FF9: 3}
F4_HOTSPOTS = {
    0x1FF4: 0, 0x1FF5: 1, 0x1FF6: 2, 0x1FF7: 3,
    0x1FF8: 4, 0x1FF9: 5, 0x1FFA: 6, 0x1FFB: 7,
}

# Bank typically containing the reset vector at power-on — the highest bank.
# For E0, "bank" is a tuple of (slot0_slice, slot1_slice, slot2_slice) since
# the format selects three different 1K slices simultaneously. The last 1K
# is fixed at slice 7; we don't track it.
_DEFAULT_BANK = {
    KIND_2K:   0,
    KIND_4K:   0,
    KIND_F8:   1,
    KIND_F8SC: 1,
    KIND_F6:   3,
    KIND_F6SC: 3,
    KIND_F4:   7,
    KIND_F4SC: 7,
    KIND_E0:   0,                   # placeholder; E0 uses `slice_slots` instead
}

# E0 cart: 8K split into 8 × 1K slices. Three slots at $1000-$13FF /
# $1400-$17FF / $1800-$1BFF map to one of the 8 slices each, via hotspot
# reads/writes:
#   $1FE0..$1FE7  → set slot 0 to slice 0..7
#   $1FE8..$1FEF  → set slot 1 to slice 0..7
#   $1FF0..$1FF7  → set slot 2 to slice 0..7
# The fourth slot at $1C00-$1FFF is hard-wired to slice 7 (which holds
# the reset vector at $FFFC/$FFFD = $1FFC/$1FFD in cart-window terms).
E0_HOTSPOT_BASE = (0x1FE0, 0x1FE8, 0x1FF0)
E0_FIXED_SLICE  = 7

# F8SC on-cart RAM layout (P5b). 128 bytes, write at $1000-$107F and read
# at $1080-$10FF. Both areas alias the same 128-byte buffer via
# `addr & 0x7F`.
_SC_RAM_BYTES   = 128
_SC_WRITE_BASE  = 0x1000     # write area: $1000-$107F
_SC_READ_BASE   = 0x1080     # read  area: $1080-$10FF
_SC_AREA_MASK   = 0x1F80     # high bits = $1000 or $1080
_SC_OFFSET_MASK = 0x007F     # low 7 bits index into the 128-byte buffer


# --------------------------------------------------------------------------- #
# Cart state
# --------------------------------------------------------------------------- #

class Cart:
    """Mutable cart state: ROM image, kind tag, and (for bank-switched
    carts) the currently-mapped bank index. For F8SC, also carries
    the 128-byte on-cart RAM buffer.

    Mutable because hotspot reads in real hardware change bank state as a
    side-effect of the read itself; modelling that in a pure-functional
    NamedTuple would require returning a new world from `peek`, which
    would ripple through every memory-access call site in the CPU.
    """

    __slots__ = ("kind", "rom", "current_bank", "ram", "slice_slots")

    def __init__(self, kind: int, rom: jnp.ndarray) -> None:
        self.kind = kind
        self.rom = rom
        self.current_bank = _DEFAULT_BANK[kind]
        # Plain Python `bytearray` for the 128 B on-cart RAM: fast to
        # mutate per write, and we never need gradient flow through cart
        # RAM (SOFT mode doesn't yet model bank state at all — see
        # STATUS.md P7f-e).
        self.ram = bytearray(_SC_RAM_BYTES) if kind in _SC_KINDS else bytearray(0)
        # P5c — E0: three mutable 1K-slice indices. The fourth slot is
        # fixed at slice 7. Default boot state is (0, 0, 0) — Parker
        # Bros titles all start their reset vector inside slot 3 (the
        # fixed slice) so any initial slot-0..2 mapping works.
        self.slice_slots = [0, 0, 0] if kind == KIND_E0 else []

    def __repr__(self) -> str:
        names = {KIND_2K: "2K", KIND_4K: "4K",
                 KIND_F8: "F8", KIND_F8SC: "F8SC",
                 KIND_F6: "F6", KIND_F6SC: "F6SC",
                 KIND_F4: "F4", KIND_F4SC: "F4SC",
                 KIND_E0: "E0"}
        if self.kind == KIND_E0:
            return (f"Cart({names[self.kind]}, rom_size={len(self.rom)}, "
                    f"slices={tuple(self.slice_slots)})")
        return (f"Cart({names[self.kind]}, "
                f"rom_size={len(self.rom)}, bank={self.current_bank})")


def make_cart(rom: jnp.ndarray, *, kind: int | None = None) -> Cart:
    """Build a `Cart`, auto-detecting the kind from ROM size by default.

    Pass `kind=KIND_F8SC` to override the auto-detection for an 8K cart
    you know to be F8SC (size alone can't distinguish F8 from F8SC).
    Signature-based detection for the ambiguous formats is a deferred
    follow-up; for now we assume the canonical mapping in `_SIZE_TO_KIND`
    and trust the caller for overrides.
    """
    n = len(rom)
    if kind is None:
        if n not in _SIZE_TO_KIND:
            raise ValueError(
                f"unrecognised ROM size {n} bytes. "
                f"P5 supports sizes {sorted(_SIZE_TO_KIND.keys())}."
            )
        kind = _SIZE_TO_KIND[n]
    else:
        expected = _SC_EXPECTED_SIZE.get(kind)
        if expected is not None and n != expected:
            raise ValueError(
                f"kind {kind!r} requires a {expected}-byte ROM (got {n}).")
    if rom.dtype != jnp.uint8:
        rom = rom.astype(jnp.uint8)
    return Cart(kind=kind, rom=rom)


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

    # P5c — E0: four 1K slots, three mutable + one fixed at slice 7.
    # Reads compute (slice, in-slice-offset) from the cart-window
    # address; hotspot reads (slot 3 only, $1FE0-$1FF7) ALSO trigger
    # a slice swap as a side-effect of the read.
    if cart.kind == KIND_E0:
        value = int(cart.rom[_e0_offset(cart, a)])
        _maybe_switch_e0(cart, a)
        return value

    # P5b — F8SC / F6SC / F4SC on-cart RAM dispatch. The read area at
    # $1080-$10FF returns the 128-byte RAM buffer (mirrored every 128
    # bytes). The write area at $1000-$107F passes through to ROM on
    # reads (real hardware blocks ROM here, but most software only
    # *reads* from the read window — a value from the bank is the
    # de-facto "open bus" behaviour Stella exposes). Hotspots fire
    # only outside the RAM window.
    if cart.kind in _SC_KINDS and (a & 0x1F80) == _SC_READ_BASE:
        return int(cart.ram[a & _SC_OFFSET_MASK])

    bank_offset = cart.current_bank * 0x1000
    value = int(cart.rom[bank_offset + (a & 0x0FFF)])
    _maybe_switch_bank(cart, a)
    return value


def cart_poke(cart: Cart, addr: int, value: int) -> None:
    """ROM is read-only, so the value is discarded. But hotspot accesses
    still fire on writes, switching the bank.

    P5b — F8SC / F6SC / F4SC writes to the RAM-write area
    ($1000-$107F) store into the 128-byte on-cart RAM buffer; writes
    elsewhere fall through to the normal "ROM is read-only / hotspots
    might fire" path.

    P5c — E0 hotspots also fire on writes (every cart access).
    """
    a = addr & 0x1FFF
    if cart.kind in _SC_KINDS and (a & 0x1F80) == _SC_WRITE_BASE:
        cart.ram[a & _SC_OFFSET_MASK] = value & 0xFF
        return
    if cart.kind == KIND_E0:
        _maybe_switch_e0(cart, a)
        return
    _maybe_switch_bank(cart, a)


# --------------------------------------------------------------------------- #
# Internal bank-switch logic
# --------------------------------------------------------------------------- #

def _maybe_switch_bank(cart: Cart, masked_addr: int) -> None:
    if cart.kind in (KIND_F8, KIND_F8SC):
        if masked_addr in F8_HOTSPOTS:
            cart.current_bank = F8_HOTSPOTS[masked_addr]
    elif cart.kind in (KIND_F6, KIND_F6SC):
        if masked_addr in F6_HOTSPOTS:
            cart.current_bank = F6_HOTSPOTS[masked_addr]
    elif cart.kind in (KIND_F4, KIND_F4SC):
        if masked_addr in F4_HOTSPOTS:
            cart.current_bank = F4_HOTSPOTS[masked_addr]


def _e0_offset(cart: Cart, masked_addr: int) -> int:
    """Translate a cart-window address `$1000-$1FFF` to the absolute
    ROM offset, honouring E0's three mutable slice slots and the
    fixed-slice-7 fourth slot."""
    slot = (masked_addr - 0x1000) >> 10            # 0..3
    in_slice = masked_addr & 0x03FF                # offset within the 1K slice
    if slot < 3:
        slice_idx = cart.slice_slots[slot]
    else:
        slice_idx = E0_FIXED_SLICE
    return slice_idx * 0x0400 + in_slice


def _maybe_switch_e0(cart: Cart, masked_addr: int) -> None:
    """E0 hotspot side-effect: $1FE0..$1FE7 → set slot 0, $1FE8..$1FEF
    → slot 1, $1FF0..$1FF7 → slot 2. Hotspots fire on any cart access
    (read or write) — same convention as F8/F6/F4."""
    for slot_idx, base in enumerate(E0_HOTSPOT_BASE):
        if base <= masked_addr <= base + 7:
            cart.slice_slots[slot_idx] = masked_addr - base
            return
