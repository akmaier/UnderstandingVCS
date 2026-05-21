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

P2 status: this module implements the address decode and the RAM/ROM
peek/poke. TIA and RIOT regions are stubs (read 0, ignore writes); proper
behaviour lands in P3 (TIA), P4 (RIOT), and P5 (bank-switching cartridges).

A second, simpler memory model — a flat 65,536-byte `jnp.ndarray` — is
supported too, so the P1 unit tests can keep building tiny programs at
arbitrary addresses without needing a Bus. `peek` / `poke` dispatch on
the world type.
"""

from __future__ import annotations

from typing import NamedTuple, Union

import jax.numpy as jnp


# --------------------------------------------------------------------------- #
# Bus type
# --------------------------------------------------------------------------- #

class Bus(NamedTuple):
    """6507 system bus state.

    Attributes
    ----------
    ram : jnp.ndarray (128,) uint8
        128 bytes of RIOT internal RAM.
    rom : jnp.ndarray (4096,) uint8
        Cartridge ROM image. P2 supports flat 4K ROMs only; bank-switched
        cartridges (8K, 16K, 32K, …) land in P5.
    """
    ram: jnp.ndarray
    rom: jnp.ndarray


def initial_bus(rom: Union[jnp.ndarray, None] = None) -> Bus:
    """Build a `Bus` with all-zero RAM and the given ROM (default: all zeros)."""
    if rom is None:
        rom = jnp.zeros((4096,), dtype=jnp.uint8)
    if rom.shape != (4096,):
        raise ValueError(
            f"P2 expects a flat 4K ROM; got shape {rom.shape}. "
            f"Bank-switched ROMs land in P5."
        )
    if rom.dtype != jnp.uint8:
        rom = rom.astype(jnp.uint8)
    return Bus(ram=jnp.zeros((128,), dtype=jnp.uint8), rom=rom)


# --------------------------------------------------------------------------- #
# peek / poke — type-dispatched between Bus and flat memory
# --------------------------------------------------------------------------- #

# A "world" is whatever the CPU steps against. Either a real Bus or — for
# the P1 unit tests — a flat 65,536-byte jnp.ndarray.
World = Union[Bus, jnp.ndarray]


def peek(world: World, addr: int) -> int:
    """Read a byte from `world` at the 16-bit address `addr`.

    For a Bus, the 6507 address decode is honoured (13-bit mirror, then
    TIA / RIOT / RAM / ROM region selection).
    For a flat `jnp.ndarray`, the byte at `addr & 0xFFFF` is returned.
    """
    if isinstance(world, Bus):
        return _bus_peek(world, addr)
    return int(world[addr & 0xFFFF])


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

def _bus_peek(bus: Bus, addr: int) -> int:
    addr = addr & 0x1FFF  # 6507 13-bit mirror
    if addr & 0x1000:
        # Cartridge ROM ($1000–$1FFF).
        return int(bus.rom[addr & 0x0FFF])
    if not (addr & 0x80):
        # TIA region (A7=0). Most TIA registers are write-only; the readable
        # subset (INPT*, CXM*, etc.) lands in P3. Return 0 for now.
        return 0
    if addr & 0x200:
        # RIOT I/O (A9=1). Stub for P2; proper timer + ports land in P4.
        return 0
    # RIOT RAM (A7=1, A9=0). 128 bytes, mirrored at offset addr & 0x7F.
    return int(bus.ram[addr & 0x7F])


def _bus_poke(bus: Bus, addr: int, value: int) -> Bus:
    addr = addr & 0x1FFF
    if addr & 0x1000:
        return bus  # ROM is read-only.
    if not (addr & 0x80):
        return bus  # TIA write-stub (P3 will trigger real TIA side-effects).
    if addr & 0x200:
        return bus  # RIOT I/O write-stub (P4).
    return bus._replace(
        ram=bus.ram.at[addr & 0x7F].set(jnp.uint8(value & 0xFF))
    )
