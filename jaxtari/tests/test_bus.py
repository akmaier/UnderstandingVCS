"""P2 tests — 6507 bus + 13-bit address decode + RAM/ROM/TIA/RIOT regions.

These tests work directly against `jaxtari.bus.Bus`. A second group runs the
P1 `step()` through a Bus to verify that the same CPU code that passes the
flat-memory P1 tests also works against the proper bus.
"""

import jax.numpy as jnp

from jaxtari.bus import Bus, initial_bus, peek, poke
from jaxtari.cpu.m6502 import step
from jaxtari.cpu.tables import FLAG_U
from jaxtari.types import CPUState, initial_cpu_state


# --------------------------------------------------------------------------- #
# Direct peek / poke against a Bus
# --------------------------------------------------------------------------- #

def test_bus_ram_read_write_at_canonical_address():
    bus = initial_bus()
    bus = poke(bus, 0x0080, 0x42)   # canonical RAM slot
    assert peek(bus, 0x0080)[0] == 0x42
    # Internal storage is at offset 0 of the 128 B RAM.
    assert int(bus.ram[0x00]) == 0x42


def test_bus_ram_mirror_at_stack_page():
    """$0180 (RAM mirror in the stack page) should alias $0080."""
    bus = initial_bus()
    bus = poke(bus, 0x0180, 0x55)
    assert peek(bus, 0x0080)[0] == 0x55
    assert peek(bus, 0x0180)[0] == 0x55


def test_bus_ram_uses_low_7_bits_of_address():
    """All addresses with A12=0, A7=1, A9=0 land in the same 128 B RAM."""
    bus = initial_bus()
    bus = poke(bus, 0x0098, 0xAB)        # RAM index 0x18
    assert peek(bus, 0x0098)[0] == 0xAB
    assert peek(bus, 0x0198)[0] == 0xAB   # mirror in stack page
    assert peek(bus, 0x0498)[0] == 0xAB   # mirror at A9=A10=1 still hits RAM (A9=0 here)


def test_bus_13bit_mirror_high_addresses_wrap():
    """Addresses outside the 8K window wrap via the 13-bit address bus."""
    rom = jnp.zeros((4096,), dtype=jnp.uint8).at[0x100].set(jnp.uint8(0xEE))
    bus = initial_bus(rom)
    # ROM byte at ROM offset 0x100 = bus address 0x1100
    assert peek(bus, 0x1100)[0] == 0xEE
    # 13-bit mirror: $F100 → $1100
    assert peek(bus, 0xF100)[0] == 0xEE
    # 13-bit mirror: $9100 → $1100
    assert peek(bus, 0x9100)[0] == 0xEE


def test_bus_tia_region_reads_zero_and_writes_ignored():
    bus = initial_bus()
    assert peek(bus, 0x0000)[0] == 0      # TIA register $00 (P2 stub)
    assert peek(bus, 0x007F)[0] == 0      # last TIA register in the bank
    bus2 = poke(bus, 0x0001, 0xFF)        # write should be silently dropped
    # RAM is unchanged (TIA write doesn't leak into RAM).
    assert bool(jnp.array_equal(bus.ram, bus2.ram))


def test_bus_riot_io_region_does_not_corrupt_ram():
    """As of P4 the RIOT region returns real values (port reads, timer
    status, etc.) — see test_riot.py for the actual semantics. What still
    holds: RIOT writes do not leak into the RAM bank."""
    bus = initial_bus()
    bus2 = poke(bus, 0x0284, 0xFF)
    assert bool(jnp.array_equal(bus.ram, bus2.ram))


def test_bus_rom_is_read_only():
    rom = jnp.full((4096,), 0xAA, dtype=jnp.uint8)
    bus = initial_bus(rom)
    assert peek(bus, 0x1000)[0] == 0xAA
    bus2 = poke(bus, 0x1000, 0x55)        # ROM write — value is discarded
    assert peek(bus2, 0x1000)[0] == 0xAA
    # ROM array is structurally unchanged (cart.rom is the same underlying array).
    assert bool(jnp.array_equal(bus.cart.rom, bus2.cart.rom))


def test_bus_rejects_unrecognised_rom_size():
    import pytest
    with pytest.raises(ValueError, match="unrecognised ROM size"):
        initial_bus(jnp.zeros((3000,), dtype=jnp.uint8))


def test_bus_accepts_bank_switched_rom_sizes():
    """Sizes 2K/4K/8K/16K/32K all build a valid Bus as of P5."""
    for size in (2048, 4096, 8192, 16384, 32768):
        bus = initial_bus(jnp.zeros((size,), dtype=jnp.uint8))
        assert bus.cart.rom.shape == (size,)


# --------------------------------------------------------------------------- #
# CPU running through the Bus
# --------------------------------------------------------------------------- #

def _state(**fields) -> CPUState:
    s = initial_cpu_state()
    return s._replace(**{
        k: jnp.asarray(v).astype(getattr(s, k).dtype) for k, v in fields.items()
    })


def test_step_lda_immediate_via_bus():
    """`LDA #$42` at ROM $F000 (= $1000 after mirror) sets A=$42."""
    rom = jnp.zeros((4096,), dtype=jnp.uint8)
    rom = rom.at[0].set(jnp.uint8(0xA9))   # LDA imm
    rom = rom.at[1].set(jnp.uint8(0x42))
    bus = initial_bus(rom)
    s = _state(PC=0xF000)
    s2, bus2 = step(s, bus)
    assert int(s2.A) == 0x42
    assert int(s2.PC) == 0xF002
    assert int(s2.cycles) == 2
    # No RAM changes.
    assert bool(jnp.array_equal(bus.ram, bus2.ram))


def test_step_sta_then_lda_round_trip_via_bus():
    """STA $80 / LDA $80 should round-trip through RAM."""
    rom = jnp.zeros((4096,), dtype=jnp.uint8)
    # Program at ROM offset 0 (= bus $1000 = mirrored $F000):
    #   $A9 $77         LDA #$77
    #   $85 $80         STA $80
    #   $A9 $00         LDA #$00       ; clobber A
    #   $A5 $80         LDA $80
    program = [0xA9, 0x77, 0x85, 0x80, 0xA9, 0x00, 0xA5, 0x80]
    for i, b in enumerate(program):
        rom = rom.at[i].set(jnp.uint8(b))
    bus = initial_bus(rom)
    s = _state(PC=0xF000)
    for _ in range(4):
        s, bus = step(s, bus)
    assert int(s.A) == 0x77
    # RAM offset 0 (canonical $80) holds $77.
    assert int(bus.ram[0]) == 0x77


def test_step_jsr_rts_via_bus_stack_lives_in_ram():
    """JSR pushes to $01FD/$01FC which are RAM mirror addresses; RTS pops them.

    With initial SP=$FD, JSR pushes 2 bytes so SP→$FB. The pushed bytes land
    at $01FD and $01FC, which both mirror to RAM (since $01xx with A7=1 is
    in the RAM mirror band).
    """
    rom = jnp.zeros((4096,), dtype=jnp.uint8)
    # At $F000: JSR $F005; at $F005: RTS
    program = {0: 0x20, 1: 0x05, 2: 0xF0, 5: 0x60}
    for off, b in program.items():
        rom = rom.at[off].set(jnp.uint8(b))
    bus = initial_bus(rom)
    s = _state(PC=0xF000, SP=0xFD)
    s, bus = step(s, bus)            # JSR
    assert int(s.PC) == 0xF005
    assert int(s.SP) == 0xFB
    # The pushed return-address bytes are now visible in RAM at offsets
    # 0x7D and 0x7C (mirror of $01FD / $01FC).
    assert int(bus.ram[0x7D]) == 0xF0   # high
    assert int(bus.ram[0x7C]) == 0x02   # low
    s, bus = step(s, bus)            # RTS
    assert int(s.PC) == 0xF003
    assert int(s.SP) == 0xFD


# --------------------------------------------------------------------------- #
# Floating-bus quirk (PXC1-x round 3 — TIA reads OR in last-bus-byte noise on
# the un-driven data lines, matching xitari's `System::peek/poke` + the real
# 1A05 hardware).
# --------------------------------------------------------------------------- #

def test_bus_data_bus_state_initial():
    """Fresh bus: `data_bus_state` is 0."""
    bus = initial_bus()
    assert bus.data_bus_state == 0


def test_bus_data_bus_state_updated_by_ram_write():
    bus = initial_bus()
    bus = poke(bus, 0x0080, 0x48)
    assert bus.data_bus_state == 0x48


def test_bus_data_bus_state_updated_by_ram_read():
    bus = initial_bus()
    bus = poke(bus, 0x0080, 0xA5)
    # After the poke, data_bus_state is 0xA5. Reading another RAM cell
    # (value 0) updates it to 0.
    _, bus = peek(bus, 0x0081)
    assert bus.data_bus_state == 0


def test_bus_tia_read_or_noise_into_undriven_bits():
    """Floating-bus: CXM0P (reg 0, driven mask $C0) returns the driven
    bits + the low 6 of `data_bus_state`."""
    bus = initial_bus()
    # Seed noise via a RAM poke. Noise = 0x48 = 0100_1000 → low6 = 0x08.
    bus = poke(bus, 0x0080, 0x48)
    value, _ = peek(bus, 0x0000)        # CXM0P. No collisions → driven=0.
    assert value == 0x08, f"expected noise bits 0x08, got {value:#04x}"


def test_bus_tia_read_full_noise_on_unused_register():
    """Reg $0F has no driven bits — TIA returns full floating-bus byte."""
    bus = initial_bus()
    bus = poke(bus, 0x0080, 0xFE)
    value, _ = peek(bus, 0x000F)
    assert value == 0xFE


def test_bus_inpt4_d7_high_with_noise():
    """INPT4 (reg $0C, driven mask $80): D7 from trigger latch, D6-D0
    from noise. Default INPT init is 0x80 → trigger up, so result is
    0x80 | low7(noise)."""
    bus = initial_bus()
    bus = poke(bus, 0x0080, 0x73)       # noise = 0x73
    value, _ = peek(bus, 0x000C)
    # D7 driven (1, trigger idle) + D6..D0 from noise (0x73 & 0x7F = 0x73).
    assert value == 0x80 | (0x73 & 0x7F)
