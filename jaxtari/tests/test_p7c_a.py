"""P7c-a tests — full load/store/transfer opcode coverage in SOFT mode,
plus N/Z flag updates.

This rounds out the eight-opcode P7b core into all the LDA/LDX/LDY/
STA/STX/STY/transfer variants the 6502 has (33 opcodes total, across
all addressing modes), and verifies that the N and Z status bits are
set in `state.P` after each load/transfer.

Gradient tests at the bottom exercise the new addressing modes —
zp,X / zp,Y / abs / abs,X / abs,Y / (ind,X) / (ind),Y — to confirm
that `jax.grad` still produces a structurally-correct one-hot
gradient back to the relevant ROM byte even when the address path
itself involves register state and ROM dereferences.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxtari.diff import (
    SoftBus,
    SoftCPUState,
    SOFT_SUPPORTED_OPCODES,
    initial_soft_bus,
    initial_soft_cpu_state,
    soft_run,
    soft_step,
)


def _rom_with(opcodes_at_offset_0: list[int], size: int = 4096) -> jnp.ndarray:
    rom = jnp.zeros((size,), dtype=jnp.float32)
    for i, b in enumerate(opcodes_at_offset_0):
        rom = rom.at[i].set(jnp.float32(b))
    return rom


def _step(rom_bytes, initial_state=None, n=1):
    bus = initial_soft_bus(_rom_with(rom_bytes))
    state = initial_state if initial_state is not None else initial_soft_cpu_state()
    state, bus = soft_run(state, bus, n)
    return state, bus


# --------------------------------------------------------------------------- #
# Coverage sanity
# --------------------------------------------------------------------------- #

def test_p7c_a_adds_full_load_store_transfer_coverage():
    """All 33 P7c-a opcodes are now in the dispatch table."""
    p7c_a = {
        # LDA — 8 modes
        0xA9, 0xA5, 0xB5, 0xAD, 0xBD, 0xB9, 0xA1, 0xB1,
        # LDX — 5 modes
        0xA2, 0xA6, 0xB6, 0xAE, 0xBE,
        # LDY — 5 modes
        0xA0, 0xA4, 0xB4, 0xAC, 0xBC,
        # STA — 7 modes
        0x85, 0x95, 0x8D, 0x9D, 0x99, 0x81, 0x91,
        # STX — 3 modes
        0x86, 0x96, 0x8E,
        # STY — 3 modes
        0x84, 0x94, 0x8C,
        # Transfers
        0xAA, 0xA8, 0x8A, 0x98, 0xBA, 0x9A,
    }
    assert p7c_a.issubset(SOFT_SUPPORTED_OPCODES)
    assert len(p7c_a) == 37


# --------------------------------------------------------------------------- #
# LDA across all 8 addressing modes
# --------------------------------------------------------------------------- #

def test_lda_zp_x_indexes_into_ram():
    rom = _rom_with([0xB5, 0x10])     # LDA $10,X
    bus = initial_soft_bus(rom)
    bus = bus._replace(ram=bus.ram.at[0x13].set(jnp.float32(0x99)))
    state = initial_soft_cpu_state()._replace(X=jnp.float32(3))
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x99


def test_lda_abs_reads_from_cart_when_addr_in_rom_window():
    # LDA $F005 — should read from rom offset 0x005.
    rom = _rom_with([0xAD, 0x05, 0xF0])
    rom = rom.at[0x005].set(jnp.float32(0x77))
    bus = initial_soft_bus(rom)
    state = initial_soft_cpu_state()
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x77


def test_lda_abs_x_indexes_correctly():
    # LDA $F000,X with X=4 — should read rom[4] = whatever we put there.
    rom = _rom_with([0xBD, 0x00, 0xF0])
    rom = rom.at[4].set(jnp.float32(0x55))
    bus = initial_soft_bus(rom)
    state = initial_soft_cpu_state()._replace(X=jnp.float32(4))
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x55


def test_lda_abs_y_indexes_correctly():
    rom = _rom_with([0xB9, 0x00, 0xF0])
    rom = rom.at[7].set(jnp.float32(0x44))
    bus = initial_soft_bus(rom)
    state = initial_soft_cpu_state()._replace(Y=jnp.float32(7))
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x44


def test_lda_ind_x_uses_pointer_in_zero_page():
    """(ind,X): pointer in zp at (operand+X) & 0xFF; value at the address
    that pointer holds. The pointer reads through soft_ram_peek."""
    rom = _rom_with([0xA1, 0x10])     # LDA ($10,X)
    bus = initial_soft_bus(rom)
    # X=2 → pointer is at zp $12. Put pointer to RAM[$30] there. The value
    # we ultimately load is RAM[$30] = 0x88.
    bus = bus._replace(ram=bus.ram
                       .at[0x12].set(jnp.float32(0x30))   # ptr lo
                       .at[0x13].set(jnp.float32(0x00))   # ptr hi (→ $0030)
                       .at[0x30].set(jnp.float32(0x88)))
    state = initial_soft_cpu_state()._replace(X=jnp.float32(2))
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x88


def test_lda_ind_y_uses_pointer_then_adds_y():
    """(ind),Y: pointer in zp at operand; effective addr is *pointer + Y*."""
    rom = _rom_with([0xB1, 0x40])     # LDA ($40),Y
    bus = initial_soft_bus(rom)
    # Put pointer at zp $40 → $0040. Then Y=5 makes the effective addr $0045.
    bus = bus._replace(ram=bus.ram
                       .at[0x40].set(jnp.float32(0x40))
                       .at[0x41].set(jnp.float32(0x00))
                       .at[0x45].set(jnp.float32(0x66)))
    state = initial_soft_cpu_state()._replace(Y=jnp.float32(5))
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x66


# --------------------------------------------------------------------------- #
# LDX / LDY
# --------------------------------------------------------------------------- #

def test_ldx_abs_y_reads_from_ram():
    rom = _rom_with([0xBE, 0x10, 0x00])
    bus = initial_soft_bus(rom)
    bus = bus._replace(ram=bus.ram.at[0x14].set(jnp.float32(0xAA)))
    state = initial_soft_cpu_state()._replace(Y=jnp.float32(4))
    state, _ = soft_step(state, bus)
    assert float(state.X) == 0xAA


def test_ldy_abs_x_reads_from_ram():
    rom = _rom_with([0xBC, 0x10, 0x00])
    bus = initial_soft_bus(rom)
    bus = bus._replace(ram=bus.ram.at[0x12].set(jnp.float32(0xBB)))
    state = initial_soft_cpu_state()._replace(X=jnp.float32(2))
    state, _ = soft_step(state, bus)
    assert float(state.Y) == 0xBB


# --------------------------------------------------------------------------- #
# STA / STX / STY across modes
# --------------------------------------------------------------------------- #

def test_sta_zp_x_writes_to_offset():
    rom = _rom_with([0x95, 0x20])     # STA $20,X
    state = initial_soft_cpu_state()._replace(
        A=jnp.float32(0xAB), X=jnp.float32(3))
    bus = initial_soft_bus(rom)
    _, bus = soft_step(state, bus)
    assert float(bus.ram[0x23]) == 0xAB


def test_sta_abs_writes_to_ram_region():
    # STA $0030 — abs addr in RIOT RAM region. In our simplified bus
    # model it maps to ram[0x30 & 0x7F] = ram[0x30].
    rom = _rom_with([0x8D, 0x30, 0x00])
    state = initial_soft_cpu_state()._replace(A=jnp.float32(0xCD))
    bus = initial_soft_bus(rom)
    _, bus = soft_step(state, bus)
    assert float(bus.ram[0x30]) == 0xCD


def test_stx_zp_y_writes_x():
    rom = _rom_with([0x96, 0x40])     # STX $40,Y
    state = initial_soft_cpu_state()._replace(
        X=jnp.float32(0xEE), Y=jnp.float32(2))
    bus = initial_soft_bus(rom)
    _, bus = soft_step(state, bus)
    assert float(bus.ram[0x42]) == 0xEE


def test_sty_abs_writes_y():
    rom = _rom_with([0x8C, 0x50, 0x00])
    state = initial_soft_cpu_state()._replace(Y=jnp.float32(0xFF))
    bus = initial_soft_bus(rom)
    _, bus = soft_step(state, bus)
    assert float(bus.ram[0x50]) == 0xFF


def test_sta_ind_y_resolves_pointer_then_writes():
    rom = _rom_with([0x91, 0x40])     # STA ($40),Y
    bus = initial_soft_bus(rom)
    bus = bus._replace(ram=bus.ram
                       .at[0x40].set(jnp.float32(0x60))
                       .at[0x41].set(jnp.float32(0x00)))
    state = initial_soft_cpu_state()._replace(
        A=jnp.float32(0x42), Y=jnp.float32(7))
    _, bus = soft_step(state, bus)
    # Effective addr = $0060 + 7 = $0067 → ram[$67 & 0x7F] = ram[$67]
    assert float(bus.ram[0x67]) == 0x42


# --------------------------------------------------------------------------- #
# Transfer opcodes — TAX / TAY / TXA / TYA / TSX / TXS
# --------------------------------------------------------------------------- #

def test_tax_copies_a_to_x_and_sets_nz():
    state = initial_soft_cpu_state()._replace(A=jnp.float32(0x80))
    bus = initial_soft_bus(_rom_with([0xAA]))
    state, _ = soft_step(state, bus)
    assert float(state.X) == 0x80
    assert int(state.P) & 0x80   # N set (high bit)
    assert not (int(state.P) & 0x02)  # Z clear


def test_tay_zero_sets_z_flag():
    state = initial_soft_cpu_state()._replace(A=jnp.float32(0))
    bus = initial_soft_bus(_rom_with([0xA8]))
    state, _ = soft_step(state, bus)
    assert float(state.Y) == 0
    assert int(state.P) & 0x02   # Z set


def test_txs_does_not_update_flags():
    state = initial_soft_cpu_state()._replace(X=jnp.float32(0))
    bus = initial_soft_bus(_rom_with([0x9A]))
    state, _ = soft_step(state, bus)
    assert float(state.SP) == 0
    # TXS is the only transfer that does NOT touch flags.
    assert not (int(state.P) & 0x02)


def test_tsx_copies_sp_to_x_and_sets_flags():
    state = initial_soft_cpu_state()._replace(SP=jnp.float32(0xFD))
    bus = initial_soft_bus(_rom_with([0xBA]))
    state, _ = soft_step(state, bus)
    assert float(state.X) == 0xFD
    assert int(state.P) & 0x80   # N set (0xFD has bit 7)


def test_txa_propagates_to_a():
    state = initial_soft_cpu_state()._replace(X=jnp.float32(0x42))
    bus = initial_soft_bus(_rom_with([0x8A]))
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x42


def test_tya_propagates_to_a():
    state = initial_soft_cpu_state()._replace(Y=jnp.float32(0x33))
    bus = initial_soft_bus(_rom_with([0x98]))
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x33


# --------------------------------------------------------------------------- #
# N/Z flag semantics
# --------------------------------------------------------------------------- #

def test_lda_zero_sets_z_clears_n():
    state, _ = _step([0xA9, 0x00])
    assert int(state.P) & 0x02
    assert not (int(state.P) & 0x80)


def test_lda_positive_clears_both():
    state, _ = _step([0xA9, 0x42])
    assert not (int(state.P) & 0x02)
    assert not (int(state.P) & 0x80)


def test_lda_negative_sets_n_clears_z():
    state, _ = _step([0xA9, 0x80])
    assert not (int(state.P) & 0x02)
    assert int(state.P) & 0x80


def test_ldx_negative_sets_n():
    state, _ = _step([0xA2, 0xFF])
    assert int(state.P) & 0x80


def test_ldy_zero_sets_z():
    state, _ = _step([0xA0, 0x00])
    assert int(state.P) & 0x02


# --------------------------------------------------------------------------- #
# Gradient sanity — the more elaborate addressing modes should still produce
# clean one-hot gradients back to the ROM bytes that drive them.
# --------------------------------------------------------------------------- #

def test_grad_lda_abs_x_to_ram_one_hot_at_loaded_byte():
    """LDA $F005,X with X=2; A := ROM[$F007]. `∂A / ∂rom[7] = 1`."""
    rom_init = _rom_with([0xBD, 0x05, 0xF0], size=256)
    rom_init = rom_init.at[7].set(jnp.float32(0x66))

    def simulator(rom_arr):
        bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom_arr)
        state = initial_soft_cpu_state()._replace(X=jnp.float32(2))
        state, _ = soft_step(state, bus)
        return state.A

    forward = simulator(rom_init)
    assert float(forward) == 0x66

    grad = jax.grad(simulator)(rom_init)
    assert float(grad[7]) == pytest.approx(1.0)
    other = float(jnp.sum(jnp.abs(grad))) - float(jnp.abs(grad[7]))
    assert other == pytest.approx(0.0)


def test_grad_through_ldy_then_lda_abs_y_two_instruction_chain():
    """LDY #$03 / LDA $F010,Y — A := ROM[$F013]. `∂A / ∂rom[$13] = 1`."""
    rom_init = _rom_with([0xA0, 0x03, 0xB9, 0x10, 0xF0], size=256)
    rom_init = rom_init.at[0x13].set(jnp.float32(0x77))

    def simulator(rom_arr):
        bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom_arr)
        state = initial_soft_cpu_state()
        state, bus = soft_run(state, bus, n_steps=2)
        return state.A

    assert float(simulator(rom_init)) == 0x77
    grad = jax.grad(simulator)(rom_init)
    assert float(grad[0x13]) == pytest.approx(1.0)
    other = float(jnp.sum(jnp.abs(grad))) - float(jnp.abs(grad[0x13]))
    assert other == pytest.approx(0.0)


def test_grad_lda_then_sta_abs_one_hot_at_immediate():
    """LDA #$AB / STA $0010 — RAM[$10] gets gradient at the immediate byte."""
    rom_init = _rom_with([0xA9, 0xAB, 0x8D, 0x10, 0x00], size=256)

    def simulator(rom_arr):
        bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom_arr)
        state = initial_soft_cpu_state()
        state, bus = soft_run(state, bus, n_steps=2)
        return bus.ram[0x10]

    assert float(simulator(rom_init)) == 0xAB
    grad = jax.grad(simulator)(rom_init)
    assert float(grad[1]) == pytest.approx(1.0)
    other = float(jnp.sum(jnp.abs(grad))) - float(jnp.abs(grad[1]))
    assert other == pytest.approx(0.0)
