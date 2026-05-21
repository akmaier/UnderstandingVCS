"""P7c-f tests — RTI, and the milestone check that SOFT mode now covers
the full documented NMOS opcode set.

RTI is the last documented opcode; with it in the table, `soft_step`
dispatches all 151 NMOS opcodes (+ the USBC $EB alias) with real
behaviour. This file verifies RTI's pop sequence, the JSR↔RTS vs.
interrupt↔RTI distinction (RTI does *not* add 1 to the popped PC), and
asserts the total opcode count.
"""

import jax.numpy as jnp

from jaxtari.diff import (
    SOFT_SUPPORTED_OPCODES,
    initial_soft_bus,
    initial_soft_cpu_state,
    soft_run,
    soft_step,
)


def _rom_with(bytes_: list[int], size: int = 4096) -> jnp.ndarray:
    rom = jnp.zeros((size,), dtype=jnp.float32)
    for i, b in enumerate(bytes_):
        rom = rom.at[i].set(jnp.float32(b))
    return rom


# --------------------------------------------------------------------------- #
# Milestone: the full documented NMOS opcode set
# --------------------------------------------------------------------------- #

def test_soft_mode_covers_all_151_documented_nmos_opcodes():
    """The documented NMOS 6502 has 151 opcodes; xitari also implements
    the undocumented USBC ($EB). SOFT mode should now handle all of
    them — 152 total."""
    assert len(SOFT_SUPPORTED_OPCODES) == 152
    assert 0x40 in SOFT_SUPPORTED_OPCODES   # RTI — the P7c-f addition
    assert 0xEB in SOFT_SUPPORTED_OPCODES   # USBC alias


def test_every_documented_opcode_group_is_present():
    """Spot-check one representative opcode from each instruction family
    so a regression that drops a whole group is caught."""
    representatives = {
        0xA9: "LDA", 0x85: "STA", 0xAA: "TAX",
        0x69: "ADC", 0xE9: "SBC", 0x29: "AND",
        0x09: "ORA", 0x49: "EOR", 0xC9: "CMP",
        0x24: "BIT", 0x0A: "ASL", 0x4A: "LSR",
        0x2A: "ROL", 0x6A: "ROR", 0xD0: "BNE",
        0x4C: "JMP", 0x20: "JSR", 0x60: "RTS",
        0x48: "PHA", 0x28: "PLP", 0x18: "CLC",
        0xE6: "INC", 0xC6: "DEC", 0xE8: "INX",
        0x00: "BRK", 0x40: "RTI", 0xEA: "NOP",
    }
    for opcode in representatives:
        assert opcode in SOFT_SUPPORTED_OPCODES


# --------------------------------------------------------------------------- #
# RTI
# --------------------------------------------------------------------------- #

def test_rti_pops_p_and_pc():
    """Hand-prime the stack with a status byte and a return address,
    then RTI should restore them. Stack layout (push order = P, PCH,
    PCL bottom-up): SP points below PCL after the pushes."""
    bus = initial_soft_bus(_rom_with([0x40]))     # RTI at $F000
    # We will RTI to $1234 with P = 0x21 (→ forced to 0x31 by RTI).
    # Stack grows down from $01FD. The 6502 interrupt pushes PCH, PCL,
    # then P — so on the stack from high addr to low: PCH, PCL, P.
    # _pop8 increments SP first, so it reads $01FB, $01FC, $01FD.
    state = initial_soft_cpu_state()._replace(SP=jnp.float32(0xFA))
    bus = bus._replace(ram=bus.ram
                       .at[0x7B].set(jnp.float32(0x21))   # P  at $01FB
                       .at[0x7C].set(jnp.float32(0x34))   # PCL at $01FC
                       .at[0x7D].set(jnp.float32(0x12)))  # PCH at $01FD
    state, _ = soft_step(state, bus)
    assert float(state.PC) == 0x1234
    # RTI forces B (0x10) and U (0x20) on → 0x21 | 0x30 = 0x31.
    assert int(state.P) == 0x31
    assert float(state.SP) == 0xFD


def test_rti_does_not_add_one_to_pc():
    """Unlike RTS (which returns to address+1), RTI returns to the exact
    popped address. Pushing $2000 → RTI lands on $2000, not $2001."""
    bus = initial_soft_bus(_rom_with([0x40]))
    state = initial_soft_cpu_state()._replace(SP=jnp.float32(0xFA))
    bus = bus._replace(ram=bus.ram
                       .at[0x7B].set(jnp.float32(0x00))   # P
                       .at[0x7C].set(jnp.float32(0x00))   # PCL
                       .at[0x7D].set(jnp.float32(0x20)))  # PCH
    state, _ = soft_step(state, bus)
    assert float(state.PC) == 0x2000


def test_rti_costs_6_cycles():
    bus = initial_soft_bus(_rom_with([0x40]))
    state = initial_soft_cpu_state()._replace(SP=jnp.float32(0xFA))
    state, _ = soft_step(state, bus)
    assert float(state.cycles) == 6.0


def test_brk_still_halts_in_place_as_sentinel():
    """P7c-f keeps BRK as the end-of-trace sentinel — it must NOT have
    grown a proper interrupt sequence."""
    bus = initial_soft_bus(_rom_with([0x00]))
    state = initial_soft_cpu_state()
    pc_before = float(state.PC)
    state, _ = soft_step(state, bus)
    assert float(state.PC) == pc_before


def test_jsr_rti_pair_runs_without_raising():
    """A JSR followed (at the target) by RTI exercises push16-then-pop
    in the opposite pairing from JSR/RTS. We only assert it runs and
    the SP returns to a sane value — RTI consumes one extra byte (the
    status) so the SP arithmetic differs from RTS."""
    rom = _rom_with([0x20, 0x10, 0xF0])           # JSR $F010
    rom = rom.at[0x010].set(jnp.float32(0x40))    # RTI at $F010
    bus = initial_soft_bus(rom)
    state = initial_soft_cpu_state()
    state, bus = soft_step(state, bus)            # JSR — pushes 2 bytes
    assert float(state.PC) == 0xF010
    state, bus = soft_step(state, bus)            # RTI — pops 3 bytes
    # RTI popped 3 bytes where JSR pushed 2, so SP ends one above start.
    assert float(state.SP) == 0xFE
