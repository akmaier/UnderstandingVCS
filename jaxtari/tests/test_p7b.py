"""P7b tests — SOFT-mode `step()` with end-to-end gradient flow.

Forward-behaviour tests assert each handled opcode does the right
thing in float arithmetic. Gradient tests assert `jax.grad` through
a 1-instruction or multi-instruction trace produces a structurally
correct gradient back to ROM bytes.

The headline test, `test_grad_lda_imm_then_sta_zp_one_hot_at_immediate`,
is the end-to-end XAI demo this whole project is built around: run a
two-instruction program (`LDA #$42 / STA $80`) in SOFT mode, take
`jax.grad` of `RAM[$80]` w.r.t. the ROM bytes, and assert the
gradient is one-hot at the position of the immediate operand. That's
"this ROM byte explains this RAM cell" in code.
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
    soft_ram_peek,
    soft_rom_peek,
    soft_run,
    soft_step,
)


def _rom_with(opcodes_at_offset_0: list[int], size: int = 4096) -> jnp.ndarray:
    """Build a (size,) float32 ROM with the given byte sequence at offset 0
    (= CPU address \$F000 after the 13-bit mirror)."""
    rom = jnp.zeros((size,), dtype=jnp.float32)
    for i, b in enumerate(opcodes_at_offset_0):
        rom = rom.at[i].set(jnp.float32(b))
    return rom


# --------------------------------------------------------------------------- #
# Type sanity
# --------------------------------------------------------------------------- #

def test_initial_soft_cpu_state_defaults_match_hard_reset():
    s = initial_soft_cpu_state()
    assert float(s.A) == 0.0
    assert float(s.X) == 0.0
    assert float(s.SP) == 0xFD
    assert float(s.PC) == 0xF000
    assert float(s.P) == 0x34
    assert float(s.cycles) == 0.0


def test_initial_soft_bus_has_zero_ram_and_carries_rom():
    rom = jnp.arange(16, dtype=jnp.float32)
    bus = initial_soft_bus(rom)
    assert bus.ram.shape == (128,)
    assert float(bus.ram.sum()) == 0.0
    assert jnp.array_equal(bus.rom, rom)


def test_soft_supported_opcode_set_contains_p7b_core():
    """The P7b core (NOP, LDA imm/zp, LDX imm, STA zp, STX zp, JMP abs,
    BRK) must remain present even after P7c-a … P7c-f extend the set."""
    p7b_core = {0x00, 0xEA, 0xA9, 0xA5, 0xA2, 0x85, 0x86, 0x4C}
    assert p7b_core.issubset(SOFT_SUPPORTED_OPCODES)


# --------------------------------------------------------------------------- #
# Forward behaviour — one instruction at a time
# --------------------------------------------------------------------------- #

def test_nop_advances_pc_and_cycles():
    bus = initial_soft_bus(_rom_with([0xEA]))
    state = initial_soft_cpu_state()
    state, _ = soft_step(state, bus)
    assert float(state.PC) == 0xF001
    assert float(state.cycles) == 2.0


def test_lda_imm_loads_a_with_immediate_value():
    bus = initial_soft_bus(_rom_with([0xA9, 0x42]))
    state = initial_soft_cpu_state()
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x42
    assert float(state.PC) == 0xF002


def test_ldx_imm_loads_x():
    bus = initial_soft_bus(_rom_with([0xA2, 0x33]))
    state = initial_soft_cpu_state()
    state, _ = soft_step(state, bus)
    assert float(state.X) == 0x33


def test_sta_zp_writes_a_to_ram():
    """LDA #\$42 / STA \$80 — RAM[\$80] should hold \$42 afterwards."""
    bus = initial_soft_bus(_rom_with([0xA9, 0x42, 0x85, 0x00]))   # STA \$00 to keep zp in range
    state = initial_soft_cpu_state()
    state, bus = soft_step(state, bus)        # LDA #$42
    state, bus = soft_step(state, bus)        # STA $00
    assert float(state.A) == 0x42
    assert float(bus.ram[0]) == 0x42


def test_stx_zp_writes_x_to_ram():
    bus = initial_soft_bus(_rom_with([0xA2, 0x77, 0x86, 0x10]))
    state = initial_soft_cpu_state()
    state, bus = soft_step(state, bus)        # LDX #$77
    state, bus = soft_step(state, bus)        # STX $10
    assert float(bus.ram[0x10]) == 0x77


def test_lda_zp_reads_from_ram():
    """Set RAM[\$05] = 0x99 directly, then LDA \$05 should load 0x99 into A."""
    bus = initial_soft_bus(_rom_with([0xA5, 0x05]))
    bus = bus._replace(ram=bus.ram.at[0x05].set(jnp.float32(0x99)))
    state = initial_soft_cpu_state()
    state, _ = soft_step(state, bus)
    assert float(state.A) == 0x99


def test_jmp_abs_sets_pc_from_operand():
    bus = initial_soft_bus(_rom_with([0x4C, 0x34, 0x12]))           # JMP $1234
    state = initial_soft_cpu_state()
    state, _ = soft_step(state, bus)
    assert float(state.PC) == 0x1234


def test_brk_halts_in_place():
    bus = initial_soft_bus(_rom_with([0x00]))
    state = initial_soft_cpu_state()
    pc_before = float(state.PC)
    state, _ = soft_step(state, bus)
    assert float(state.PC) == pc_before        # BRK doesn't advance in the sentinel


def test_default_branch_advances_one_byte_for_unhandled_opcode():
    """Opcodes outside SOFT_SUPPORTED_OPCODES fall through to a PC+1
    advance — the trace stays differentiable but the forward result
    is wrong (the test asserts only the structural behaviour)."""
    bus = initial_soft_bus(_rom_with([0xFF]))  # 0xFF is unhandled in P7b
    state = initial_soft_cpu_state()
    state, _ = soft_step(state, bus)
    assert float(state.PC) == 0xF001


# --------------------------------------------------------------------------- #
# soft_run — multi-instruction traces
# --------------------------------------------------------------------------- #

def test_soft_run_executes_fixed_number_of_instructions():
    bus = initial_soft_bus(_rom_with([0xEA] * 5))   # 5 NOPs
    state = initial_soft_cpu_state()
    state, _ = soft_run(state, bus, n_steps=5)
    assert float(state.PC) == 0xF005
    assert float(state.cycles) == 10.0


# --------------------------------------------------------------------------- #
# Gradient tests — the whole point
# --------------------------------------------------------------------------- #

def test_grad_lda_imm_one_hot_at_immediate_address():
    """`A = ROM[PC+1]` ⇒ `∂A / ∂rom` is one-hot at ROM offset 1."""
    rom_init = _rom_with([0xA9, 0x42], size=256)

    def simulator(rom_arr):
        bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom_arr)
        state = initial_soft_cpu_state()
        state, _ = soft_step(state, bus)
        return state.A

    grad = jax.grad(simulator)(rom_init)
    # `∂A / ∂rom[i]` is 1 at i=1, 0 elsewhere (LDA #imm has no other
    # ROM dependency on the immediate path).
    assert float(grad[1]) == pytest.approx(1.0)
    other_sum = float(jnp.sum(jnp.abs(grad))) - float(jnp.abs(grad[1]))
    assert other_sum == pytest.approx(0.0)


def test_grad_ldx_imm_one_hot_at_immediate_address():
    rom_init = _rom_with([0xA2, 0x33], size=256)

    def simulator(rom_arr):
        bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom_arr)
        state = initial_soft_cpu_state()
        state, _ = soft_step(state, bus)
        return state.X

    grad = jax.grad(simulator)(rom_init)
    assert float(grad[1]) == pytest.approx(1.0)
    other = float(jnp.sum(jnp.abs(grad))) - float(jnp.abs(grad[1]))
    assert other == pytest.approx(0.0)


def test_grad_lda_imm_then_sta_zp_one_hot_at_immediate():
    """**The headline XAI demo.** A two-instruction program writes a
    ROM byte into RAM; the gradient of the resulting RAM cell w.r.t.
    the ROM array is one-hot at exactly the immediate-operand position.

    Program: LDA #\$42 / STA \$00. RAM[0] should become \$42; gradient
    of RAM[0] w.r.t. rom[] should be 1 at rom[1] (the immediate value)
    and 0 elsewhere.
    """
    rom_init = _rom_with([0xA9, 0x42, 0x85, 0x00], size=256)

    def simulator(rom_arr):
        bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom_arr)
        state = initial_soft_cpu_state()
        state, bus = soft_run(state, bus, n_steps=2)
        return bus.ram[0]

    forward = simulator(rom_init)
    assert float(forward) == 0x42

    grad = jax.grad(simulator)(rom_init)
    # The IMMEDIATE byte (rom[1]) is what flows into A and then into
    # RAM[0]. Its gradient should be exactly 1.0.
    assert float(grad[1]) == pytest.approx(1.0)
    # The other ROM bytes (opcodes + zp address) don't appear in the
    # arithmetic path of the *value* — only of the *control flow* and
    # *destination address*, which are integer-decoded via .astype.
    # → their gradient should be 0.
    other = float(jnp.sum(jnp.abs(grad))) - float(jnp.abs(grad[1]))
    assert other == pytest.approx(0.0)


def test_grad_lda_zp_then_to_a_back_to_ram_cell():
    """Loading from a known-valued RAM cell into A and reading A out
    again — the gradient w.r.t. the RAM cell should be 1.0 (the path
    `RAM → A → output` is the identity).
    """
    rom_init = _rom_with([0xA5, 0x05], size=256)
    ram_init = jnp.zeros((128,), dtype=jnp.float32).at[0x05].set(jnp.float32(0x99))

    def simulator(ram_arr):
        bus = SoftBus(ram=ram_arr, rom=rom_init)
        state = initial_soft_cpu_state()
        state, _ = soft_step(state, bus)
        return state.A

    grad = jax.grad(simulator)(ram_init)
    assert float(grad[0x05]) == pytest.approx(1.0)
    other = float(jnp.sum(jnp.abs(grad))) - float(jnp.abs(grad[0x05]))
    assert other == pytest.approx(0.0)


def test_grad_through_loop_of_lda_imm_only_last_immediate_matters():
    """Run two consecutive `LDA #imm` instructions. The final A holds
    the *second* immediate, so only that ROM byte should have non-zero
    gradient — the first LDA's immediate gets overwritten."""
    rom_init = _rom_with([0xA9, 0x11, 0xA9, 0x22], size=256)

    def simulator(rom_arr):
        bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom_arr)
        state = initial_soft_cpu_state()
        state, _ = soft_run(state, bus, n_steps=2)
        return state.A

    assert float(simulator(rom_init)) == 0x22
    grad = jax.grad(simulator)(rom_init)
    assert float(grad[3]) == pytest.approx(1.0)   # second immediate
    assert float(grad[1]) == pytest.approx(0.0)   # first immediate — overwritten
    other = float(jnp.sum(jnp.abs(grad))) - float(jnp.abs(grad[3]))
    assert other == pytest.approx(0.0)
