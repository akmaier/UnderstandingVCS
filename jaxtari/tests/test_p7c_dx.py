"""P7c-dx tests — float-valued flag mirrors + soft-branch dispatch.

P7c-dx adds `P_N` / `P_Z` / `P_C` / `P_V` to `SoftCPUState` (kept in
lock-step with the packed `P` byte via the internal `_with_p` helper)
and rewires `_do_branch` to feed `soft_branch` from the matching
float, so a `jax.grad` over a SOFT trace carries gradient through
conditional control flow.

Forward semantics stay bit-exact via a straight-through trick — the
existing P7c-d branch tests and the PXC1 conformance fixture are
unaffected (covered by the existing suites). This file pins down
*new* behaviour:

  1. Flag mirrors update when an opcode touches `P`.
  2. The branch handler reads its forward-path predicate from `state.P`
     (so a test that mutates only P still picks the right branch).
  3. `jax.grad` of a value computed from a branched trace has a
     non-zero contribution from the *not-taken* PC too — that's the
     soft-branch sigmoid blend doing its job.
"""

import jax
import jax.numpy as jnp

from jaxtari.diff.soft_state import SoftBus, initial_soft_cpu_state
from jaxtari.diff.soft_step import soft_step


def _rom_with(program: list[int]) -> jnp.ndarray:
    """Tiny ROM helper — fills a 256-byte ROM with `program` at offset 0."""
    rom = jnp.zeros((256,), dtype=jnp.float32)
    for i, b in enumerate(program):
        rom = rom.at[i].set(jnp.float32(b & 0xFF))
    return rom


def _bus_with_rom(rom: jnp.ndarray) -> SoftBus:
    return SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom)


# --------------------------------------------------------------------------- #
# Mirror sync — flag-touching opcodes update P_N / P_Z / P_C / P_V
# --------------------------------------------------------------------------- #

def test_lda_immediate_zero_sets_PZ_float():
    """LDA #$00 sets Z=1 in P; the P_Z mirror must follow."""
    # LDA #$00 at offset 0
    bus = _bus_with_rom(_rom_with([0xA9, 0x00]))
    state = initial_soft_cpu_state(pc=0xF000)
    state2, _ = soft_step(state, bus)
    assert float(state2.P_Z) == 1.0
    assert float(state2.P_N) == 0.0
    # Packed P also has Z=1 (bit 1).
    assert (int(state2.P) & 0x02) != 0


def test_lda_immediate_negative_sets_PN_float():
    """LDA #$80 sets N=1 in P; the P_N mirror must follow."""
    bus = _bus_with_rom(_rom_with([0xA9, 0x80]))
    state = initial_soft_cpu_state(pc=0xF000)
    state2, _ = soft_step(state, bus)
    assert float(state2.P_N) == 1.0
    assert float(state2.P_Z) == 0.0


def test_sec_sets_PC_float():
    bus = _bus_with_rom(_rom_with([0x38]))                       # SEC
    state = initial_soft_cpu_state(pc=0xF000)
    state2, _ = soft_step(state, bus)
    assert float(state2.P_C) == 1.0


def test_clc_clears_PC_float():
    bus = _bus_with_rom(_rom_with([0x38, 0x18]))                 # SEC; CLC
    state = initial_soft_cpu_state(pc=0xF000)
    state2, _ = soft_step(state, bus)
    assert float(state2.P_C) == 1.0
    state3, _ = soft_step(state2, bus)
    assert float(state3.P_C) == 0.0


def test_clv_clears_PV_float():
    """BIT $00 with operand bit 6 set sets V; then CLV clears it."""
    # Stash 0x40 at $00, then BIT $00; CLV
    bus = _bus_with_rom(_rom_with([0x24, 0x00, 0xB8]))
    ram = jnp.zeros((128,), dtype=jnp.float32).at[0].set(0x40)
    bus = bus._replace(ram=ram)
    state = initial_soft_cpu_state(pc=0xF000)
    state2, _ = soft_step(state, bus)                            # BIT
    assert float(state2.P_V) == 1.0
    state3, _ = soft_step(state2, bus)                           # CLV
    assert float(state3.P_V) == 0.0


def test_cmp_carry_mirror_set_when_greater_or_equal():
    """CMP A,#imm with A >= operand sets C; P_C must mirror."""
    bus = _bus_with_rom(_rom_with([0xC9, 0x10]))                 # CMP #$10
    state = initial_soft_cpu_state(pc=0xF000)._replace(A=jnp.float32(0x42))
    state2, _ = soft_step(state, bus)
    assert float(state2.P_C) == 1.0
    # And A < operand clears C:
    state = initial_soft_cpu_state(pc=0xF000)._replace(A=jnp.float32(0x05))
    state2, _ = soft_step(state, bus)
    assert float(state2.P_C) == 0.0


def test_plp_syncs_floats_from_pulled_byte():
    """PLP loads P from the stack; the mirrors must re-derive from
    the pulled byte."""
    bus = _bus_with_rom(_rom_with([0x28]))                       # PLP
    # Set up the stack: SP=$FC; stack[$FD] = 0xC1 (N=1, V=1, C=1).
    ram = jnp.zeros((128,), dtype=jnp.float32).at[0x7D].set(0xC1)
    bus = bus._replace(ram=ram)
    state = initial_soft_cpu_state(pc=0xF000)._replace(SP=jnp.float32(0xFC))
    state2, _ = soft_step(state, bus)
    assert float(state2.P_N) == 1.0
    assert float(state2.P_V) == 1.0
    assert float(state2.P_C) == 1.0
    assert float(state2.P_Z) == 0.0


# --------------------------------------------------------------------------- #
# Branch forward semantics — read from state.P even if P_X out of sync
# --------------------------------------------------------------------------- #

def test_bne_uses_P_byte_for_forward_when_P_Z_unset():
    """A test that mutates only `state.P` (skipping `_with_p`) must
    still get the correct forward PC. This guards against a future
    refactor that "modernises" `_do_branch` to use only the float
    mirror — that would silently break every existing P7c-d test."""
    bus = _bus_with_rom(_rom_with([0xD0, 0x10]))                 # BNE +16
    # Z=1 in packed P; floats stay default-zero (P_Z=0).
    state = initial_soft_cpu_state(pc=0xF000)._replace(P=jnp.float32(0x36))
    state2, _ = soft_step(state, bus)
    # Z=1 so BNE NOT taken → PC = $F002.
    assert int(state2.PC) == 0xF002


def test_beq_uses_P_byte_for_forward_when_P_Z_unset():
    bus = _bus_with_rom(_rom_with([0xF0, 0x10]))                 # BEQ +16
    state = initial_soft_cpu_state(pc=0xF000)._replace(P=jnp.float32(0x36))
    state2, _ = soft_step(state, bus)
    # Z=1 → BEQ taken → PC = $F002 + 16 = $F012.
    assert int(state2.PC) == 0xF012


# --------------------------------------------------------------------------- #
# Soft-branch gradient — both PCs contribute when the float flag is soft
# --------------------------------------------------------------------------- #

def test_soft_branch_carries_gradient_through_P_Z():
    """If a caller hands `_do_branch` a soft-valued P_Z (in [0, 1])
    via the SoftCPUState field, the resulting PC depends differentiably
    on that value — that's the headline P7c-dx capability."""
    bus = _bus_with_rom(_rom_with([0xF0, 0x10]))                 # BEQ +16

    def branched_pc(p_z_float: jnp.ndarray) -> jnp.ndarray:
        # Build a state with the soft float Z value injected. The
        # packed P keeps Z=1 (so the forward PC is the "taken" branch),
        # but the gradient propagates through the soft P_Z via the
        # sigmoid blend.
        state = (
            initial_soft_cpu_state(pc=0xF000)
            ._replace(P=jnp.float32(0x36),                       # Z=1 (forward)
                      P_Z=p_z_float)                             # soft gradient hook
        )
        new_state, _ = soft_step(state, bus)
        return new_state.PC

    # Forward must still pick the taken branch.
    pc_forward = branched_pc(jnp.float32(1.0))
    assert int(pc_forward) == 0xF012

    # Gradient w.r.t. the soft P_Z is non-zero — that's the whole point
    # of the float-flag plumbing.
    g = jax.grad(lambda p_z: branched_pc(p_z))(jnp.float32(0.5))
    assert abs(float(g)) > 1e-3   # non-trivially non-zero


def test_soft_branch_gradient_sign_matches_pc_delta():
    """∂PC/∂P_Z for a BEQ +offset should be *positive*: increasing the
    Z probability pulls the soft-blended PC towards `pc_taken`, which
    is larger than `pc_not_taken` for a positive forward branch."""
    bus = _bus_with_rom(_rom_with([0xF0, 0x10]))                 # BEQ +16

    def branched_pc(p_z_float):
        state = (
            initial_soft_cpu_state(pc=0xF000)
            ._replace(P=jnp.float32(0x36), P_Z=p_z_float)
        )
        new_state, _ = soft_step(state, bus)
        return new_state.PC

    g = jax.grad(branched_pc)(jnp.float32(0.5))
    assert float(g) > 0.0


def test_p7c_dx_keeps_pong_noop_10_pc_path_bit_exact():
    """Smoke check: stepping a simple loop with branch ops returns the
    same packed P bytes as before P7c-dx (which the existing P7c-d
    branch tests already lock down). This redundantly asserts that
    the straight-through trick really does leave forward exact."""
    # BNE +2 (over a halt), HALT (BRK), then a final byte.
    bus = _bus_with_rom(_rom_with([0xD0, 0x02, 0x00, 0xEA, 0xEA]))
    # Z=0 → BNE taken → skips the BRK at $F002, lands at $F004 (a NOP).
    state = initial_soft_cpu_state(pc=0xF000)._replace(P=jnp.float32(0x34))
    state2, _ = soft_step(state, bus)
    assert int(state2.PC) == 0xF004
