"""SOFT-mode `step()` — a differentiable parallel to `cpu.m6502.step`.

`soft_step(state, bus)` executes one 6502 instruction with the SOFT
primitives from `jaxtari.diff`: memory access goes through
`soft_rom_peek` (one-hot dot product on the ROM, gradient-friendly);
register state is `float32`; opcode dispatch is `jax.lax.switch` over
all 256 opcodes. The HARD `step()` is left untouched — SOFT mode is a
parallel execution path.

**Opcode coverage in P7b (subset to prove gradient flow):**

  NOP            (\$EA)  no effect, PC++, cycles+=2
  LDA #imm       (\$A9)  A = peek(PC+1)
  LDA \$zp       (\$A5)  A = peek(peek(PC+1))                   ← double indirection
  LDX #imm       (\$A2)  X = peek(PC+1)
  STA \$zp       (\$85)  RAM[peek(PC+1)] = A
  STX \$zp       (\$86)  RAM[peek(PC+1)] = X
  JMP \$abs      (\$4C)  PC = peek(PC+1) | (peek(PC+2) << 8)
  BRK            (\$00)  used as an "end of trace" sentinel — halts in place

All other opcodes fall through to `_branch_default` which advances PC
by 1 and bumps cycles. This is intentionally lenient — a real ROM
that hits an unhandled opcode will produce wrong forward behaviour
but **will not** raise during `jax.grad` tracing, so the function
remains differentiable. Extending the handler table to the other
~140 opcodes is the natural P7c / P7d work.

What this implementation does NOT (yet) do:
  - Status flag updates (N/Z/C/V) — registers move, flags stay 0
  - TIA / RIOT I/O writes — STA \$0xxx etc. silently drop into the
    RAM region (no side-effects on TIA registers or framebuffer)
  - Stack push/pop, JSR/RTS, BRK/RTI proper interrupt sequence
  - Bank switching for F8/F6/F4 carts
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxtari.diff.soft_state import SoftBus, SoftCPUState


# --------------------------------------------------------------------------- #
# Differentiable memory access
# --------------------------------------------------------------------------- #

def soft_rom_peek(rom: jnp.ndarray, addr) -> jnp.ndarray:
    """Differentiable cart byte read — `one_hot(addr) · rom`.

    `rom` is any 1-D numeric array. `addr` is a scalar (int or float);
    it's treated as the offset *inside the ROM* (the 13-bit address
    mirror is applied at the call site, not here).
    """
    n = rom.shape[0]
    one_hot = jax.nn.one_hot(addr, n, dtype=jnp.float32)
    return jnp.dot(one_hot, rom.astype(jnp.float32))


def soft_ram_peek(ram: jnp.ndarray, addr) -> jnp.ndarray:
    """Differentiable RAM read — same one-hot trick, on the 128-byte RAM."""
    one_hot = jax.nn.one_hot(addr, ram.shape[0], dtype=jnp.float32)
    return jnp.dot(one_hot, ram.astype(jnp.float32))


def _cart_addr(pc_offset: jnp.ndarray) -> jnp.ndarray:
    """Convert a CPU bus address to a 0..4095 ROM offset.

    Real bus does the full 13-bit mirror + cart-window check; for the
    SOFT subset we assume the program lives in the cart region and
    do the simpler `addr & 0x0FFF` to keep the gradient path narrow.
    """
    return pc_offset.astype(jnp.int32) & 0x0FFF


# --------------------------------------------------------------------------- #
# Per-opcode branch handlers
# --------------------------------------------------------------------------- #
# Each handler has the signature `(state, bus) → (new_state, new_bus)` so
# `jax.lax.switch` can dispatch on opcode index.

def _branch_default(state: SoftCPUState, bus: SoftBus):
    """Unhandled opcode: advance PC by 1, bump cycles. Doesn't raise so
    the trace stays differentiable; forward result will be wrong if a
    real ROM hits this."""
    return state._replace(
        PC=state.PC + 1.0,
        cycles=state.cycles + 2.0,
    ), bus


def _branch_brk(state: SoftCPUState, bus: SoftBus):
    """End-of-trace sentinel — halt in place. PC doesn't advance so
    a run-until-BRK loop terminates by hitting the same instruction
    twice (callers should break out of the loop when PC stops changing)."""
    return state._replace(cycles=state.cycles + 7.0), bus


def _branch_nop(state: SoftCPUState, bus: SoftBus):
    return state._replace(
        PC=state.PC + 1.0,
        cycles=state.cycles + 2.0,
    ), bus


def _branch_lda_imm(state: SoftCPUState, bus: SoftBus):
    """LDA #imm: A = ROM[PC + 1]."""
    rom_off = _cart_addr(state.PC + 1.0)
    operand = soft_rom_peek(bus.rom, rom_off)
    return state._replace(
        A=operand,
        PC=state.PC + 2.0,
        cycles=state.cycles + 2.0,
    ), bus


def _branch_lda_zp(state: SoftCPUState, bus: SoftBus):
    """LDA \$zp: zp = ROM[PC+1]; A = RAM[zp]. The zp address itself
    rides through `soft_ram_peek`'s one-hot, so a gradient at the
    output flows back to ROM[PC+1] **and** to RAM[zp]."""
    rom_off = _cart_addr(state.PC + 1.0)
    zp_float = soft_rom_peek(bus.rom, rom_off)
    zp_int = zp_float.astype(jnp.int32) & 0x7F
    operand = soft_ram_peek(bus.ram, zp_int)
    return state._replace(
        A=operand,
        PC=state.PC + 2.0,
        cycles=state.cycles + 3.0,
    ), bus


def _branch_ldx_imm(state: SoftCPUState, bus: SoftBus):
    rom_off = _cart_addr(state.PC + 1.0)
    operand = soft_rom_peek(bus.rom, rom_off)
    return state._replace(
        X=operand,
        PC=state.PC + 2.0,
        cycles=state.cycles + 2.0,
    ), bus


def _branch_sta_zp(state: SoftCPUState, bus: SoftBus):
    """STA \$zp: zp = ROM[PC+1]; RAM[zp] = A."""
    rom_off = _cart_addr(state.PC + 1.0)
    zp_float = soft_rom_peek(bus.rom, rom_off)
    zp_int = zp_float.astype(jnp.int32) & 0x7F
    new_ram = bus.ram.at[zp_int].set(state.A)
    new_state = state._replace(
        PC=state.PC + 2.0,
        cycles=state.cycles + 3.0,
    )
    return new_state, bus._replace(ram=new_ram)


def _branch_stx_zp(state: SoftCPUState, bus: SoftBus):
    rom_off = _cart_addr(state.PC + 1.0)
    zp_float = soft_rom_peek(bus.rom, rom_off)
    zp_int = zp_float.astype(jnp.int32) & 0x7F
    new_ram = bus.ram.at[zp_int].set(state.X)
    new_state = state._replace(
        PC=state.PC + 2.0,
        cycles=state.cycles + 3.0,
    )
    return new_state, bus._replace(ram=new_ram)


def _branch_jmp_abs(state: SoftCPUState, bus: SoftBus):
    """JMP \$abs: PC = ROM[PC+1] | (ROM[PC+2] << 8)."""
    rom_lo = _cart_addr(state.PC + 1.0)
    rom_hi = _cart_addr(state.PC + 2.0)
    lo = soft_rom_peek(bus.rom, rom_lo)
    hi = soft_rom_peek(bus.rom, rom_hi)
    new_pc = lo + hi * 256.0
    return state._replace(
        PC=new_pc,
        cycles=state.cycles + 3.0,
    ), bus


# --------------------------------------------------------------------------- #
# Dispatch table
# --------------------------------------------------------------------------- #

_HANDLERS = [_branch_default] * 256
_HANDLERS[0x00] = _branch_brk
_HANDLERS[0xEA] = _branch_nop
_HANDLERS[0xA9] = _branch_lda_imm
_HANDLERS[0xA5] = _branch_lda_zp
_HANDLERS[0xA2] = _branch_ldx_imm
_HANDLERS[0x85] = _branch_sta_zp
_HANDLERS[0x86] = _branch_stx_zp
_HANDLERS[0x4C] = _branch_jmp_abs

# Opcodes handled with full behaviour (rather than default). Exposed so
# tests + a future P7c can introspect coverage.
SOFT_SUPPORTED_OPCODES = frozenset({0x00, 0xEA, 0xA9, 0xA5, 0xA2, 0x85, 0x86, 0x4C})


# --------------------------------------------------------------------------- #
# Top-level step
# --------------------------------------------------------------------------- #

def soft_step(state: SoftCPUState, bus: SoftBus):
    """Execute one instruction.

    Memory access is differentiable (one-hot dot product on ROM and
    RAM). Dispatch is `jax.lax.switch` over the 256-way opcode table.
    Returns `(new_state, new_bus)`.
    """
    rom_off = _cart_addr(state.PC)
    opcode_float = soft_rom_peek(bus.rom, rom_off)
    opcode_int = opcode_float.astype(jnp.int32)
    return jax.lax.switch(opcode_int, _HANDLERS, state, bus)


def soft_run(state: SoftCPUState, bus: SoftBus, n_steps: int):
    """Execute exactly `n_steps` instructions in sequence (Python loop
    — gets unrolled by jax.jit if you wrap this in jit). Useful when the
    trace length is known statically, which is the common XAI case
    ("run 1000 instructions and tell me which ROM bytes affected the
    output").
    """
    for _ in range(n_steps):
        state, bus = soft_step(state, bus)
    return state, bus
