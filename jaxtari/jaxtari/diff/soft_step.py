"""SOFT-mode `step()` — a differentiable parallel to `cpu.m6502.step`.

`soft_step(state, bus)` executes one 6502 instruction with the SOFT
primitives from `jaxtari.diff`: memory access goes through
`soft_rom_peek` / `soft_ram_peek` (one-hot dot products, gradient-friendly);
register state is `float32`; opcode dispatch is `jax.lax.switch` over
all 256 opcodes. The HARD `step()` is left untouched — SOFT mode is a
parallel execution path.

**Opcode coverage:**

  P7b (the original 8-opcode core):
    NOP            ($EA)
    LDA #imm/zp    ($A9 / $A5)
    LDA $zp        ($A5)
    LDX #imm       ($A2)
    STA $zp        ($85)
    STX $zp        ($86)
    JMP $abs       ($4C)
    BRK            ($00)

  P7c-a (full load/store/transfer + N/Z flag updates):
    LDA — all 8 addressing modes  ($A9 $A5 $B5 $AD $BD $B9 $A1 $B1)
    LDX — all 5 addressing modes  ($A2 $A6 $B6 $AE $BE)
    LDY — all 5 addressing modes  ($A0 $A4 $B4 $AC $BC)
    STA — all 7 addressing modes  ($85 $95 $8D $9D $99 $81 $91)
    STX — all 3 addressing modes  ($86 $96 $8E)
    STY — all 3 addressing modes  ($84 $94 $8C)
    Transfers — TAX, TAY, TXA, TYA, TSX, TXS
                  ($AA $A8 $8A $98 $BA $9A)

All other opcodes fall through to `_branch_default` which advances PC
by 1 and bumps cycles. This is intentionally lenient — a real ROM
that hits an unhandled opcode will produce wrong forward behaviour
but **will not** raise during `jax.grad` tracing, so the function
remains differentiable. Extending the handler table is P7c-b … P7c-f.

What this implementation does NOT (yet) do:
  - P7c-b: ADC/SBC/AND/ORA/EOR/CMP/CPX/CPY/BIT + N/Z/C/V flag updates
  - P7c-c: ASL/LSR/ROL/ROR + N/Z/C flags
  - P7c-d: branches via `soft_branch` + JMP (ind) + JSR / RTS
  - P7c-e: stack push/pop, status-flag opcodes, INC/DEC/INX/INY/DEX/DEY
  - P7c-f: BRK/RTI proper interrupt sequence, TIA/RIOT writes via SOFT
    bus dispatch, cart hotspot bank-switching

SOFT-mode read/write address dispatch is simplified compared to the
HARD bus: cart-range reads use `soft_rom_peek`, everything else maps
into the 128-byte RAM array via `addr & 0x7F`. TIA/RIOT register
behaviour is therefore wrong forward but the trace stays
gradient-clean. Real bus dispatch lands in P7c-f.
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


def _cart_addr(pc_offset) -> jnp.ndarray:
    """Convert a CPU bus address to a 0..4095 ROM offset.

    Real bus does the full 13-bit mirror + cart-window check; for the
    SOFT subset we assume the program lives in the cart region and
    do the simpler `addr & 0x0FFF` to keep the gradient path narrow.
    """
    return jnp.asarray(pc_offset).astype(jnp.int32) & 0x0FFF


def _bus_read(bus: SoftBus, addr) -> jnp.ndarray:
    """SOFT-mode differentiable bus read with cart-vs-RAM decode.

    Cart range ($1000-$1FFF after 13-bit mirror): `soft_rom_peek`.
    Everything else: `soft_ram_peek(addr & 0x7F)` — a simplified model
    that collapses TIA / RIOT / RAM into a single 128-byte array.
    Proper TIA / RIOT register dispatch is P7c-f.

    `jnp.where` picks the right branch and the gradient flows through
    both — the wrong branch contributes 0 because its `one_hot` weight
    is unused, but the cost is two peeks instead of one. The clean
    gradient is the point.
    """
    a_int       = jnp.asarray(addr).astype(jnp.int32) & 0x1FFF
    is_cart     = (a_int & 0x1000) != 0
    cart_offset = a_int & 0x0FFF
    ram_offset  = a_int & 0x7F
    return jnp.where(
        is_cart,
        soft_rom_peek(bus.rom, cart_offset),
        soft_ram_peek(bus.ram, ram_offset),
    )


def _bus_write(bus: SoftBus, addr, value: jnp.ndarray) -> SoftBus:
    """SOFT-mode differentiable bus write.

    For P7c-a the model is: ROM writes are silently dropped; everything
    else mutates `bus.ram[addr & 0x7F]`. TIA register writes therefore
    land in RAM rather than affecting the framebuffer — that's wrong
    forward behaviour but keeps the trace gradient-clean. P7c-f adds
    real TIA / RIOT / cart-hotspot routing.
    """
    a_int      = jnp.asarray(addr).astype(jnp.int32) & 0x1FFF
    ram_offset = a_int & 0x7F
    is_cart    = (a_int & 0x1000) != 0
    current    = soft_ram_peek(bus.ram, ram_offset)
    new_val    = jnp.where(is_cart, current, value)
    new_ram    = bus.ram.at[ram_offset].set(new_val)
    return bus._replace(ram=new_ram)


# --------------------------------------------------------------------------- #
# N / Z flag helpers
# --------------------------------------------------------------------------- #
# Flag bits use the standard 6502 packing (see jaxtari.cpu.tables):
#   bit 7 N, bit 6 V, bit 5 U (always 1), bit 4 B,
#   bit 3 D, bit 2 I, bit 1 Z, bit 0 C.
# In SOFT mode the P register is Float32; we cast to int32 to flip bits,
# then cast back. Flag gradients are *stopped* at this cast; the gradient
# on register values (A, X, Y) is preserved because they're written from
# the operand path, not from P.

_NZ_CLEAR_MASK = 0xFF ^ (0x80 | 0x02)   # clear N and Z, keep the rest


def _set_nz(p: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    """Set the N and Z flag bits in P from `value` (a single byte).

    Z = 1 iff value & 0xFF == 0.
    N = value bit 7.
    """
    v_int = value.astype(jnp.int32) & 0xFF
    p_int = p.astype(jnp.int32) & 0xFF
    p_int = p_int & _NZ_CLEAR_MASK
    z_bit = jnp.where(v_int == 0, 0x02, 0)
    n_bit = v_int & 0x80
    return (p_int | z_bit | n_bit).astype(jnp.float32)


# --------------------------------------------------------------------------- #
# Addressing-mode resolvers — operand value for reads, effective address for
# stores. The HARD path uses RESOLVERS[mode] from cpu.addressing; the SOFT
# path needs differentiable variants that route through soft_rom_peek and
# (for indirect modes) soft_ram_peek for the pointer load.
# --------------------------------------------------------------------------- #

def _operand_byte(bus: SoftBus, pc_plus_one: jnp.ndarray) -> jnp.ndarray:
    """Read the byte at ROM[PC+1] (the typical operand fetch)."""
    return soft_rom_peek(bus.rom, _cart_addr(pc_plus_one))


def _operand_word(bus: SoftBus, pc_plus_one: jnp.ndarray) -> jnp.ndarray:
    """Read the little-endian 16-bit word at ROM[PC+1..PC+2]."""
    lo = soft_rom_peek(bus.rom, _cart_addr(pc_plus_one))
    hi = soft_rom_peek(bus.rom, _cart_addr(pc_plus_one + 1.0))
    return lo + hi * 256.0


def _addr_zp(state: SoftCPUState, bus: SoftBus) -> jnp.ndarray:
    return _operand_byte(bus, state.PC + 1.0).astype(jnp.int32) & 0xFF


def _addr_zp_x(state: SoftCPUState, bus: SoftBus) -> jnp.ndarray:
    base = _operand_byte(bus, state.PC + 1.0).astype(jnp.int32)
    return (base + state.X.astype(jnp.int32)) & 0xFF


def _addr_zp_y(state: SoftCPUState, bus: SoftBus) -> jnp.ndarray:
    base = _operand_byte(bus, state.PC + 1.0).astype(jnp.int32)
    return (base + state.Y.astype(jnp.int32)) & 0xFF


def _addr_abs(state: SoftCPUState, bus: SoftBus) -> jnp.ndarray:
    return _operand_word(bus, state.PC + 1.0).astype(jnp.int32) & 0xFFFF


def _addr_abs_x(state: SoftCPUState, bus: SoftBus) -> jnp.ndarray:
    base = _operand_word(bus, state.PC + 1.0).astype(jnp.int32)
    return (base + state.X.astype(jnp.int32)) & 0xFFFF


def _addr_abs_y(state: SoftCPUState, bus: SoftBus) -> jnp.ndarray:
    base = _operand_word(bus, state.PC + 1.0).astype(jnp.int32)
    return (base + state.Y.astype(jnp.int32)) & 0xFFFF


def _addr_ind_x(state: SoftCPUState, bus: SoftBus) -> jnp.ndarray:
    """(zp,X): pointer at zero-page (zp+X) & 0xFF; LE word at that pointer."""
    zp   = (_operand_byte(bus, state.PC + 1.0).astype(jnp.int32)
            + state.X.astype(jnp.int32)) & 0xFF
    lo   = soft_ram_peek(bus.ram, zp & 0x7F)
    hi   = soft_ram_peek(bus.ram, (zp + 1) & 0x7F)
    return (lo.astype(jnp.int32) + hi.astype(jnp.int32) * 256) & 0xFFFF


def _addr_ind_y(state: SoftCPUState, bus: SoftBus) -> jnp.ndarray:
    """(zp),Y: pointer at zero-page zp; LE word at that pointer, then + Y."""
    zp   = _operand_byte(bus, state.PC + 1.0).astype(jnp.int32) & 0xFF
    lo   = soft_ram_peek(bus.ram, zp & 0x7F)
    hi   = soft_ram_peek(bus.ram, (zp + 1) & 0x7F)
    base = lo.astype(jnp.int32) + hi.astype(jnp.int32) * 256
    return (base + state.Y.astype(jnp.int32)) & 0xFFFF


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


def _branch_jmp_abs(state: SoftCPUState, bus: SoftBus):
    """JMP $abs: PC = ROM[PC+1] | (ROM[PC+2] << 8)."""
    new_pc = _operand_word(bus, state.PC + 1.0)
    return state._replace(
        PC=new_pc,
        cycles=state.cycles + 3.0,
    ), bus


# --- Load helpers + handlers ----------------------------------------------- #

def _do_load(state: SoftCPUState, bus: SoftBus, reg: str,
             value: jnp.ndarray, instr_len: float, cycles: float):
    """Common path for LDA / LDX / LDY — register := value; N/Z update;
    PC advance; cycles bump. `reg` is one of "A", "X", "Y"."""
    new_p = _set_nz(state.P, value)
    fields = {
        "P":      new_p,
        "PC":     state.PC + instr_len,
        "cycles": state.cycles + cycles,
        reg:      value,
    }
    return state._replace(**fields), bus


# LDA (immediate, zp, zp,X, abs, abs,X, abs,Y, (ind,X), (ind),Y) -------------
def _branch_lda_imm(state, bus):
    val = _operand_byte(bus, state.PC + 1.0)
    return _do_load(state, bus, "A", val, 2.0, 2.0)

def _branch_lda_zp(state, bus):
    val = _bus_read(bus, _addr_zp(state, bus))
    return _do_load(state, bus, "A", val, 2.0, 3.0)

def _branch_lda_zp_x(state, bus):
    val = _bus_read(bus, _addr_zp_x(state, bus))
    return _do_load(state, bus, "A", val, 2.0, 4.0)

def _branch_lda_abs(state, bus):
    val = _bus_read(bus, _addr_abs(state, bus))
    return _do_load(state, bus, "A", val, 3.0, 4.0)

def _branch_lda_abs_x(state, bus):
    val = _bus_read(bus, _addr_abs_x(state, bus))
    return _do_load(state, bus, "A", val, 3.0, 4.0)

def _branch_lda_abs_y(state, bus):
    val = _bus_read(bus, _addr_abs_y(state, bus))
    return _do_load(state, bus, "A", val, 3.0, 4.0)

def _branch_lda_ind_x(state, bus):
    val = _bus_read(bus, _addr_ind_x(state, bus))
    return _do_load(state, bus, "A", val, 2.0, 6.0)

def _branch_lda_ind_y(state, bus):
    val = _bus_read(bus, _addr_ind_y(state, bus))
    return _do_load(state, bus, "A", val, 2.0, 5.0)


# LDX (immediate, zp, zp,Y, abs, abs,Y) -------------------------------------
def _branch_ldx_imm(state, bus):
    val = _operand_byte(bus, state.PC + 1.0)
    return _do_load(state, bus, "X", val, 2.0, 2.0)

def _branch_ldx_zp(state, bus):
    val = _bus_read(bus, _addr_zp(state, bus))
    return _do_load(state, bus, "X", val, 2.0, 3.0)

def _branch_ldx_zp_y(state, bus):
    val = _bus_read(bus, _addr_zp_y(state, bus))
    return _do_load(state, bus, "X", val, 2.0, 4.0)

def _branch_ldx_abs(state, bus):
    val = _bus_read(bus, _addr_abs(state, bus))
    return _do_load(state, bus, "X", val, 3.0, 4.0)

def _branch_ldx_abs_y(state, bus):
    val = _bus_read(bus, _addr_abs_y(state, bus))
    return _do_load(state, bus, "X", val, 3.0, 4.0)


# LDY (immediate, zp, zp,X, abs, abs,X) -------------------------------------
def _branch_ldy_imm(state, bus):
    val = _operand_byte(bus, state.PC + 1.0)
    return _do_load(state, bus, "Y", val, 2.0, 2.0)

def _branch_ldy_zp(state, bus):
    val = _bus_read(bus, _addr_zp(state, bus))
    return _do_load(state, bus, "Y", val, 2.0, 3.0)

def _branch_ldy_zp_x(state, bus):
    val = _bus_read(bus, _addr_zp_x(state, bus))
    return _do_load(state, bus, "Y", val, 2.0, 4.0)

def _branch_ldy_abs(state, bus):
    val = _bus_read(bus, _addr_abs(state, bus))
    return _do_load(state, bus, "Y", val, 3.0, 4.0)

def _branch_ldy_abs_x(state, bus):
    val = _bus_read(bus, _addr_abs_x(state, bus))
    return _do_load(state, bus, "Y", val, 3.0, 4.0)


# --- Store helpers + handlers ---------------------------------------------- #

def _do_store(state: SoftCPUState, bus: SoftBus, addr: jnp.ndarray,
              value: jnp.ndarray, instr_len: float, cycles: float):
    """Common path for STA / STX / STY — no flag changes, just write."""
    new_bus = _bus_write(bus, addr, value)
    return state._replace(
        PC=state.PC + instr_len,
        cycles=state.cycles + cycles,
    ), new_bus


# STA (zp, zp,X, abs, abs,X, abs,Y, (ind,X), (ind),Y) ------------------------
def _branch_sta_zp(state, bus):
    return _do_store(state, bus, _addr_zp(state, bus), state.A, 2.0, 3.0)

def _branch_sta_zp_x(state, bus):
    return _do_store(state, bus, _addr_zp_x(state, bus), state.A, 2.0, 4.0)

def _branch_sta_abs(state, bus):
    return _do_store(state, bus, _addr_abs(state, bus), state.A, 3.0, 4.0)

def _branch_sta_abs_x(state, bus):
    return _do_store(state, bus, _addr_abs_x(state, bus), state.A, 3.0, 5.0)

def _branch_sta_abs_y(state, bus):
    return _do_store(state, bus, _addr_abs_y(state, bus), state.A, 3.0, 5.0)

def _branch_sta_ind_x(state, bus):
    return _do_store(state, bus, _addr_ind_x(state, bus), state.A, 2.0, 6.0)

def _branch_sta_ind_y(state, bus):
    return _do_store(state, bus, _addr_ind_y(state, bus), state.A, 2.0, 6.0)


# STX (zp, zp,Y, abs) -------------------------------------------------------
def _branch_stx_zp(state, bus):
    return _do_store(state, bus, _addr_zp(state, bus), state.X, 2.0, 3.0)

def _branch_stx_zp_y(state, bus):
    return _do_store(state, bus, _addr_zp_y(state, bus), state.X, 2.0, 4.0)

def _branch_stx_abs(state, bus):
    return _do_store(state, bus, _addr_abs(state, bus), state.X, 3.0, 4.0)


# STY (zp, zp,X, abs) -------------------------------------------------------
def _branch_sty_zp(state, bus):
    return _do_store(state, bus, _addr_zp(state, bus), state.Y, 2.0, 3.0)

def _branch_sty_zp_x(state, bus):
    return _do_store(state, bus, _addr_zp_x(state, bus), state.Y, 2.0, 4.0)

def _branch_sty_abs(state, bus):
    return _do_store(state, bus, _addr_abs(state, bus), state.Y, 3.0, 4.0)


# --- Transfer handlers (TAX, TAY, TXA, TYA, TSX, TXS) ---------------------- #

def _branch_tax(state, bus):
    return _do_load(state, bus, "X", state.A, 1.0, 2.0)

def _branch_tay(state, bus):
    return _do_load(state, bus, "Y", state.A, 1.0, 2.0)

def _branch_txa(state, bus):
    return _do_load(state, bus, "A", state.X, 1.0, 2.0)

def _branch_tya(state, bus):
    return _do_load(state, bus, "A", state.Y, 1.0, 2.0)

def _branch_tsx(state, bus):
    return _do_load(state, bus, "X", state.SP, 1.0, 2.0)

def _branch_txs(state, bus):
    """TXS moves X→SP **without** affecting N/Z (the only flag-silent transfer)."""
    return state._replace(
        SP=state.X,
        PC=state.PC + 1.0,
        cycles=state.cycles + 2.0,
    ), bus


# --------------------------------------------------------------------------- #
# Dispatch table
# --------------------------------------------------------------------------- #

_HANDLERS = [_branch_default] * 256
# P7b core
_HANDLERS[0x00] = _branch_brk
_HANDLERS[0xEA] = _branch_nop
_HANDLERS[0x4C] = _branch_jmp_abs
# P7c-a — LDA
_HANDLERS[0xA9] = _branch_lda_imm
_HANDLERS[0xA5] = _branch_lda_zp
_HANDLERS[0xB5] = _branch_lda_zp_x
_HANDLERS[0xAD] = _branch_lda_abs
_HANDLERS[0xBD] = _branch_lda_abs_x
_HANDLERS[0xB9] = _branch_lda_abs_y
_HANDLERS[0xA1] = _branch_lda_ind_x
_HANDLERS[0xB1] = _branch_lda_ind_y
# P7c-a — LDX
_HANDLERS[0xA2] = _branch_ldx_imm
_HANDLERS[0xA6] = _branch_ldx_zp
_HANDLERS[0xB6] = _branch_ldx_zp_y
_HANDLERS[0xAE] = _branch_ldx_abs
_HANDLERS[0xBE] = _branch_ldx_abs_y
# P7c-a — LDY
_HANDLERS[0xA0] = _branch_ldy_imm
_HANDLERS[0xA4] = _branch_ldy_zp
_HANDLERS[0xB4] = _branch_ldy_zp_x
_HANDLERS[0xAC] = _branch_ldy_abs
_HANDLERS[0xBC] = _branch_ldy_abs_x
# P7c-a — STA
_HANDLERS[0x85] = _branch_sta_zp
_HANDLERS[0x95] = _branch_sta_zp_x
_HANDLERS[0x8D] = _branch_sta_abs
_HANDLERS[0x9D] = _branch_sta_abs_x
_HANDLERS[0x99] = _branch_sta_abs_y
_HANDLERS[0x81] = _branch_sta_ind_x
_HANDLERS[0x91] = _branch_sta_ind_y
# P7c-a — STX
_HANDLERS[0x86] = _branch_stx_zp
_HANDLERS[0x96] = _branch_stx_zp_y
_HANDLERS[0x8E] = _branch_stx_abs
# P7c-a — STY
_HANDLERS[0x84] = _branch_sty_zp
_HANDLERS[0x94] = _branch_sty_zp_x
_HANDLERS[0x8C] = _branch_sty_abs
# P7c-a — Transfers
_HANDLERS[0xAA] = _branch_tax
_HANDLERS[0xA8] = _branch_tay
_HANDLERS[0x8A] = _branch_txa
_HANDLERS[0x98] = _branch_tya
_HANDLERS[0xBA] = _branch_tsx
_HANDLERS[0x9A] = _branch_txs

# Opcodes handled with full behaviour (rather than default). Exposed so
# tests + a future P7c can introspect coverage.
SOFT_SUPPORTED_OPCODES = frozenset({
    # P7b core
    0x00, 0xEA, 0x4C,
    # P7c-a — LDA / LDX / LDY / STA / STX / STY / transfers
    0xA9, 0xA5, 0xB5, 0xAD, 0xBD, 0xB9, 0xA1, 0xB1,
    0xA2, 0xA6, 0xB6, 0xAE, 0xBE,
    0xA0, 0xA4, 0xB4, 0xAC, 0xBC,
    0x85, 0x95, 0x8D, 0x9D, 0x99, 0x81, 0x91,
    0x86, 0x96, 0x8E,
    0x84, 0x94, 0x8C,
    0xAA, 0xA8, 0x8A, 0x98, 0xBA, 0x9A,
})


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
