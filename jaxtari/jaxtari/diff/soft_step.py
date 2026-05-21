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

_NZ_CLEAR_MASK   = 0xFF ^ (0x80 | 0x02)               # clear N and Z
_NZC_CLEAR_MASK  = 0xFF ^ (0x80 | 0x02 | 0x01)        # clear N, Z, C
_NZCV_CLEAR_MASK = 0xFF ^ (0x80 | 0x40 | 0x02 | 0x01) # clear N, Z, C, V


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


def _set_nzc(p: jnp.ndarray, value: jnp.ndarray, carry: jnp.ndarray) -> jnp.ndarray:
    """Set N, Z (from value) and C (from carry boolean / int) in P."""
    v_int = value.astype(jnp.int32) & 0xFF
    p_int = p.astype(jnp.int32) & 0xFF
    p_int = p_int & _NZC_CLEAR_MASK
    z_bit = jnp.where(v_int == 0, 0x02, 0)
    n_bit = v_int & 0x80
    c_bit = (carry.astype(jnp.int32) & 0x01)
    return (p_int | z_bit | n_bit | c_bit).astype(jnp.float32)


def _set_nzcv(p: jnp.ndarray, value: jnp.ndarray,
              carry: jnp.ndarray, overflow: jnp.ndarray) -> jnp.ndarray:
    """Set all four arithmetic flags. Used by ADC and SBC."""
    v_int = value.astype(jnp.int32) & 0xFF
    p_int = p.astype(jnp.int32) & 0xFF
    p_int = p_int & _NZCV_CLEAR_MASK
    z_bit = jnp.where(v_int == 0, 0x02, 0)
    n_bit = v_int & 0x80
    c_bit = carry.astype(jnp.int32) & 0x01
    v_bit = jnp.where(overflow != 0, 0x40, 0)
    return (p_int | z_bit | n_bit | c_bit | v_bit).astype(jnp.float32)


def _set_bit_flags(p: jnp.ndarray, a_and_operand: jnp.ndarray,
                   operand: jnp.ndarray) -> jnp.ndarray:
    """For BIT: Z from `A AND operand`; N from operand bit 7; V from operand bit 6."""
    p_int  = p.astype(jnp.int32) & 0xFF
    p_int  = p_int & _NZCV_CLEAR_MASK | (p_int & 0x01)  # keep C unchanged
    and_v  = a_and_operand.astype(jnp.int32) & 0xFF
    op_int = operand.astype(jnp.int32) & 0xFF
    z_bit  = jnp.where(and_v == 0, 0x02, 0)
    n_bit  = op_int & 0x80
    v_bit  = op_int & 0x40
    return (p_int | z_bit | n_bit | v_bit).astype(jnp.float32)


def _read_carry(p: jnp.ndarray) -> jnp.ndarray:
    """Extract the C bit from P as a 0/1 int32 scalar."""
    return p.astype(jnp.int32) & 0x01


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
# P7c-b — arithmetic and logic (ADC / SBC / AND / ORA / EOR / CMP / CPX /
# CPY / BIT). For ADC and SBC, BCD mode is NOT implemented in SOFT — the
# binary path always runs regardless of the D flag, matching xitari's
# behaviour with D=0 (Atari 2600 ROMs almost never set decimal mode and the
# 2A03 ignored it entirely). Real BCD support for SOFT mode is a P7c-bx
# follow-up if it's ever needed.
# --------------------------------------------------------------------------- #

def _operand_for_mode(state, bus, mode: str) -> jnp.ndarray:
    """Resolve an operand BYTE for one of the byte-yielding modes used by
    arithmetic/logic ops. `mode` is one of: 'imm', 'zp', 'zp_x', 'zp_y',
    'abs', 'abs_x', 'abs_y', 'ind_x', 'ind_y'."""
    if mode == "imm":
        return _operand_byte(bus, state.PC + 1.0)
    if mode == "zp":
        return _bus_read(bus, _addr_zp(state, bus))
    if mode == "zp_x":
        return _bus_read(bus, _addr_zp_x(state, bus))
    if mode == "zp_y":
        return _bus_read(bus, _addr_zp_y(state, bus))
    if mode == "abs":
        return _bus_read(bus, _addr_abs(state, bus))
    if mode == "abs_x":
        return _bus_read(bus, _addr_abs_x(state, bus))
    if mode == "abs_y":
        return _bus_read(bus, _addr_abs_y(state, bus))
    if mode == "ind_x":
        return _bus_read(bus, _addr_ind_x(state, bus))
    if mode == "ind_y":
        return _bus_read(bus, _addr_ind_y(state, bus))
    raise ValueError(f"unknown mode {mode!r}")


# ADC / SBC --------------------------------------------------------------- #

def _adc_step(state: SoftCPUState, bus: SoftBus, operand: jnp.ndarray,
              instr_len: float, cycles: float):
    """Binary-mode ADC. Float arithmetic on the value so the operand
    gradient flows; flag updates go via int cast."""
    c_in   = _read_carry(state.P).astype(jnp.float32)
    sum_   = state.A + operand + c_in
    new_a  = sum_ - jnp.floor(sum_ / 256.0) * 256.0           # & 0xFF, diff-clean
    carry  = (sum_.astype(jnp.int32) > 0xFF).astype(jnp.int32)
    # Signed overflow: (A^new_a) & (operand^new_a) & 0x80
    a_int       = state.A.astype(jnp.int32) & 0xFF
    op_int      = operand.astype(jnp.int32) & 0xFF
    new_a_int   = new_a.astype(jnp.int32) & 0xFF
    overflow    = ((a_int ^ new_a_int) & (op_int ^ new_a_int)) & 0x80
    new_p       = _set_nzcv(state.P, new_a, carry, overflow)
    return state._replace(
        A=new_a,
        P=new_p,
        PC=state.PC + instr_len,
        cycles=state.cycles + cycles,
    ), bus


def _sbc_step(state: SoftCPUState, bus: SoftBus, operand: jnp.ndarray,
              instr_len: float, cycles: float):
    """Binary-mode SBC. Implemented as ADC with one's-complement operand
    so the carry/overflow logic stays uniform with _adc_step."""
    op_inv = 255.0 - operand
    c_in   = _read_carry(state.P).astype(jnp.float32)
    sum_   = state.A + op_inv + c_in
    new_a  = sum_ - jnp.floor(sum_ / 256.0) * 256.0
    carry  = (sum_.astype(jnp.int32) > 0xFF).astype(jnp.int32)
    a_int       = state.A.astype(jnp.int32) & 0xFF
    op_inv_int  = op_inv.astype(jnp.int32) & 0xFF
    new_a_int   = new_a.astype(jnp.int32) & 0xFF
    overflow    = ((a_int ^ new_a_int) & (op_inv_int ^ new_a_int)) & 0x80
    new_p       = _set_nzcv(state.P, new_a, carry, overflow)
    return state._replace(
        A=new_a,
        P=new_p,
        PC=state.PC + instr_len,
        cycles=state.cycles + cycles,
    ), bus


# ADC handlers (8 modes)
def _branch_adc_imm(s, b):    return _adc_step(s, b, _operand_for_mode(s, b, "imm"),   2.0, 2.0)
def _branch_adc_zp(s, b):     return _adc_step(s, b, _operand_for_mode(s, b, "zp"),    2.0, 3.0)
def _branch_adc_zp_x(s, b):   return _adc_step(s, b, _operand_for_mode(s, b, "zp_x"),  2.0, 4.0)
def _branch_adc_abs(s, b):    return _adc_step(s, b, _operand_for_mode(s, b, "abs"),   3.0, 4.0)
def _branch_adc_abs_x(s, b):  return _adc_step(s, b, _operand_for_mode(s, b, "abs_x"), 3.0, 4.0)
def _branch_adc_abs_y(s, b):  return _adc_step(s, b, _operand_for_mode(s, b, "abs_y"), 3.0, 4.0)
def _branch_adc_ind_x(s, b):  return _adc_step(s, b, _operand_for_mode(s, b, "ind_x"), 2.0, 6.0)
def _branch_adc_ind_y(s, b):  return _adc_step(s, b, _operand_for_mode(s, b, "ind_y"), 2.0, 5.0)

# SBC handlers (8 modes + USBC alias $EB)
def _branch_sbc_imm(s, b):    return _sbc_step(s, b, _operand_for_mode(s, b, "imm"),   2.0, 2.0)
def _branch_sbc_zp(s, b):     return _sbc_step(s, b, _operand_for_mode(s, b, "zp"),    2.0, 3.0)
def _branch_sbc_zp_x(s, b):   return _sbc_step(s, b, _operand_for_mode(s, b, "zp_x"),  2.0, 4.0)
def _branch_sbc_abs(s, b):    return _sbc_step(s, b, _operand_for_mode(s, b, "abs"),   3.0, 4.0)
def _branch_sbc_abs_x(s, b):  return _sbc_step(s, b, _operand_for_mode(s, b, "abs_x"), 3.0, 4.0)
def _branch_sbc_abs_y(s, b):  return _sbc_step(s, b, _operand_for_mode(s, b, "abs_y"), 3.0, 4.0)
def _branch_sbc_ind_x(s, b):  return _sbc_step(s, b, _operand_for_mode(s, b, "ind_x"), 2.0, 6.0)
def _branch_sbc_ind_y(s, b):  return _sbc_step(s, b, _operand_for_mode(s, b, "ind_y"), 2.0, 5.0)


# AND / ORA / EOR --------------------------------------------------------- #
# Bitwise ops are integer-only; the operand gradient is stopped at the int
# cast. For XAI we still get gradient on the *destination* register through
# the soft_rom_peek path that fetched the operand byte.

def _bitop_step(state, bus, operand, op, instr_len, cycles):
    a_int    = state.A.astype(jnp.int32) & 0xFF
    op_int   = operand.astype(jnp.int32) & 0xFF
    new_a_int = op(a_int, op_int) & 0xFF
    new_a    = new_a_int.astype(jnp.float32)
    return state._replace(
        A=new_a,
        P=_set_nz(state.P, new_a),
        PC=state.PC + instr_len,
        cycles=state.cycles + cycles,
    ), bus


def _branch_and_imm(s, b):   return _bitop_step(s, b, _operand_for_mode(s, b, "imm"),   lambda a, o: a & o, 2.0, 2.0)
def _branch_and_zp(s, b):    return _bitop_step(s, b, _operand_for_mode(s, b, "zp"),    lambda a, o: a & o, 2.0, 3.0)
def _branch_and_zp_x(s, b):  return _bitop_step(s, b, _operand_for_mode(s, b, "zp_x"),  lambda a, o: a & o, 2.0, 4.0)
def _branch_and_abs(s, b):   return _bitop_step(s, b, _operand_for_mode(s, b, "abs"),   lambda a, o: a & o, 3.0, 4.0)
def _branch_and_abs_x(s, b): return _bitop_step(s, b, _operand_for_mode(s, b, "abs_x"), lambda a, o: a & o, 3.0, 4.0)
def _branch_and_abs_y(s, b): return _bitop_step(s, b, _operand_for_mode(s, b, "abs_y"), lambda a, o: a & o, 3.0, 4.0)
def _branch_and_ind_x(s, b): return _bitop_step(s, b, _operand_for_mode(s, b, "ind_x"), lambda a, o: a & o, 2.0, 6.0)
def _branch_and_ind_y(s, b): return _bitop_step(s, b, _operand_for_mode(s, b, "ind_y"), lambda a, o: a & o, 2.0, 5.0)

def _branch_ora_imm(s, b):   return _bitop_step(s, b, _operand_for_mode(s, b, "imm"),   lambda a, o: a | o, 2.0, 2.0)
def _branch_ora_zp(s, b):    return _bitop_step(s, b, _operand_for_mode(s, b, "zp"),    lambda a, o: a | o, 2.0, 3.0)
def _branch_ora_zp_x(s, b):  return _bitop_step(s, b, _operand_for_mode(s, b, "zp_x"),  lambda a, o: a | o, 2.0, 4.0)
def _branch_ora_abs(s, b):   return _bitop_step(s, b, _operand_for_mode(s, b, "abs"),   lambda a, o: a | o, 3.0, 4.0)
def _branch_ora_abs_x(s, b): return _bitop_step(s, b, _operand_for_mode(s, b, "abs_x"), lambda a, o: a | o, 3.0, 4.0)
def _branch_ora_abs_y(s, b): return _bitop_step(s, b, _operand_for_mode(s, b, "abs_y"), lambda a, o: a | o, 3.0, 4.0)
def _branch_ora_ind_x(s, b): return _bitop_step(s, b, _operand_for_mode(s, b, "ind_x"), lambda a, o: a | o, 2.0, 6.0)
def _branch_ora_ind_y(s, b): return _bitop_step(s, b, _operand_for_mode(s, b, "ind_y"), lambda a, o: a | o, 2.0, 5.0)

def _branch_eor_imm(s, b):   return _bitop_step(s, b, _operand_for_mode(s, b, "imm"),   lambda a, o: a ^ o, 2.0, 2.0)
def _branch_eor_zp(s, b):    return _bitop_step(s, b, _operand_for_mode(s, b, "zp"),    lambda a, o: a ^ o, 2.0, 3.0)
def _branch_eor_zp_x(s, b):  return _bitop_step(s, b, _operand_for_mode(s, b, "zp_x"),  lambda a, o: a ^ o, 2.0, 4.0)
def _branch_eor_abs(s, b):   return _bitop_step(s, b, _operand_for_mode(s, b, "abs"),   lambda a, o: a ^ o, 3.0, 4.0)
def _branch_eor_abs_x(s, b): return _bitop_step(s, b, _operand_for_mode(s, b, "abs_x"), lambda a, o: a ^ o, 3.0, 4.0)
def _branch_eor_abs_y(s, b): return _bitop_step(s, b, _operand_for_mode(s, b, "abs_y"), lambda a, o: a ^ o, 3.0, 4.0)
def _branch_eor_ind_x(s, b): return _bitop_step(s, b, _operand_for_mode(s, b, "ind_x"), lambda a, o: a ^ o, 2.0, 6.0)
def _branch_eor_ind_y(s, b): return _bitop_step(s, b, _operand_for_mode(s, b, "ind_y"), lambda a, o: a ^ o, 2.0, 5.0)


# CMP / CPX / CPY --------------------------------------------------------- #

def _compare_step(state, bus, reg_value, operand, instr_len, cycles):
    """Set N, Z, C from `reg_value - operand` (8-bit unsigned)."""
    reg_int   = reg_value.astype(jnp.int32) & 0xFF
    op_int    = operand.astype(jnp.int32) & 0xFF
    diff_int  = (reg_int - op_int) & 0xFF
    diff_f    = diff_int.astype(jnp.float32)
    carry     = (reg_int >= op_int).astype(jnp.int32)
    return state._replace(
        P=_set_nzc(state.P, diff_f, carry),
        PC=state.PC + instr_len,
        cycles=state.cycles + cycles,
    ), bus


def _branch_cmp_imm(s, b):   return _compare_step(s, b, s.A, _operand_for_mode(s, b, "imm"),   2.0, 2.0)
def _branch_cmp_zp(s, b):    return _compare_step(s, b, s.A, _operand_for_mode(s, b, "zp"),    2.0, 3.0)
def _branch_cmp_zp_x(s, b):  return _compare_step(s, b, s.A, _operand_for_mode(s, b, "zp_x"),  2.0, 4.0)
def _branch_cmp_abs(s, b):   return _compare_step(s, b, s.A, _operand_for_mode(s, b, "abs"),   3.0, 4.0)
def _branch_cmp_abs_x(s, b): return _compare_step(s, b, s.A, _operand_for_mode(s, b, "abs_x"), 3.0, 4.0)
def _branch_cmp_abs_y(s, b): return _compare_step(s, b, s.A, _operand_for_mode(s, b, "abs_y"), 3.0, 4.0)
def _branch_cmp_ind_x(s, b): return _compare_step(s, b, s.A, _operand_for_mode(s, b, "ind_x"), 2.0, 6.0)
def _branch_cmp_ind_y(s, b): return _compare_step(s, b, s.A, _operand_for_mode(s, b, "ind_y"), 2.0, 5.0)

def _branch_cpx_imm(s, b):   return _compare_step(s, b, s.X, _operand_for_mode(s, b, "imm"),  2.0, 2.0)
def _branch_cpx_zp(s, b):    return _compare_step(s, b, s.X, _operand_for_mode(s, b, "zp"),   2.0, 3.0)
def _branch_cpx_abs(s, b):   return _compare_step(s, b, s.X, _operand_for_mode(s, b, "abs"),  3.0, 4.0)

def _branch_cpy_imm(s, b):   return _compare_step(s, b, s.Y, _operand_for_mode(s, b, "imm"),  2.0, 2.0)
def _branch_cpy_zp(s, b):    return _compare_step(s, b, s.Y, _operand_for_mode(s, b, "zp"),   2.0, 3.0)
def _branch_cpy_abs(s, b):   return _compare_step(s, b, s.Y, _operand_for_mode(s, b, "abs"),  3.0, 4.0)


# BIT --------------------------------------------------------------------- #

def _bit_step(state, bus, operand, instr_len, cycles):
    """Z from A AND operand; N from operand bit 7; V from operand bit 6."""
    a_int   = state.A.astype(jnp.int32) & 0xFF
    op_int  = operand.astype(jnp.int32) & 0xFF
    and_v   = (a_int & op_int).astype(jnp.float32)
    return state._replace(
        P=_set_bit_flags(state.P, and_v, operand),
        PC=state.PC + instr_len,
        cycles=state.cycles + cycles,
    ), bus


def _branch_bit_zp(s, b):  return _bit_step(s, b, _operand_for_mode(s, b, "zp"),  2.0, 3.0)
def _branch_bit_abs(s, b): return _bit_step(s, b, _operand_for_mode(s, b, "abs"), 3.0, 4.0)


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
# P7c-b — ADC
_HANDLERS[0x69] = _branch_adc_imm
_HANDLERS[0x65] = _branch_adc_zp
_HANDLERS[0x75] = _branch_adc_zp_x
_HANDLERS[0x6D] = _branch_adc_abs
_HANDLERS[0x7D] = _branch_adc_abs_x
_HANDLERS[0x79] = _branch_adc_abs_y
_HANDLERS[0x61] = _branch_adc_ind_x
_HANDLERS[0x71] = _branch_adc_ind_y
# P7c-b — SBC (+ USBC $EB alias)
_HANDLERS[0xE9] = _branch_sbc_imm
_HANDLERS[0xE5] = _branch_sbc_zp
_HANDLERS[0xF5] = _branch_sbc_zp_x
_HANDLERS[0xED] = _branch_sbc_abs
_HANDLERS[0xFD] = _branch_sbc_abs_x
_HANDLERS[0xF9] = _branch_sbc_abs_y
_HANDLERS[0xE1] = _branch_sbc_ind_x
_HANDLERS[0xF1] = _branch_sbc_ind_y
_HANDLERS[0xEB] = _branch_sbc_imm                # USBC alias
# P7c-b — AND
_HANDLERS[0x29] = _branch_and_imm
_HANDLERS[0x25] = _branch_and_zp
_HANDLERS[0x35] = _branch_and_zp_x
_HANDLERS[0x2D] = _branch_and_abs
_HANDLERS[0x3D] = _branch_and_abs_x
_HANDLERS[0x39] = _branch_and_abs_y
_HANDLERS[0x21] = _branch_and_ind_x
_HANDLERS[0x31] = _branch_and_ind_y
# P7c-b — ORA
_HANDLERS[0x09] = _branch_ora_imm
_HANDLERS[0x05] = _branch_ora_zp
_HANDLERS[0x15] = _branch_ora_zp_x
_HANDLERS[0x0D] = _branch_ora_abs
_HANDLERS[0x1D] = _branch_ora_abs_x
_HANDLERS[0x19] = _branch_ora_abs_y
_HANDLERS[0x01] = _branch_ora_ind_x
_HANDLERS[0x11] = _branch_ora_ind_y
# P7c-b — EOR
_HANDLERS[0x49] = _branch_eor_imm
_HANDLERS[0x45] = _branch_eor_zp
_HANDLERS[0x55] = _branch_eor_zp_x
_HANDLERS[0x4D] = _branch_eor_abs
_HANDLERS[0x5D] = _branch_eor_abs_x
_HANDLERS[0x59] = _branch_eor_abs_y
_HANDLERS[0x41] = _branch_eor_ind_x
_HANDLERS[0x51] = _branch_eor_ind_y
# P7c-b — CMP
_HANDLERS[0xC9] = _branch_cmp_imm
_HANDLERS[0xC5] = _branch_cmp_zp
_HANDLERS[0xD5] = _branch_cmp_zp_x
_HANDLERS[0xCD] = _branch_cmp_abs
_HANDLERS[0xDD] = _branch_cmp_abs_x
_HANDLERS[0xD9] = _branch_cmp_abs_y
_HANDLERS[0xC1] = _branch_cmp_ind_x
_HANDLERS[0xD1] = _branch_cmp_ind_y
# P7c-b — CPX / CPY
_HANDLERS[0xE0] = _branch_cpx_imm
_HANDLERS[0xE4] = _branch_cpx_zp
_HANDLERS[0xEC] = _branch_cpx_abs
_HANDLERS[0xC0] = _branch_cpy_imm
_HANDLERS[0xC4] = _branch_cpy_zp
_HANDLERS[0xCC] = _branch_cpy_abs
# P7c-b — BIT
_HANDLERS[0x24] = _branch_bit_zp
_HANDLERS[0x2C] = _branch_bit_abs

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
    # P7c-b — ADC / SBC / AND / ORA / EOR / CMP / CPX / CPY / BIT
    0x69, 0x65, 0x75, 0x6D, 0x7D, 0x79, 0x61, 0x71,
    0xE9, 0xE5, 0xF5, 0xED, 0xFD, 0xF9, 0xE1, 0xF1, 0xEB,
    0x29, 0x25, 0x35, 0x2D, 0x3D, 0x39, 0x21, 0x31,
    0x09, 0x05, 0x15, 0x0D, 0x1D, 0x19, 0x01, 0x11,
    0x49, 0x45, 0x55, 0x4D, 0x5D, 0x59, 0x41, 0x51,
    0xC9, 0xC5, 0xD5, 0xCD, 0xDD, 0xD9, 0xC1, 0xD1,
    0xE0, 0xE4, 0xEC,
    0xC0, 0xC4, 0xCC,
    0x24, 0x2C,
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
