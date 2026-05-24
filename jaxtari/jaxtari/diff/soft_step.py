"""SOFT-mode `step()` — a differentiable parallel to `cpu.m6502.step`.

`soft_step(state, bus)` executes one 6502 instruction with the SOFT
primitives from `jaxtari.diff`: memory access goes through
`soft_rom_peek` / `soft_ram_peek` (one-hot dot products, gradient-friendly);
register state is `float32`; opcode dispatch is `jax.lax.switch` over
all 256 opcodes. The HARD `step()` is left untouched — SOFT mode is a
parallel execution path.

**Opcode coverage — COMPLETE as of P7c-f.** All 151 documented NMOS
6502 opcodes plus the undocumented USBC ($EB) alias are handled:

  P7b  : NOP, the original LDA/LDX/STA/STX core, JMP abs, BRK
  P7c-a: full load/store/transfer (LDA/LDX/LDY/STA/STX/STY all modes,
         TAX/TAY/TXA/TYA/TSX/TXS) + N/Z flag updates
  P7c-b: ADC/SBC (binary + BCD via P7c-bx; USBC alias), AND/ORA/EOR,
         CMP/CPX/CPY, BIT + N/Z/C/V flag updates
  P7c-c: ASL/LSR/ROL/ROR (accumulator + 4 memory modes each)
  P7c-d: 8 conditional branches, JMP indirect, JSR / RTS
  P7c-e: PHA/PHP/PLA/PLP, CLC/SEC/CLI/SEI/CLV/CLD/SED, INC/DEC,
         INX/INY/DEX/DEY
  P7c-f: RTI

Any opcode outside the documented set still falls through to
`_branch_default` (PC += 1, cycles += 2) — non-raising so the trace
stays differentiable.

**Known SOFT-mode simplifications (deferred):**
  - Branch predicates are HARD (`jnp.where` on the flag bit): forward
    PC is exact, but gradient through the predicate is broken at the
    int-cast of `P`. A float-valued flag representation would let
    `soft_branch` carry that gradient. (P7c-dx)
  - BRK stays the "end-of-trace sentinel" from P7b rather than running
    the proper interrupt sequence — the useful semantics for fixed-
    length XAI traces.
  - Bus dispatch is simplified: cart-range reads use `soft_rom_peek`,
    everything else maps into the 128-byte RAM array via `addr & 0x7F`.
    TIA / RIOT register writes therefore land in RAM rather than
    affecting chip state or a framebuffer. Real TIA / RIOT / cart-
    hotspot dispatch + a differentiable TIA is **P7f**.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxtari.diff.rom_as_weights import RomTensor
from jaxtari.diff.soft_branch import soft_branch
from jaxtari.diff.soft_state import SoftBus, SoftCPUState


# --------------------------------------------------------------------------- #
# Differentiable memory access
# --------------------------------------------------------------------------- #

def _rom_array(rom) -> jnp.ndarray:
    """Return the underlying float array whether `rom` is a raw
    `jnp.ndarray` or a `RomTensor` wrapper (P7d). Both are valid
    `SoftBus.rom` payloads."""
    return rom.rom if isinstance(rom, RomTensor) else rom


def soft_rom_peek(rom, addr) -> jnp.ndarray:
    """Differentiable cart byte read — `one_hot(addr) · rom`.

    `rom` may be a raw 1-D numeric array or a `RomTensor` (P7d). `addr`
    is a scalar (int or float). The address is wrapped modulo the
    ROM size so the cart mirrors correctly in the 4 KB $F000-$FFFF
    window — a 2 KB cart like `pong.bin` is mirrored twice (P8-cx
    needed this for IRQ-vector reads at $FFFE/$FFFF, which a naive
    `& 0x0FFF` would index past the ROM). Cart sizes are powers of 2
    (2K/4K/8K/16K/32K), so the `& (n - 1)` wrap is exact.
    """
    arr = _rom_array(rom)
    n = arr.shape[0]
    # `%` works for any ROM size (including the small fixtures the unit
    # tests use); for production cart sizes (powers of 2) JAX lowers it
    # to a bitwise AND on the integer path anyway.
    addr_wrapped = jnp.asarray(addr).astype(jnp.int32) % n
    one_hot = jax.nn.one_hot(addr_wrapped, n, dtype=jnp.float32)
    return jnp.dot(one_hot, arr.astype(jnp.float32))


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
    Proper TIA / RIOT register dispatch is P7c-f / P7f-dx (deferred).

    `jnp.where` picks the right branch and the gradient flows through
    both — the wrong branch contributes 0 because its `one_hot` weight
    is unused, but the cost is two peeks instead of one. The clean
    gradient is the point.

    **P7f-dx attempt (reverted)**: an earlier change in this batch
    wired `soft_collision_registers` into the $30-$37 range so SOFT
    programs could `LDA $30` for CXM0P etc. That broke the existing
    SOFT bus collapse contract — several P7c-c / P7c-d tests use the
    $30 cell as a normal RAM byte (which is what the SOFT collapse
    has always promised). Proper TIA/RAM separation in SOFT mode is
    the architectural prerequisite — see STATUS.md P7f-dx for the
    deferral rationale. Callers that want collision values today
    should call `soft_collision_registers(bus)` directly.
    """
    a_int       = jnp.asarray(addr).astype(jnp.int32) & 0x1FFF
    is_cart     = (a_int & 0x1000) != 0
    is_riot_io  = (~is_cart) & ((a_int & 0x80) != 0) & ((a_int & 0x200) != 0)
    cart_offset = a_int & 0x0FFF
    ram_offset  = a_int & 0x7F

    cart_val = soft_rom_peek(bus.rom, cart_offset)
    ram_val  = soft_ram_peek(bus.ram, ram_offset)

    # P8-cx: RIOT I/O reads ($0280-$029F band, decoded by `addr & 0x07`):
    #   reg 4 (INTIM) → bus.riot_intim
    #   reg 5 (INSTAT) → bit 7 = riot_expired latch
    #   anything else (SWCHA/SWACNT/SWCHB/SWBCNT) → 0xFF, the "all
    #   console switches released, all joystick directions released"
    #   default. Falling through to RAM here was the bug that kept
    #   Pong's init reading RESET-pressed because a prior VSYNC write
    #   to $02 had clobbered ram[2] (the SOFT collapse stores TIA $02
    #   and reads RIOT $0282 in the same cell). Returning a constant
    #   gives Pong an idle controller — enough for its init to proceed.
    reg = a_int & 0x07
    riot_val = jnp.where(
        reg == 4, bus.riot_intim,
        jnp.where(reg == 5,
                  jnp.where(bus.riot_expired != 0, jnp.float32(0x80), jnp.float32(0)),
                  jnp.float32(0xFF)))

    return jnp.where(is_cart, cart_val,
                     jnp.where(is_riot_io, riot_val, ram_val))


# Prescaler-shift lookup for TIM*T addresses: low 2 bits select
# 1/8/64/1024 cycles per tick (= 2^0/3/6/10).
_TIMT_SHIFTS = jnp.array([0, 3, 6, 10], dtype=jnp.float32)


def _bus_write(bus: SoftBus, addr, value: jnp.ndarray) -> SoftBus:
    """SOFT-mode differentiable bus write.

    Writes to ROM are silently dropped. Writes to RIOT TIM*T (detected
    by `(addr & 0x14) == 0x14`, matching the HARD RIOT decode) load the
    P8-cx timer fields — INTIM = value, prescaler from the low 2 bits,
    residual cycles reset, expired flag cleared. Everything else
    mutates `bus.ram[addr & 0x7F]` as before.

    All branches are evaluated and selected with `jnp.where` so the
    gradient stays clean.
    """
    a_int      = jnp.asarray(addr).astype(jnp.int32) & 0x1FFF
    ram_offset = a_int & 0x7F
    is_cart    = (a_int & 0x1000) != 0
    # P8-cx: TIM*T detection must ALSO require the RIOT I/O region
    # (A7=1, A9=1) — `(addr & 0x14) == 0x14` alone matches innocuous
    # stack addresses like $01FD (bits 2 + 4 happen to be set), and
    # without the region guard every BRK push got reinterpreted as a
    # timer load. Round-trip stack pushes / pops then pulled zeros and
    # any RTI landed at PC=$0000.
    is_riot_io = (~is_cart) & ((a_int & 0x80) != 0) & ((a_int & 0x200) != 0)
    is_tim_t   = is_riot_io & ((a_int & 0x14) == 0x14)

    # RAM path: keep current value if this is a cart or TIM*T write.
    current   = soft_ram_peek(bus.ram, ram_offset)
    new_val   = jnp.where(is_cart | is_tim_t, current, value)
    new_ram   = bus.ram.at[ram_offset].set(new_val)

    # P8-cx: timer load on TIM*T.
    new_intim          = jnp.where(is_tim_t, value, bus.riot_intim)
    new_prescaler_shft = jnp.where(
        is_tim_t, _TIMT_SHIFTS[a_int & 0x03], bus.riot_prescaler_shift)
    new_residual       = jnp.where(is_tim_t, jnp.float32(0), bus.riot_residual_cycles)
    new_expired        = jnp.where(is_tim_t, jnp.float32(0), bus.riot_expired)

    return bus._replace(
        ram=new_ram,
        riot_intim=new_intim,
        riot_prescaler_shift=new_prescaler_shft,
        riot_residual_cycles=new_residual,
        riot_expired=new_expired,
    )


def _advance_riot_timer(bus: SoftBus, cpu_cycles) -> SoftBus:
    """P8-cx: tick the RIOT timer by `cpu_cycles` CPU cycles.

    Simplified model — pre-expiration only: INTIM decrements once per
    `2 ** prescaler_shift` cycles; on reaching 0 the `riot_expired`
    latch fires and INTIM stays at 0. The HARD RIOT switches to one-
    tick-per-cycle after expiration; SOFT mode floors at 0 because
    every Atari ROM polling INTIM cares about the *reaches-zero*
    moment, not the post-expiration wrap-around timing.
    """
    prescaler   = jnp.int32(1) << bus.riot_prescaler_shift.astype(jnp.int32)
    total       = bus.riot_residual_cycles.astype(jnp.int32) + cpu_cycles.astype(jnp.int32)
    ticks       = total // prescaler
    residual    = (total - ticks * prescaler).astype(jnp.float32)

    intim_i     = bus.riot_intim.astype(jnp.int32)
    new_intim_i = jnp.maximum(intim_i - ticks, 0)
    new_expired = (intim_i <= ticks).astype(jnp.float32)
    # `riot_expired` latches — once set, stays set until the next TIM*T
    # write clears it.
    new_expired = jnp.where(bus.riot_expired != 0, bus.riot_expired, new_expired)

    return bus._replace(
        riot_intim=new_intim_i.astype(jnp.float32),
        riot_residual_cycles=residual,
        riot_expired=new_expired,
    )


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
# P7c-dx — float-valued flag mirrors.
#
# `SoftCPUState` now carries `P_N` / `P_Z` / `P_C` / `P_V` alongside the
# packed `P` byte. Every `state._replace(P=...)` site routes through
# `_with_p` instead — that single helper recomputes the four floats
# from the new packed byte, keeping them in lock-step. The packed P
# stays the single source of truth so PHP / PLP / RTI continue to work
# without separate logic.
#
# The hook that *uses* the floats is `_do_branch`: it feeds the matching
# float into `soft_branch`, so the PC blend becomes sigmoid-smooth — a
# `jax.grad` over a trace gets a meaningful gradient through the branch
# predicate (weighted by the sigmoid's slope at the flag boundary).
# Forward correctness is preserved with a straight-through round, so
# `pong_noop_10` and every existing test still pass bit-exact.
# --------------------------------------------------------------------------- #

def _float_flags_from_p(p: jnp.ndarray):
    """Split a packed status byte into its (N, Z, C, V) float mirrors.
    Each return value is a 0.0/1.0 float — int-extracted, so the
    gradient is zero w.r.t. the underlying packed P (as expected for a
    structural flag bit), but the float itself slots cleanly into
    `soft_branch`."""
    pi = p.astype(jnp.int32) & 0xFF
    return (
        ((pi >> 7) & 1).astype(jnp.float32),    # N (bit 7)
        ((pi >> 1) & 1).astype(jnp.float32),    # Z (bit 1)
        ((pi >> 0) & 1).astype(jnp.float32),    # C (bit 0)
        ((pi >> 6) & 1).astype(jnp.float32),    # V (bit 6)
    )


def _with_p(state: SoftCPUState, new_p: jnp.ndarray, **other_fields) -> SoftCPUState:
    """Drop-in replacement for `state._replace(P=new_p, **other_fields)`
    that also syncs the four float-flag mirrors from the new packed P.
    Use this *every* time a flag-touching opcode updates P — the
    packed byte stays the source of truth; the floats are derived.
    """
    n_f, z_f, c_f, v_f = _float_flags_from_p(new_p)
    return state._replace(P=new_p, P_N=n_f, P_Z=z_f, P_C=c_f, P_V=v_f,
                          **other_fields)


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
    """BRK — proper interrupt sequence: push PC+2, push P|B|U, set the
    I flag, jump to the IRQ vector at $FFFE/$FFFF.

    P8-cx note: P7b through P7c had BRK as an "end-of-trace sentinel"
    (halt in place) because the early XAI traces didn't hit it. Real
    Atari ROMs use BRK intentionally — Pong's vertical-blank kernel
    sits on a BRK at $F262, branching into the IRQ handler at the
    vector — so SOFT mode now performs the proper sequence. 7 cycles.
    Uses `_push8` (defined later in the module, looked up at call
    time)."""
    return_addr = state.PC + 2.0
    ra_int = return_addr.astype(jnp.int32) & 0xFFFF
    hi = ((ra_int >> 8) & 0xFF).astype(jnp.float32)
    lo = (ra_int & 0xFF).astype(jnp.float32)
    # Push the return address (high first, then low — RTI pops low then high)
    bus, sp = _push8(bus, state.SP, hi)
    bus, sp = _push8(bus, sp, lo)
    # Push P with B and U forced on (the BRK-pushed copy of P).
    p_pushed = (state.P.astype(jnp.int32) | 0x30).astype(jnp.float32)
    bus, sp = _push8(bus, sp, p_pushed)
    # Set I flag (keep U set as an invariant).
    new_p = (state.P.astype(jnp.int32) | 0x04 | 0x20).astype(jnp.float32)
    # Jump to the IRQ vector at $FFFE / $FFFF. (`_bus_read` handles
    # the 2K/4K cart mirror via the wrap inside `soft_rom_peek`.)
    irq_lo = _bus_read(bus, 0xFFFE)
    irq_hi = _bus_read(bus, 0xFFFF)
    new_pc = irq_lo + irq_hi * 256.0
    return _with_p(state, new_p,
        SP=sp,
        PC=new_pc,
        cycles=state.cycles + 7.0,
    ), bus


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
        "PC":     state.PC + instr_len,
        "cycles": state.cycles + cycles,
        reg:      value,
    }
    return _with_p(state, new_p, **fields), bus


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
# CPY / BIT).
#
# P7c-bx extends ADC / SBC with BCD (decimal-mode) support: both the
# binary and the BCD result are computed, then `jnp.where` selects on the
# D flag. The binary path stays float arithmetic (gradient-clean); the
# BCD path is integer nibble arithmetic (gradient breaks at the int
# cast — acceptable, decimal mode is rare on the 2600). BCD formulas
# match xitari/M6502Hi.ins (see jaxtari.cpu.alu adc / sbc).
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

def _bcd_decode(byte_int: jnp.ndarray) -> jnp.ndarray:
    """Two BCD digits packed in a byte → integer 0..99."""
    return ((byte_int >> 4) & 0x0F) * 10 + (byte_int & 0x0F)


def _bcd_encode(n: jnp.ndarray) -> jnp.ndarray:
    """Integer → BCD byte. `n` is reduced mod 100 first."""
    n = n % 100
    return ((n // 10) << 4) | (n % 10)


def _decimal_flag(p: jnp.ndarray) -> jnp.ndarray:
    """The D (decimal-mode) flag as a boolean."""
    return (p.astype(jnp.int32) & 0x08) != 0


def _adc_step(state: SoftCPUState, bus: SoftBus, operand: jnp.ndarray,
              instr_len: float, cycles: float):
    """ADC with D-flag dispatch. The binary path is float arithmetic so
    the operand gradient flows; the BCD path is integer (gradient breaks
    at the int cast — decimal mode is rare on the 2600)."""
    c_in    = _read_carry(state.P).astype(jnp.float32)
    decimal = _decimal_flag(state.P)
    a_int   = state.A.astype(jnp.int32) & 0xFF
    op_int  = operand.astype(jnp.int32) & 0xFF
    c_int   = c_in.astype(jnp.int32)

    # --- Binary path (gradient-clean) ---
    sum_bin   = state.A + operand + c_in
    na_bin    = sum_bin - jnp.floor(sum_bin / 256.0) * 256.0
    carry_bin = (sum_bin.astype(jnp.int32) > 0xFF).astype(jnp.int32)

    # --- BCD path (integer) ---
    bcd_sum   = _bcd_decode(a_int) + _bcd_decode(op_int) + c_int
    na_bcd    = _bcd_encode(bcd_sum).astype(jnp.float32)
    carry_bcd = (bcd_sum > 99).astype(jnp.int32)

    new_a  = jnp.where(decimal, na_bcd, na_bin)
    carry  = jnp.where(decimal, carry_bcd, carry_bin)
    # Overflow — (A^r) & (operand^r) & 0x80 — the single formula matches
    # both the binary signed-range check and xitari's BCD V convention.
    na_int   = new_a.astype(jnp.int32) & 0xFF
    overflow = ((a_int ^ na_int) & (op_int ^ na_int)) & 0x80
    new_p    = _set_nzcv(state.P, new_a, carry, overflow)
    return _with_p(state, new_p,
        A=new_a,
        PC=state.PC + instr_len,
        cycles=state.cycles + cycles,
    ), bus


def _sbc_step(state: SoftCPUState, bus: SoftBus, operand: jnp.ndarray,
              instr_len: float, cycles: float):
    """SBC with D-flag dispatch. Binary path = ADC with one's-complement
    operand; BCD path follows xitari's M6502Hi.ins case 0xe9."""
    c_in    = _read_carry(state.P).astype(jnp.float32)
    decimal = _decimal_flag(state.P)
    a_int   = state.A.astype(jnp.int32) & 0xFF
    op_int  = operand.astype(jnp.int32) & 0xFF
    c_int   = c_in.astype(jnp.int32)

    # --- Binary path (gradient-clean) — ADC with ~operand ---
    op_inv     = 255.0 - operand
    sum_bin    = state.A + op_inv + c_in
    na_bin     = sum_bin - jnp.floor(sum_bin / 256.0) * 256.0
    carry_bin  = (sum_bin.astype(jnp.int32) > 0xFF).astype(jnp.int32)
    op_inv_int = op_inv.astype(jnp.int32) & 0xFF
    na_bin_int = na_bin.astype(jnp.int32) & 0xFF
    v_bin      = ((a_int ^ na_bin_int) & (op_inv_int ^ na_bin_int)) & 0x80

    # --- BCD path (integer) ---
    diff       = _bcd_decode(a_int) - _bcd_decode(op_int) - (1 - c_int)
    diff_wrap  = jnp.where(diff < 0, diff + 100, diff)
    na_bcd     = _bcd_encode(diff_wrap).astype(jnp.float32)
    # Carry uses the ORIGINAL bytes (xitari convention), pre-op carry inverted.
    carry_bcd  = (a_int >= (op_int + (1 - c_int))).astype(jnp.int32)
    na_bcd_int = na_bcd.astype(jnp.int32) & 0xFF
    v_bcd      = ((a_int ^ na_bcd_int) & (op_int ^ na_bcd_int)) & 0x80

    new_a    = jnp.where(decimal, na_bcd, na_bin)
    carry    = jnp.where(decimal, carry_bcd, carry_bin)
    overflow = jnp.where(decimal, v_bcd, v_bin)
    new_p    = _set_nzcv(state.P, new_a, carry, overflow)
    return _with_p(state, new_p,
        A=new_a,
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
    return _with_p(state, _set_nz(state.P, new_a),
        A=new_a,
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
    return _with_p(state, _set_nzc(state.P, diff_f, carry),
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
    return _with_p(state, _set_bit_flags(state.P, and_v, operand),
        PC=state.PC + instr_len,
        cycles=state.cycles + cycles,
    ), bus


def _branch_bit_zp(s, b):  return _bit_step(s, b, _operand_for_mode(s, b, "zp"),  2.0, 3.0)
def _branch_bit_abs(s, b): return _bit_step(s, b, _operand_for_mode(s, b, "abs"), 3.0, 4.0)


# --------------------------------------------------------------------------- #
# P7c-c — shifts and rotates (ASL / LSR / ROL / ROR).
# Each has 5 addressing modes: accumulator + zp/zp,X/abs/abs,X. The memory
# modes are RMW (read-modify-write) — read, shift, write back. The value
# computation goes via int cast (bitwise ops), so the *value* gradient
# breaks here; the address gradient (via _addr_*) still flows.
# --------------------------------------------------------------------------- #

def _asl_value(p, value):
    v_int  = value.astype(jnp.int32) & 0xFF
    new_c  = (v_int >> 7) & 1
    result = (v_int << 1) & 0xFF
    new_v  = result.astype(jnp.float32)
    return new_v, _set_nzc(p, new_v, new_c)


def _lsr_value(p, value):
    v_int  = value.astype(jnp.int32) & 0xFF
    new_c  = v_int & 1
    result = (v_int >> 1) & 0x7F
    new_v  = result.astype(jnp.float32)
    return new_v, _set_nzc(p, new_v, new_c)


def _rol_value(p, value):
    v_int  = value.astype(jnp.int32) & 0xFF
    c_in   = _read_carry(p)
    new_c  = (v_int >> 7) & 1
    result = ((v_int << 1) | c_in) & 0xFF
    new_v  = result.astype(jnp.float32)
    return new_v, _set_nzc(p, new_v, new_c)


def _ror_value(p, value):
    v_int  = value.astype(jnp.int32) & 0xFF
    c_in   = _read_carry(p)
    new_c  = v_int & 1
    result = ((v_int >> 1) | (c_in << 7)) & 0xFF
    new_v  = result.astype(jnp.float32)
    return new_v, _set_nzc(p, new_v, new_c)


def _shift_acc(state, bus, value_op):
    """Accumulator-mode shift: result goes back into A. 1 byte, 2 cycles."""
    new_a, new_p = value_op(state.P, state.A)
    return _with_p(state, new_p,
        A=new_a,
        PC=state.PC + 1.0,
        cycles=state.cycles + 2.0,
    ), bus


def _shift_memory(state, bus, addr_resolver, value_op, instr_len, cycles):
    """Memory-mode shift: read addr, transform, write back."""
    addr = addr_resolver(state, bus)
    value = _bus_read(bus, addr)
    new_value, new_p = value_op(state.P, value)
    new_bus = _bus_write(bus, addr, new_value)
    return _with_p(state, new_p,
        PC=state.PC + instr_len,
        cycles=state.cycles + cycles,
    ), new_bus


# ASL
def _branch_asl_acc(s, b):   return _shift_acc(s, b, _asl_value)
def _branch_asl_zp(s, b):    return _shift_memory(s, b, _addr_zp,    _asl_value, 2.0, 5.0)
def _branch_asl_zp_x(s, b):  return _shift_memory(s, b, _addr_zp_x,  _asl_value, 2.0, 6.0)
def _branch_asl_abs(s, b):   return _shift_memory(s, b, _addr_abs,   _asl_value, 3.0, 6.0)
def _branch_asl_abs_x(s, b): return _shift_memory(s, b, _addr_abs_x, _asl_value, 3.0, 7.0)

# LSR
def _branch_lsr_acc(s, b):   return _shift_acc(s, b, _lsr_value)
def _branch_lsr_zp(s, b):    return _shift_memory(s, b, _addr_zp,    _lsr_value, 2.0, 5.0)
def _branch_lsr_zp_x(s, b):  return _shift_memory(s, b, _addr_zp_x,  _lsr_value, 2.0, 6.0)
def _branch_lsr_abs(s, b):   return _shift_memory(s, b, _addr_abs,   _lsr_value, 3.0, 6.0)
def _branch_lsr_abs_x(s, b): return _shift_memory(s, b, _addr_abs_x, _lsr_value, 3.0, 7.0)

# ROL
def _branch_rol_acc(s, b):   return _shift_acc(s, b, _rol_value)
def _branch_rol_zp(s, b):    return _shift_memory(s, b, _addr_zp,    _rol_value, 2.0, 5.0)
def _branch_rol_zp_x(s, b):  return _shift_memory(s, b, _addr_zp_x,  _rol_value, 2.0, 6.0)
def _branch_rol_abs(s, b):   return _shift_memory(s, b, _addr_abs,   _rol_value, 3.0, 6.0)
def _branch_rol_abs_x(s, b): return _shift_memory(s, b, _addr_abs_x, _rol_value, 3.0, 7.0)

# ROR
def _branch_ror_acc(s, b):   return _shift_acc(s, b, _ror_value)
def _branch_ror_zp(s, b):    return _shift_memory(s, b, _addr_zp,    _ror_value, 2.0, 5.0)
def _branch_ror_zp_x(s, b):  return _shift_memory(s, b, _addr_zp_x,  _ror_value, 2.0, 6.0)
def _branch_ror_abs(s, b):   return _shift_memory(s, b, _addr_abs,   _ror_value, 3.0, 6.0)
def _branch_ror_abs_x(s, b): return _shift_memory(s, b, _addr_abs_x, _ror_value, 3.0, 7.0)


# --------------------------------------------------------------------------- #
# P7c-d — branches, JMP (indirect), JSR / RTS.
#
# P7c-dx: branches now feed `soft_branch` from the per-flag float mirrors
# (`P_N` / `P_Z` / `P_C` / `P_V`) that `_with_p` keeps in sync with the
# packed `P` byte. The PC blend is sigmoid-gated; a straight-through
# trick keeps the *forward* PC bit-exact (so PXC1 conformance is
# unaffected) while the gradient flows through both branch destinations.
# A reverse-mode pass over `soft_run` therefore now carries a usable
# gradient through conditional control flow — that's the whole point of
# P7c-dx.
# --------------------------------------------------------------------------- #

def _wrap_byte(v: jnp.ndarray) -> jnp.ndarray:
    """Reduce a float to its 0..255 residue without an int round-trip."""
    return v - jnp.floor(v / 256.0) * 256.0


def _push8(bus: SoftBus, sp: jnp.ndarray, value: jnp.ndarray):
    """Push a byte to the stack ($0100 + SP). Returns (new_bus, new_sp).

    The 6507 stack is 128 B shared with zero-page RAM; `_bus_write`'s
    `addr & 0x7F` collapse means a push with SP in the conventional
    upper half lands in the SOFT RAM array."""
    addr = 0x0100 + (sp.astype(jnp.int32) & 0xFF)
    new_bus = _bus_write(bus, addr, value)
    return new_bus, _wrap_byte(sp - 1.0)


def _pop8(bus: SoftBus, sp: jnp.ndarray):
    """Pull a byte from the stack. Returns (value, new_sp)."""
    new_sp = _wrap_byte(sp + 1.0)
    addr = 0x0100 + (new_sp.astype(jnp.int32) & 0xFF)
    return _bus_read(bus, addr), new_sp


def _signed_offset(offset_byte: jnp.ndarray) -> jnp.ndarray:
    """Interpret a 0..255 byte as a signed −128..+127 branch displacement."""
    o = offset_byte.astype(jnp.int32) & 0xFF
    return jnp.where(o >= 128, o - 256, o).astype(jnp.float32)


def _straight_through(soft: jnp.ndarray, hard: jnp.ndarray) -> jnp.ndarray:
    """Forward = `hard`; backward = ∂soft. Lets `_do_branch` keep the
    PXC1 bit-exact forward semantics (the new PC is whatever the hard
    branch chose) while routing the gradient through the sigmoid-
    blended `soft` value, so a `jax.grad` of a trace gets a meaningful
    contribution from both branch destinations."""
    return soft + jax.lax.stop_gradient(hard - soft)


def _do_branch(state: SoftCPUState, bus: SoftBus,
               flag_attr: str, flag_mask: int, take_when_set: bool):
    """Conditional branch.

    `flag_attr` is one of `"P_N"`, `"P_Z"`, `"P_C"`, `"P_V"` (the
    float-valued mirror `_with_p` keeps in sync with `P`) and
    `flag_mask` is the matching packed-byte bit (0x80 / 0x02 / 0x01 /
    0x40). Reads the relative displacement at ROM[PC+1]; if the
    predicate holds, PC = (PC+2) + signed_offset, else PC += 2; +1
    cycle when the branch is taken (page-cross +1 is not modelled).

    Forward semantics come from the packed `state.P` byte (so tests
    that only set `P` keep working bit-exact, and PXC1 conformance is
    unaffected). The *gradient* runs through `state.P_X`'s float
    mirror via `soft_branch`: a caller that injects a soft flag value
    into `state.P_X` for an XAI experiment gets gradient through the
    sigmoid blend of `pc_not_taken` and `pc_taken`. The two paths are
    glued together with a straight-through trick — forward = `pc_hard`,
    backward = ∂`pc_soft`.
    """
    offset       = _signed_offset(_operand_byte(bus, state.PC + 1.0))
    pc_not_taken = state.PC + 2.0
    pc_taken     = pc_not_taken + offset

    # Float flag for the gradient path — what the caller can override.
    flag_soft  = getattr(state, flag_attr)
    logit_sign = 1.0 if take_when_set else -1.0
    flag_logit = logit_sign * (2.0 * flag_soft - 1.0)

    # Soft PC — sigmoid-blended.
    pc_soft = soft_branch(flag_logit, pc_not_taken, pc_taken, alpha=10.0)

    # Hard PC: read from `state.P` directly so a test that mutates only
    # P (and not P_X) still picks the right branch on forward.
    flag_set  = (state.P.astype(jnp.int32) & flag_mask) != 0
    take_hard = flag_set if take_when_set else jnp.logical_not(flag_set)
    pc_hard   = jnp.where(take_hard, pc_taken, pc_not_taken)

    new_pc = _straight_through(pc_soft, pc_hard)

    # Cycles +1 when taken. Same straight-through dance — hard for
    # forward, soft (sigmoid weight) for gradient.
    extra_hard = jnp.where(take_hard, 1.0, 0.0)
    extra_soft = jax.nn.sigmoid(10.0 * flag_logit)       # 0..1
    extra      = _straight_through(extra_soft, extra_hard)

    return state._replace(
        PC=new_pc,
        cycles=state.cycles + 2.0 + extra,
    ), bus


# Conditional branch handlers — pass both the float-flag field name
# (gradient path) and the packed-byte mask (forward path).
def _branch_bpl(s, b): return _do_branch(s, b, "P_N", 0x80, False)   # branch if N=0
def _branch_bmi(s, b): return _do_branch(s, b, "P_N", 0x80, True)    # branch if N=1
def _branch_bvc(s, b): return _do_branch(s, b, "P_V", 0x40, False)   # branch if V=0
def _branch_bvs(s, b): return _do_branch(s, b, "P_V", 0x40, True)    # branch if V=1
def _branch_bcc(s, b): return _do_branch(s, b, "P_C", 0x01, False)   # branch if C=0
def _branch_bcs(s, b): return _do_branch(s, b, "P_C", 0x01, True)    # branch if C=1
def _branch_bne(s, b): return _do_branch(s, b, "P_Z", 0x02, False)   # branch if Z=0
def _branch_beq(s, b): return _do_branch(s, b, "P_Z", 0x02, True)    # branch if Z=1


def _branch_jmp_ind(state: SoftCPUState, bus: SoftBus):
    """JMP ($abs): PC = word at the pointer. Reproduces the NMOS
    page-wrap bug — when the pointer's low byte is $FF, the high byte
    is fetched from $xx00 of the *same* page, not the next."""
    ptr      = _addr_abs(state, bus)
    ptr_lo   = ptr
    ptr_hi   = (ptr & 0xFF00) | ((ptr + 1) & 0x00FF)        # page-wrap bug
    lo       = _bus_read(bus, ptr_lo)
    hi       = _bus_read(bus, ptr_hi)
    new_pc   = lo + hi * 256.0
    return state._replace(
        PC=new_pc,
        cycles=state.cycles + 5.0,
    ), bus


def _branch_jsr(state: SoftCPUState, bus: SoftBus):
    """JSR $abs: push (PC+2) high then low; PC = operand. 6 cycles."""
    target      = _operand_word(bus, state.PC + 1.0)
    return_addr = state.PC + 2.0                            # last byte of JSR
    ra_int      = return_addr.astype(jnp.int32) & 0xFFFF
    hi          = ((ra_int >> 8) & 0xFF).astype(jnp.float32)
    lo          = (ra_int & 0xFF).astype(jnp.float32)
    bus, sp     = _push8(bus, state.SP, hi)
    bus, sp     = _push8(bus, sp, lo)
    return state._replace(
        SP=sp,
        PC=target,
        cycles=state.cycles + 6.0,
    ), bus


def _branch_rts(state: SoftCPUState, bus: SoftBus):
    """RTS: pull low then high; PC = word + 1. 6 cycles."""
    lo, sp  = _pop8(bus, state.SP)
    hi, sp  = _pop8(bus, sp)
    ret     = lo + hi * 256.0
    return state._replace(
        SP=sp,
        PC=ret + 1.0,
        cycles=state.cycles + 6.0,
    ), bus


# --------------------------------------------------------------------------- #
# P7c-e — stack push/pull (PHA/PHP/PLA/PLP), status-flag opcodes
# (CLC/SEC/CLI/SEI/CLV/CLD/SED), INC/DEC memory, INX/INY/DEX/DEY.
# --------------------------------------------------------------------------- #

def _branch_pha(state: SoftCPUState, bus: SoftBus):
    """PHA: push A. 1 byte, 3 cycles."""
    new_bus, sp = _push8(bus, state.SP, state.A)
    return state._replace(
        SP=sp,
        PC=state.PC + 1.0,
        cycles=state.cycles + 3.0,
    ), new_bus


def _branch_php(state: SoftCPUState, bus: SoftBus):
    """PHP: push P with B (0x10) and U (0x20) forced on."""
    p_pushed   = (state.P.astype(jnp.int32) | 0x30).astype(jnp.float32)
    new_bus, sp = _push8(bus, state.SP, p_pushed)
    return state._replace(
        SP=sp,
        PC=state.PC + 1.0,
        cycles=state.cycles + 3.0,
    ), new_bus


def _branch_pla(state: SoftCPUState, bus: SoftBus):
    """PLA: pull A, set N/Z. 4 cycles."""
    value, sp = _pop8(bus, state.SP)
    return _with_p(state, _set_nz(state.P, value),
        A=value,
        SP=sp,
        PC=state.PC + 1.0,
        cycles=state.cycles + 4.0,
    ), bus


def _branch_plp(state: SoftCPUState, bus: SoftBus):
    """PLP: pull P, force B and U on (xitari PSLockupTable convention)."""
    popped, sp = _pop8(bus, state.SP)
    new_p = (popped.astype(jnp.int32) | 0x30).astype(jnp.float32)
    return _with_p(state, new_p,
        SP=sp,
        PC=state.PC + 1.0,
        cycles=state.cycles + 4.0,
    ), bus


def _set_flag(state: SoftCPUState, flag_mask: int, set_it: bool):
    """Set or clear a single P bit; advance PC by 1, +2 cycles."""
    p_int = state.P.astype(jnp.int32)
    new_p = ((p_int | flag_mask) if set_it
             else (p_int & (0xFF ^ flag_mask))).astype(jnp.float32)
    return _with_p(state, new_p,
        PC=state.PC + 1.0,
        cycles=state.cycles + 2.0,
    )


def _branch_clc(s, b): return _set_flag(s, 0x01, False), b   # clear C
def _branch_sec(s, b): return _set_flag(s, 0x01, True),  b   # set C
def _branch_cli(s, b): return _set_flag(s, 0x04, False), b   # clear I
def _branch_sei(s, b): return _set_flag(s, 0x04, True),  b   # set I
def _branch_clv(s, b): return _set_flag(s, 0x40, False), b   # clear V
def _branch_cld(s, b): return _set_flag(s, 0x08, False), b   # clear D
def _branch_sed(s, b): return _set_flag(s, 0x08, True),  b   # set D


def _inc_value(value):
    """value + 1, wrapped to a byte. Kept float-clean so the gradient
    on the increment path is preserved."""
    return _wrap_byte(value + 1.0)


def _dec_value(value):
    return _wrap_byte(value - 1.0)


def _incdec_memory(state, bus, addr_resolver, value_op, instr_len, cycles):
    """RMW INC/DEC on memory — read, +/-1, write back, set N/Z."""
    addr = addr_resolver(state, bus)
    value = _bus_read(bus, addr)
    new_value = value_op(value)
    new_bus = _bus_write(bus, addr, new_value)
    return _with_p(state, _set_nz(state.P, new_value),
        PC=state.PC + instr_len,
        cycles=state.cycles + cycles,
    ), new_bus


# INC memory
def _branch_inc_zp(s, b):    return _incdec_memory(s, b, _addr_zp,    _inc_value, 2.0, 5.0)
def _branch_inc_zp_x(s, b):  return _incdec_memory(s, b, _addr_zp_x,  _inc_value, 2.0, 6.0)
def _branch_inc_abs(s, b):   return _incdec_memory(s, b, _addr_abs,   _inc_value, 3.0, 6.0)
def _branch_inc_abs_x(s, b): return _incdec_memory(s, b, _addr_abs_x, _inc_value, 3.0, 7.0)

# DEC memory
def _branch_dec_zp(s, b):    return _incdec_memory(s, b, _addr_zp,    _dec_value, 2.0, 5.0)
def _branch_dec_zp_x(s, b):  return _incdec_memory(s, b, _addr_zp_x,  _dec_value, 2.0, 6.0)
def _branch_dec_abs(s, b):   return _incdec_memory(s, b, _addr_abs,   _dec_value, 3.0, 6.0)
def _branch_dec_abs_x(s, b): return _incdec_memory(s, b, _addr_abs_x, _dec_value, 3.0, 7.0)


def _incdec_reg(state, reg: str, value_op):
    """INX/INY/DEX/DEY — register +/-1, set N/Z. 1 byte, 2 cycles."""
    cur = getattr(state, reg)
    new_v = value_op(cur)
    fields = {
        "PC": state.PC + 1.0,
        "cycles": state.cycles + 2.0,
        reg: new_v,
    }
    return _with_p(state, _set_nz(state.P, new_v), **fields)


def _branch_inx(s, b): return _incdec_reg(s, "X", _inc_value), b
def _branch_iny(s, b): return _incdec_reg(s, "Y", _inc_value), b
def _branch_dex(s, b): return _incdec_reg(s, "X", _dec_value), b
def _branch_dey(s, b): return _incdec_reg(s, "Y", _dec_value), b


# --------------------------------------------------------------------------- #
# P7c-f — RTI. This completes the documented NMOS opcode set in SOFT mode
# (151 opcodes + the USBC $EB alias).
#
# BRK is intentionally NOT given its proper interrupt sequence — it stays
# the "end-of-trace sentinel" introduced in P7b (`_branch_brk`, halt in
# place). For fixed-length XAI traces ("run N instructions, attribute the
# output") a sentinel that re-executes harmlessly is the useful
# semantics; a proper BRK would jump to the IRQ vector and keep running.
# RTI has no such tension and gets the real sequence.
#
# What remains for SOFT mode beyond P7c is **P7f**: routing SOFT-mode
# writes through real TIA / RIOT register dispatch and cart-hotspot
# bank-switching (today `_bus_write` collapses all non-cart writes into
# the 128-byte RAM array), plus a differentiable TIA so `jax.grad` can
# flow from a framebuffer pixel back to ROM. That is a chip-level
# re-implementation, not opcode work, hence its own phase.
# --------------------------------------------------------------------------- #

def _branch_rti(state: SoftCPUState, bus: SoftBus):
    """RTI: pop P (force B+U on), then pop PC low + high. Unlike RTS
    there is no +1 — RTI returns to the exact pushed address. 6 cycles."""
    popped_p, sp = _pop8(bus, state.SP)
    lo, sp       = _pop8(bus, sp)
    hi, sp       = _pop8(bus, sp)
    new_p        = (popped_p.astype(jnp.int32) | 0x30).astype(jnp.float32)
    new_pc       = lo + hi * 256.0
    return _with_p(state, new_p,
        SP=sp,
        PC=new_pc,
        cycles=state.cycles + 6.0,
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
# P7c-c — ASL (acc + 4 mem modes)
_HANDLERS[0x0A] = _branch_asl_acc
_HANDLERS[0x06] = _branch_asl_zp
_HANDLERS[0x16] = _branch_asl_zp_x
_HANDLERS[0x0E] = _branch_asl_abs
_HANDLERS[0x1E] = _branch_asl_abs_x
# P7c-c — LSR
_HANDLERS[0x4A] = _branch_lsr_acc
_HANDLERS[0x46] = _branch_lsr_zp
_HANDLERS[0x56] = _branch_lsr_zp_x
_HANDLERS[0x4E] = _branch_lsr_abs
_HANDLERS[0x5E] = _branch_lsr_abs_x
# P7c-c — ROL
_HANDLERS[0x2A] = _branch_rol_acc
_HANDLERS[0x26] = _branch_rol_zp
_HANDLERS[0x36] = _branch_rol_zp_x
_HANDLERS[0x2E] = _branch_rol_abs
_HANDLERS[0x3E] = _branch_rol_abs_x
# P7c-c — ROR
_HANDLERS[0x6A] = _branch_ror_acc
_HANDLERS[0x66] = _branch_ror_zp
_HANDLERS[0x76] = _branch_ror_zp_x
_HANDLERS[0x6E] = _branch_ror_abs
_HANDLERS[0x7E] = _branch_ror_abs_x
# P7c-d — conditional branches
_HANDLERS[0x10] = _branch_bpl
_HANDLERS[0x30] = _branch_bmi
_HANDLERS[0x50] = _branch_bvc
_HANDLERS[0x70] = _branch_bvs
_HANDLERS[0x90] = _branch_bcc
_HANDLERS[0xB0] = _branch_bcs
_HANDLERS[0xD0] = _branch_bne
_HANDLERS[0xF0] = _branch_beq
# P7c-d — JMP indirect + JSR / RTS
_HANDLERS[0x6C] = _branch_jmp_ind
_HANDLERS[0x20] = _branch_jsr
_HANDLERS[0x60] = _branch_rts
# P7c-e — stack push/pull
_HANDLERS[0x48] = _branch_pha
_HANDLERS[0x08] = _branch_php
_HANDLERS[0x68] = _branch_pla
_HANDLERS[0x28] = _branch_plp
# P7c-e — status-flag opcodes
_HANDLERS[0x18] = _branch_clc
_HANDLERS[0x38] = _branch_sec
_HANDLERS[0x58] = _branch_cli
_HANDLERS[0x78] = _branch_sei
_HANDLERS[0xB8] = _branch_clv
_HANDLERS[0xD8] = _branch_cld
_HANDLERS[0xF8] = _branch_sed
# P7c-e — INC memory
_HANDLERS[0xE6] = _branch_inc_zp
_HANDLERS[0xF6] = _branch_inc_zp_x
_HANDLERS[0xEE] = _branch_inc_abs
_HANDLERS[0xFE] = _branch_inc_abs_x
# P7c-e — DEC memory
_HANDLERS[0xC6] = _branch_dec_zp
_HANDLERS[0xD6] = _branch_dec_zp_x
_HANDLERS[0xCE] = _branch_dec_abs
_HANDLERS[0xDE] = _branch_dec_abs_x
# P7c-e — INX/INY/DEX/DEY
_HANDLERS[0xE8] = _branch_inx
_HANDLERS[0xC8] = _branch_iny
_HANDLERS[0xCA] = _branch_dex
_HANDLERS[0x88] = _branch_dey
# P7c-f — RTI (completes the documented NMOS opcode set)
_HANDLERS[0x40] = _branch_rti

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
    # P7c-c — shifts and rotates (ASL/LSR/ROL/ROR, 5 modes each)
    0x0A, 0x06, 0x16, 0x0E, 0x1E,
    0x4A, 0x46, 0x56, 0x4E, 0x5E,
    0x2A, 0x26, 0x36, 0x2E, 0x3E,
    0x6A, 0x66, 0x76, 0x6E, 0x7E,
    # P7c-d — branches + JMP indirect + JSR / RTS
    0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0,
    0x6C, 0x20, 0x60,
    # P7c-e — stack, status flags, INC/DEC, INX/INY/DEX/DEY
    0x48, 0x08, 0x68, 0x28,
    0x18, 0x38, 0x58, 0x78, 0xB8, 0xD8, 0xF8,
    0xE6, 0xF6, 0xEE, 0xFE,
    0xC6, 0xD6, 0xCE, 0xDE,
    0xE8, 0xC8, 0xCA, 0x88,
    # P7c-f — RTI (completes the 151-opcode documented NMOS set)
    0x40,
})


# --------------------------------------------------------------------------- #
# Top-level step
# --------------------------------------------------------------------------- #

def soft_step(state: SoftCPUState, bus: SoftBus):
    """Execute one instruction.

    Memory access is differentiable (one-hot dot product on ROM and
    RAM). Dispatch is `jax.lax.switch` over the 256-way opcode table.
    Returns `(new_state, new_bus)`. After the opcode handler runs, the
    RIOT timer (P8-cx) is ticked by the instruction's cycle count.
    """
    rom_off = _cart_addr(state.PC)
    opcode_float = soft_rom_peek(bus.rom, rom_off)
    opcode_int = opcode_float.astype(jnp.int32)
    cycles_before = state.cycles
    new_state, new_bus = jax.lax.switch(opcode_int, _HANDLERS, state, bus)
    new_bus = _advance_riot_timer(new_bus, new_state.cycles - cycles_before)
    return new_state, new_bus


def soft_run(state: SoftCPUState, bus: SoftBus, n_steps: int):
    """Execute exactly `n_steps` instructions in sequence (Python loop
    — gets unrolled by jax.jit if you wrap this in jit). Use this for
    short traces (~tens to a few hundred instructions); for thousands
    of instructions on a real ROM, `soft_run_scan` is dramatically
    faster — it lowers to a single compiled loop body instead of an
    unrolled trace whose size grows with `n_steps`.
    """
    for _ in range(n_steps):
        state, bus = soft_step(state, bus)
    return state, bus


def soft_run_scan(state: SoftCPUState, bus: SoftBus, n_steps: int):
    """Execute `n_steps` instructions via `jax.lax.scan`.

    P8-c: when SOFT mode runs a real Atari ROM, the trace is many
    thousands of instructions — Pong alone runs ~6,000 instructions per
    frame. The naive Python-loop `soft_run` unrolls into a JAX trace
    whose size grows with `n_steps`, so a 5,000-step run becomes a
    5,000-deep computation graph that takes tens of seconds to trace
    and minutes to JIT-compile.

    `soft_run_scan` instead compiles the per-step body **once** and
    runs it `n_steps` times under `jax.lax.scan` — trace size is
    constant, compilation is O(1), and the per-step overhead drops by
    orders of magnitude. The function is `jax.jit`-wrapped so a second
    call at the same `n_steps` reuses the cached compilation.
    """
    def _step(carry, _):
        s, b = carry
        s, b = soft_step(s, b)
        return (s, b), None

    @jax.jit
    def _run(s, b):
        (s, b), _ = jax.lax.scan(_step, (s, b), None, length=n_steps)
        return s, b

    return _run(state, bus)
