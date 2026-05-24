# SOFT-mode `step()` — a differentiable parallel to `CPU.step`.
#
# `soft_step!(state, bus)` executes one 6502 instruction with the SOFT
# primitives from `JuTari.Diff`: memory access goes through
# `soft_rom_peek` / `soft_ram_peek` (one-hot dot product on the ROM/RAM,
# gradient-friendly); register state is Float32; opcode dispatch is a
# Julia branch table over a constant `_HANDLERS` vector. The HARD `step`
# is left untouched — SOFT mode is a parallel execution path.
#
# Opcode coverage:
#
#   P7b (the original 8-opcode core):
#     NOP            ($EA)
#     LDA #imm/zp    ($A9 / $A5)
#     LDX #imm       ($A2)
#     STA $zp        ($85)
#     STX $zp        ($86)
#     JMP $abs       ($4C)
#     BRK            ($00)
#
#   P7c-a (full load/store/transfer + N/Z flag updates):
#     LDA — all 8 addressing modes  ($A9 $A5 $B5 $AD $BD $B9 $A1 $B1)
#     LDX — all 5 addressing modes  ($A2 $A6 $B6 $AE $BE)
#     LDY — all 5 addressing modes  ($A0 $A4 $B4 $AC $BC)
#     STA — all 7 addressing modes  ($85 $95 $8D $9D $99 $81 $91)
#     STX — all 3 addressing modes  ($86 $96 $8E)
#     STY — all 3 addressing modes  ($84 $94 $8C)
#     Transfers — TAX, TAY, TXA, TYA, TSX, TXS
#
# All other opcodes fall through to `_branch_default!` which advances
# PC by 1 and bumps cycles. This is intentionally lenient — a real ROM
# that hits an unhandled opcode will produce wrong forward behaviour
# but **will not** raise during a Zygote pullback, so the function
# remains differentiable.
#
# What this implementation does NOT (yet) do:
#   - P7c-b: ADC/SBC/AND/ORA/EOR/CMP/CPX/CPY/BIT + N/Z/C/V flags
#   - P7c-c: ASL/LSR/ROL/ROR + flags
#   - P7c-d: branches via soft_branch + JMP (ind) + JSR/RTS
#   - P7c-e: stack push/pop, SEC/CLC/SEI/CLI/SED/CLD/CLV, INC/DEC/INX/INY/DEX/DEY
#   - P7c-f: BRK/RTI proper sequence + TIA/RIOT writes via SOFT bus dispatch
#     + cart bank-switching
#
# SOFT-mode read/write address dispatch is simplified compared to the
# HARD bus: cart-range reads use `soft_rom_peek`, everything else maps
# into the 128-byte RAM array via `addr & 0x7F`. TIA / RIOT register
# behaviour is therefore wrong forward but the trace stays
# gradient-clean. Real bus dispatch lands in P7c-f.


# --------------------------------------------------------------------------- #
# Differentiable memory access
# --------------------------------------------------------------------------- #

"""
    soft_rom_peek(rom, addr) -> Float32

Differentiable cart byte read — `one_hot(addr) · rom`. The one-hot is
built by a broadcast comparison (no `setindex!`) so the call is
Zygote-differentiable w.r.t. `rom` — see P7e.
"""
function soft_rom_peek(rom::AbstractVector{<:Real}, addr::Real)
    n = length(rom)
    # P8-cx: wrap the address modulo the ROM size so 2K / 4K / 8K carts
    # mirror correctly across the 4 KB $F000-$FFFF window (the IRQ
    # vector at $FFFE/$FFFF must read from the last two bytes of the
    # cart, regardless of cart size). `mod` works for any ROM size,
    # including the small fixtures the unit tests use.
    idx = mod(Int(addr), n)
    one_hot = Float32.((0:n - 1) .== idx)
    return _dot(one_hot, Float32.(rom))
end

"""
    soft_ram_peek(ram, addr) -> Float32

Differentiable RAM read — same Zygote-friendly broadcast one-hot.
"""
function soft_ram_peek(ram::AbstractVector{<:Real}, addr::Real)
    n = length(ram)
    one_hot = Float32.((0:n - 1) .== Int(addr))
    return _dot(one_hot, Float32.(ram))
end

@inline _cart_addr(pc_offset::Real) = Int(pc_offset) & 0x0FFF

"""
    _bus_read(bus, addr) -> Float32

SOFT-mode bus read with cart-vs-RAM decode (13-bit mirror; cart range
\$1000-\$1FFF goes through soft_rom_peek). P8-cx dispatches RIOT I/O
reads at \$0280-\$029F: INTIM (reg=4) → `bus.riot_intim`, INSTAT (reg=5)
→ bit 7 from the `riot_expired` latch, anything else → \$FF (all
switches released — the right default for SOFT execution without an
input layer). RAM is the fallback.
"""
function _bus_read(bus::SoftBus, addr::Real)
    a_int = Int(addr) & 0x1FFF
    if (a_int & 0x1000) != 0
        return soft_rom_peek(bus.rom, a_int & 0x0FFF)
    end
    # RIOT I/O region: A7=1, A9=1 → $0280-$029F
    if (a_int & 0x80) != 0 && (a_int & 0x200) != 0
        reg = a_int & 0x07
        reg == 4 && return bus.riot_intim
        reg == 5 && return bus.riot_expired != 0f0 ? Float32(0x80) : 0f0
        return Float32(0xFF)
    end
    return soft_ram_peek(bus.ram, a_int & 0x7F)
end

# Prescaler-shift lookup for TIM*T addresses: low 2 bits select
# 1/8/64/1024 cycles per tick (= 2^0/3/6/10).
const _TIMT_SHIFTS = Int[0, 3, 6, 10]

"""
    _bus_write!(bus, addr, value) -> nothing

SOFT-mode bus write. ROM writes are silently dropped. Writes to RIOT
TIM*T (in the RIOT I/O region AND `(addr & 0x14) == 0x14`) load the
P8-cx timer fields. Everything else mutates `bus.ram[(addr & 0x7F) + 1]`.
"""
function _bus_write!(bus::SoftBus, addr::Real, value::Real)
    a_int = Int(addr) & 0x1FFF
    if (a_int & 0x1000) != 0
        return nothing                              # ROM write — drop
    end
    # P8-cx: TIM*T detection must include the RIOT-region guard —
    # without it innocuous stack addresses like $01FD match.
    is_riot_io = (a_int & 0x80) != 0 && (a_int & 0x200) != 0
    if is_riot_io && (a_int & 0x14) == 0x14
        bus.riot_intim            = Float32(value)
        bus.riot_prescaler_shift  = Float32(_TIMT_SHIFTS[(a_int & 0x03) + 1])
        bus.riot_residual_cycles  = 0f0
        bus.riot_expired          = 0f0
        return nothing
    end
    bus.ram[(a_int & 0x7F) + 1] = Float32(value)
    return nothing
end

"""
    _advance_riot_timer!(bus, cpu_cycles) -> nothing

P8-cx: tick the RIOT timer. Pre-expiration only — INTIM decrements
once per `2^prescaler_shift` cycles and the `riot_expired` latch
fires when INTIM reaches 0. The HARD RIOT switches to one-tick-per-
cycle post-expiration; SOFT mode floors at 0 because every Atari ROM
polling INTIM cares about the *reaches-zero* moment.
"""
function _advance_riot_timer!(bus::SoftBus, cpu_cycles::Real)
    prescaler = 1 << Int(bus.riot_prescaler_shift)
    total     = Int(bus.riot_residual_cycles) + Int(cpu_cycles)
    ticks     = total ÷ prescaler
    bus.riot_residual_cycles = Float32(total - ticks * prescaler)

    intim_i = Int(bus.riot_intim)
    new_intim = max(intim_i - ticks, 0)
    expired_now = intim_i <= ticks ? 1f0 : 0f0
    bus.riot_intim   = Float32(new_intim)
    # Latch: once expired stays expired (until next TIM*T write).
    if bus.riot_expired == 0f0
        bus.riot_expired = expired_now
    end
    return nothing
end


# --------------------------------------------------------------------------- #
# N/Z flag helpers
# --------------------------------------------------------------------------- #
# Standard 6502 packing (see CPU.CPUTables): bit 7 N, bit 6 V, bit 5 U
# (always 1), bit 4 B, bit 3 D, bit 2 I, bit 1 Z, bit 0 C. In SOFT mode
# the P register is Float32; we cast to Int for bit twiddling, then
# cast back. Flag gradients are *stopped* at the cast; register-value
# gradients are preserved.

const _NZ_CLEAR_MASK   = 0xFF ⊻ (0x80 | 0x02)               # 0x7D
const _NZC_CLEAR_MASK  = 0xFF ⊻ (0x80 | 0x02 | 0x01)        # 0x7C
const _NZCV_CLEAR_MASK = 0xFF ⊻ (0x80 | 0x40 | 0x02 | 0x01) # 0x3C

"""
    _set_nz(p, value) -> Float32

Update N and Z in P from `value`. Z = 1 iff value == 0; N = bit 7.
"""
function _set_nz(p::Real, value::Real)
    v_int = Int(value) & 0xFF
    p_int = Int(p) & 0xFF
    p_int = p_int & _NZ_CLEAR_MASK
    z_bit = v_int == 0 ? 0x02 : 0
    n_bit = v_int & 0x80
    return Float32(p_int | z_bit | n_bit)
end

"""
    _set_nzc(p, value, carry) -> Float32

Set N, Z (from value) and C (from `carry` 0/1) in P.
"""
function _set_nzc(p::Real, value::Real, carry::Integer)
    v_int = Int(value) & 0xFF
    p_int = Int(p) & 0xFF
    p_int = p_int & _NZC_CLEAR_MASK
    z_bit = v_int == 0 ? 0x02 : 0
    n_bit = v_int & 0x80
    c_bit = carry & 0x01
    return Float32(p_int | z_bit | n_bit | c_bit)
end

"""
    _set_nzcv(p, value, carry, overflow) -> Float32

Set all four arithmetic flags. Used by ADC and SBC.
"""
function _set_nzcv(p::Real, value::Real, carry::Integer, overflow::Integer)
    v_int = Int(value) & 0xFF
    p_int = Int(p) & 0xFF
    p_int = p_int & _NZCV_CLEAR_MASK
    z_bit = v_int == 0 ? 0x02 : 0
    n_bit = v_int & 0x80
    c_bit = carry & 0x01
    v_bit = overflow != 0 ? 0x40 : 0
    return Float32(p_int | z_bit | n_bit | c_bit | v_bit)
end

"""
    _set_bit_flags(p, a_and_operand, operand) -> Float32

For BIT: Z from `A AND operand`; N from operand bit 7; V from operand bit 6.
C is left untouched.
"""
function _set_bit_flags(p::Real, a_and_operand::Real, operand::Real)
    p_int  = Int(p) & 0xFF
    p_int  = (p_int & _NZCV_CLEAR_MASK) | (p_int & 0x01)   # preserve C
    and_v  = Int(a_and_operand) & 0xFF
    op_int = Int(operand) & 0xFF
    z_bit  = and_v == 0 ? 0x02 : 0
    n_bit  = op_int & 0x80
    v_bit  = op_int & 0x40
    return Float32(p_int | z_bit | n_bit | v_bit)
end

@inline _read_carry(p::Real) = Int(p) & 0x01


# --------------------------------------------------------------------------- #
# Addressing-mode resolvers — return the operand byte (reads) or the
# effective address (stores). Mirrors jaxtari's _addr_zp / _addr_abs_x /
# _addr_ind_y etc.
# --------------------------------------------------------------------------- #

@inline _operand_byte(bus::SoftBus, pc_plus_one::Real) =
    soft_rom_peek(bus.rom, _cart_addr(pc_plus_one))

function _operand_word(bus::SoftBus, pc_plus_one::Real)
    lo = soft_rom_peek(bus.rom, _cart_addr(pc_plus_one))
    hi = soft_rom_peek(bus.rom, _cart_addr(pc_plus_one + 1f0))
    return lo + hi * 256f0
end

@inline _addr_zp(state::SoftCPUState, bus::SoftBus) =
    Int(_operand_byte(bus, state.PC + 1f0)) & 0xFF

@inline _addr_zp_x(state::SoftCPUState, bus::SoftBus) =
    (Int(_operand_byte(bus, state.PC + 1f0)) + Int(state.X)) & 0xFF

@inline _addr_zp_y(state::SoftCPUState, bus::SoftBus) =
    (Int(_operand_byte(bus, state.PC + 1f0)) + Int(state.Y)) & 0xFF

@inline _addr_abs(state::SoftCPUState, bus::SoftBus) =
    Int(_operand_word(bus, state.PC + 1f0)) & 0xFFFF

@inline _addr_abs_x(state::SoftCPUState, bus::SoftBus) =
    (Int(_operand_word(bus, state.PC + 1f0)) + Int(state.X)) & 0xFFFF

@inline _addr_abs_y(state::SoftCPUState, bus::SoftBus) =
    (Int(_operand_word(bus, state.PC + 1f0)) + Int(state.Y)) & 0xFFFF

function _addr_ind_x(state::SoftCPUState, bus::SoftBus)
    zp = (Int(_operand_byte(bus, state.PC + 1f0)) + Int(state.X)) & 0xFF
    lo = soft_ram_peek(bus.ram, zp & 0x7F)
    hi = soft_ram_peek(bus.ram, (zp + 1) & 0x7F)
    return (Int(lo) + Int(hi) * 256) & 0xFFFF
end

function _addr_ind_y(state::SoftCPUState, bus::SoftBus)
    zp = Int(_operand_byte(bus, state.PC + 1f0)) & 0xFF
    lo = soft_ram_peek(bus.ram, zp & 0x7F)
    hi = soft_ram_peek(bus.ram, (zp + 1) & 0x7F)
    base = Int(lo) + Int(hi) * 256
    return (base + Int(state.Y)) & 0xFFFF
end


# --------------------------------------------------------------------------- #
# Per-opcode branch handlers
# --------------------------------------------------------------------------- #
# Each handler mutates `state` and `bus` in place.

function _branch_default!(state::SoftCPUState, bus::SoftBus)
    state.PC     += 1f0
    state.cycles += 2f0
    return nothing
end

function _branch_brk!(state::SoftCPUState, bus::SoftBus)
    # P8-cx: proper interrupt sequence (was an "end-of-trace sentinel"
    # in P7b–P7c). Real Atari ROMs use BRK intentionally — Pong's
    # vertical-blank kernel sits on a BRK at $F262, branching into the
    # IRQ handler at the $FFFE/$FFFF vector. 7 cycles.
    return_addr = state.PC + 2f0
    ra_int = Int(return_addr) & 0xFFFF
    hi = Float32((ra_int >> 8) & 0xFF)
    lo = Float32(ra_int & 0xFF)
    sp = _push8!(bus, state.SP, hi)
    sp = _push8!(bus, sp, lo)
    # Push P with B and U forced on.
    p_pushed = Float32(Int(state.P) | 0x30)
    sp = _push8!(bus, sp, p_pushed)
    # Set I flag (U stays set as invariant).
    state.SP = sp
    state.P  = Float32((Int(state.P) | 0x04 | 0x20) & 0xFF)
    # Jump to IRQ vector at $FFFE / $FFFF.
    irq_lo = _bus_read(bus, 0xFFFE)
    irq_hi = _bus_read(bus, 0xFFFF)
    state.PC = irq_lo + irq_hi * 256f0
    state.cycles += 7f0
    return nothing
end

function _branch_nop!(state::SoftCPUState, bus::SoftBus)
    state.PC     += 1f0
    state.cycles += 2f0
    return nothing
end

function _branch_jmp_abs!(state::SoftCPUState, bus::SoftBus)
    state.PC      = _operand_word(bus, state.PC + 1f0)
    state.cycles += 3f0
    return nothing
end


# --- Load helpers + handlers ----------------------------------------------- #

"""
    _do_load!(state, bus, reg, value, instr_len, cycles)

Common path for LDA / LDX / LDY (and the flag-setting transfers).
`reg` is one of `:A`, `:X`, `:Y`, `:SP`.
"""
function _do_load!(state::SoftCPUState, bus::SoftBus, reg::Symbol,
                   value::Real, instr_len::Real, cycles::Real)
    state.P = _set_nz(state.P, value)
    if     reg === :A;  state.A  = Float32(value)
    elseif reg === :X;  state.X  = Float32(value)
    elseif reg === :Y;  state.Y  = Float32(value)
    elseif reg === :SP; state.SP = Float32(value)
    end
    state.PC     += Float32(instr_len)
    state.cycles += Float32(cycles)
    return nothing
end

# LDA
_branch_lda_imm!(state, bus)   = _do_load!(state, bus, :A, _operand_byte(bus, state.PC + 1f0), 2, 2)
_branch_lda_zp!(state, bus)    = _do_load!(state, bus, :A, _bus_read(bus, _addr_zp(state, bus)), 2, 3)
_branch_lda_zp_x!(state, bus)  = _do_load!(state, bus, :A, _bus_read(bus, _addr_zp_x(state, bus)), 2, 4)
_branch_lda_abs!(state, bus)   = _do_load!(state, bus, :A, _bus_read(bus, _addr_abs(state, bus)), 3, 4)
_branch_lda_abs_x!(state, bus) = _do_load!(state, bus, :A, _bus_read(bus, _addr_abs_x(state, bus)), 3, 4)
_branch_lda_abs_y!(state, bus) = _do_load!(state, bus, :A, _bus_read(bus, _addr_abs_y(state, bus)), 3, 4)
_branch_lda_ind_x!(state, bus) = _do_load!(state, bus, :A, _bus_read(bus, _addr_ind_x(state, bus)), 2, 6)
_branch_lda_ind_y!(state, bus) = _do_load!(state, bus, :A, _bus_read(bus, _addr_ind_y(state, bus)), 2, 5)

# LDX
_branch_ldx_imm!(state, bus)   = _do_load!(state, bus, :X, _operand_byte(bus, state.PC + 1f0), 2, 2)
_branch_ldx_zp!(state, bus)    = _do_load!(state, bus, :X, _bus_read(bus, _addr_zp(state, bus)), 2, 3)
_branch_ldx_zp_y!(state, bus)  = _do_load!(state, bus, :X, _bus_read(bus, _addr_zp_y(state, bus)), 2, 4)
_branch_ldx_abs!(state, bus)   = _do_load!(state, bus, :X, _bus_read(bus, _addr_abs(state, bus)), 3, 4)
_branch_ldx_abs_y!(state, bus) = _do_load!(state, bus, :X, _bus_read(bus, _addr_abs_y(state, bus)), 3, 4)

# LDY
_branch_ldy_imm!(state, bus)   = _do_load!(state, bus, :Y, _operand_byte(bus, state.PC + 1f0), 2, 2)
_branch_ldy_zp!(state, bus)    = _do_load!(state, bus, :Y, _bus_read(bus, _addr_zp(state, bus)), 2, 3)
_branch_ldy_zp_x!(state, bus)  = _do_load!(state, bus, :Y, _bus_read(bus, _addr_zp_x(state, bus)), 2, 4)
_branch_ldy_abs!(state, bus)   = _do_load!(state, bus, :Y, _bus_read(bus, _addr_abs(state, bus)), 3, 4)
_branch_ldy_abs_x!(state, bus) = _do_load!(state, bus, :Y, _bus_read(bus, _addr_abs_x(state, bus)), 3, 4)


# --- Store helpers + handlers ---------------------------------------------- #

"""
    _do_store!(state, bus, addr, value, instr_len, cycles)

Common path for STA / STX / STY — no flag changes.
"""
function _do_store!(state::SoftCPUState, bus::SoftBus, addr::Real,
                    value::Real, instr_len::Real, cycles::Real)
    _bus_write!(bus, addr, value)
    state.PC     += Float32(instr_len)
    state.cycles += Float32(cycles)
    return nothing
end

# STA
_branch_sta_zp!(state, bus)    = _do_store!(state, bus, _addr_zp(state, bus),    state.A, 2, 3)
_branch_sta_zp_x!(state, bus)  = _do_store!(state, bus, _addr_zp_x(state, bus),  state.A, 2, 4)
_branch_sta_abs!(state, bus)   = _do_store!(state, bus, _addr_abs(state, bus),   state.A, 3, 4)
_branch_sta_abs_x!(state, bus) = _do_store!(state, bus, _addr_abs_x(state, bus), state.A, 3, 5)
_branch_sta_abs_y!(state, bus) = _do_store!(state, bus, _addr_abs_y(state, bus), state.A, 3, 5)
_branch_sta_ind_x!(state, bus) = _do_store!(state, bus, _addr_ind_x(state, bus), state.A, 2, 6)
_branch_sta_ind_y!(state, bus) = _do_store!(state, bus, _addr_ind_y(state, bus), state.A, 2, 6)

# STX
_branch_stx_zp!(state, bus)   = _do_store!(state, bus, _addr_zp(state, bus),   state.X, 2, 3)
_branch_stx_zp_y!(state, bus) = _do_store!(state, bus, _addr_zp_y(state, bus), state.X, 2, 4)
_branch_stx_abs!(state, bus)  = _do_store!(state, bus, _addr_abs(state, bus),  state.X, 3, 4)

# STY
_branch_sty_zp!(state, bus)   = _do_store!(state, bus, _addr_zp(state, bus),   state.Y, 2, 3)
_branch_sty_zp_x!(state, bus) = _do_store!(state, bus, _addr_zp_x(state, bus), state.Y, 2, 4)
_branch_sty_abs!(state, bus)  = _do_store!(state, bus, _addr_abs(state, bus),  state.Y, 3, 4)


# --- Transfer handlers ----------------------------------------------------- #

_branch_tax!(state, bus) = _do_load!(state, bus, :X, state.A,  1, 2)
_branch_tay!(state, bus) = _do_load!(state, bus, :Y, state.A,  1, 2)
_branch_txa!(state, bus) = _do_load!(state, bus, :A, state.X,  1, 2)
_branch_tya!(state, bus) = _do_load!(state, bus, :A, state.Y,  1, 2)
_branch_tsx!(state, bus) = _do_load!(state, bus, :X, state.SP, 1, 2)

function _branch_txs!(state::SoftCPUState, bus::SoftBus)
    # TXS is the only transfer that does NOT touch flags.
    state.SP      = state.X
    state.PC     += 1f0
    state.cycles += 2f0
    return nothing
end


# --------------------------------------------------------------------------- #
# P7c-b — arithmetic and logic (ADC / SBC / AND / ORA / EOR / CMP / CPX /
# CPY / BIT). P7c-bx adds BCD (decimal-mode) ADC/SBC: the D flag selects
# between the binary and BCD result. BCD formulas match xitari's
# M6502Hi.ins decimal convention (the same one CPU.ALU uses).
# --------------------------------------------------------------------------- #

function _operand_for_mode(state::SoftCPUState, bus::SoftBus, mode::Symbol)
    if mode === :imm;   return _operand_byte(bus, state.PC + 1f0)
    elseif mode === :zp;    return _bus_read(bus, _addr_zp(state, bus))
    elseif mode === :zp_x;  return _bus_read(bus, _addr_zp_x(state, bus))
    elseif mode === :zp_y;  return _bus_read(bus, _addr_zp_y(state, bus))
    elseif mode === :abs;   return _bus_read(bus, _addr_abs(state, bus))
    elseif mode === :abs_x; return _bus_read(bus, _addr_abs_x(state, bus))
    elseif mode === :abs_y; return _bus_read(bus, _addr_abs_y(state, bus))
    elseif mode === :ind_x; return _bus_read(bus, _addr_ind_x(state, bus))
    elseif mode === :ind_y; return _bus_read(bus, _addr_ind_y(state, bus))
    end
    error("unknown mode $mode")
end


# ADC / SBC --------------------------------------------------------------- #

@inline _bcd_decode(b::Integer) = ((b >> 4) & 0x0F) * 10 + (b & 0x0F)

function _bcd_encode(n::Integer)
    n = mod(n, 100)
    return ((n ÷ 10) << 4) | (n % 10)
end

@inline _decimal_flag(p::Real) = (Int(p) & 0x08) != 0

function _adc_step!(state::SoftCPUState, bus::SoftBus, operand::Real,
                    instr_len::Real, cycles::Real)
    c_in   = Float32(_read_carry(state.P))
    c_int  = Int(c_in)
    a_int  = Int(state.A) & 0xFF
    op_int = Int(operand) & 0xFF
    if _decimal_flag(state.P)
        bcd_sum = _bcd_decode(a_int) + _bcd_decode(op_int) + c_int
        new_a   = Float32(_bcd_encode(bcd_sum))
        carry   = bcd_sum > 99 ? 1 : 0
    else
        sum_  = state.A + Float32(operand) + c_in
        new_a = sum_ - floor(sum_ / 256f0) * 256f0
        carry = Int(sum_) > 0xFF ? 1 : 0
    end
    na_int   = Int(new_a) & 0xFF
    overflow = ((a_int ⊻ na_int) & (op_int ⊻ na_int)) & 0x80
    state.A      = new_a
    state.P      = _set_nzcv(state.P, new_a, carry, overflow)
    state.PC    += Float32(instr_len)
    state.cycles += Float32(cycles)
    return nothing
end

function _sbc_step!(state::SoftCPUState, bus::SoftBus, operand::Real,
                    instr_len::Real, cycles::Real)
    c_in   = Float32(_read_carry(state.P))
    c_int  = Int(c_in)
    a_int  = Int(state.A) & 0xFF
    op_int = Int(operand) & 0xFF
    if _decimal_flag(state.P)
        diff = _bcd_decode(a_int) - _bcd_decode(op_int) - (1 - c_int)
        diff < 0 && (diff += 100)
        new_a    = Float32(_bcd_encode(diff))
        carry    = a_int >= (op_int + (1 - c_int)) ? 1 : 0
        na_int   = Int(new_a) & 0xFF
        overflow = ((a_int ⊻ na_int) & (op_int ⊻ na_int)) & 0x80
    else
        op_inv   = 255f0 - Float32(operand)
        sum_     = state.A + op_inv + c_in
        new_a    = sum_ - floor(sum_ / 256f0) * 256f0
        carry    = Int(sum_) > 0xFF ? 1 : 0
        op_inv_i = Int(op_inv) & 0xFF
        na_int   = Int(new_a) & 0xFF
        overflow = ((a_int ⊻ na_int) & (op_inv_i ⊻ na_int)) & 0x80
    end
    state.A      = new_a
    state.P      = _set_nzcv(state.P, new_a, carry, overflow)
    state.PC    += Float32(instr_len)
    state.cycles += Float32(cycles)
    return nothing
end

# ADC handlers
_branch_adc_imm!(s, b)    = _adc_step!(s, b, _operand_for_mode(s, b, :imm),   2, 2)
_branch_adc_zp!(s, b)     = _adc_step!(s, b, _operand_for_mode(s, b, :zp),    2, 3)
_branch_adc_zp_x!(s, b)   = _adc_step!(s, b, _operand_for_mode(s, b, :zp_x),  2, 4)
_branch_adc_abs!(s, b)    = _adc_step!(s, b, _operand_for_mode(s, b, :abs),   3, 4)
_branch_adc_abs_x!(s, b)  = _adc_step!(s, b, _operand_for_mode(s, b, :abs_x), 3, 4)
_branch_adc_abs_y!(s, b)  = _adc_step!(s, b, _operand_for_mode(s, b, :abs_y), 3, 4)
_branch_adc_ind_x!(s, b)  = _adc_step!(s, b, _operand_for_mode(s, b, :ind_x), 2, 6)
_branch_adc_ind_y!(s, b)  = _adc_step!(s, b, _operand_for_mode(s, b, :ind_y), 2, 5)

# SBC handlers + USBC ($EB alias)
_branch_sbc_imm!(s, b)    = _sbc_step!(s, b, _operand_for_mode(s, b, :imm),   2, 2)
_branch_sbc_zp!(s, b)     = _sbc_step!(s, b, _operand_for_mode(s, b, :zp),    2, 3)
_branch_sbc_zp_x!(s, b)   = _sbc_step!(s, b, _operand_for_mode(s, b, :zp_x),  2, 4)
_branch_sbc_abs!(s, b)    = _sbc_step!(s, b, _operand_for_mode(s, b, :abs),   3, 4)
_branch_sbc_abs_x!(s, b)  = _sbc_step!(s, b, _operand_for_mode(s, b, :abs_x), 3, 4)
_branch_sbc_abs_y!(s, b)  = _sbc_step!(s, b, _operand_for_mode(s, b, :abs_y), 3, 4)
_branch_sbc_ind_x!(s, b)  = _sbc_step!(s, b, _operand_for_mode(s, b, :ind_x), 2, 6)
_branch_sbc_ind_y!(s, b)  = _sbc_step!(s, b, _operand_for_mode(s, b, :ind_y), 2, 5)


# Bitwise — AND / ORA / EOR ----------------------------------------------- #

function _bitop_step!(state::SoftCPUState, bus::SoftBus, operand::Real,
                      op::Function, instr_len::Real, cycles::Real)
    a_int      = Int(state.A) & 0xFF
    op_int     = Int(operand) & 0xFF
    new_a_int  = op(a_int, op_int) & 0xFF
    state.A      = Float32(new_a_int)
    state.P      = _set_nz(state.P, new_a_int)
    state.PC    += Float32(instr_len)
    state.cycles += Float32(cycles)
    return nothing
end

_branch_and_imm!(s, b)   = _bitop_step!(s, b, _operand_for_mode(s, b, :imm),   &, 2, 2)
_branch_and_zp!(s, b)    = _bitop_step!(s, b, _operand_for_mode(s, b, :zp),    &, 2, 3)
_branch_and_zp_x!(s, b)  = _bitop_step!(s, b, _operand_for_mode(s, b, :zp_x),  &, 2, 4)
_branch_and_abs!(s, b)   = _bitop_step!(s, b, _operand_for_mode(s, b, :abs),   &, 3, 4)
_branch_and_abs_x!(s, b) = _bitop_step!(s, b, _operand_for_mode(s, b, :abs_x), &, 3, 4)
_branch_and_abs_y!(s, b) = _bitop_step!(s, b, _operand_for_mode(s, b, :abs_y), &, 3, 4)
_branch_and_ind_x!(s, b) = _bitop_step!(s, b, _operand_for_mode(s, b, :ind_x), &, 2, 6)
_branch_and_ind_y!(s, b) = _bitop_step!(s, b, _operand_for_mode(s, b, :ind_y), &, 2, 5)

_branch_ora_imm!(s, b)   = _bitop_step!(s, b, _operand_for_mode(s, b, :imm),   |, 2, 2)
_branch_ora_zp!(s, b)    = _bitop_step!(s, b, _operand_for_mode(s, b, :zp),    |, 2, 3)
_branch_ora_zp_x!(s, b)  = _bitop_step!(s, b, _operand_for_mode(s, b, :zp_x),  |, 2, 4)
_branch_ora_abs!(s, b)   = _bitop_step!(s, b, _operand_for_mode(s, b, :abs),   |, 3, 4)
_branch_ora_abs_x!(s, b) = _bitop_step!(s, b, _operand_for_mode(s, b, :abs_x), |, 3, 4)
_branch_ora_abs_y!(s, b) = _bitop_step!(s, b, _operand_for_mode(s, b, :abs_y), |, 3, 4)
_branch_ora_ind_x!(s, b) = _bitop_step!(s, b, _operand_for_mode(s, b, :ind_x), |, 2, 6)
_branch_ora_ind_y!(s, b) = _bitop_step!(s, b, _operand_for_mode(s, b, :ind_y), |, 2, 5)

_branch_eor_imm!(s, b)   = _bitop_step!(s, b, _operand_for_mode(s, b, :imm),   ⊻, 2, 2)
_branch_eor_zp!(s, b)    = _bitop_step!(s, b, _operand_for_mode(s, b, :zp),    ⊻, 2, 3)
_branch_eor_zp_x!(s, b)  = _bitop_step!(s, b, _operand_for_mode(s, b, :zp_x),  ⊻, 2, 4)
_branch_eor_abs!(s, b)   = _bitop_step!(s, b, _operand_for_mode(s, b, :abs),   ⊻, 3, 4)
_branch_eor_abs_x!(s, b) = _bitop_step!(s, b, _operand_for_mode(s, b, :abs_x), ⊻, 3, 4)
_branch_eor_abs_y!(s, b) = _bitop_step!(s, b, _operand_for_mode(s, b, :abs_y), ⊻, 3, 4)
_branch_eor_ind_x!(s, b) = _bitop_step!(s, b, _operand_for_mode(s, b, :ind_x), ⊻, 2, 6)
_branch_eor_ind_y!(s, b) = _bitop_step!(s, b, _operand_for_mode(s, b, :ind_y), ⊻, 2, 5)


# CMP / CPX / CPY --------------------------------------------------------- #

function _compare_step!(state::SoftCPUState, bus::SoftBus, reg_value::Real,
                        operand::Real, instr_len::Real, cycles::Real)
    reg_int   = Int(reg_value) & 0xFF
    op_int    = Int(operand) & 0xFF
    diff_int  = (reg_int - op_int) & 0xFF
    carry     = reg_int >= op_int ? 1 : 0
    state.P      = _set_nzc(state.P, diff_int, carry)
    state.PC    += Float32(instr_len)
    state.cycles += Float32(cycles)
    return nothing
end

_branch_cmp_imm!(s, b)   = _compare_step!(s, b, s.A, _operand_for_mode(s, b, :imm),   2, 2)
_branch_cmp_zp!(s, b)    = _compare_step!(s, b, s.A, _operand_for_mode(s, b, :zp),    2, 3)
_branch_cmp_zp_x!(s, b)  = _compare_step!(s, b, s.A, _operand_for_mode(s, b, :zp_x),  2, 4)
_branch_cmp_abs!(s, b)   = _compare_step!(s, b, s.A, _operand_for_mode(s, b, :abs),   3, 4)
_branch_cmp_abs_x!(s, b) = _compare_step!(s, b, s.A, _operand_for_mode(s, b, :abs_x), 3, 4)
_branch_cmp_abs_y!(s, b) = _compare_step!(s, b, s.A, _operand_for_mode(s, b, :abs_y), 3, 4)
_branch_cmp_ind_x!(s, b) = _compare_step!(s, b, s.A, _operand_for_mode(s, b, :ind_x), 2, 6)
_branch_cmp_ind_y!(s, b) = _compare_step!(s, b, s.A, _operand_for_mode(s, b, :ind_y), 2, 5)

_branch_cpx_imm!(s, b)   = _compare_step!(s, b, s.X, _operand_for_mode(s, b, :imm),  2, 2)
_branch_cpx_zp!(s, b)    = _compare_step!(s, b, s.X, _operand_for_mode(s, b, :zp),   2, 3)
_branch_cpx_abs!(s, b)   = _compare_step!(s, b, s.X, _operand_for_mode(s, b, :abs),  3, 4)

_branch_cpy_imm!(s, b)   = _compare_step!(s, b, s.Y, _operand_for_mode(s, b, :imm),  2, 2)
_branch_cpy_zp!(s, b)    = _compare_step!(s, b, s.Y, _operand_for_mode(s, b, :zp),   2, 3)
_branch_cpy_abs!(s, b)   = _compare_step!(s, b, s.Y, _operand_for_mode(s, b, :abs),  3, 4)


# BIT --------------------------------------------------------------------- #

function _bit_step!(state::SoftCPUState, bus::SoftBus, operand::Real,
                    instr_len::Real, cycles::Real)
    a_int   = Int(state.A) & 0xFF
    op_int  = Int(operand) & 0xFF
    and_v   = a_int & op_int
    state.P      = _set_bit_flags(state.P, and_v, operand)
    state.PC    += Float32(instr_len)
    state.cycles += Float32(cycles)
    return nothing
end

_branch_bit_zp!(s, b)  = _bit_step!(s, b, _operand_for_mode(s, b, :zp),  2, 3)
_branch_bit_abs!(s, b) = _bit_step!(s, b, _operand_for_mode(s, b, :abs), 3, 4)


# --------------------------------------------------------------------------- #
# P7c-c — shifts and rotates (ASL / LSR / ROL / ROR). Each has 5 modes:
# accumulator + zp / zp,X / abs / abs,X. Memory modes are RMW.
# --------------------------------------------------------------------------- #

function _asl_value(p::Real, value::Real)
    v_int  = Int(value) & 0xFF
    new_c  = (v_int >> 7) & 1
    result = (v_int << 1) & 0xFF
    return Float32(result), _set_nzc(p, result, new_c)
end

function _lsr_value(p::Real, value::Real)
    v_int  = Int(value) & 0xFF
    new_c  = v_int & 1
    result = (v_int >> 1) & 0x7F
    return Float32(result), _set_nzc(p, result, new_c)
end

function _rol_value(p::Real, value::Real)
    v_int  = Int(value) & 0xFF
    c_in   = _read_carry(p)
    new_c  = (v_int >> 7) & 1
    result = ((v_int << 1) | c_in) & 0xFF
    return Float32(result), _set_nzc(p, result, new_c)
end

function _ror_value(p::Real, value::Real)
    v_int  = Int(value) & 0xFF
    c_in   = _read_carry(p)
    new_c  = v_int & 1
    result = ((v_int >> 1) | (c_in << 7)) & 0xFF
    return Float32(result), _set_nzc(p, result, new_c)
end

function _shift_acc!(state::SoftCPUState, bus::SoftBus, value_op::Function)
    new_a, new_p = value_op(state.P, state.A)
    state.A      = new_a
    state.P      = new_p
    state.PC    += 1f0
    state.cycles += 2f0
    return nothing
end

function _shift_memory!(state::SoftCPUState, bus::SoftBus,
                        addr_resolver::Function, value_op::Function,
                        instr_len::Real, cycles::Real)
    addr = addr_resolver(state, bus)
    value = _bus_read(bus, addr)
    new_value, new_p = value_op(state.P, value)
    _bus_write!(bus, addr, new_value)
    state.P      = new_p
    state.PC    += Float32(instr_len)
    state.cycles += Float32(cycles)
    return nothing
end

# ASL
_branch_asl_acc!(s, b)   = _shift_acc!(s, b, _asl_value)
_branch_asl_zp!(s, b)    = _shift_memory!(s, b, _addr_zp,    _asl_value, 2, 5)
_branch_asl_zp_x!(s, b)  = _shift_memory!(s, b, _addr_zp_x,  _asl_value, 2, 6)
_branch_asl_abs!(s, b)   = _shift_memory!(s, b, _addr_abs,   _asl_value, 3, 6)
_branch_asl_abs_x!(s, b) = _shift_memory!(s, b, _addr_abs_x, _asl_value, 3, 7)

# LSR
_branch_lsr_acc!(s, b)   = _shift_acc!(s, b, _lsr_value)
_branch_lsr_zp!(s, b)    = _shift_memory!(s, b, _addr_zp,    _lsr_value, 2, 5)
_branch_lsr_zp_x!(s, b)  = _shift_memory!(s, b, _addr_zp_x,  _lsr_value, 2, 6)
_branch_lsr_abs!(s, b)   = _shift_memory!(s, b, _addr_abs,   _lsr_value, 3, 6)
_branch_lsr_abs_x!(s, b) = _shift_memory!(s, b, _addr_abs_x, _lsr_value, 3, 7)

# ROL
_branch_rol_acc!(s, b)   = _shift_acc!(s, b, _rol_value)
_branch_rol_zp!(s, b)    = _shift_memory!(s, b, _addr_zp,    _rol_value, 2, 5)
_branch_rol_zp_x!(s, b)  = _shift_memory!(s, b, _addr_zp_x,  _rol_value, 2, 6)
_branch_rol_abs!(s, b)   = _shift_memory!(s, b, _addr_abs,   _rol_value, 3, 6)
_branch_rol_abs_x!(s, b) = _shift_memory!(s, b, _addr_abs_x, _rol_value, 3, 7)

# ROR
_branch_ror_acc!(s, b)   = _shift_acc!(s, b, _ror_value)
_branch_ror_zp!(s, b)    = _shift_memory!(s, b, _addr_zp,    _ror_value, 2, 5)
_branch_ror_zp_x!(s, b)  = _shift_memory!(s, b, _addr_zp_x,  _ror_value, 2, 6)
_branch_ror_abs!(s, b)   = _shift_memory!(s, b, _addr_abs,   _ror_value, 3, 6)
_branch_ror_abs_x!(s, b) = _shift_memory!(s, b, _addr_abs_x, _ror_value, 3, 7)


# --------------------------------------------------------------------------- #
# P7c-d — branches, JMP (indirect), JSR / RTS.
#
# Conditional branches use a HARD predicate; the forward PC is exact.
# Gradient through the branch predicate is broken at the int-cast of P;
# a float-valued flag representation (so soft_branch can be wired into
# the default handlers) is deferred as P7c-dx.
# --------------------------------------------------------------------------- #

@inline _wrap_byte(v::Real) = Float32(v) - floor(Float32(v) / 256f0) * 256f0

function _push8!(bus::SoftBus, sp::Real, value::Real)
    addr = 0x0100 + (Int(sp) & 0xFF)
    _bus_write!(bus, addr, value)
    return _wrap_byte(Float32(sp) - 1f0)
end

function _pop8(bus::SoftBus, sp::Real)
    new_sp = _wrap_byte(Float32(sp) + 1f0)
    addr = 0x0100 + (Int(new_sp) & 0xFF)
    return _bus_read(bus, addr), new_sp
end

function _signed_offset(offset_byte::Real)
    o = Int(offset_byte) & 0xFF
    return Float32(o >= 128 ? o - 256 : o)
end

function _do_branch!(state::SoftCPUState, bus::SoftBus,
                     flag_mask::Integer, take_when_set::Bool)
    offset       = _signed_offset(_operand_byte(bus, state.PC + 1f0))
    pc_not_taken = state.PC + 2f0
    pc_taken     = pc_not_taken + offset
    flag_set     = (Int(state.P) & flag_mask) != 0
    take         = take_when_set ? flag_set : !flag_set
    state.PC     = take ? pc_taken : pc_not_taken
    state.cycles += take ? 3f0 : 2f0
    return nothing
end

_branch_bpl!(s, b) = _do_branch!(s, b, 0x80, false)
_branch_bmi!(s, b) = _do_branch!(s, b, 0x80, true)
_branch_bvc!(s, b) = _do_branch!(s, b, 0x40, false)
_branch_bvs!(s, b) = _do_branch!(s, b, 0x40, true)
_branch_bcc!(s, b) = _do_branch!(s, b, 0x01, false)
_branch_bcs!(s, b) = _do_branch!(s, b, 0x01, true)
_branch_bne!(s, b) = _do_branch!(s, b, 0x02, false)
_branch_beq!(s, b) = _do_branch!(s, b, 0x02, true)

function _branch_jmp_ind!(state::SoftCPUState, bus::SoftBus)
    ptr    = _addr_abs(state, bus)
    ptr_lo = ptr
    ptr_hi = (ptr & 0xFF00) | ((ptr + 1) & 0x00FF)   # NMOS page-wrap bug
    lo     = _bus_read(bus, ptr_lo)
    hi     = _bus_read(bus, ptr_hi)
    state.PC      = lo + hi * 256f0
    state.cycles += 5f0
    return nothing
end

function _branch_jsr!(state::SoftCPUState, bus::SoftBus)
    target      = _operand_word(bus, state.PC + 1f0)
    ra_int      = Int(state.PC + 2f0) & 0xFFFF        # last byte of JSR
    hi          = Float32((ra_int >> 8) & 0xFF)
    lo          = Float32(ra_int & 0xFF)
    sp          = _push8!(bus, state.SP, hi)
    sp          = _push8!(bus, sp, lo)
    state.SP      = sp
    state.PC      = target
    state.cycles += 6f0
    return nothing
end

function _branch_rts!(state::SoftCPUState, bus::SoftBus)
    lo, sp = _pop8(bus, state.SP)
    hi, sp = _pop8(bus, sp)
    ret    = lo + hi * 256f0
    state.SP      = sp
    state.PC      = ret + 1f0
    state.cycles += 6f0
    return nothing
end


# --------------------------------------------------------------------------- #
# P7c-e — stack push/pull, status-flag opcodes, INC/DEC, INX/INY/DEX/DEY.
# --------------------------------------------------------------------------- #

function _branch_pha!(state::SoftCPUState, bus::SoftBus)
    state.SP      = _push8!(bus, state.SP, state.A)
    state.PC     += 1f0
    state.cycles += 3f0
    return nothing
end

function _branch_php!(state::SoftCPUState, bus::SoftBus)
    p_pushed = Float32(Int(state.P) | 0x30)        # force B + U
    state.SP      = _push8!(bus, state.SP, p_pushed)
    state.PC     += 1f0
    state.cycles += 3f0
    return nothing
end

function _branch_pla!(state::SoftCPUState, bus::SoftBus)
    value, sp = _pop8(bus, state.SP)
    state.A       = value
    state.SP      = sp
    state.P       = _set_nz(state.P, value)
    state.PC     += 1f0
    state.cycles += 4f0
    return nothing
end

function _branch_plp!(state::SoftCPUState, bus::SoftBus)
    popped, sp = _pop8(bus, state.SP)
    state.SP      = sp
    state.P       = Float32(Int(popped) | 0x30)    # force B + U
    state.PC     += 1f0
    state.cycles += 4f0
    return nothing
end

function _set_flag!(state::SoftCPUState, flag_mask::Integer, set_it::Bool)
    p_int = Int(state.P)
    state.P       = Float32(set_it ? (p_int | flag_mask) :
                                     (p_int & (0xFF ⊻ flag_mask)))
    state.PC     += 1f0
    state.cycles += 2f0
    return nothing
end

_branch_clc!(s, b) = _set_flag!(s, 0x01, false)
_branch_sec!(s, b) = _set_flag!(s, 0x01, true)
_branch_cli!(s, b) = _set_flag!(s, 0x04, false)
_branch_sei!(s, b) = _set_flag!(s, 0x04, true)
_branch_clv!(s, b) = _set_flag!(s, 0x40, false)
_branch_cld!(s, b) = _set_flag!(s, 0x08, false)
_branch_sed!(s, b) = _set_flag!(s, 0x08, true)

@inline _inc_value(value::Real) = _wrap_byte(Float32(value) + 1f0)
@inline _dec_value(value::Real) = _wrap_byte(Float32(value) - 1f0)

function _incdec_memory!(state::SoftCPUState, bus::SoftBus,
                         addr_resolver::Function, value_op::Function,
                         instr_len::Real, cycles::Real)
    addr = addr_resolver(state, bus)
    value = _bus_read(bus, addr)
    new_value = value_op(value)
    _bus_write!(bus, addr, new_value)
    state.P       = _set_nz(state.P, new_value)
    state.PC     += Float32(instr_len)
    state.cycles += Float32(cycles)
    return nothing
end

_branch_inc_zp!(s, b)    = _incdec_memory!(s, b, _addr_zp,    _inc_value, 2, 5)
_branch_inc_zp_x!(s, b)  = _incdec_memory!(s, b, _addr_zp_x,  _inc_value, 2, 6)
_branch_inc_abs!(s, b)   = _incdec_memory!(s, b, _addr_abs,   _inc_value, 3, 6)
_branch_inc_abs_x!(s, b) = _incdec_memory!(s, b, _addr_abs_x, _inc_value, 3, 7)

_branch_dec_zp!(s, b)    = _incdec_memory!(s, b, _addr_zp,    _dec_value, 2, 5)
_branch_dec_zp_x!(s, b)  = _incdec_memory!(s, b, _addr_zp_x,  _dec_value, 2, 6)
_branch_dec_abs!(s, b)   = _incdec_memory!(s, b, _addr_abs,   _dec_value, 3, 6)
_branch_dec_abs_x!(s, b) = _incdec_memory!(s, b, _addr_abs_x, _dec_value, 3, 7)

function _incdec_reg!(state::SoftCPUState, reg::Symbol, value_op::Function)
    cur = reg === :X ? state.X : state.Y
    new_v = value_op(cur)
    if reg === :X; state.X = new_v else state.Y = new_v end
    state.P       = _set_nz(state.P, new_v)
    state.PC     += 1f0
    state.cycles += 2f0
    return nothing
end

_branch_inx!(s, b) = _incdec_reg!(s, :X, _inc_value)
_branch_iny!(s, b) = _incdec_reg!(s, :Y, _inc_value)
_branch_dex!(s, b) = _incdec_reg!(s, :X, _dec_value)
_branch_dey!(s, b) = _incdec_reg!(s, :Y, _dec_value)


# --------------------------------------------------------------------------- #
# P7c-f — RTI. Completes the documented NMOS opcode set (151 opcodes +
# the USBC $EB alias). BRK intentionally stays the end-of-trace sentinel
# from P7b — see the soft_step docstring in the jaxtari twin for the
# rationale. Routing SOFT writes through real TIA / RIOT / cart dispatch
# (and a differentiable TIA) is the separate P7f phase.
# --------------------------------------------------------------------------- #

function _branch_rti!(state::SoftCPUState, bus::SoftBus)
    popped_p, sp = _pop8(bus, state.SP)
    lo, sp       = _pop8(bus, sp)
    hi, sp       = _pop8(bus, sp)
    state.SP      = sp
    state.P       = Float32(Int(popped_p) | 0x30)   # force B + U
    state.PC      = lo + hi * 256f0                 # no +1, unlike RTS
    state.cycles += 6f0
    return nothing
end


# --------------------------------------------------------------------------- #
# Dispatch table
# --------------------------------------------------------------------------- #

const _HANDLERS = let
    h = Function[_branch_default! for _ in 1:256]
    # P7b core
    h[0x00 + 1] = _branch_brk!
    h[0xEA + 1] = _branch_nop!
    h[0x4C + 1] = _branch_jmp_abs!
    # P7c-a — LDA
    h[0xA9 + 1] = _branch_lda_imm!
    h[0xA5 + 1] = _branch_lda_zp!
    h[0xB5 + 1] = _branch_lda_zp_x!
    h[0xAD + 1] = _branch_lda_abs!
    h[0xBD + 1] = _branch_lda_abs_x!
    h[0xB9 + 1] = _branch_lda_abs_y!
    h[0xA1 + 1] = _branch_lda_ind_x!
    h[0xB1 + 1] = _branch_lda_ind_y!
    # P7c-a — LDX
    h[0xA2 + 1] = _branch_ldx_imm!
    h[0xA6 + 1] = _branch_ldx_zp!
    h[0xB6 + 1] = _branch_ldx_zp_y!
    h[0xAE + 1] = _branch_ldx_abs!
    h[0xBE + 1] = _branch_ldx_abs_y!
    # P7c-a — LDY
    h[0xA0 + 1] = _branch_ldy_imm!
    h[0xA4 + 1] = _branch_ldy_zp!
    h[0xB4 + 1] = _branch_ldy_zp_x!
    h[0xAC + 1] = _branch_ldy_abs!
    h[0xBC + 1] = _branch_ldy_abs_x!
    # P7c-a — STA
    h[0x85 + 1] = _branch_sta_zp!
    h[0x95 + 1] = _branch_sta_zp_x!
    h[0x8D + 1] = _branch_sta_abs!
    h[0x9D + 1] = _branch_sta_abs_x!
    h[0x99 + 1] = _branch_sta_abs_y!
    h[0x81 + 1] = _branch_sta_ind_x!
    h[0x91 + 1] = _branch_sta_ind_y!
    # P7c-a — STX
    h[0x86 + 1] = _branch_stx_zp!
    h[0x96 + 1] = _branch_stx_zp_y!
    h[0x8E + 1] = _branch_stx_abs!
    # P7c-a — STY
    h[0x84 + 1] = _branch_sty_zp!
    h[0x94 + 1] = _branch_sty_zp_x!
    h[0x8C + 1] = _branch_sty_abs!
    # P7c-a — Transfers
    h[0xAA + 1] = _branch_tax!
    h[0xA8 + 1] = _branch_tay!
    h[0x8A + 1] = _branch_txa!
    h[0x98 + 1] = _branch_tya!
    h[0xBA + 1] = _branch_tsx!
    h[0x9A + 1] = _branch_txs!
    # P7c-b — ADC
    h[0x69 + 1] = _branch_adc_imm!;   h[0x65 + 1] = _branch_adc_zp!
    h[0x75 + 1] = _branch_adc_zp_x!;  h[0x6D + 1] = _branch_adc_abs!
    h[0x7D + 1] = _branch_adc_abs_x!; h[0x79 + 1] = _branch_adc_abs_y!
    h[0x61 + 1] = _branch_adc_ind_x!; h[0x71 + 1] = _branch_adc_ind_y!
    # P7c-b — SBC (+ USBC $EB)
    h[0xE9 + 1] = _branch_sbc_imm!;   h[0xE5 + 1] = _branch_sbc_zp!
    h[0xF5 + 1] = _branch_sbc_zp_x!;  h[0xED + 1] = _branch_sbc_abs!
    h[0xFD + 1] = _branch_sbc_abs_x!; h[0xF9 + 1] = _branch_sbc_abs_y!
    h[0xE1 + 1] = _branch_sbc_ind_x!; h[0xF1 + 1] = _branch_sbc_ind_y!
    h[0xEB + 1] = _branch_sbc_imm!     # USBC alias
    # P7c-b — AND
    h[0x29 + 1] = _branch_and_imm!;   h[0x25 + 1] = _branch_and_zp!
    h[0x35 + 1] = _branch_and_zp_x!;  h[0x2D + 1] = _branch_and_abs!
    h[0x3D + 1] = _branch_and_abs_x!; h[0x39 + 1] = _branch_and_abs_y!
    h[0x21 + 1] = _branch_and_ind_x!; h[0x31 + 1] = _branch_and_ind_y!
    # P7c-b — ORA
    h[0x09 + 1] = _branch_ora_imm!;   h[0x05 + 1] = _branch_ora_zp!
    h[0x15 + 1] = _branch_ora_zp_x!;  h[0x0D + 1] = _branch_ora_abs!
    h[0x1D + 1] = _branch_ora_abs_x!; h[0x19 + 1] = _branch_ora_abs_y!
    h[0x01 + 1] = _branch_ora_ind_x!; h[0x11 + 1] = _branch_ora_ind_y!
    # P7c-b — EOR
    h[0x49 + 1] = _branch_eor_imm!;   h[0x45 + 1] = _branch_eor_zp!
    h[0x55 + 1] = _branch_eor_zp_x!;  h[0x4D + 1] = _branch_eor_abs!
    h[0x5D + 1] = _branch_eor_abs_x!; h[0x59 + 1] = _branch_eor_abs_y!
    h[0x41 + 1] = _branch_eor_ind_x!; h[0x51 + 1] = _branch_eor_ind_y!
    # P7c-b — CMP
    h[0xC9 + 1] = _branch_cmp_imm!;   h[0xC5 + 1] = _branch_cmp_zp!
    h[0xD5 + 1] = _branch_cmp_zp_x!;  h[0xCD + 1] = _branch_cmp_abs!
    h[0xDD + 1] = _branch_cmp_abs_x!; h[0xD9 + 1] = _branch_cmp_abs_y!
    h[0xC1 + 1] = _branch_cmp_ind_x!; h[0xD1 + 1] = _branch_cmp_ind_y!
    # P7c-b — CPX / CPY / BIT
    h[0xE0 + 1] = _branch_cpx_imm!; h[0xE4 + 1] = _branch_cpx_zp!; h[0xEC + 1] = _branch_cpx_abs!
    h[0xC0 + 1] = _branch_cpy_imm!; h[0xC4 + 1] = _branch_cpy_zp!; h[0xCC + 1] = _branch_cpy_abs!
    h[0x24 + 1] = _branch_bit_zp!;  h[0x2C + 1] = _branch_bit_abs!
    # P7c-c — shifts and rotates
    h[0x0A + 1] = _branch_asl_acc!;   h[0x06 + 1] = _branch_asl_zp!
    h[0x16 + 1] = _branch_asl_zp_x!;  h[0x0E + 1] = _branch_asl_abs!
    h[0x1E + 1] = _branch_asl_abs_x!
    h[0x4A + 1] = _branch_lsr_acc!;   h[0x46 + 1] = _branch_lsr_zp!
    h[0x56 + 1] = _branch_lsr_zp_x!;  h[0x4E + 1] = _branch_lsr_abs!
    h[0x5E + 1] = _branch_lsr_abs_x!
    h[0x2A + 1] = _branch_rol_acc!;   h[0x26 + 1] = _branch_rol_zp!
    h[0x36 + 1] = _branch_rol_zp_x!;  h[0x2E + 1] = _branch_rol_abs!
    h[0x3E + 1] = _branch_rol_abs_x!
    h[0x6A + 1] = _branch_ror_acc!;   h[0x66 + 1] = _branch_ror_zp!
    h[0x76 + 1] = _branch_ror_zp_x!;  h[0x6E + 1] = _branch_ror_abs!
    h[0x7E + 1] = _branch_ror_abs_x!
    # P7c-d — branches
    h[0x10 + 1] = _branch_bpl!; h[0x30 + 1] = _branch_bmi!
    h[0x50 + 1] = _branch_bvc!; h[0x70 + 1] = _branch_bvs!
    h[0x90 + 1] = _branch_bcc!; h[0xB0 + 1] = _branch_bcs!
    h[0xD0 + 1] = _branch_bne!; h[0xF0 + 1] = _branch_beq!
    # P7c-d — JMP indirect + JSR / RTS
    h[0x6C + 1] = _branch_jmp_ind!
    h[0x20 + 1] = _branch_jsr!
    h[0x60 + 1] = _branch_rts!
    # P7c-e — stack push/pull
    h[0x48 + 1] = _branch_pha!; h[0x08 + 1] = _branch_php!
    h[0x68 + 1] = _branch_pla!; h[0x28 + 1] = _branch_plp!
    # P7c-e — status-flag opcodes
    h[0x18 + 1] = _branch_clc!; h[0x38 + 1] = _branch_sec!
    h[0x58 + 1] = _branch_cli!; h[0x78 + 1] = _branch_sei!
    h[0xB8 + 1] = _branch_clv!; h[0xD8 + 1] = _branch_cld!
    h[0xF8 + 1] = _branch_sed!
    # P7c-e — INC memory
    h[0xE6 + 1] = _branch_inc_zp!;  h[0xF6 + 1] = _branch_inc_zp_x!
    h[0xEE + 1] = _branch_inc_abs!; h[0xFE + 1] = _branch_inc_abs_x!
    # P7c-e — DEC memory
    h[0xC6 + 1] = _branch_dec_zp!;  h[0xD6 + 1] = _branch_dec_zp_x!
    h[0xCE + 1] = _branch_dec_abs!; h[0xDE + 1] = _branch_dec_abs_x!
    # P7c-e — INX/INY/DEX/DEY
    h[0xE8 + 1] = _branch_inx!; h[0xC8 + 1] = _branch_iny!
    h[0xCA + 1] = _branch_dex!; h[0x88 + 1] = _branch_dey!
    # P7c-f — RTI (completes the documented NMOS opcode set)
    h[0x40 + 1] = _branch_rti!
    h
end

# Opcodes handled with full behaviour (rather than default).
const SOFT_SUPPORTED_OPCODES = Set{UInt8}([
    # P7b core
    0x00, 0xEA, 0x4C,
    # P7c-a — load/store/transfer
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
    # P7c-c — shifts and rotates
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
])


# --------------------------------------------------------------------------- #
# Top-level step
# --------------------------------------------------------------------------- #

"""
    soft_step!(state::SoftCPUState, bus::SoftBus) -> nothing

Execute one instruction. Memory access is differentiable (one-hot dot
product on ROM and RAM). Dispatch is a Julia table lookup over the
256-way opcode handler list. Mutates `state` and `bus.ram` in place.
"""
function soft_step!(state::SoftCPUState, bus::SoftBus)
    rom_off = _cart_addr(state.PC)
    opcode_float = soft_rom_peek(bus.rom, rom_off)
    opcode_int = Int(opcode_float) & 0xFF
    cycles_before = state.cycles
    _HANDLERS[opcode_int + 1](state, bus)
    # P8-cx: tick the RIOT timer by this instruction's cycle count.
    _advance_riot_timer!(bus, state.cycles - cycles_before)
    return nothing
end

"""
    soft_run!(state::SoftCPUState, bus::SoftBus, n_steps::Integer) -> nothing

Execute exactly `n_steps` instructions in sequence.
"""
function soft_run!(state::SoftCPUState, bus::SoftBus, n_steps::Integer)
    for _ in 1:n_steps
        soft_step!(state, bus)
    end
    return nothing
end


# --------------------------------------------------------------------------- #
# P7e-x — functional soft_step (Zygote-differentiable)
# --------------------------------------------------------------------------- #
#
# The mutating `soft_step!` / `soft_run!` are not Zygote-differentiable
# (Zygote rejects `setfield!` / `setindex!`). For end-to-end gradient
# through a Julia trace we need a pure-functional `soft_step(state, bus)
# -> (new_state, new_bus)` that builds *new* struct instances and new
# RAM vectors instead of mutating in place.
#
# This block adds that path. The handlers below are 1:1 functional
# mirrors of the mutating ones — same semantics, just immutable in
# style. Opcodes not yet rewritten fall through to `_func_default`, a
# lenient continuation (PC += 1, cycles += 2) so the trace stays
# differentiable even past an unhandled opcode.
#
# Initial coverage is the "P7b core" + the most-common P7c-a opcodes —
# enough to exercise loads, stores, transfers, the simplest flag ops
# and an unconditional JMP, and to drive a meaningful XAI trace through
# `soft_run`. Extending coverage to the full 151-opcode set is a
# documented follow-up in STATUS.md (the mutating `soft_step!` retains
# full coverage for non-AD use).
#
# The single non-obvious primitive is `_set_ram` — Zygote can't trace
# `bus.ram[i] = v`, so we build a new vector via broadcast:
#   new_ram = (1 .- mask) .* bus.ram .+ mask .* v
# which IS traceable; the gradient flows back through both the
# selection mask and the value.

"""
    update_state(state; A=…, X=…, Y=…, SP=…, PC=…, P=…, cycles=…,
                        P_N=…, P_Z=…, P_C=…, P_V=…) -> SoftCPUState

Functional "with-modifications" constructor for `SoftCPUState`. Any
omitted field defaults to the original value. Zygote can differentiate
through this because it's just a struct constructor call.

P7c-dx: also threads through the float-flag mirrors so callers that
need to bump P + the mirrors atomically can do so in one call (see
`_with_p` for the standard pattern).
"""
function update_state(s::SoftCPUState;
                      A::Real      = s.A,
                      X::Real      = s.X,
                      Y::Real      = s.Y,
                      SP::Real     = s.SP,
                      PC::Real     = s.PC,
                      P::Real      = s.P,
                      cycles::Real = s.cycles,
                      P_N::Real    = s.P_N,
                      P_Z::Real    = s.P_Z,
                      P_C::Real    = s.P_C,
                      P_V::Real    = s.P_V)
    return SoftCPUState(Float32(A), Float32(X), Float32(Y), Float32(SP),
                         Float32(PC), Float32(P), Float32(cycles),
                         Float32(P_N), Float32(P_Z), Float32(P_C), Float32(P_V))
end

"""
    _float_flags_from_p(p) -> (n_f, z_f, c_f, v_f)

Split a packed status byte into its (N, Z, C, V) float mirrors. Each
return value is a 0.0 / 1.0 Float32 — integer-extracted, so the
gradient is zero w.r.t. the underlying packed P (as expected for a
structural flag bit), but the float itself slots cleanly into
`soft_branch`.
"""
function _float_flags_from_p(p::Real)
    pi = Int(p) & 0xFF
    return (Float32((pi >> 7) & 1),   # N (bit 7)
            Float32((pi >> 1) & 1),   # Z (bit 1)
            Float32(pi & 0x01),       # C (bit 0)
            Float32((pi >> 6) & 1))   # V (bit 6)
end

"""
    _with_p(state, new_p; other_kwargs...) -> SoftCPUState

Functional update of `state` with a new packed `P` byte AND the four
float-flag mirrors re-derived from it. Use this every time a
flag-touching functional handler bumps `P` — the packed byte stays
the source of truth; the floats are derived. Pass any other
`update_state` kwargs through.
"""
function _with_p(state::SoftCPUState, new_p::Real; kwargs...)
    n_f, z_f, c_f, v_f = _float_flags_from_p(new_p)
    return update_state(state; P=Float32(new_p),
                        P_N=n_f, P_Z=z_f, P_C=c_f, P_V=v_f,
                        kwargs...)
end

"""
    update_bus(bus; ram=…, rom=…, riot_intim=…, …) -> SoftBus

Functional with-modifications constructor for `SoftBus`. ROM defaults
to the input (the differentiability target — usually unchanged within
a step).
"""
function update_bus(b::SoftBus;
                    ram::AbstractVector{<:Real} = b.ram,
                    rom::AbstractVector{<:Real} = b.rom,
                    riot_intim::Real           = b.riot_intim,
                    riot_prescaler_shift::Real = b.riot_prescaler_shift,
                    riot_residual_cycles::Real = b.riot_residual_cycles,
                    riot_expired::Real         = b.riot_expired)
    return SoftBus(Vector{Float32}(ram), Vector{Float32}(rom),
                    Float32(riot_intim), Float32(riot_prescaler_shift),
                    Float32(riot_residual_cycles), Float32(riot_expired))
end

"""
    _set_ram(ram, idx, value) -> Vector{Float32}

Functional single-cell RAM write — broadcast a one-hot select-and-set,
returning a fresh vector. The construction is Zygote-friendly: the
gradient flows back to `value` (one-hot weighted) and to the original
`ram` (one-cold weighted).
"""
function _set_ram(ram::AbstractVector{<:Real}, idx::Integer, value::Real)
    n    = length(ram)
    mask = Float32.((0:n - 1) .== idx)
    return (1f0 .- mask) .* Float32.(ram) .+ mask .* Float32(value)
end


# --- Functional bus dispatch ------------------------------------------------ #

"""
    _func_bus_read(bus, addr) -> Float32

Functional sibling of `_bus_read` — same dispatch (cart / RIOT I/O /
RAM), differentiable on the cart and RAM paths.
"""
function _func_bus_read(bus::SoftBus, addr::Real)
    a_int = Int(addr) & 0x1FFF
    if (a_int & 0x1000) != 0
        return soft_rom_peek(bus.rom, a_int & 0x0FFF)
    end
    if (a_int & 0x80) != 0 && (a_int & 0x200) != 0
        reg = a_int & 0x07
        reg == 4 && return bus.riot_intim
        reg == 5 && return bus.riot_expired != 0f0 ? Float32(0x80) : 0f0
        return Float32(0xFF)
    end
    return soft_ram_peek(bus.ram, a_int & 0x7F)
end

"""
    _func_bus_write(bus, addr, value) -> SoftBus

Functional sibling of `_bus_write!`. Returns a *new* `SoftBus` for the
write — ROM writes pass through unchanged; RIOT TIM*T writes load the
timer fields; everything else lands in `bus.ram[(addr & 0x7F) + 1]`.
"""
function _func_bus_write(bus::SoftBus, addr::Real, value::Real)
    a_int = Int(addr) & 0x1FFF
    if (a_int & 0x1000) != 0
        return bus                                          # ROM — drop
    end
    is_riot_io = (a_int & 0x80) != 0 && (a_int & 0x200) != 0
    if is_riot_io && (a_int & 0x14) == 0x14
        return update_bus(bus;
            riot_intim           = Float32(value),
            riot_prescaler_shift = Float32(_TIMT_SHIFTS[(a_int & 0x03) + 1]),
            riot_residual_cycles = 0f0,
            riot_expired         = 0f0)
    end
    new_ram = _set_ram(bus.ram, a_int & 0x7F, value)
    return update_bus(bus; ram = new_ram)
end


# --- Functional handlers — full 151-opcode coverage (P7e-x extension) ------ #
#
# Every handler returns (new_state, new_bus). The naming matches the
# mutating handlers but with no `!`. The extension fills in everything
# the mutating `_HANDLERS` table covers — load/store/transfer (every
# mode), ADC/SBC (binary + BCD), AND/ORA/EOR, CMP/CPX/CPY, BIT, ASL/LSR/
# ROL/ROR (acc + 4 memory modes), 8 conditional branches, JMP indirect,
# JSR / RTS, PHA / PHP / PLA / PLP, all 7 status-flag opcodes, INC/DEC
# memory + INX/INY/DEX/DEY, RTI. The full documented NMOS set + USBC
# ($EB) alias.
#
# Implementation notes:
#   * `_func_bus_write` returns a new SoftBus (no mutation), so RMW
#     handlers thread the bus through `(state, bus) → (s', b')` cleanly.
#   * `_func_push8` / `_func_pop8` are the stack helpers (also pure).
#   * `Int(...)` casts inside handlers break the gradient at structural
#     positions (same convention as the mutating side) — flag bits used
#     by branches and conditional flags. The float-flag mirrors needed
#     for full gradient through branches are jaxtari's P7c-dx; jutari's
#     equivalent lands as a follow-up to this extension.
#   * Every handler keeps the same cycle counts and PC advance as its
#     mutating sibling (forward parity with `soft_step!`).

_func_default(state, bus) = update_state(state; PC=state.PC + 1f0,
                                                cycles=state.cycles + 2f0), bus

_func_nop(state, bus)     = update_state(state; PC=state.PC + 1f0,
                                                cycles=state.cycles + 2f0), bus

function _func_brk(state, bus)
    # End-of-trace sentinel (consistent with `_branch_brk!`'s historic
    # P7b role — full BRK→IRQ sequence is the next deferral after this
    # extension).
    return update_state(state; cycles=state.cycles + 7f0), bus
end

function _func_jmp_abs(state, bus)
    new_pc = _operand_word(bus, state.PC + 1f0)
    return update_state(state; PC=new_pc, cycles=state.cycles + 3f0), bus
end

# --- Load / store helpers --------------------------------------------------- #

# Reads through the functional bus dispatch.
@inline _func_read_for_mode(state, bus, mode::Symbol) =
    mode === :imm   ? _operand_byte(bus, state.PC + 1f0) :
    mode === :zp    ? _func_bus_read(bus, _addr_zp(state, bus)) :
    mode === :zp_x  ? _func_bus_read(bus, _addr_zp_x(state, bus)) :
    mode === :zp_y  ? _func_bus_read(bus, _addr_zp_y(state, bus)) :
    mode === :abs   ? _func_bus_read(bus, _addr_abs(state, bus)) :
    mode === :abs_x ? _func_bus_read(bus, _addr_abs_x(state, bus)) :
    mode === :abs_y ? _func_bus_read(bus, _addr_abs_y(state, bus)) :
    mode === :ind_x ? _func_bus_read(bus, _addr_ind_x(state, bus)) :
    mode === :ind_y ? _func_bus_read(bus, _addr_ind_y(state, bus)) :
    error("_func_read_for_mode: unknown mode $mode")

# Load helpers — register := value; N/Z update; PC advance; cycle bump.
# Routes through `_with_p` so the float-flag mirrors (P_N / P_Z / P_C /
# P_V) stay in sync with the packed P byte — that's the P7c-dx hook
# `_func_do_branch` reads from for the gradient path.
function _func_load(state, bus, reg::Symbol, value::Real,
                    instr_len::Real, cycles::Real)
    new_p = _set_nz(state.P, value)
    nt = (A=state.A, X=state.X, Y=state.Y, SP=state.SP)
    new_reg_values = NamedTuple{(:A, :X, :Y, :SP)}((
        reg == :A  ? Float32(value) : nt.A,
        reg == :X  ? Float32(value) : nt.X,
        reg == :Y  ? Float32(value) : nt.Y,
        reg == :SP ? Float32(value) : nt.SP))
    return _with_p(state, new_p;
        A=new_reg_values.A, X=new_reg_values.X, Y=new_reg_values.Y,
        SP=new_reg_values.SP,
        PC=state.PC + Float32(instr_len),
        cycles=state.cycles + Float32(cycles),
    ), bus
end

# LDA (all 8 modes)
_func_lda_imm(state, bus)   = _func_load(state, bus, :A, _func_read_for_mode(state, bus, :imm),   2, 2)
_func_lda_zp(state, bus)    = _func_load(state, bus, :A, _func_read_for_mode(state, bus, :zp),    2, 3)
_func_lda_zp_x(state, bus)  = _func_load(state, bus, :A, _func_read_for_mode(state, bus, :zp_x),  2, 4)
_func_lda_abs(state, bus)   = _func_load(state, bus, :A, _func_read_for_mode(state, bus, :abs),   3, 4)
_func_lda_abs_x(state, bus) = _func_load(state, bus, :A, _func_read_for_mode(state, bus, :abs_x), 3, 4)
_func_lda_abs_y(state, bus) = _func_load(state, bus, :A, _func_read_for_mode(state, bus, :abs_y), 3, 4)
_func_lda_ind_x(state, bus) = _func_load(state, bus, :A, _func_read_for_mode(state, bus, :ind_x), 2, 6)
_func_lda_ind_y(state, bus) = _func_load(state, bus, :A, _func_read_for_mode(state, bus, :ind_y), 2, 5)

# LDX (5 modes: imm/zp/zp,Y/abs/abs,Y)
_func_ldx_imm(state, bus)   = _func_load(state, bus, :X, _func_read_for_mode(state, bus, :imm),   2, 2)
_func_ldx_zp(state, bus)    = _func_load(state, bus, :X, _func_read_for_mode(state, bus, :zp),    2, 3)
_func_ldx_zp_y(state, bus)  = _func_load(state, bus, :X, _func_read_for_mode(state, bus, :zp_y),  2, 4)
_func_ldx_abs(state, bus)   = _func_load(state, bus, :X, _func_read_for_mode(state, bus, :abs),   3, 4)
_func_ldx_abs_y(state, bus) = _func_load(state, bus, :X, _func_read_for_mode(state, bus, :abs_y), 3, 4)

# LDY (5 modes: imm/zp/zp,X/abs/abs,X)
_func_ldy_imm(state, bus)   = _func_load(state, bus, :Y, _func_read_for_mode(state, bus, :imm),   2, 2)
_func_ldy_zp(state, bus)    = _func_load(state, bus, :Y, _func_read_for_mode(state, bus, :zp),    2, 3)
_func_ldy_zp_x(state, bus)  = _func_load(state, bus, :Y, _func_read_for_mode(state, bus, :zp_x),  2, 4)
_func_ldy_abs(state, bus)   = _func_load(state, bus, :Y, _func_read_for_mode(state, bus, :abs),   3, 4)
_func_ldy_abs_x(state, bus) = _func_load(state, bus, :Y, _func_read_for_mode(state, bus, :abs_x), 3, 4)

# Store helpers — no flag changes.
function _func_store(state, bus, addr::Real, value::Real,
                     instr_len::Real, cycles::Real)
    new_bus = _func_bus_write(bus, addr, value)
    return update_state(state;
        PC=state.PC + Float32(instr_len),
        cycles=state.cycles + Float32(cycles),
    ), new_bus
end

# STA (7 modes)
_func_sta_zp(state, bus)    = _func_store(state, bus, _addr_zp(state, bus),    state.A, 2, 3)
_func_sta_zp_x(state, bus)  = _func_store(state, bus, _addr_zp_x(state, bus),  state.A, 2, 4)
_func_sta_abs(state, bus)   = _func_store(state, bus, _addr_abs(state, bus),   state.A, 3, 4)
_func_sta_abs_x(state, bus) = _func_store(state, bus, _addr_abs_x(state, bus), state.A, 3, 5)
_func_sta_abs_y(state, bus) = _func_store(state, bus, _addr_abs_y(state, bus), state.A, 3, 5)
_func_sta_ind_x(state, bus) = _func_store(state, bus, _addr_ind_x(state, bus), state.A, 2, 6)
_func_sta_ind_y(state, bus) = _func_store(state, bus, _addr_ind_y(state, bus), state.A, 2, 6)

# STX (3 modes)
_func_stx_zp(state, bus)    = _func_store(state, bus, _addr_zp(state, bus),    state.X, 2, 3)
_func_stx_zp_y(state, bus)  = _func_store(state, bus, _addr_zp_y(state, bus),  state.X, 2, 4)
_func_stx_abs(state, bus)   = _func_store(state, bus, _addr_abs(state, bus),   state.X, 3, 4)

# STY (3 modes)
_func_sty_zp(state, bus)    = _func_store(state, bus, _addr_zp(state, bus),    state.Y, 2, 3)
_func_sty_zp_x(state, bus)  = _func_store(state, bus, _addr_zp_x(state, bus),  state.Y, 2, 4)
_func_sty_abs(state, bus)   = _func_store(state, bus, _addr_abs(state, bus),   state.Y, 3, 4)

# Transfers (TAX/TAY/TXA/TYA/TSX touch N/Z; TXS does not).
_func_tax(state, bus) = _func_load(state, bus, :X, state.A,  1, 2)
_func_tay(state, bus) = _func_load(state, bus, :Y, state.A,  1, 2)
_func_txa(state, bus) = _func_load(state, bus, :A, state.X,  1, 2)
_func_tya(state, bus) = _func_load(state, bus, :A, state.Y,  1, 2)
_func_tsx(state, bus) = _func_load(state, bus, :X, state.SP, 1, 2)

function _func_txs(state, bus)
    return update_state(state;
        SP=state.X, PC=state.PC + 1f0, cycles=state.cycles + 2f0,
    ), bus
end

# --- ALU helpers (ADC/SBC + binary, AND/ORA/EOR, CMP, BIT) ----------------- #

function _func_adc_step(state, bus, operand::Real,
                        instr_len::Real, cycles::Real)
    c_in   = Float32(_read_carry(state.P))
    c_int  = Int(c_in)
    a_int  = Int(state.A) & 0xFF
    op_int = Int(operand) & 0xFF
    if _decimal_flag(state.P)
        bcd_sum = _bcd_decode(a_int) + _bcd_decode(op_int) + c_int
        new_a   = Float32(_bcd_encode(bcd_sum))
        carry   = bcd_sum > 99 ? 1 : 0
    else
        sum_  = state.A + Float32(operand) + c_in
        new_a = sum_ - floor(sum_ / 256f0) * 256f0
        carry = Int(sum_) > 0xFF ? 1 : 0
    end
    na_int   = Int(new_a) & 0xFF
    overflow = ((a_int ⊻ na_int) & (op_int ⊻ na_int)) & 0x80
    new_p    = _set_nzcv(state.P, new_a, carry, overflow)
    return _with_p(state, new_p;
        A=new_a,
        PC=state.PC + Float32(instr_len),
        cycles=state.cycles + Float32(cycles),
    ), bus
end

function _func_sbc_step(state, bus, operand::Real,
                        instr_len::Real, cycles::Real)
    c_in   = Float32(_read_carry(state.P))
    c_int  = Int(c_in)
    a_int  = Int(state.A) & 0xFF
    op_int = Int(operand) & 0xFF
    if _decimal_flag(state.P)
        diff = _bcd_decode(a_int) - _bcd_decode(op_int) - (1 - c_int)
        diff < 0 && (diff += 100)
        new_a    = Float32(_bcd_encode(diff))
        carry    = a_int >= (op_int + (1 - c_int)) ? 1 : 0
        na_int   = Int(new_a) & 0xFF
        overflow = ((a_int ⊻ na_int) & (op_int ⊻ na_int)) & 0x80
    else
        op_inv   = 255f0 - Float32(operand)
        sum_     = state.A + op_inv + c_in
        new_a    = sum_ - floor(sum_ / 256f0) * 256f0
        carry    = Int(sum_) > 0xFF ? 1 : 0
        op_inv_i = Int(op_inv) & 0xFF
        na_int   = Int(new_a) & 0xFF
        overflow = ((a_int ⊻ na_int) & (op_inv_i ⊻ na_int)) & 0x80
    end
    new_p = _set_nzcv(state.P, new_a, carry, overflow)
    return _with_p(state, new_p;
        A=new_a,
        PC=state.PC + Float32(instr_len),
        cycles=state.cycles + Float32(cycles),
    ), bus
end

function _func_bitop_step(state, bus, operand::Real, op::Function,
                          instr_len::Real, cycles::Real)
    a_int     = Int(state.A) & 0xFF
    op_int    = Int(operand) & 0xFF
    new_a_int = op(a_int, op_int) & 0xFF
    new_a     = Float32(new_a_int)
    new_p     = _set_nz(state.P, new_a_int)
    return _with_p(state, new_p;
        A=new_a,
        PC=state.PC + Float32(instr_len),
        cycles=state.cycles + Float32(cycles),
    ), bus
end

function _func_compare_step(state, bus, reg_value::Real, operand::Real,
                            instr_len::Real, cycles::Real)
    reg_int  = Int(reg_value) & 0xFF
    op_int   = Int(operand) & 0xFF
    diff_int = (reg_int - op_int) & 0xFF
    carry    = reg_int >= op_int ? 1 : 0
    new_p    = _set_nzc(state.P, diff_int, carry)
    return _with_p(state, new_p;
        PC=state.PC + Float32(instr_len),
        cycles=state.cycles + Float32(cycles),
    ), bus
end

function _func_bit_step(state, bus, operand::Real,
                        instr_len::Real, cycles::Real)
    a_int  = Int(state.A) & 0xFF
    op_int = Int(operand) & 0xFF
    and_v  = a_int & op_int
    new_p  = _set_bit_flags(state.P, and_v, operand)
    return _with_p(state, new_p;
        PC=state.PC + Float32(instr_len),
        cycles=state.cycles + Float32(cycles),
    ), bus
end

# ADC (8 modes)
_func_adc_imm(s, b)   = _func_adc_step(s, b, _func_read_for_mode(s, b, :imm),   2, 2)
_func_adc_zp(s, b)    = _func_adc_step(s, b, _func_read_for_mode(s, b, :zp),    2, 3)
_func_adc_zp_x(s, b)  = _func_adc_step(s, b, _func_read_for_mode(s, b, :zp_x),  2, 4)
_func_adc_abs(s, b)   = _func_adc_step(s, b, _func_read_for_mode(s, b, :abs),   3, 4)
_func_adc_abs_x(s, b) = _func_adc_step(s, b, _func_read_for_mode(s, b, :abs_x), 3, 4)
_func_adc_abs_y(s, b) = _func_adc_step(s, b, _func_read_for_mode(s, b, :abs_y), 3, 4)
_func_adc_ind_x(s, b) = _func_adc_step(s, b, _func_read_for_mode(s, b, :ind_x), 2, 6)
_func_adc_ind_y(s, b) = _func_adc_step(s, b, _func_read_for_mode(s, b, :ind_y), 2, 5)

# SBC (8 modes + USBC alias on $EB)
_func_sbc_imm(s, b)   = _func_sbc_step(s, b, _func_read_for_mode(s, b, :imm),   2, 2)
_func_sbc_zp(s, b)    = _func_sbc_step(s, b, _func_read_for_mode(s, b, :zp),    2, 3)
_func_sbc_zp_x(s, b)  = _func_sbc_step(s, b, _func_read_for_mode(s, b, :zp_x),  2, 4)
_func_sbc_abs(s, b)   = _func_sbc_step(s, b, _func_read_for_mode(s, b, :abs),   3, 4)
_func_sbc_abs_x(s, b) = _func_sbc_step(s, b, _func_read_for_mode(s, b, :abs_x), 3, 4)
_func_sbc_abs_y(s, b) = _func_sbc_step(s, b, _func_read_for_mode(s, b, :abs_y), 3, 4)
_func_sbc_ind_x(s, b) = _func_sbc_step(s, b, _func_read_for_mode(s, b, :ind_x), 2, 6)
_func_sbc_ind_y(s, b) = _func_sbc_step(s, b, _func_read_for_mode(s, b, :ind_y), 2, 5)

# AND
_func_and_imm(s, b)   = _func_bitop_step(s, b, _func_read_for_mode(s, b, :imm),   &, 2, 2)
_func_and_zp(s, b)    = _func_bitop_step(s, b, _func_read_for_mode(s, b, :zp),    &, 2, 3)
_func_and_zp_x(s, b)  = _func_bitop_step(s, b, _func_read_for_mode(s, b, :zp_x),  &, 2, 4)
_func_and_abs(s, b)   = _func_bitop_step(s, b, _func_read_for_mode(s, b, :abs),   &, 3, 4)
_func_and_abs_x(s, b) = _func_bitop_step(s, b, _func_read_for_mode(s, b, :abs_x), &, 3, 4)
_func_and_abs_y(s, b) = _func_bitop_step(s, b, _func_read_for_mode(s, b, :abs_y), &, 3, 4)
_func_and_ind_x(s, b) = _func_bitop_step(s, b, _func_read_for_mode(s, b, :ind_x), &, 2, 6)
_func_and_ind_y(s, b) = _func_bitop_step(s, b, _func_read_for_mode(s, b, :ind_y), &, 2, 5)

# ORA
_func_ora_imm(s, b)   = _func_bitop_step(s, b, _func_read_for_mode(s, b, :imm),   |, 2, 2)
_func_ora_zp(s, b)    = _func_bitop_step(s, b, _func_read_for_mode(s, b, :zp),    |, 2, 3)
_func_ora_zp_x(s, b)  = _func_bitop_step(s, b, _func_read_for_mode(s, b, :zp_x),  |, 2, 4)
_func_ora_abs(s, b)   = _func_bitop_step(s, b, _func_read_for_mode(s, b, :abs),   |, 3, 4)
_func_ora_abs_x(s, b) = _func_bitop_step(s, b, _func_read_for_mode(s, b, :abs_x), |, 3, 4)
_func_ora_abs_y(s, b) = _func_bitop_step(s, b, _func_read_for_mode(s, b, :abs_y), |, 3, 4)
_func_ora_ind_x(s, b) = _func_bitop_step(s, b, _func_read_for_mode(s, b, :ind_x), |, 2, 6)
_func_ora_ind_y(s, b) = _func_bitop_step(s, b, _func_read_for_mode(s, b, :ind_y), |, 2, 5)

# EOR
_func_eor_imm(s, b)   = _func_bitop_step(s, b, _func_read_for_mode(s, b, :imm),   ⊻, 2, 2)
_func_eor_zp(s, b)    = _func_bitop_step(s, b, _func_read_for_mode(s, b, :zp),    ⊻, 2, 3)
_func_eor_zp_x(s, b)  = _func_bitop_step(s, b, _func_read_for_mode(s, b, :zp_x),  ⊻, 2, 4)
_func_eor_abs(s, b)   = _func_bitop_step(s, b, _func_read_for_mode(s, b, :abs),   ⊻, 3, 4)
_func_eor_abs_x(s, b) = _func_bitop_step(s, b, _func_read_for_mode(s, b, :abs_x), ⊻, 3, 4)
_func_eor_abs_y(s, b) = _func_bitop_step(s, b, _func_read_for_mode(s, b, :abs_y), ⊻, 3, 4)
_func_eor_ind_x(s, b) = _func_bitop_step(s, b, _func_read_for_mode(s, b, :ind_x), ⊻, 2, 6)
_func_eor_ind_y(s, b) = _func_bitop_step(s, b, _func_read_for_mode(s, b, :ind_y), ⊻, 2, 5)

# CMP
_func_cmp_imm(s, b)   = _func_compare_step(s, b, s.A, _func_read_for_mode(s, b, :imm),   2, 2)
_func_cmp_zp(s, b)    = _func_compare_step(s, b, s.A, _func_read_for_mode(s, b, :zp),    2, 3)
_func_cmp_zp_x(s, b)  = _func_compare_step(s, b, s.A, _func_read_for_mode(s, b, :zp_x),  2, 4)
_func_cmp_abs(s, b)   = _func_compare_step(s, b, s.A, _func_read_for_mode(s, b, :abs),   3, 4)
_func_cmp_abs_x(s, b) = _func_compare_step(s, b, s.A, _func_read_for_mode(s, b, :abs_x), 3, 4)
_func_cmp_abs_y(s, b) = _func_compare_step(s, b, s.A, _func_read_for_mode(s, b, :abs_y), 3, 4)
_func_cmp_ind_x(s, b) = _func_compare_step(s, b, s.A, _func_read_for_mode(s, b, :ind_x), 2, 6)
_func_cmp_ind_y(s, b) = _func_compare_step(s, b, s.A, _func_read_for_mode(s, b, :ind_y), 2, 5)

# CPX / CPY
_func_cpx_imm(s, b) = _func_compare_step(s, b, s.X, _func_read_for_mode(s, b, :imm), 2, 2)
_func_cpx_zp(s, b)  = _func_compare_step(s, b, s.X, _func_read_for_mode(s, b, :zp),  2, 3)
_func_cpx_abs(s, b) = _func_compare_step(s, b, s.X, _func_read_for_mode(s, b, :abs), 3, 4)

_func_cpy_imm(s, b) = _func_compare_step(s, b, s.Y, _func_read_for_mode(s, b, :imm), 2, 2)
_func_cpy_zp(s, b)  = _func_compare_step(s, b, s.Y, _func_read_for_mode(s, b, :zp),  2, 3)
_func_cpy_abs(s, b) = _func_compare_step(s, b, s.Y, _func_read_for_mode(s, b, :abs), 3, 4)

# BIT
_func_bit_zp(s, b)  = _func_bit_step(s, b, _func_read_for_mode(s, b, :zp),  2, 3)
_func_bit_abs(s, b) = _func_bit_step(s, b, _func_read_for_mode(s, b, :abs), 3, 4)


# --- Shift / rotate helpers ------------------------------------------------- #
#
# `_func_shift_acc` writes the result back to A; `_func_shift_memory` does
# a read-modify-write through the functional bus dispatch. Both routines
# share `_asl_value` / `_lsr_value` / `_rol_value` / `_ror_value` with
# the mutating side — these are pure already.

function _func_shift_acc(state, bus, value_op::Function)
    new_a, new_p = value_op(state.P, state.A)
    return _with_p(state, new_p;
        A=new_a,
        PC=state.PC + 1f0, cycles=state.cycles + 2f0,
    ), bus
end

function _func_shift_memory(state, bus, addr_resolver::Function,
                            value_op::Function,
                            instr_len::Real, cycles::Real)
    addr             = addr_resolver(state, bus)
    value            = _func_bus_read(bus, addr)
    new_value, new_p = value_op(state.P, value)
    new_bus          = _func_bus_write(bus, addr, new_value)
    return _with_p(state, new_p;
        PC=state.PC + Float32(instr_len),
        cycles=state.cycles + Float32(cycles),
    ), new_bus
end

# ASL
_func_asl_acc(s, b)   = _func_shift_acc(s, b, _asl_value)
_func_asl_zp(s, b)    = _func_shift_memory(s, b, _addr_zp,    _asl_value, 2, 5)
_func_asl_zp_x(s, b)  = _func_shift_memory(s, b, _addr_zp_x,  _asl_value, 2, 6)
_func_asl_abs(s, b)   = _func_shift_memory(s, b, _addr_abs,   _asl_value, 3, 6)
_func_asl_abs_x(s, b) = _func_shift_memory(s, b, _addr_abs_x, _asl_value, 3, 7)

# LSR
_func_lsr_acc(s, b)   = _func_shift_acc(s, b, _lsr_value)
_func_lsr_zp(s, b)    = _func_shift_memory(s, b, _addr_zp,    _lsr_value, 2, 5)
_func_lsr_zp_x(s, b)  = _func_shift_memory(s, b, _addr_zp_x,  _lsr_value, 2, 6)
_func_lsr_abs(s, b)   = _func_shift_memory(s, b, _addr_abs,   _lsr_value, 3, 6)
_func_lsr_abs_x(s, b) = _func_shift_memory(s, b, _addr_abs_x, _lsr_value, 3, 7)

# ROL
_func_rol_acc(s, b)   = _func_shift_acc(s, b, _rol_value)
_func_rol_zp(s, b)    = _func_shift_memory(s, b, _addr_zp,    _rol_value, 2, 5)
_func_rol_zp_x(s, b)  = _func_shift_memory(s, b, _addr_zp_x,  _rol_value, 2, 6)
_func_rol_abs(s, b)   = _func_shift_memory(s, b, _addr_abs,   _rol_value, 3, 6)
_func_rol_abs_x(s, b) = _func_shift_memory(s, b, _addr_abs_x, _rol_value, 3, 7)

# ROR
_func_ror_acc(s, b)   = _func_shift_acc(s, b, _ror_value)
_func_ror_zp(s, b)    = _func_shift_memory(s, b, _addr_zp,    _ror_value, 2, 5)
_func_ror_zp_x(s, b)  = _func_shift_memory(s, b, _addr_zp_x,  _ror_value, 2, 6)
_func_ror_abs(s, b)   = _func_shift_memory(s, b, _addr_abs,   _ror_value, 3, 6)
_func_ror_abs_x(s, b) = _func_shift_memory(s, b, _addr_abs_x, _ror_value, 3, 7)


# --- Branch, JMP indirect, JSR / RTS / RTI --------------------------------- #

"""
    _func_do_branch(state, bus, flag_field, flag_mask, take_when_set)

Conditional branch.

`flag_field` is one of `:P_N`, `:P_Z`, `:P_C`, `:P_V` — the
float-valued mirror `_with_p` keeps in sync with `P`. `flag_mask`
is the matching packed-byte bit (\$80 / \$02 / \$01 / \$40). Reads
the relative displacement at ROM[PC+1]; if the predicate holds,
PC = (PC+2) + signed_offset, else PC += 2; +1 cycle when the branch
is taken (page-cross penalty isn't modelled).

**P7c-dx**: forward semantics come from the packed `state.P` byte
(so PXC1 conformance + existing tests stay bit-exact), but the
*gradient* runs through `state.<flag_field>` via `soft_branch` — a
caller that injects a soft Z / N / C / V into `state.P_X` for an XAI
experiment gets gradient through the sigmoid-blended PC.

The two paths are glued together with a straight-through trick —
forward = `pc_hard`, backward = ∂`pc_soft`.
"""
function _func_do_branch(state, bus, flag_field::Symbol,
                         flag_mask::Integer, take_when_set::Bool)
    offset       = _signed_offset(_operand_byte(bus, state.PC + 1f0))
    pc_not_taken = state.PC + 2f0
    pc_taken     = pc_not_taken + offset

    # Float flag for the gradient path — what the caller can override.
    # Zygote traces dot-access on mutable structs cleanly; runtime-Symbol
    # getfield can be opaque to the rrule generator, so dispatch by name.
    flag_soft  = flag_field === :P_N ? state.P_N :
                 flag_field === :P_Z ? state.P_Z :
                 flag_field === :P_C ? state.P_C :
                 flag_field === :P_V ? state.P_V :
                 error("_func_do_branch: unknown flag_field $flag_field")
    logit_sign = take_when_set ? 1f0 : -1f0
    flag_logit = logit_sign * (2f0 * flag_soft - 1f0)

    # Soft PC — sigmoid-blended for the gradient.
    pc_soft = soft_branch(flag_logit, pc_not_taken, pc_taken; alpha=10.0)

    # Hard PC — read from packed `state.P` directly so tests that
    # mutate only P (and not P_X) still pick the right branch.
    flag_set  = (Int(state.P) & flag_mask) != 0
    take_hard = take_when_set ? flag_set : !flag_set
    pc_hard   = take_hard ? pc_taken : pc_not_taken

    # Straight-through: forward = hard PC, backward = soft (sigmoid) PC.
    # `_stop_gradient` makes the additive offset opaque to Zygote, so
    # the backward pass sees only the `pc_soft` term — matching the
    # jaxtari `pc_soft + jax.lax.stop_gradient(pc_hard - pc_soft)`
    # straight-through construction.
    new_pc = pc_soft + _stop_gradient(pc_hard - pc_soft)

    # Cycles +1 when taken. Same straight-through dance.
    extra_hard = take_hard ? 1f0 : 0f0
    g          = 1f0 / (1f0 + exp(-10f0 * flag_logit))   # sigmoid
    extra_soft = g
    extra      = extra_soft + _stop_gradient(extra_hard - extra_soft)
    return update_state(state;
        PC=new_pc, cycles=state.cycles + 2f0 + extra,
    ), bus
end

_func_bpl(s, b) = _func_do_branch(s, b, :P_N, 0x80, false)   # branch if N=0
_func_bmi(s, b) = _func_do_branch(s, b, :P_N, 0x80, true)    # branch if N=1
_func_bvc(s, b) = _func_do_branch(s, b, :P_V, 0x40, false)   # branch if V=0
_func_bvs(s, b) = _func_do_branch(s, b, :P_V, 0x40, true)    # branch if V=1
_func_bcc(s, b) = _func_do_branch(s, b, :P_C, 0x01, false)   # branch if C=0
_func_bcs(s, b) = _func_do_branch(s, b, :P_C, 0x01, true)    # branch if C=1
_func_bne(s, b) = _func_do_branch(s, b, :P_Z, 0x02, false)   # branch if Z=0
_func_beq(s, b) = _func_do_branch(s, b, :P_Z, 0x02, true)    # branch if Z=1

function _func_jmp_ind(state, bus)
    ptr    = _addr_abs(state, bus)
    ptr_lo = ptr
    ptr_hi = (ptr & 0xFF00) | ((ptr + 1) & 0x00FF)   # NMOS page-wrap bug
    lo     = _func_bus_read(bus, ptr_lo)
    hi     = _func_bus_read(bus, ptr_hi)
    return update_state(state;
        PC=lo + hi * 256f0, cycles=state.cycles + 5f0,
    ), bus
end

# Stack helpers (pure).
function _func_push8(bus::SoftBus, sp::Real, value::Real)
    addr    = 0x0100 + (Int(sp) & 0xFF)
    new_bus = _func_bus_write(bus, addr, value)
    new_sp  = _wrap_byte(Float32(sp) - 1f0)
    return new_bus, new_sp
end

function _func_pop8(bus::SoftBus, sp::Real)
    new_sp = _wrap_byte(Float32(sp) + 1f0)
    addr   = 0x0100 + (Int(new_sp) & 0xFF)
    return _func_bus_read(bus, addr), new_sp
end

function _func_jsr(state, bus)
    target = _operand_word(bus, state.PC + 1f0)
    ra_int = Int(state.PC + 2f0) & 0xFFFF        # last byte of JSR
    hi     = Float32((ra_int >> 8) & 0xFF)
    lo     = Float32(ra_int & 0xFF)
    bus, sp = _func_push8(bus, state.SP, hi)
    bus, sp = _func_push8(bus, sp, lo)
    return update_state(state;
        SP=sp, PC=target, cycles=state.cycles + 6f0,
    ), bus
end

function _func_rts(state, bus)
    lo, sp = _func_pop8(bus, state.SP)
    hi, sp = _func_pop8(bus, sp)
    ret    = lo + hi * 256f0
    return update_state(state;
        SP=sp, PC=ret + 1f0, cycles=state.cycles + 6f0,
    ), bus
end


# --- Stack push/pull, status flags, INC/DEC, INX/INY/DEX/DEY --------------- #

function _func_pha(state, bus)
    new_bus, sp = _func_push8(bus, state.SP, state.A)
    return update_state(state;
        SP=sp, PC=state.PC + 1f0, cycles=state.cycles + 3f0,
    ), new_bus
end

function _func_php(state, bus)
    p_pushed     = Float32(Int(state.P) | 0x30)        # force B + U
    new_bus, sp  = _func_push8(bus, state.SP, p_pushed)
    return update_state(state;
        SP=sp, PC=state.PC + 1f0, cycles=state.cycles + 3f0,
    ), new_bus
end

function _func_pla(state, bus)
    value, sp = _func_pop8(bus, state.SP)
    new_p     = _set_nz(state.P, value)
    return _with_p(state, new_p;
        A=value, SP=sp,
        PC=state.PC + 1f0, cycles=state.cycles + 4f0,
    ), bus
end

function _func_plp(state, bus)
    popped, sp = _func_pop8(bus, state.SP)
    new_p      = Float32(Int(popped) | 0x30)           # force B + U
    return _with_p(state, new_p;
        SP=sp,
        PC=state.PC + 1f0, cycles=state.cycles + 4f0,
    ), bus
end

# Single-bit flag toggles (CLC/SEC + CLI/SEI/CLV/CLD/SED).
function _func_set_flag(state, bus, mask::Integer, set_it::Bool)
    p_int = Int(state.P) & 0xFF
    new_p = set_it ? (p_int | mask) : (p_int & (0xFF ⊻ mask))
    return _with_p(state, Float32(new_p);
        PC=state.PC + 1f0, cycles=state.cycles + 2f0,
    ), bus
end

_func_clc(state, bus) = _func_set_flag(state, bus, 0x01, false)
_func_sec(state, bus) = _func_set_flag(state, bus, 0x01, true)
_func_cli(state, bus) = _func_set_flag(state, bus, 0x04, false)
_func_sei(state, bus) = _func_set_flag(state, bus, 0x04, true)
_func_clv(state, bus) = _func_set_flag(state, bus, 0x40, false)
_func_cld(state, bus) = _func_set_flag(state, bus, 0x08, false)
_func_sed(state, bus) = _func_set_flag(state, bus, 0x08, true)

function _func_incdec_memory(state, bus, addr_resolver::Function,
                             value_op::Function,
                             instr_len::Real, cycles::Real)
    addr      = addr_resolver(state, bus)
    value     = _func_bus_read(bus, addr)
    new_value = value_op(value)
    new_bus   = _func_bus_write(bus, addr, new_value)
    new_p     = _set_nz(state.P, new_value)
    return _with_p(state, new_p;
        PC=state.PC + Float32(instr_len),
        cycles=state.cycles + Float32(cycles),
    ), new_bus
end

_func_inc_zp(s, b)    = _func_incdec_memory(s, b, _addr_zp,    _inc_value, 2, 5)
_func_inc_zp_x(s, b)  = _func_incdec_memory(s, b, _addr_zp_x,  _inc_value, 2, 6)
_func_inc_abs(s, b)   = _func_incdec_memory(s, b, _addr_abs,   _inc_value, 3, 6)
_func_inc_abs_x(s, b) = _func_incdec_memory(s, b, _addr_abs_x, _inc_value, 3, 7)

_func_dec_zp(s, b)    = _func_incdec_memory(s, b, _addr_zp,    _dec_value, 2, 5)
_func_dec_zp_x(s, b)  = _func_incdec_memory(s, b, _addr_zp_x,  _dec_value, 2, 6)
_func_dec_abs(s, b)   = _func_incdec_memory(s, b, _addr_abs,   _dec_value, 3, 6)
_func_dec_abs_x(s, b) = _func_incdec_memory(s, b, _addr_abs_x, _dec_value, 3, 7)

function _func_incdec_reg(state, bus, reg::Symbol, value_op::Function)
    cur   = reg === :X ? state.X : state.Y
    new_v = value_op(cur)
    new_p = _set_nz(state.P, new_v)
    if reg === :X
        return _with_p(state, new_p;
            X=new_v,
            PC=state.PC + 1f0, cycles=state.cycles + 2f0,
        ), bus
    else
        return _with_p(state, new_p;
            Y=new_v,
            PC=state.PC + 1f0, cycles=state.cycles + 2f0,
        ), bus
    end
end

_func_inx(s, b) = _func_incdec_reg(s, b, :X, _inc_value)
_func_iny(s, b) = _func_incdec_reg(s, b, :Y, _inc_value)
_func_dex(s, b) = _func_incdec_reg(s, b, :X, _dec_value)
_func_dey(s, b) = _func_incdec_reg(s, b, :Y, _dec_value)


# --- RTI (completes the 151-opcode documented NMOS set) -------------------- #

function _func_rti(state, bus)
    popped_p, sp = _func_pop8(bus, state.SP)
    lo, sp       = _func_pop8(bus, sp)
    hi, sp       = _func_pop8(bus, sp)
    new_p        = Float32(Int(popped_p) | 0x30)   # force B + U
    return _with_p(state, new_p;
        SP=sp,
        PC=lo + hi * 256f0,                        # no +1, unlike RTS
        cycles=state.cycles + 6f0,
    ), bus
end


# Dispatch table — functional analogue of `_HANDLERS`. Opcodes not
# listed fall through to `_func_default`. Extending coverage is a
# matter of writing the handler + adding the entry.
const _FUNC_HANDLERS = let h = Function[_func_default for _ in 1:256]
    # P7b core
    h[0x00 + 1] = _func_brk
    h[0xEA + 1] = _func_nop
    h[0x4C + 1] = _func_jmp_abs
    # LDA
    h[0xA9 + 1] = _func_lda_imm
    h[0xA5 + 1] = _func_lda_zp
    h[0xB5 + 1] = _func_lda_zp_x
    h[0xAD + 1] = _func_lda_abs
    h[0xBD + 1] = _func_lda_abs_x
    h[0xB9 + 1] = _func_lda_abs_y
    h[0xA1 + 1] = _func_lda_ind_x
    h[0xB1 + 1] = _func_lda_ind_y
    # LDX
    h[0xA2 + 1] = _func_ldx_imm
    h[0xA6 + 1] = _func_ldx_zp
    h[0xB6 + 1] = _func_ldx_zp_y
    h[0xAE + 1] = _func_ldx_abs
    h[0xBE + 1] = _func_ldx_abs_y
    # LDY
    h[0xA0 + 1] = _func_ldy_imm
    h[0xA4 + 1] = _func_ldy_zp
    h[0xB4 + 1] = _func_ldy_zp_x
    h[0xAC + 1] = _func_ldy_abs
    h[0xBC + 1] = _func_ldy_abs_x
    # STA
    h[0x85 + 1] = _func_sta_zp
    h[0x95 + 1] = _func_sta_zp_x
    h[0x8D + 1] = _func_sta_abs
    h[0x9D + 1] = _func_sta_abs_x
    h[0x99 + 1] = _func_sta_abs_y
    h[0x81 + 1] = _func_sta_ind_x
    h[0x91 + 1] = _func_sta_ind_y
    # STX
    h[0x86 + 1] = _func_stx_zp
    h[0x96 + 1] = _func_stx_zp_y
    h[0x8E + 1] = _func_stx_abs
    # STY
    h[0x84 + 1] = _func_sty_zp
    h[0x94 + 1] = _func_sty_zp_x
    h[0x8C + 1] = _func_sty_abs
    # Transfers
    h[0xAA + 1] = _func_tax
    h[0xA8 + 1] = _func_tay
    h[0x8A + 1] = _func_txa
    h[0x98 + 1] = _func_tya
    h[0xBA + 1] = _func_tsx
    h[0x9A + 1] = _func_txs
    # ADC
    h[0x69 + 1] = _func_adc_imm;   h[0x65 + 1] = _func_adc_zp
    h[0x75 + 1] = _func_adc_zp_x;  h[0x6D + 1] = _func_adc_abs
    h[0x7D + 1] = _func_adc_abs_x; h[0x79 + 1] = _func_adc_abs_y
    h[0x61 + 1] = _func_adc_ind_x; h[0x71 + 1] = _func_adc_ind_y
    # SBC (+ USBC $EB)
    h[0xE9 + 1] = _func_sbc_imm;   h[0xE5 + 1] = _func_sbc_zp
    h[0xF5 + 1] = _func_sbc_zp_x;  h[0xED + 1] = _func_sbc_abs
    h[0xFD + 1] = _func_sbc_abs_x; h[0xF9 + 1] = _func_sbc_abs_y
    h[0xE1 + 1] = _func_sbc_ind_x; h[0xF1 + 1] = _func_sbc_ind_y
    h[0xEB + 1] = _func_sbc_imm
    # AND
    h[0x29 + 1] = _func_and_imm;   h[0x25 + 1] = _func_and_zp
    h[0x35 + 1] = _func_and_zp_x;  h[0x2D + 1] = _func_and_abs
    h[0x3D + 1] = _func_and_abs_x; h[0x39 + 1] = _func_and_abs_y
    h[0x21 + 1] = _func_and_ind_x; h[0x31 + 1] = _func_and_ind_y
    # ORA
    h[0x09 + 1] = _func_ora_imm;   h[0x05 + 1] = _func_ora_zp
    h[0x15 + 1] = _func_ora_zp_x;  h[0x0D + 1] = _func_ora_abs
    h[0x1D + 1] = _func_ora_abs_x; h[0x19 + 1] = _func_ora_abs_y
    h[0x01 + 1] = _func_ora_ind_x; h[0x11 + 1] = _func_ora_ind_y
    # EOR
    h[0x49 + 1] = _func_eor_imm;   h[0x45 + 1] = _func_eor_zp
    h[0x55 + 1] = _func_eor_zp_x;  h[0x4D + 1] = _func_eor_abs
    h[0x5D + 1] = _func_eor_abs_x; h[0x59 + 1] = _func_eor_abs_y
    h[0x41 + 1] = _func_eor_ind_x; h[0x51 + 1] = _func_eor_ind_y
    # CMP
    h[0xC9 + 1] = _func_cmp_imm;   h[0xC5 + 1] = _func_cmp_zp
    h[0xD5 + 1] = _func_cmp_zp_x;  h[0xCD + 1] = _func_cmp_abs
    h[0xDD + 1] = _func_cmp_abs_x; h[0xD9 + 1] = _func_cmp_abs_y
    h[0xC1 + 1] = _func_cmp_ind_x; h[0xD1 + 1] = _func_cmp_ind_y
    # CPX / CPY / BIT
    h[0xE0 + 1] = _func_cpx_imm; h[0xE4 + 1] = _func_cpx_zp; h[0xEC + 1] = _func_cpx_abs
    h[0xC0 + 1] = _func_cpy_imm; h[0xC4 + 1] = _func_cpy_zp; h[0xCC + 1] = _func_cpy_abs
    h[0x24 + 1] = _func_bit_zp;  h[0x2C + 1] = _func_bit_abs
    # Shifts / rotates
    h[0x0A + 1] = _func_asl_acc;   h[0x06 + 1] = _func_asl_zp
    h[0x16 + 1] = _func_asl_zp_x;  h[0x0E + 1] = _func_asl_abs
    h[0x1E + 1] = _func_asl_abs_x
    h[0x4A + 1] = _func_lsr_acc;   h[0x46 + 1] = _func_lsr_zp
    h[0x56 + 1] = _func_lsr_zp_x;  h[0x4E + 1] = _func_lsr_abs
    h[0x5E + 1] = _func_lsr_abs_x
    h[0x2A + 1] = _func_rol_acc;   h[0x26 + 1] = _func_rol_zp
    h[0x36 + 1] = _func_rol_zp_x;  h[0x2E + 1] = _func_rol_abs
    h[0x3E + 1] = _func_rol_abs_x
    h[0x6A + 1] = _func_ror_acc;   h[0x66 + 1] = _func_ror_zp
    h[0x76 + 1] = _func_ror_zp_x;  h[0x6E + 1] = _func_ror_abs
    h[0x7E + 1] = _func_ror_abs_x
    # Branches
    h[0x10 + 1] = _func_bpl; h[0x30 + 1] = _func_bmi
    h[0x50 + 1] = _func_bvc; h[0x70 + 1] = _func_bvs
    h[0x90 + 1] = _func_bcc; h[0xB0 + 1] = _func_bcs
    h[0xD0 + 1] = _func_bne; h[0xF0 + 1] = _func_beq
    # JMP indirect + JSR / RTS
    h[0x6C + 1] = _func_jmp_ind
    h[0x20 + 1] = _func_jsr
    h[0x60 + 1] = _func_rts
    # Stack push/pull
    h[0x48 + 1] = _func_pha; h[0x08 + 1] = _func_php
    h[0x68 + 1] = _func_pla; h[0x28 + 1] = _func_plp
    # Status flags
    h[0x18 + 1] = _func_clc; h[0x38 + 1] = _func_sec
    h[0x58 + 1] = _func_cli; h[0x78 + 1] = _func_sei
    h[0xB8 + 1] = _func_clv; h[0xD8 + 1] = _func_cld
    h[0xF8 + 1] = _func_sed
    # INC memory
    h[0xE6 + 1] = _func_inc_zp;  h[0xF6 + 1] = _func_inc_zp_x
    h[0xEE + 1] = _func_inc_abs; h[0xFE + 1] = _func_inc_abs_x
    # DEC memory
    h[0xC6 + 1] = _func_dec_zp;  h[0xD6 + 1] = _func_dec_zp_x
    h[0xCE + 1] = _func_dec_abs; h[0xDE + 1] = _func_dec_abs_x
    # INX/INY/DEX/DEY
    h[0xE8 + 1] = _func_inx; h[0xC8 + 1] = _func_iny
    h[0xCA + 1] = _func_dex; h[0x88 + 1] = _func_dey
    # RTI
    h[0x40 + 1] = _func_rti
    h
end

"""
    soft_step(state::SoftCPUState, bus::SoftBus) -> (SoftCPUState, SoftBus)

Functional sibling of `soft_step!`. Builds a *new* `SoftCPUState` and
(when needed) a new `SoftBus` instead of mutating in place — Zygote
can therefore differentiate through it.

Opcode coverage as of the **P7e-x extension**: the **full 151-opcode
documented NMOS set** + the USBC (\$EB) alias, matching the mutating
`soft_step!` 1:1 — load/store/transfer (every mode), ADC/SBC
(binary + BCD), AND/ORA/EOR, CMP/CPX/CPY, BIT, ASL/LSR/ROL/ROR
(acc + 4 memory modes each), 8 conditional branches, JMP indirect,
JSR / RTS, PHA/PHP/PLA/PLP, all 7 status-flag opcodes (CLC/SEC/CLI/
SEI/CLV/CLD/SED), INC/DEC memory (4 modes), INX/INY/DEX/DEY, RTI.
BRK keeps the end-of-trace sentinel role from P7b (the full IRQ
sequence is the next deferral).

Anything outside that set falls through to `_func_default` (PC += 1,
cycles += 2) — the trace stays gradient-clean but forward behaviour
past an unhandled opcode is wrong.
"""
function soft_step(state::SoftCPUState, bus::SoftBus)
    rom_off       = _cart_addr(state.PC)
    opcode_float  = soft_rom_peek(bus.rom, rom_off)
    opcode_int    = Int(opcode_float) & 0xFF
    return _FUNC_HANDLERS[opcode_int + 1](state, bus)
end

"""
    soft_run(state::SoftCPUState, bus::SoftBus, n_steps::Integer)
        -> (SoftCPUState, SoftBus)

Run `n_steps` functional steps. Zygote-differentiable end-to-end — a
gradient at any output field of the returned state (or bus.ram) flows
back to the ROM bytes that fed every memory access along the trace.
"""
function soft_run(state::SoftCPUState, bus::SoftBus, n_steps::Integer)
    for _ in 1:n_steps
        state, bus = soft_step(state, bus)
    end
    return state, bus
end
