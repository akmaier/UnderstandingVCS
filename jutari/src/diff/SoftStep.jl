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

Differentiable cart byte read — `one_hot(addr) · rom`.
"""
function soft_rom_peek(rom::AbstractVector{<:Real}, addr::Real)
    n = length(rom)
    one_hot = zeros(Float32, n)
    one_hot[Int(addr) + 1] = 1f0
    return _dot(one_hot, Float32.(rom))
end

"""
    soft_ram_peek(ram, addr) -> Float32

Differentiable RAM read.
"""
function soft_ram_peek(ram::AbstractVector{<:Real}, addr::Real)
    n = length(ram)
    one_hot = zeros(Float32, n)
    one_hot[Int(addr) + 1] = 1f0
    return _dot(one_hot, Float32.(ram))
end

@inline _cart_addr(pc_offset::Real) = Int(pc_offset) & 0x0FFF

"""
    _bus_read(bus, addr) -> Float32

SOFT-mode bus read with cart-vs-RAM decode (13-bit mirror; cart range
\$1000-\$1FFF goes through soft_rom_peek, everything else maps into the
128 B RAM via `addr & 0x7F`). Proper TIA / RIOT register dispatch is
P7c-f.
"""
function _bus_read(bus::SoftBus, addr::Real)
    a_int = Int(addr) & 0x1FFF
    if (a_int & 0x1000) != 0
        return soft_rom_peek(bus.rom, a_int & 0x0FFF)
    else
        return soft_ram_peek(bus.ram, a_int & 0x7F)
    end
end

"""
    _bus_write!(bus, addr, value) -> nothing

SOFT-mode bus write. ROM writes are silently dropped; everything else
mutates `bus.ram[(addr & 0x7F) + 1]`. P7c-f adds real TIA/RIOT/cart
dispatch.
"""
function _bus_write!(bus::SoftBus, addr::Real, value::Real)
    a_int = Int(addr) & 0x1FFF
    if (a_int & 0x1000) == 0
        bus.ram[(a_int & 0x7F) + 1] = Float32(value)
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
# CPY / BIT). Binary-mode ADC/SBC only — BCD is a P7c-bx deferral.
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

function _adc_step!(state::SoftCPUState, bus::SoftBus, operand::Real,
                    instr_len::Real, cycles::Real)
    c_in       = Float32(_read_carry(state.P))
    sum_       = state.A + Float32(operand) + c_in
    new_a      = sum_ - floor(sum_ / 256f0) * 256f0
    carry      = Int(sum_) > 0xFF ? 1 : 0
    a_int      = Int(state.A) & 0xFF
    op_int     = Int(operand) & 0xFF
    new_a_int  = Int(new_a) & 0xFF
    overflow   = ((a_int ⊻ new_a_int) & (op_int ⊻ new_a_int)) & 0x80
    state.A      = new_a
    state.P      = _set_nzcv(state.P, new_a, carry, overflow)
    state.PC    += Float32(instr_len)
    state.cycles += Float32(cycles)
    return nothing
end

function _sbc_step!(state::SoftCPUState, bus::SoftBus, operand::Real,
                    instr_len::Real, cycles::Real)
    op_inv     = 255f0 - Float32(operand)
    c_in       = Float32(_read_carry(state.P))
    sum_       = state.A + op_inv + c_in
    new_a      = sum_ - floor(sum_ / 256f0) * 256f0
    carry      = Int(sum_) > 0xFF ? 1 : 0
    a_int      = Int(state.A) & 0xFF
    op_inv_i   = Int(op_inv) & 0xFF
    new_a_int  = Int(new_a) & 0xFF
    overflow   = ((a_int ⊻ new_a_int) & (op_inv_i ⊻ new_a_int)) & 0x80
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
    _HANDLERS[opcode_int + 1](state, bus)
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
