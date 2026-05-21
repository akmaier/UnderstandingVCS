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

const _NZ_CLEAR_MASK = 0xFF ⊻ (0x80 | 0x02)   # = 0x7D — keep all but N and Z

"""
    _set_nz(p, value) -> Float32

Update N and Z in P from `value` (a single byte). Z = 1 iff value == 0;
N = bit 7.
"""
function _set_nz(p::Real, value::Real)
    v_int = Int(value) & 0xFF
    p_int = Int(p) & 0xFF
    p_int = p_int & _NZ_CLEAR_MASK
    z_bit = v_int == 0 ? 0x02 : 0
    n_bit = v_int & 0x80
    return Float32(p_int | z_bit | n_bit)
end


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
