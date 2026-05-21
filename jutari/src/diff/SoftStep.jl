# SOFT-mode `step()` — a differentiable parallel to `CPU.step`.
#
# `soft_step!(state, bus)` executes one 6502 instruction with the SOFT
# primitives from `JuTari.Diff`: memory access goes through
# `soft_rom_peek` (one-hot dot product on the ROM, gradient-friendly);
# register state is Float32; opcode dispatch is a Julia branch table
# over a constant `_HANDLERS` vector. The HARD `step` is left untouched
# — SOFT mode is a parallel execution path.
#
# Opcode coverage in P7b (subset to prove gradient flow):
#
#   NOP            ($EA)  no effect, PC++, cycles+=2
#   LDA #imm       ($A9)  A = peek(PC+1)
#   LDA $zp        ($A5)  A = peek(peek(PC+1))                   ← double indirection
#   LDX #imm       ($A2)  X = peek(PC+1)
#   STA $zp        ($85)  RAM[peek(PC+1)] = A
#   STX $zp        ($86)  RAM[peek(PC+1)] = X
#   JMP $abs       ($4C)  PC = peek(PC+1) | (peek(PC+2) << 8)
#   BRK            ($00)  used as an "end of trace" sentinel — halts in place
#
# All other opcodes fall through to `_branch_default` which advances
# PC by 1 and bumps cycles. This is intentionally lenient — a real ROM
# that hits an unhandled opcode will produce wrong forward behaviour
# but **will not** raise during a Zygote pullback, so the function
# remains differentiable.
#
# What this implementation does NOT (yet) do:
#   - Status flag updates (N/Z/C/V) — registers move, flags stay 0
#   - TIA / RIOT I/O writes — STA $0xxx etc. silently drop into the
#     RAM region (no side-effects on TIA registers or framebuffer)
#   - Stack push/pop, JSR/RTS, BRK/RTI proper interrupt sequence
#   - Bank switching for F8/F6/F4 carts
#   - ChainRulesCore rrules — present for forward only; Zygote.jl
#     hookup is the next milestone


# --------------------------------------------------------------------------- #
# Differentiable memory access
# --------------------------------------------------------------------------- #

"""
    soft_rom_peek(rom, addr) -> Float32

Differentiable cart byte read — `one_hot(addr) · rom`. `rom` is any
1-D numeric vector; `addr` is a scalar (int or float) treated as the
offset *inside the ROM* (the 13-bit address mirror is applied at the
call site, not here).
"""
function soft_rom_peek(rom::AbstractVector{<:Real}, addr::Real)
    n = length(rom)
    one_hot = zeros(Float32, n)
    one_hot[Int(addr) + 1] = 1f0
    return _dot(one_hot, Float32.(rom))
end

"""
    soft_ram_peek(ram, addr) -> Float32

Differentiable RAM read — same one-hot trick, on the 128-byte RAM.
"""
function soft_ram_peek(ram::AbstractVector{<:Real}, addr::Real)
    n = length(ram)
    one_hot = zeros(Float32, n)
    one_hot[Int(addr) + 1] = 1f0
    return _dot(one_hot, Float32.(ram))
end

# Convert a CPU bus address to a 0..4095 ROM offset. Real bus does the
# full 13-bit mirror + cart-window check; for the SOFT subset we assume
# the program lives in the cart region and do the simpler `addr & 0x0FFF`.
@inline _cart_addr(pc_offset::Real) = Int(pc_offset) & 0x0FFF


# --------------------------------------------------------------------------- #
# Per-opcode branch handlers
# --------------------------------------------------------------------------- #
# Each handler takes `(state, bus)` and mutates both in place. The
# dispatch table calls one handler per `soft_step!` invocation.

# Unhandled opcode: advance PC by 1, bump cycles. Doesn't raise so the
# trace stays differentiable; forward result will be wrong if a real
# ROM hits this.
function _branch_default!(state::SoftCPUState, bus::SoftBus)
    state.PC     += 1f0
    state.cycles += 2f0
    return nothing
end

# End-of-trace sentinel — halt in place. PC doesn't advance so a
# run-until-BRK loop terminates by hitting the same instruction twice.
function _branch_brk!(state::SoftCPUState, bus::SoftBus)
    state.cycles += 7f0
    return nothing
end

function _branch_nop!(state::SoftCPUState, bus::SoftBus)
    state.PC     += 1f0
    state.cycles += 2f0
    return nothing
end

# LDA #imm: A = ROM[PC + 1]
function _branch_lda_imm!(state::SoftCPUState, bus::SoftBus)
    rom_off = _cart_addr(state.PC + 1f0)
    state.A      = soft_rom_peek(bus.rom, rom_off)
    state.PC    += 2f0
    state.cycles += 2f0
    return nothing
end

# LDA $zp: zp = ROM[PC+1]; A = RAM[zp]. The zp address itself rides
# through `soft_ram_peek`'s one-hot, so a gradient at the output flows
# back to ROM[PC+1] **and** to RAM[zp].
function _branch_lda_zp!(state::SoftCPUState, bus::SoftBus)
    rom_off = _cart_addr(state.PC + 1f0)
    zp_float = soft_rom_peek(bus.rom, rom_off)
    zp_int   = Int(zp_float) & 0x7F
    state.A      = soft_ram_peek(bus.ram, zp_int)
    state.PC    += 2f0
    state.cycles += 3f0
    return nothing
end

function _branch_ldx_imm!(state::SoftCPUState, bus::SoftBus)
    rom_off = _cart_addr(state.PC + 1f0)
    state.X      = soft_rom_peek(bus.rom, rom_off)
    state.PC    += 2f0
    state.cycles += 2f0
    return nothing
end

# STA $zp: zp = ROM[PC+1]; RAM[zp] = A
function _branch_sta_zp!(state::SoftCPUState, bus::SoftBus)
    rom_off = _cart_addr(state.PC + 1f0)
    zp_float = soft_rom_peek(bus.rom, rom_off)
    zp_int   = Int(zp_float) & 0x7F
    bus.ram[zp_int + 1] = state.A
    state.PC    += 2f0
    state.cycles += 3f0
    return nothing
end

function _branch_stx_zp!(state::SoftCPUState, bus::SoftBus)
    rom_off = _cart_addr(state.PC + 1f0)
    zp_float = soft_rom_peek(bus.rom, rom_off)
    zp_int   = Int(zp_float) & 0x7F
    bus.ram[zp_int + 1] = state.X
    state.PC    += 2f0
    state.cycles += 3f0
    return nothing
end

# JMP $abs: PC = ROM[PC+1] | (ROM[PC+2] << 8)
function _branch_jmp_abs!(state::SoftCPUState, bus::SoftBus)
    rom_lo = _cart_addr(state.PC + 1f0)
    rom_hi = _cart_addr(state.PC + 2f0)
    lo = soft_rom_peek(bus.rom, rom_lo)
    hi = soft_rom_peek(bus.rom, rom_hi)
    state.PC     = lo + hi * 256f0
    state.cycles += 3f0
    return nothing
end


# --------------------------------------------------------------------------- #
# Dispatch table
# --------------------------------------------------------------------------- #

const _HANDLERS = let
    h = Function[_branch_default! for _ in 1:256]
    h[0x00 + 1] = _branch_brk!
    h[0xEA + 1] = _branch_nop!
    h[0xA9 + 1] = _branch_lda_imm!
    h[0xA5 + 1] = _branch_lda_zp!
    h[0xA2 + 1] = _branch_ldx_imm!
    h[0x85 + 1] = _branch_sta_zp!
    h[0x86 + 1] = _branch_stx_zp!
    h[0x4C + 1] = _branch_jmp_abs!
    h
end

# Opcodes handled with full behaviour (rather than default). Exposed so
# tests + a future P7c can introspect coverage.
const SOFT_SUPPORTED_OPCODES =
    Set{UInt8}([0x00, 0xEA, 0xA9, 0xA5, 0xA2, 0x85, 0x86, 0x4C])


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

Execute exactly `n_steps` instructions in sequence. Useful when the
trace length is known statically, which is the common XAI case ("run
1000 instructions and tell me which ROM bytes affected the output").
"""
function soft_run!(state::SoftCPUState, bus::SoftBus, n_steps::Integer)
    for _ in 1:n_steps
        soft_step!(state, bus)
    end
    return nothing
end
