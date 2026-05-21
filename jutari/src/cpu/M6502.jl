"""
    CPU

6502 fetch–decode–execute step.

Phase P1a status: load / store / transfer opcodes are implemented
(LDA, LDX, LDY, STA, STX, STY, TAX, TAY, TXA, TYA, TSX, TXS — 37 opcodes
total). All other opcodes fall through to the P0 stub that advances PC by 1
and bumps `cycles` by the base count. P1b–P1f will fill the rest.
"""
module CPU

include("Tables.jl")
include("Addressing.jl")
include("ALU.jl")

using .CPUTables: ADDRESSING_MODE_TABLE, CYCLE_TABLE
using .Addressing: resolve, instruction_length
using .ALU: set_zn!
using ..Types: CPUState

export step

# Opcode → P1a mnemonic. Anything not here falls through to the stub path.
const P1A_OPCODES = Dict{UInt8, Symbol}(
    # LDA
    0xA9 => :LDA, 0xA5 => :LDA, 0xB5 => :LDA, 0xAD => :LDA,
    0xBD => :LDA, 0xB9 => :LDA, 0xA1 => :LDA, 0xB1 => :LDA,
    # LDX
    0xA2 => :LDX, 0xA6 => :LDX, 0xB6 => :LDX, 0xAE => :LDX, 0xBE => :LDX,
    # LDY
    0xA0 => :LDY, 0xA4 => :LDY, 0xB4 => :LDY, 0xAC => :LDY, 0xBC => :LDY,
    # STA
    0x85 => :STA, 0x95 => :STA, 0x8D => :STA,
    0x9D => :STA, 0x99 => :STA, 0x81 => :STA, 0x91 => :STA,
    # STX
    0x86 => :STX, 0x96 => :STX, 0x8E => :STX,
    # STY
    0x84 => :STY, 0x94 => :STY, 0x8C => :STY,
    # Transfers (implied, 1 byte, 2 cycles)
    0xAA => :TAX, 0xA8 => :TAY,
    0x8A => :TXA, 0x98 => :TYA,
    0xBA => :TSX, 0x9A => :TXS,
)

@inline _peek(memory::Vector{UInt8}, addr::Integer) =
    memory[(Int(addr) & 0xFFFF) + 1]

@inline function _stub_advance!(state::CPUState, base_cycles::Integer)
    state.PC = UInt16((Int(state.PC) + 1) & 0xFFFF)
    state.cycles += UInt64(base_cycles)
    return nothing
end

@inline function _advance_pc!(state::CPUState, mode::UInt8)
    state.PC = UInt16((Int(state.PC) + instruction_length(mode)) & 0xFFFF)
    return nothing
end

"""
    step(state::CPUState, memory::Vector{UInt8}) -> (state, memory)

Execute one 6502 instruction. Mutates `state` and `memory` in place; returns
them for convenience.
"""
function step(state::CPUState, memory::Vector{UInt8})
    pc = Int(state.PC) & 0xFFFF
    opcode = memory[pc + 1]
    mode = ADDRESSING_MODE_TABLE[Int(opcode) + 1]
    base_cycles = Int(CYCLE_TABLE[Int(opcode) + 1])

    mnemonic = get(P1A_OPCODES, opcode, nothing)
    if mnemonic === nothing
        _stub_advance!(state, base_cycles)
        return state, memory
    end

    extra_cycles = 0

    # --- Transfers (implied) ----------------------------------------------
    if mnemonic === :TAX
        state.X = state.A; set_zn!(state, state.X); _advance_pc!(state, mode)
    elseif mnemonic === :TAY
        state.Y = state.A; set_zn!(state, state.Y); _advance_pc!(state, mode)
    elseif mnemonic === :TXA
        state.A = state.X; set_zn!(state, state.A); _advance_pc!(state, mode)
    elseif mnemonic === :TYA
        state.A = state.Y; set_zn!(state, state.A); _advance_pc!(state, mode)
    elseif mnemonic === :TSX
        state.X = state.SP; set_zn!(state, state.X); _advance_pc!(state, mode)
    elseif mnemonic === :TXS
        # TXS is the only transfer that does NOT touch flags.
        state.SP = state.X; _advance_pc!(state, mode)

    # --- Loads ------------------------------------------------------------
    elseif mnemonic === :LDA
        addr, page = resolve(mode, state, memory)
        state.A = _peek(memory, addr); set_zn!(state, state.A)
        _advance_pc!(state, mode); page && (extra_cycles += 1)
    elseif mnemonic === :LDX
        addr, page = resolve(mode, state, memory)
        state.X = _peek(memory, addr); set_zn!(state, state.X)
        _advance_pc!(state, mode); page && (extra_cycles += 1)
    elseif mnemonic === :LDY
        addr, page = resolve(mode, state, memory)
        state.Y = _peek(memory, addr); set_zn!(state, state.Y)
        _advance_pc!(state, mode); page && (extra_cycles += 1)

    # --- Stores (no page-cross penalty) -----------------------------------
    elseif mnemonic === :STA
        addr, _ = resolve(mode, state, memory)
        memory[(Int(addr) & 0xFFFF) + 1] = state.A; _advance_pc!(state, mode)
    elseif mnemonic === :STX
        addr, _ = resolve(mode, state, memory)
        memory[(Int(addr) & 0xFFFF) + 1] = state.X; _advance_pc!(state, mode)
    elseif mnemonic === :STY
        addr, _ = resolve(mode, state, memory)
        memory[(Int(addr) & 0xFFFF) + 1] = state.Y; _advance_pc!(state, mode)
    end

    state.cycles += UInt64(base_cycles + extra_cycles)
    return state, memory
end

end # module
