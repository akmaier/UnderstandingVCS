"""
    CPU

6502 fetch–decode–execute step.

Implemented so far (PORTING_PLAN.md §5):
- P1a:  load (LDA, LDX, LDY), store (STA, STX, STY),
        transfer (TAX, TAY, TXA, TYA, TSX, TXS).
- P1b1: bitwise (AND, ORA, EOR), compare (CMP, CPX, CPY), BIT.
- P1b2: arithmetic (ADC, SBC including BCD/decimal mode) + undocumented
        USBC (0xEB) alias for SBC immediate.

Pending: P1c (ASL/LSR/ROL/ROR), P1d (branches + JMP/JSR/RTS), P1e (stack +
status flags), P1f (BRK/RTI/IRQ/NMI/RESET + cycle fine print). Unknown
opcodes fall through to the stub (PC += 1, cycles += base).
"""
module CPU

include("Tables.jl")
include("Addressing.jl")
include("ALU.jl")

using .CPUTables: ADDRESSING_MODE_TABLE, CYCLE_TABLE
using .Addressing: resolve, instruction_length
using .ALU: set_zn!, compare_flags!, bit_flags!, adc!, sbc!
using ..Types: CPUState

export step

# Opcode → mnemonic. Anything not in this dict falls through to the stub.
const OPCODES = Dict{UInt8, Symbol}(
    # --- P1a ---------------------------------------------------------------
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
    # STX / STY
    0x86 => :STX, 0x96 => :STX, 0x8E => :STX,
    0x84 => :STY, 0x94 => :STY, 0x8C => :STY,
    # Transfers
    0xAA => :TAX, 0xA8 => :TAY, 0x8A => :TXA, 0x98 => :TYA, 0xBA => :TSX, 0x9A => :TXS,

    # --- P1b1 --------------------------------------------------------------
    # AND
    0x29 => :AND, 0x25 => :AND, 0x35 => :AND, 0x2D => :AND,
    0x3D => :AND, 0x39 => :AND, 0x21 => :AND, 0x31 => :AND,
    # ORA
    0x09 => :ORA, 0x05 => :ORA, 0x15 => :ORA, 0x0D => :ORA,
    0x1D => :ORA, 0x19 => :ORA, 0x01 => :ORA, 0x11 => :ORA,
    # EOR
    0x49 => :EOR, 0x45 => :EOR, 0x55 => :EOR, 0x4D => :EOR,
    0x5D => :EOR, 0x59 => :EOR, 0x41 => :EOR, 0x51 => :EOR,
    # CMP
    0xC9 => :CMP, 0xC5 => :CMP, 0xD5 => :CMP, 0xCD => :CMP,
    0xDD => :CMP, 0xD9 => :CMP, 0xC1 => :CMP, 0xD1 => :CMP,
    # CPX / CPY
    0xE0 => :CPX, 0xE4 => :CPX, 0xEC => :CPX,
    0xC0 => :CPY, 0xC4 => :CPY, 0xCC => :CPY,
    # BIT
    0x24 => :BIT, 0x2C => :BIT,

    # --- P1b2 --------------------------------------------------------------
    # ADC
    0x69 => :ADC, 0x65 => :ADC, 0x75 => :ADC, 0x6D => :ADC,
    0x7D => :ADC, 0x79 => :ADC, 0x61 => :ADC, 0x71 => :ADC,
    # SBC + undocumented USBC (0xEB)
    0xE9 => :SBC, 0xE5 => :SBC, 0xF5 => :SBC, 0xED => :SBC,
    0xFD => :SBC, 0xF9 => :SBC, 0xE1 => :SBC, 0xF1 => :SBC,
    0xEB => :SBC,
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

    mnemonic = get(OPCODES, opcode, nothing)
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

    # --- Bitwise A-ops ----------------------------------------------------
    elseif mnemonic === :AND
        addr, page = resolve(mode, state, memory)
        state.A = state.A & _peek(memory, addr); set_zn!(state, state.A)
        _advance_pc!(state, mode); page && (extra_cycles += 1)
    elseif mnemonic === :ORA
        addr, page = resolve(mode, state, memory)
        state.A = state.A | _peek(memory, addr); set_zn!(state, state.A)
        _advance_pc!(state, mode); page && (extra_cycles += 1)
    elseif mnemonic === :EOR
        addr, page = resolve(mode, state, memory)
        state.A = state.A ⊻ _peek(memory, addr); set_zn!(state, state.A)
        _advance_pc!(state, mode); page && (extra_cycles += 1)

    # --- Compares ---------------------------------------------------------
    elseif mnemonic === :CMP
        addr, page = resolve(mode, state, memory)
        compare_flags!(state, state.A, _peek(memory, addr))
        _advance_pc!(state, mode); page && (extra_cycles += 1)
    elseif mnemonic === :CPX
        addr, _ = resolve(mode, state, memory)
        compare_flags!(state, state.X, _peek(memory, addr))
        _advance_pc!(state, mode)
    elseif mnemonic === :CPY
        addr, _ = resolve(mode, state, memory)
        compare_flags!(state, state.Y, _peek(memory, addr))
        _advance_pc!(state, mode)

    # --- BIT --------------------------------------------------------------
    elseif mnemonic === :BIT
        addr, _ = resolve(mode, state, memory)
        bit_flags!(state, state.A, _peek(memory, addr))
        _advance_pc!(state, mode)

    # --- ADC / SBC --------------------------------------------------------
    elseif mnemonic === :ADC
        addr, page = resolve(mode, state, memory)
        adc!(state, _peek(memory, addr))
        _advance_pc!(state, mode); page && (extra_cycles += 1)
    elseif mnemonic === :SBC
        addr, page = resolve(mode, state, memory)
        sbc!(state, _peek(memory, addr))
        _advance_pc!(state, mode); page && (extra_cycles += 1)
    end

    state.cycles += UInt64(base_cycles + extra_cycles)
    return state, memory
end

end # module
