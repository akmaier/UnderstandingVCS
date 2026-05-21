"""
    CPU

6502 fetch–decode–execute step.

Implemented so far (PORTING_PLAN.md §5):
- P1a:  load (LDA, LDX, LDY), store (STA, STX, STY),
        transfer (TAX, TAY, TXA, TYA, TSX, TXS).
- P1b1: bitwise (AND, ORA, EOR), compare (CMP, CPX, CPY), BIT.
- P1b2: arithmetic (ADC, SBC including BCD/decimal mode) + undocumented
        USBC (0xEB) alias for SBC immediate.
- P1c:  shifts/rotates (ASL, LSR, ROL, ROR) in accumulator + memory modes.
- P1d:  conditional branches (BPL/BMI/BVC/BVS/BCC/BCS/BNE/BEQ), JMP
        (absolute + indirect with page-wrap bug), JSR, RTS. Introduces the
        push8!/pop8! stack helpers reused by P1e and P1f.
- P1e:  stack push/pull (PHA/PHP/PLA/PLP), status-flag setters/clearers
        (SEC/CLC/SEI/CLI/SED/CLD/CLV), and NOP. PHP pushes P with B set;
        PLP forces B and U on pull (xitari/Stella convention).

Pending: P1f (BRK/RTI/IRQ/NMI/RESET, INC/DEC/INX/INY/DEX/DEY, cycle
fine print). Unknown opcodes fall through to the stub
(PC += 1, cycles += base).
"""
module CPU

include("Tables.jl")
include("Addressing.jl")
include("ALU.jl")

using .CPUTables: ADDRESSING_MODE_TABLE, CYCLE_TABLE,
                  ADDR_IMPLIED, ADDR_INDIRECT,
                  FLAG_C, FLAG_N, FLAG_V, FLAG_Z, FLAG_I, FLAG_D, FLAG_B, FLAG_U
using .Addressing: resolve, instruction_length
using .ALU: set_zn!, compare_flags!, bit_flags!, adc!, sbc!,
            asl_op!, lsr_op!, rol_op!, ror_op!
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

    # --- P1c ---------------------------------------------------------------
    # ASL — accumulator + 4 memory modes
    0x0A => :ASL, 0x06 => :ASL, 0x16 => :ASL, 0x0E => :ASL, 0x1E => :ASL,
    # LSR
    0x4A => :LSR, 0x46 => :LSR, 0x56 => :LSR, 0x4E => :LSR, 0x5E => :LSR,
    # ROL
    0x2A => :ROL, 0x26 => :ROL, 0x36 => :ROL, 0x2E => :ROL, 0x3E => :ROL,
    # ROR
    0x6A => :ROR, 0x66 => :ROR, 0x76 => :ROR, 0x6E => :ROR, 0x7E => :ROR,

    # --- P1d ---------------------------------------------------------------
    # Conditional branches
    0x10 => :BPL, 0x30 => :BMI, 0x50 => :BVC, 0x70 => :BVS,
    0x90 => :BCC, 0xB0 => :BCS, 0xD0 => :BNE, 0xF0 => :BEQ,
    # JMP (absolute + indirect with page-wrap bug)
    0x4C => :JMP, 0x6C => :JMP,
    # Subroutine call / return
    0x20 => :JSR, 0x60 => :RTS,

    # --- P1e ---------------------------------------------------------------
    0x48 => :PHA, 0x08 => :PHP,
    0x68 => :PLA, 0x28 => :PLP,
    0x18 => :CLC, 0x38 => :SEC,
    0x58 => :CLI, 0x78 => :SEI,
    0xB8 => :CLV,
    0xD8 => :CLD, 0xF8 => :SED,
    0xEA => :NOP,
)

# Branch opcode → (flag bit, take_when_set)
const _BRANCH_INFO = Dict{UInt8, Tuple{UInt8, Bool}}(
    0x10 => (FLAG_N, false),  # BPL
    0x30 => (FLAG_N, true),   # BMI
    0x50 => (FLAG_V, false),  # BVC
    0x70 => (FLAG_V, true),   # BVS
    0x90 => (FLAG_C, false),  # BCC
    0xB0 => (FLAG_C, true),   # BCS
    0xD0 => (FLAG_Z, false),  # BNE
    0xF0 => (FLAG_Z, true),   # BEQ
)

# Status-flag opcode → (flag bit, set_or_clear). True = set; false = clear.
const _STATUS_OP = Dict{UInt8, Tuple{UInt8, Bool}}(
    0x18 => (FLAG_C, false), 0x38 => (FLAG_C, true),
    0x58 => (FLAG_I, false), 0x78 => (FLAG_I, true),
    0xB8 => (FLAG_V, false),
    0xD8 => (FLAG_D, false), 0xF8 => (FLAG_D, true),
)

@inline _peek(memory::Vector{UInt8}, addr::Integer) =
    memory[(Int(addr) & 0xFFFF) + 1]

# --------------------------------------------------------------------------- #
# Stack helpers — shared by JSR/RTS (P1d), PHA/PLA/PHP/PLP (P1e), and
# BRK/RTI/IRQ/NMI (P1f). Stack lives at $0100 + SP; SP decrements on push.
# --------------------------------------------------------------------------- #

@inline function push8!(state::CPUState, memory::Vector{UInt8}, value::Integer)
    memory[0x0100 + Int(state.SP) + 1] = UInt8(Int(value) & 0xFF)
    state.SP = UInt8((Int(state.SP) - 1) & 0xFF)
    return nothing
end

@inline function pop8!(state::CPUState, memory::Vector{UInt8})
    state.SP = UInt8((Int(state.SP) + 1) & 0xFF)
    return memory[0x0100 + Int(state.SP) + 1]
end

@inline function push16!(state::CPUState, memory::Vector{UInt8}, value::Integer)
    # High byte first so RTS pops low then high.
    push8!(state, memory, (Int(value) >> 8) & 0xFF)
    push8!(state, memory, Int(value) & 0xFF)
    return nothing
end

@inline function pop16!(state::CPUState, memory::Vector{UInt8})
    low = UInt16(pop8!(state, memory))
    high = UInt16(pop8!(state, memory))
    return (high << 8) | low
end

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

    # --- Shifts / rotates -------------------------------------------------
    elseif mnemonic === :ASL || mnemonic === :LSR ||
           mnemonic === :ROL || mnemonic === :ROR
        op = mnemonic === :ASL ? asl_op! :
             mnemonic === :LSR ? lsr_op! :
             mnemonic === :ROL ? rol_op! : ror_op!
        if mode == ADDR_IMPLIED
            # Accumulator-mode: operand and destination are A.
            state.A = op(state, state.A)
            state.PC = UInt16((Int(state.PC) + 1) & 0xFFFF)
        else
            # Memory-mode RMW. Cycle table is already worst-case; no page-cross.
            addr, _ = resolve(mode, state, memory)
            result = op(state, _peek(memory, addr))
            memory[(Int(addr) & 0xFFFF) + 1] = result
            _advance_pc!(state, mode)
        end

    # --- Conditional branches ---------------------------------------------
    elseif mnemonic === :BPL || mnemonic === :BMI ||
           mnemonic === :BVC || mnemonic === :BVS ||
           mnemonic === :BCC || mnemonic === :BCS ||
           mnemonic === :BNE || mnemonic === :BEQ
        flag, take_when_set = _BRANCH_INFO[opcode]
        flag_set = (state.P & flag) != 0
        take = take_when_set ? flag_set : !flag_set
        if take
            target, page = resolve(mode, state, memory)
            state.PC = target
            extra_cycles += 1 + (page ? 1 : 0)
        else
            state.PC = UInt16((Int(state.PC) + 2) & 0xFFFF)
        end

    # --- JMP (absolute and indirect) --------------------------------------
    elseif mnemonic === :JMP
        target, _ = resolve(mode, state, memory)
        state.PC = target

    # --- JSR --------------------------------------------------------------
    elseif mnemonic === :JSR
        target, _ = resolve(mode, state, memory)
        # Push PC + 2 (address of the last byte of the JSR instruction).
        return_addr = UInt16((Int(state.PC) + 2) & 0xFFFF)
        push16!(state, memory, return_addr)
        state.PC = target

    # --- RTS --------------------------------------------------------------
    elseif mnemonic === :RTS
        return_addr = pop16!(state, memory)
        state.PC = UInt16((Int(return_addr) + 1) & 0xFFFF)

    # --- Stack push / pull (P1e) -------------------------------------------
    elseif mnemonic === :PHA
        push8!(state, memory, state.A)
        state.PC = UInt16((Int(state.PC) + 1) & 0xFFFF)
    elseif mnemonic === :PHP
        # PHP always pushes with bit 4 (B) set, matching xitari's "B always
        # true on 6507" convention. Bit 5 (U) is already set as an invariant.
        push8!(state, memory, state.P | FLAG_B | FLAG_U)
        state.PC = UInt16((Int(state.PC) + 1) & 0xFFFF)
    elseif mnemonic === :PLA
        state.A = pop8!(state, memory)
        set_zn!(state, state.A)
        state.PC = UInt16((Int(state.PC) + 1) & 0xFFFF)
    elseif mnemonic === :PLP
        popped = pop8!(state, memory)
        state.P = (popped | FLAG_U | FLAG_B) & UInt8(0xFF)
        state.PC = UInt16((Int(state.PC) + 1) & 0xFFFF)

    # --- Status-flag setters / clearers + NOP ------------------------------
    elseif mnemonic === :CLC || mnemonic === :SEC ||
           mnemonic === :CLI || mnemonic === :SEI ||
           mnemonic === :CLV ||
           mnemonic === :CLD || mnemonic === :SED
        flag, set_it = _STATUS_OP[opcode]
        p = set_it ? (state.P | flag) : (state.P & ~flag)
        state.P = (p | FLAG_U) & UInt8(0xFF)
        state.PC = UInt16((Int(state.PC) + 1) & 0xFFFF)
    elseif mnemonic === :NOP
        state.PC = UInt16((Int(state.PC) + 1) & 0xFFFF)
    end

    state.cycles += UInt64(base_cycles + extra_cycles)
    return state, memory
end

end # module
