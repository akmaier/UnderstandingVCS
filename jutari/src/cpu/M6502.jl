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
- P1f:  INC/DEC memory, INX/INY/DEX/DEY register, BRK (push PC+2, push
        P|B|U, set I, jump via \$FFFE/\$FFFF IRQ vector), RTI (pop P,
        pop PC). Completes the documented NMOS 6502 opcode set.

External hardware interrupts (IRQ / NMI / RESET) are not part of step() —
they require wire-level integration with the bus and land in later phases
(P3 TIA, P6 Console). Unknown opcodes still fall through to the stub
(PC += 1, cycles += base).
"""
module CPU

include("Tables.jl")
include("Addressing.jl")
include("ALU.jl")

using .CPUTables: ADDRESSING_MODE_TABLE, CYCLE_TABLE,
                  ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_Y, ADDR_IMPLIED, ADDR_INDIRECT, ADDR_INDIRECT_Y,
                  FLAG_C, FLAG_N, FLAG_V, FLAG_Z, FLAG_I, FLAG_D, FLAG_B, FLAG_U
using .Addressing: resolve, instruction_length
using .ALU: set_zn!, compare_flags!, bit_flags!, adc!, sbc!,
            asl_op!, lsr_op!, rol_op!, ror_op!
using ..Types: CPUState
# Multiple-dispatch peek / poke! so `step` accepts either a `BusState`
# (proper 6507 bus) or a flat `Vector{UInt8}` (P1-style scratch memory).
using ..Bus: peek, poke!, BusState
# TIA + RIOT timing hooks applied after each instruction when running on a Bus.
using ..TIA: tia_advance!, tia_apply_wsync!
using ..RIOT: riot_advance!

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

    # --- P1f ---------------------------------------------------------------
    # INC memory
    0xE6 => :INC, 0xF6 => :INC, 0xEE => :INC, 0xFE => :INC,
    # DEC memory
    0xC6 => :DEC, 0xD6 => :DEC, 0xCE => :DEC, 0xDE => :DEC,
    # Register inc/dec
    0xE8 => :INX, 0xC8 => :INY,
    0xCA => :DEX, 0x88 => :DEY,
    # Interrupts
    0x00 => :BRK, 0x40 => :RTI,

    # --- P1h — common undocumented NMOS 6502 opcodes -----------------------
    # Mirrors `jaxtari/jaxtari/cpu/m6502.py` P1h block. The well-behaved
    # subset (NOPs, LAX, SAX). Unstable LAX #imm ($AB) and the RMW combos
    # (DCP / ISC / RLA / RRA / SLO / SRE) stay deferred.

    # 1-byte 2-cycle NOPs (implied).
    0x1A => :NOP, 0x3A => :NOP, 0x5A => :NOP, 0x7A => :NOP,
    0xDA => :NOP, 0xFA => :NOP,
    # 2-byte 2-cycle NOPs (immediate — operand read + discarded).
    0x80 => :NOP, 0x82 => :NOP, 0x89 => :NOP, 0xC2 => :NOP, 0xE2 => :NOP,
    # 2-byte zp NOPs.
    0x04 => :NOP, 0x44 => :NOP, 0x64 => :NOP,
    # 2-byte zp,X NOPs.
    0x14 => :NOP, 0x34 => :NOP, 0x54 => :NOP, 0x74 => :NOP,
    0xD4 => :NOP, 0xF4 => :NOP,
    # 3-byte abs NOP.
    0x0C => :NOP,
    # 3-byte abs,X NOPs (+1 cycle if page-crossed, same as documented abs,X).
    0x1C => :NOP, 0x3C => :NOP, 0x5C => :NOP, 0x7C => :NOP,
    0xDC => :NOP, 0xFC => :NOP,

    # LAX — load A and X from the same operand (6 modes).
    0xA7 => :LAX, 0xB7 => :LAX, 0xAF => :LAX, 0xBF => :LAX,
    0xA3 => :LAX, 0xB3 => :LAX,

    # SAX — store A AND X (4 modes), no flag effects.
    0x87 => :SAX, 0x97 => :SAX, 0x8F => :SAX, 0x83 => :SAX,
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

# `_peek` is a local alias for the dispatched `peek` from the Bus module —
# kept so the existing call sites read the same as before this refactor.
const _peek = peek

# --------------------------------------------------------------------------- #
# Stack helpers — shared by JSR/RTS (P1d), PHA/PLA/PHP/PLP (P1e), and
# BRK/RTI/IRQ/NMI (P1f). Stack lives at $0100 + SP; SP decrements on push.
# Memory access goes through `poke!` / `peek` so the same code works against
# either a `BusState` (where stack writes land in RAM via the $0180–$01FF
# mirror) or a flat `Vector{UInt8}` (P1-style scratch memory).
# --------------------------------------------------------------------------- #

@inline function push8!(state::CPUState, memory, value::Integer)
    poke!(memory, 0x0100 + Int(state.SP), value)
    state.SP = UInt8((Int(state.SP) - 1) & 0xFF)
    return nothing
end

# Task #50: STORE / RMW with abs,X / abs,Y / (zp),Y emit an unconditional
# cycle-4 wrong-page dummy peek even when NO page cross. (LOAD only does
# it on cross — handled in the resolvers.) Mirrors jaxtari's
# `_store_rmw_wrong_page_peek`.
@inline function _store_rmw_wrong_page_peek!(memory, mode::Integer, eff::Integer,
                                              page_crossed::Bool)
    page_crossed && return                                  # resolver handled
    if mode != ADDR_ABSOLUTE_X && mode != ADDR_ABSOLUTE_Y && mode != ADDR_INDIRECT_Y
        return                                              # only these modes
    end
    _peek(memory, eff)                                      # unconditional dummy
    return
end

@inline function pop8!(state::CPUState, memory)
    state.SP = UInt8((Int(state.SP) + 1) & 0xFF)
    return peek(memory, 0x0100 + Int(state.SP))
end

@inline function push16!(state::CPUState, memory, value::Integer)
    # High byte first so RTS pops low then high.
    push8!(state, memory, (Int(value) >> 8) & 0xFF)
    push8!(state, memory, Int(value) & 0xFF)
    return nothing
end

@inline function pop16!(state::CPUState, memory)
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
    step(state::CPUState, memory) -> (state, memory)

Execute one 6502 instruction. Mutates `state` (and, where applicable, the
memory backing) in place; returns them for convenience.

`memory` may be either a `BusState` (proper 6507 bus) or a flat
`Vector{UInt8}` (P1-style scratch memory). Multiple-dispatch `peek` /
`poke!` handle the two cases. When `memory` is a `BusState`, the TIA's
scanline / frame counters are advanced by the cycles this instruction
consumed and any WSYNC stall queued by a write to \$02 is resolved
before returning.
"""
function step(state::CPUState, memory)
    pre_cycles = state.cycles
    _step_inner!(state, memory)
    delta = Int(state.cycles - pre_cycles)
    _tia_post_step!(state, memory, delta)
    return state, memory
end

# Multiple-dispatch TIA post-processing — no-op for flat memory, full
# advance + WSYNC handling for a BusState.
@inline _tia_post_step!(::CPUState, ::Vector{UInt8}, ::Integer) = nothing

@inline function _tia_post_step!(state::CPUState, bus::BusState, cycles_consumed::Integer)
    # P3i-g: TIA writes already advanced the TIA inline (in `poke!`) by
    # `bus.tia_advanced_this_instruction` cycles so the register changes
    # landed at the right sub-instruction beam position. Drain whatever
    # remains (trailing cycles after the last TIA write + internal cycles
    # that aren't visible bus ops), so per-instruction TIA advance still
    # equals the full cycle count. RIOT keeps the per-instruction model.
    drain = Int(cycles_consumed) - bus.tia_advanced_this_instruction
    drain < 0 && (drain = 0)                      # defensive — should never happen
    drain > 0 && tia_advance!(bus.tia, drain)
    riot_advance!(bus.riot, cycles_consumed)
    stall = tia_apply_wsync!(bus.tia)
    if stall != 0
        state.cycles += UInt64(stall)
        riot_advance!(bus.riot, stall)
    end
    bus.pending_tia_cycles = 0
    bus.tia_advanced_this_instruction = 0
    return nothing
end

function _step_inner!(state::CPUState, memory)
    pc = Int(state.PC) & 0xFFFF
    opcode = peek(memory, pc)
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
    # Task #50: STA/STX/STY with abs,X / abs,Y / (zp),Y emit an unconditional
    # cycle-4 wrong-page dummy peek (resolver only does it on cross — for
    # stores we ALSO need the no-cross case). Mirrors jaxtari.
    elseif mnemonic === :STA
        addr, page = resolve(mode, state, memory)
        _store_rmw_wrong_page_peek!(memory, mode, addr, page)
        poke!(memory, addr, state.A); _advance_pc!(state, mode)
    elseif mnemonic === :STX
        addr, page = resolve(mode, state, memory)
        _store_rmw_wrong_page_peek!(memory, mode, addr, page)
        poke!(memory, addr, state.X); _advance_pc!(state, mode)
    elseif mnemonic === :STY
        addr, page = resolve(mode, state, memory)
        _store_rmw_wrong_page_peek!(memory, mode, addr, page)
        poke!(memory, addr, state.Y); _advance_pc!(state, mode)

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
            # Memory-mode RMW. Task #50: NMOS 6502 emits the
            # double-write — old value first (dummy/internal cycle),
            # then new value. Plus the unconditional cycle-4 wrong-
            # page peek for abs,X mode (same as STORE).
            addr, page = resolve(mode, state, memory)
            _store_rmw_wrong_page_peek!(memory, mode, addr, page)
            value  = _peek(memory, addr)
            poke!(memory, addr, value)                  # RMW dummy write
            poke!(memory, addr, op(state, value))
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
            # Task #50: real NMOS 6502 branch-taken has a "wasted
            # opcode prefetch" of PC+2 at cycle 3; on page cross,
            # cycle 4 peeks the wrong-page address before the actual
            # corrected fetch. Both dummy peeks update the floating-
            # bus latch.
            pc_plus_2 = UInt16((Int(state.PC) + 2) & 0xFFFF)
            _peek(memory, pc_plus_2)                    # prefetch dummy
            if page
                wrong_addr = UInt16((Int(pc_plus_2) & 0xFF00) | (Int(target) & 0xFF))
                _peek(memory, wrong_addr)               # wrong-page dummy
            end
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
        # Task #50: real NMOS 6502 JSR pre-push internal cycle peeks
        # $0100+SP (the byte about to be overwritten by the PCH push)
        # and discards the result. Visible on the data bus → updates
        # the floating-bus latch.
        target, _ = resolve(mode, state, memory)
        _peek(memory, 0x0100 + Int(state.SP))          # pre-push discard
        # Push PC + 2 (address of the last byte of the JSR instruction).
        return_addr = UInt16((Int(state.PC) + 2) & 0xFFFF)
        push16!(state, memory, return_addr)
        state.PC = target

    # --- RTS --------------------------------------------------------------
    elseif mnemonic === :RTS
        # Task #50: NMOS RTS dummy peeks at cycle 3 (discard $0100+SP
        # pre-pop) and cycle 6 (discard the just-popped PCL before the
        # next instruction). Mirrors jaxtari.
        _peek(memory, 0x0100 + Int(state.SP))              # cycle-3 discard
        return_addr = pop16!(state, memory)
        new_pc = UInt16((Int(return_addr) + 1) & 0xFFFF)
        _peek(memory, UInt16(Int(new_pc) - 1))             # cycle-6 PCL discard
        state.PC = new_pc

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
        # Task #50: NMOS PLA cycle-3 discard read of $0100+SP (pre-pop).
        _peek(memory, 0x0100 + Int(state.SP))              # cycle-3 discard
        state.A = pop8!(state, memory)
        set_zn!(state, state.A)
        state.PC = UInt16((Int(state.PC) + 1) & 0xFFFF)
    elseif mnemonic === :PLP
        _peek(memory, 0x0100 + Int(state.SP))              # cycle-3 discard
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
        # Documented $EA NOP + all P1h undocumented NOPs share this
        # path. INSTRUCTION_LENGTH[mode] advances PC by the right
        # amount (1 for implied, 2 for imm/zp/zp,X, 3 for abs/abs,X).
        # Abs,X NOPs take +1 cycle when the read crosses a page,
        # matching the documented abs,X read penalty.
        if mode == ADDR_ABSOLUTE_X
            _, page = resolve(mode, state, memory)
            page && (extra_cycles += 1)
        end
        _advance_pc!(state, mode)

    # --- LAX (P1h) — load A and X from the same operand; N/Z from value ---
    elseif mnemonic === :LAX
        addr, page = resolve(mode, state, memory)
        value = _peek(memory, addr)
        state.A = value; state.X = value
        set_zn!(state, value)
        _advance_pc!(state, mode); page && (extra_cycles += 1)

    # --- SAX (P1h) — store A AND X, no flag side-effects ----------------
    elseif mnemonic === :SAX
        # Task #50: SAX is a store — emits the unconditional wrong-page
        # peek if mode is abs,X / abs,Y / (zp),Y (none of the documented
        # SAX modes hit those, but kept consistent with STA/STX/STY).
        addr, page = resolve(mode, state, memory)
        _store_rmw_wrong_page_peek!(memory, mode, addr, page)
        poke!(memory, addr, UInt8(Int(state.A) & Int(state.X) & 0xFF))
        _advance_pc!(state, mode)

    # --- INC / DEC memory (P1f) -------------------------------------------
    elseif mnemonic === :INC || mnemonic === :DEC
        # Task #50: NMOS RMW double-write + unconditional cycle-4
        # wrong-page peek for abs,X mode (same as STORE).
        addr, page = resolve(mode, state, memory)
        _store_rmw_wrong_page_peek!(memory, mode, addr, page)
        value = _peek(memory, addr)
        poke!(memory, addr, value)                       # RMW dummy write
        delta = mnemonic === :INC ? 1 : -1
        new_value = UInt8((Int(value) + delta) & 0xFF)
        poke!(memory, addr, new_value)
        set_zn!(state, new_value)
        _advance_pc!(state, mode)

    # --- Register inc/dec (P1f) -------------------------------------------
    elseif mnemonic === :INX
        state.X = UInt8((Int(state.X) + 1) & 0xFF); set_zn!(state, state.X)
        state.PC = UInt16((Int(state.PC) + 1) & 0xFFFF)
    elseif mnemonic === :INY
        state.Y = UInt8((Int(state.Y) + 1) & 0xFF); set_zn!(state, state.Y)
        state.PC = UInt16((Int(state.PC) + 1) & 0xFFFF)
    elseif mnemonic === :DEX
        state.X = UInt8((Int(state.X) - 1) & 0xFF); set_zn!(state, state.X)
        state.PC = UInt16((Int(state.PC) + 1) & 0xFFFF)
    elseif mnemonic === :DEY
        state.Y = UInt8((Int(state.Y) - 1) & 0xFF); set_zn!(state, state.Y)
        state.PC = UInt16((Int(state.PC) + 1) & 0xFFFF)

    # --- BRK / RTI (P1f) --------------------------------------------------
    elseif mnemonic === :BRK
        # Push PC + 2 (skipping the BRK signature byte), then P|B|U; set I;
        # load PC from the IRQ vector at $FFFE/$FFFF.
        #
        # Task #50: NMOS BRK cycle 2 is a discard read of PC+1 (the
        # padding byte). Visible on the data bus → updates the
        # floating-bus latch.
        _peek(memory, UInt16((Int(state.PC) + 1) & 0xFFFF))   # cycle-2 discard
        return_addr = UInt16((Int(state.PC) + 2) & 0xFFFF)
        push16!(state, memory, return_addr)
        push8!(state, memory, state.P | FLAG_B | FLAG_U)
        state.P = (state.P | FLAG_I | FLAG_U) & UInt8(0xFF)
        lo = UInt16(_peek(memory, 0xFFFE))
        hi = UInt16(_peek(memory, 0xFFFF))
        state.PC = (hi << 8) | lo
    elseif mnemonic === :RTI
        # Task #50: NMOS RTI cycle-3 discard read of $0100+SP (pre-pop).
        _peek(memory, 0x0100 + Int(state.SP))              # cycle-3 discard
        popped_p = pop8!(state, memory)
        popped_pc = pop16!(state, memory)
        state.P = (popped_p | FLAG_U | FLAG_B) & UInt8(0xFF)
        state.PC = popped_pc
    end

    state.cycles += UInt64(base_cycles + extra_cycles)
    return state, memory
end

end # module
