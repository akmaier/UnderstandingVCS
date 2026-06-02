# Phase 2b — scratch audit of `bus.pending_tia_cycles` (the per-bus-op +
# pending_tick! counter) against `CYCLE_TABLE`. This is what the threading
# in `_bus_poke!` reads to decide how much to flush. If it doesn't sum to
# CYCLE_TABLE at end-of-instruction, the threading is wrong.
#
# Run with:
#   julia --project=jutari jutari/test/scratch_cycle_audit.jl

using Printf
using JuTari
using JuTari.Types: CPUState, initial_cpu_state
using JuTari.Bus: initial_bus
using JuTari.CPU: OPCODES                    # OPCODES lives at JuTari.CPU
using JuTari.CPU.CPUTables: CYCLE_TABLE, ADDRESSING_MODE_TABLE,
                             ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_Y, ADDR_INDIRECT_Y,
                             ADDR_RELATIVE

# `_step_inner!` is NOT exported but we can reach it via fully-qualified name.
const _STEP_INNER = JuTari.CPU._step_inner!

const _BRANCH_OPS = Dict{UInt8, Tuple{Symbol, Bool}}(
    # opcode → (flag_name, take_when_set)
    0x10 => (:N, false), 0x30 => (:N, true),   # BPL / BMI
    0x50 => (:V, false), 0x70 => (:V, true),   # BVC / BVS
    0x90 => (:C, false), 0xB0 => (:C, true),   # BCC / BCS
    0xD0 => (:Z, false), 0xF0 => (:Z, true),   # BNE / BEQ
)

# Build a 4K ROM with the test opcode at offset 0 ($F000) and benign
# operand bytes (0x00) at offsets 1, 2. RESET vector at $FFFC = $F000.
function _make_bus(opcode::UInt8, op1::UInt8 = 0x00, op2::UInt8 = 0x00)
    rom = zeros(UInt8, 4096)
    rom[1] = opcode                  # $F000
    rom[2] = op1                     # $F001
    rom[3] = op2                     # $F002
    rom[0xFFFC - 0xF000 + 1] = 0x00  # reset vector lo
    rom[0xFFFD - 0xF000 + 1] = 0xF0  # reset vector hi
    return initial_bus(rom)
end

# CPU state with PC=$F000. Set P so EVERY conditional branch is NOT
# taken — set all the test flags (N=V=C=Z=1) and pick BPL/BVC/BCC/BNE
# variants accordingly. Branches taken add +1 cycle, which would mask
# the "base CYCLE_TABLE" test.
function _make_cpu(opcode::UInt8)
    s = initial_cpu_state()
    s.PC = 0xF000
    s.SP = 0xFD
    s.cycles = UInt64(0)
    if haskey(_BRANCH_OPS, opcode)
        flag, take_when_set = _BRANCH_OPS[opcode]
        # Pick P to make the branch NOT take.
        #   If `take_when_set`, branch takes when flag SET — clear all flags.
        #   If !take_when_set, branch takes when flag CLEAR — set all flags.
        s.P = take_when_set ? UInt8(0x24) : UInt8(0xE7)
    else
        s.P = UInt8(0x24)
    end
    return s
end

function audit()
    mismatches = Tuple{UInt8, Symbol, Int, Int, UInt8}[]
    for (opcode, mnemonic) in OPCODES
        bus = _make_bus(opcode)
        s   = _make_cpu(opcode)
        try
            _STEP_INNER(s, bus)
        catch e
            push!(mismatches, (opcode, mnemonic,
                               Int(CYCLE_TABLE[Int(opcode)+1]), -1,
                               ADDRESSING_MODE_TABLE[Int(opcode)+1]))
            continue
        end
        # Mid-instruction counter (NOT yet reset by _tia_post_step!).
        consumed = bus.pending_tia_cycles
        expected = Int(CYCLE_TABLE[Int(opcode) + 1])
        if consumed != expected
            push!(mismatches, (opcode, mnemonic, expected, consumed,
                               ADDRESSING_MODE_TABLE[Int(opcode)+1]))
        end
    end
    return mismatches
end

mismatches = audit()
if isempty(mismatches)
    println("ALL OK: bus.pending_tia_cycles == CYCLE_TABLE for all $(length(OPCODES)) opcodes.")
else
    println("MISMATCHES: $(length(mismatches)) of $(length(OPCODES)) opcodes diverge.\n")
    println("opcode  mnemonic  expected  actual  addressing_mode")
    println("------  --------  --------  ------  ---------------")
    for (op, mn, exp, act, mode) in sort(mismatches, by = t -> t[1])
        @printf("  0x%02X  %-7s   %-7d  %-6s  %d\n",
                op, String(mn), exp,
                act == -1 ? "EX" : string(act), mode)
    end
end
