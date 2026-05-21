"""
    Addressing

Effective-address resolution for the 6502's addressing modes. Mirrors
`jaxtari/jaxtari/cpu/addressing.py`.

Each `resolve_*` returns `(effective_addr::UInt16, page_crossed::Bool)`.
`page_crossed` only matters for absolute,X / absolute,Y / (indirect),Y modes;
loads add +1 cycle on a crossing, stores do not.
"""
module Addressing

using ..CPUTables: ADDR_IMMEDIATE, ADDR_ZERO, ADDR_ZERO_X, ADDR_ZERO_Y,
                   ADDR_ABSOLUTE, ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_Y,
                   ADDR_INDIRECT, ADDR_INDIRECT_X, ADDR_INDIRECT_Y,
                   ADDR_RELATIVE, ADDR_IMPLIED
using ..Types: CPUState

export resolve, instruction_length

@inline _peek(memory::Vector{UInt8}, addr::Integer) =
    memory[(Int(addr) & 0xFFFF) + 1]

@inline function _peek16(memory::Vector{UInt8}, addr::Integer)
    lo = UInt16(_peek(memory, addr))
    hi = UInt16(_peek(memory, addr + 1))
    return (hi << 8) | lo
end

@inline function _operand_addr(state::CPUState)
    return UInt16((Int(state.PC) + 1) & 0xFFFF)
end

function resolve_immediate(state::CPUState, memory::Vector{UInt8})
    return _operand_addr(state), false
end

function resolve_zero(state::CPUState, memory::Vector{UInt8})
    return UInt16(_peek(memory, _operand_addr(state))), false
end

function resolve_zero_x(state::CPUState, memory::Vector{UInt8})
    op = _peek(memory, _operand_addr(state))
    return UInt16((Int(op) + Int(state.X)) & 0xFF), false
end

function resolve_zero_y(state::CPUState, memory::Vector{UInt8})
    op = _peek(memory, _operand_addr(state))
    return UInt16((Int(op) + Int(state.Y)) & 0xFF), false
end

function resolve_absolute(state::CPUState, memory::Vector{UInt8})
    return _peek16(memory, _operand_addr(state)), false
end

function resolve_absolute_x(state::CPUState, memory::Vector{UInt8})
    base = _peek16(memory, _operand_addr(state))
    eff  = UInt16((Int(base) + Int(state.X)) & 0xFFFF)
    return eff, (Int(base) & 0xFF00) != (Int(eff) & 0xFF00)
end

function resolve_absolute_y(state::CPUState, memory::Vector{UInt8})
    base = _peek16(memory, _operand_addr(state))
    eff  = UInt16((Int(base) + Int(state.Y)) & 0xFFFF)
    return eff, (Int(base) & 0xFF00) != (Int(eff) & 0xFF00)
end

function resolve_indirect_x(state::CPUState, memory::Vector{UInt8})
    zp = UInt8((Int(_peek(memory, _operand_addr(state))) + Int(state.X)) & 0xFF)
    lo = UInt16(_peek(memory, zp))
    hi = UInt16(_peek(memory, UInt8((Int(zp) + 1) & 0xFF)))
    return (hi << 8) | lo, false
end

function resolve_indirect_y(state::CPUState, memory::Vector{UInt8})
    zp   = _peek(memory, _operand_addr(state))
    lo   = UInt16(_peek(memory, zp))
    hi   = UInt16(_peek(memory, UInt8((Int(zp) + 1) & 0xFF)))
    base = (hi << 8) | lo
    eff  = UInt16((Int(base) + Int(state.Y)) & 0xFFFF)
    return eff, (Int(base) & 0xFF00) != (Int(eff) & 0xFF00)
end

function resolve_relative(state::CPUState, memory::Vector{UInt8})
    offset = Int(_peek(memory, _operand_addr(state)))
    offset >= 0x80 && (offset -= 0x100)
    base = UInt16((Int(state.PC) + 2) & 0xFFFF)
    eff  = UInt16((Int(base) + offset) & 0xFFFF)
    return eff, (Int(base) & 0xFF00) != (Int(eff) & 0xFF00)
end

function resolve_indirect(state::CPUState, memory::Vector{UInt8})
    """JMP indirect with the 6502 page-wrap bug at \$xxFF."""
    ptr = _peek16(memory, _operand_addr(state))
    lo  = UInt16(_peek(memory, ptr))
    hi  = UInt16(_peek(memory, UInt16((Int(ptr) & 0xFF00) | ((Int(ptr) + 1) & 0xFF))))
    return (hi << 8) | lo, false
end

"""
    resolve(mode, state, memory) -> (UInt16, Bool)

Dispatch to the right `resolve_*` function for the given addressing-mode code.
"""
function resolve(mode::UInt8, state::CPUState, memory::Vector{UInt8})
    if mode == ADDR_IMMEDIATE;    return resolve_immediate(state, memory)
    elseif mode == ADDR_ZERO;     return resolve_zero(state, memory)
    elseif mode == ADDR_ZERO_X;   return resolve_zero_x(state, memory)
    elseif mode == ADDR_ZERO_Y;   return resolve_zero_y(state, memory)
    elseif mode == ADDR_ABSOLUTE; return resolve_absolute(state, memory)
    elseif mode == ADDR_ABSOLUTE_X; return resolve_absolute_x(state, memory)
    elseif mode == ADDR_ABSOLUTE_Y; return resolve_absolute_y(state, memory)
    elseif mode == ADDR_INDIRECT;   return resolve_indirect(state, memory)
    elseif mode == ADDR_INDIRECT_X; return resolve_indirect_x(state, memory)
    elseif mode == ADDR_INDIRECT_Y; return resolve_indirect_y(state, memory)
    elseif mode == ADDR_RELATIVE;   return resolve_relative(state, memory)
    else
        error("Addressing.resolve: implied/unknown mode $(mode)")
    end
end

"""
    instruction_length(mode) -> Int

Total instruction byte length for a given addressing mode.
"""
function instruction_length(mode::UInt8)
    if mode == ADDR_IMPLIED;      return 1
    elseif mode == ADDR_IMMEDIATE; return 2
    elseif mode == ADDR_ZERO || mode == ADDR_ZERO_X || mode == ADDR_ZERO_Y ||
           mode == ADDR_INDIRECT_X || mode == ADDR_INDIRECT_Y || mode == ADDR_RELATIVE
        return 2
    elseif mode == ADDR_ABSOLUTE || mode == ADDR_ABSOLUTE_X ||
           mode == ADDR_ABSOLUTE_Y || mode == ADDR_INDIRECT
        return 3
    else
        error("Addressing.instruction_length: unknown mode $(mode)")
    end
end

end # module
