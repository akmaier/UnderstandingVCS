module JuTari
    """
    JuTari: Differentiable Atari 2600 Emulator in Julia
    Phase 1: CPU Core (M6502)
    """

    export CPUState, create_initial_state, step_cpu, run_cpu

    const RAM_SIZE = 128
    const STACK_PAGE = 0x0100
    const RESET_VECTOR = 0xFFFC

    # Processor Status flags
    const FLAG_N = 0x80  # Negative
    const FLAG_V = 0x40  # Overflow
    const FLAG_B = 0x10  # Break
    const FLAG_D = 0x08  # Decimal
    const FLAG_I = 0x04  # Interrupt disable
    const FLAG_Z = 0x02  # Zero
    const FLAG_C = 0x01  # Carry

    """
    Mutable CPU state for standard operation
    """
    mutable struct CPUState
        A::UInt8      # Accumulator
        X::UInt8      # X index register
        Y::UInt8      # Y index register
        SP::UInt8     # Stack pointer
        PC::UInt16    # Program counter
        PS::UInt8     # Processor status register
        cycles::UInt32 # Cycle counter
    end

    """
    Create initial CPU state after RESET
    """
    function create_initial_state()
        CPUState(0x00, 0x00, 0x00, 0xFD, 0x0000, 0x34, 0)
    end

    # Addressing modes
    const ADDR_IMPLIED = 0
    const ADDR_IMMEDIATE = 1
    const ADDR_ZERO = 2
    const ADDR_ZERO_X = 3
    const ADDR_ZERO_Y = 4
    const ADDR_ABSOLUTE = 5
    const ADDR_ABSOLUTE_X = 6
    const ADDR_ABSOLUTE_Y = 7
    const ADDR_INDIRECT = 8
    const ADDR_INDIRECT_X = 9
    const ADDR_INDIRECT_Y = 10
    const ADDR_RELATIVE = 11

    # Addressing mode lookup table
    const ADDRESSING_MODE_TABLE = UInt8[
        ADDR_IMPLIED, ADDR_INDIRECT_X, ADDR_IMPLIED, ADDR_INDIRECT_X,  # 0x0?
        ADDR_ZERO, ADDR_ZERO, ADDR_ZERO, ADDR_ZERO,
        ADDR_IMPLIED, ADDR_IMMEDIATE, ADDR_IMPLIED, ADDR_IMMEDIATE,
        ADDR_ABSOLUTE, ADDR_ABSOLUTE, ADDR_ABSOLUTE, ADDR_ABSOLUTE,
        
        ADDR_RELATIVE, ADDR_INDIRECT_Y, ADDR_IMPLIED, ADDR_INDIRECT_Y,  # 0x1?
        ADDR_ZERO_X, ADDR_ZERO_X, ADDR_ZERO_X, ADDR_ZERO_X,
        ADDR_IMPLIED, ADDR_ABSOLUTE_Y, ADDR_IMPLIED, ADDR_ABSOLUTE_Y,
        ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_X,
        
        ADDR_ABSOLUTE, ADDR_INDIRECT_X, ADDR_IMPLIED, ADDR_INDIRECT_X,  # 0x2?
        ADDR_ZERO, ADDR_ZERO, ADDR_ZERO, ADDR_ZERO,
        ADDR_IMPLIED, ADDR_IMMEDIATE, ADDR_IMPLIED, ADDR_IMMEDIATE,
        ADDR_ABSOLUTE, ADDR_ABSOLUTE, ADDR_ABSOLUTE, ADDR_ABSOLUTE,
        
        ADDR_RELATIVE, ADDR_INDIRECT_Y, ADDR_IMPLIED, ADDR_INDIRECT_Y,  # 0x3?
        ADDR_ZERO_X, ADDR_ZERO_X, ADDR_ZERO_X, ADDR_ZERO_X,
        ADDR_IMPLIED, ADDR_ABSOLUTE_Y, ADDR_IMPLIED, ADDR_ABSOLUTE_Y,
        ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_X,
        
        ADDR_IMPLIED, ADDR_INDIRECT_X, ADDR_IMPLIED, ADDR_INDIRECT_X,  # 0x4?
        ADDR_ZERO, ADDR_ZERO, ADDR_ZERO, ADDR_ZERO,
        ADDR_IMPLIED, ADDR_IMMEDIATE, ADDR_IMPLIED, ADDR_IMMEDIATE,
        ADDR_ABSOLUTE, ADDR_ABSOLUTE, ADDR_ABSOLUTE, ADDR_ABSOLUTE,
        
        ADDR_RELATIVE, ADDR_INDIRECT_Y, ADDR_IMPLIED, ADDR_INDIRECT_Y,  # 0x5?
        ADDR_ZERO_X, ADDR_ZERO_X, ADDR_ZERO_X, ADDR_ZERO_X,
        ADDR_IMPLIED, ADDR_ABSOLUTE_Y, ADDR_IMPLIED, ADDR_ABSOLUTE_Y,
        ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_X,
        
        ADDR_IMPLIED, ADDR_INDIRECT_X, ADDR_IMPLIED, ADDR_INDIRECT_X,  # 0x6?
        ADDR_ZERO, ADDR_ZERO, ADDR_ZERO, ADDR_ZERO,
        ADDR_IMPLIED, ADDR_IMMEDIATE, ADDR_IMPLIED, ADDR_IMMEDIATE,
        ADDR_INDIRECT, ADDR_ABSOLUTE, ADDR_ABSOLUTE, ADDR_ABSOLUTE,
        
        ADDR_RELATIVE, ADDR_INDIRECT_Y, ADDR_IMPLIED, ADDR_INDIRECT_Y,  # 0x7?
        ADDR_ZERO_X, ADDR_ZERO_X, ADDR_ZERO_X, ADDR_ZERO_X,
        ADDR_IMPLIED, ADDR_ABSOLUTE_Y, ADDR_IMPLIED, ADDR_ABSOLUTE_Y,
        ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_X,
        
        ADDR_IMMEDIATE, ADDR_INDIRECT_X, ADDR_IMMEDIATE, ADDR_INDIRECT_X,  # 0x8?
        ADDR_ZERO, ADDR_ZERO, ADDR_ZERO, ADDR_ZERO,
        ADDR_IMPLIED, ADDR_IMMEDIATE, ADDR_IMPLIED, ADDR_IMMEDIATE,
        ADDR_ABSOLUTE, ADDR_ABSOLUTE, ADDR_ABSOLUTE, ADDR_ABSOLUTE,
        
        ADDR_RELATIVE, ADDR_INDIRECT_Y, ADDR_IMPLIED, ADDR_INDIRECT_Y,  # 0x9?
        ADDR_ZERO_X, ADDR_ZERO_X, ADDR_ZERO_Y, ADDR_ZERO_Y,
        ADDR_IMPLIED, ADDR_ABSOLUTE_Y, ADDR_IMPLIED, ADDR_ABSOLUTE_Y,
        ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_Y, ADDR_ABSOLUTE_Y,
        
        ADDR_IMMEDIATE, ADDR_INDIRECT_X, ADDR_IMMEDIATE, ADDR_INDIRECT_X,  # 0xA?
        ADDR_ZERO, ADDR_ZERO, ADDR_ZERO, ADDR_ZERO,
        ADDR_IMPLIED, ADDR_IMMEDIATE, ADDR_IMPLIED, ADDR_IMMEDIATE,
        ADDR_ABSOLUTE, ADDR_ABSOLUTE, ADDR_ABSOLUTE, ADDR_ABSOLUTE,
        
        ADDR_RELATIVE, ADDR_INDIRECT_Y, ADDR_IMPLIED, ADDR_INDIRECT_Y,  # 0xB?
        ADDR_ZERO_X, ADDR_ZERO_X, ADDR_ZERO_Y, ADDR_ZERO_Y,
        ADDR_IMPLIED, ADDR_ABSOLUTE_Y, ADDR_IMPLIED, ADDR_ABSOLUTE_Y,
        ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_Y, ADDR_ABSOLUTE_Y,
        
        ADDR_IMMEDIATE, ADDR_INDIRECT_X, ADDR_IMMEDIATE, ADDR_INDIRECT_X,  # 0xC?
        ADDR_ZERO, ADDR_ZERO, ADDR_ZERO, ADDR_ZERO,
        ADDR_IMPLIED, ADDR_IMMEDIATE, ADDR_IMPLIED, ADDR_IMMEDIATE,
        ADDR_ABSOLUTE, ADDR_ABSOLUTE, ADDR_ABSOLUTE, ADDR_ABSOLUTE,
        
        ADDR_RELATIVE, ADDR_INDIRECT_Y, ADDR_IMPLIED, ADDR_INDIRECT_Y,  # 0xD?
        ADDR_ZERO_X, ADDR_ZERO_X, ADDR_ZERO_X, ADDR_ZERO_X,
        ADDR_IMPLIED, ADDR_ABSOLUTE_Y, ADDR_IMPLIED, ADDR_ABSOLUTE_Y,
        ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_X,
        
        ADDR_IMMEDIATE, ADDR_INDIRECT_X, ADDR_IMMEDIATE, ADDR_INDIRECT_X,  # 0xE?
        ADDR_ZERO, ADDR_ZERO, ADDR_ZERO, ADDR_ZERO,
        ADDR_IMPLIED, ADDR_IMMEDIATE, ADDR_IMPLIED, ADDR_IMMEDIATE,
        ADDR_ABSOLUTE, ADDR_ABSOLUTE, ADDR_ABSOLUTE, ADDR_ABSOLUTE,
        
        ADDR_RELATIVE, ADDR_INDIRECT_Y, ADDR_IMPLIED, ADDR_INDIRECT_Y,  # 0xF?
        ADDR_ZERO_X, ADDR_ZERO_X, ADDR_ZERO_X, ADDR_ZERO_X,
        ADDR_IMPLIED, ADDR_ABSOLUTE_Y, ADDR_IMPLIED, ADDR_ABSOLUTE_Y,
        ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_X, ADDR_ABSOLUTE_X
    ]

    # Cycle count table
    const CYCLE_TABLE = UInt8[
        7, 6, 2, 8, 3, 3, 5, 5, 3, 2, 2, 2, 4, 4, 6, 6,  # 0
        2, 5, 2, 8, 4, 4, 6, 6, 2, 4, 2, 7, 4, 4, 7, 7,  # 1
        6, 6, 2, 8, 3, 3, 5, 5, 4, 2, 2, 2, 4, 4, 6, 6,  # 2
        2, 5, 2, 8, 4, 4, 6, 6, 2, 4, 2, 7, 4, 4, 7, 7,  # 3
        6, 6, 2, 8, 3, 3, 5, 5, 3, 2, 2, 2, 3, 4, 6, 6,  # 4
        2, 5, 2, 8, 4, 4, 6, 6, 2, 4, 2, 7, 4, 4, 7, 7,  # 5
        6, 6, 2, 8, 3, 3, 5, 5, 4, 2, 2, 2, 5, 4, 6, 6,  # 6
        2, 5, 2, 8, 4, 4, 6, 6, 2, 4, 2, 7, 4, 4, 7, 7,  # 7
        2, 6, 2, 6, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4,  # 8
        2, 6, 2, 6, 4, 4, 4, 4, 2, 5, 2, 5, 5, 5, 5, 5,  # 9
        2, 6, 2, 6, 3, 3, 3, 4, 2, 2, 2, 2, 4, 4, 4, 4,  # A
        2, 5, 2, 5, 4, 4, 4, 4, 2, 4, 2, 4, 4, 4, 4, 4,  # B
        2, 6, 2, 8, 3, 3, 5, 5, 2, 2, 2, 2, 4, 4, 6, 6,  # C
        2, 5, 2, 8, 4, 4, 6, 6, 2, 4, 2, 7, 4, 4, 7, 7,  # D
        2, 6, 2, 8, 3, 3, 5, 5, 2, 2, 2, 2, 4, 4, 6, 6,  # E
        2, 5, 2, 8, 4, 4, 6, 6, 2, 4, 2, 7, 4, 4, 7, 7   # F
    ]

    function step_cpu(state::CPUState, memory::Vector{UInt8})
        """Execute one CPU instruction"""
        # Fetch opcode
        opcode = memory[state.PC + 1]  # Julia is 1-indexed
        
        # Get cycles
        cycles = CYCLE_TABLE[opcode + 1]
        
        # TODO: Implement instruction execution
        # For now, just increment PC and cycles
        state.PC += 1
        state.cycles += cycles
        
        return state, memory
    end

    function run_cpu(state::CPUState, memory::Vector{UInt8}, num_cycles::Int)
        """Run CPU for specified number of cycles"""
        for _ in 1:num_cycles
            state, memory = step_cpu(state, memory)
        end
        return state, memory
    end

end # module
