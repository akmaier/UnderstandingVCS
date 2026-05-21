"""
JAXTARI: Differentiable Atari 2600 Emulator in JAX
Phase 1: CPU Core (M6502)

This module implements a cycle-accurate MOS Technology 6502 CPU emulator
using JAX arrays for full differentiability.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any
from dataclasses import dataclass
from functools import partial

# Constants
RAM_SIZE = 128
STACK_PAGE = 0x0100
RESET_VECTOR = 0xFFFC

@dataclass
class CPUState:
    """Immutable CPU state container for JAX functional updates"""
    A: jnp.uint8      # Accumulator
    X: jnp.uint8      # X index register
    Y: jnp.uint8      # Y index register
    SP: jnp.uint8     # Stack pointer
    PC: jnp.uint16    # Program counter
    PS: jnp.uint8     # Processor status register (NV-BDIZC)
    cycles: jnp.uint32 # Cycle counter

# Processor Status flags
FLAG_N = 0x80  # Negative
FLAG_V = 0x40  # Overflow
FLAG_B = 0x10  # Break
FLAG_D = 0x08  # Decimal
FLAG_I = 0x04  # Interrupt disable
FLAG_Z = 0x02  # Zero
FLAG_C = 0x01  # Carry

def ps_flags_to_byte(N, V, B, D, I, Z, C):
    """Convert individual flags to status byte"""
    return jnp.uint8(
        (N << 7) | (V << 6) | (1 << 5) | (B << 4) | 
        (D << 3) | (I << 2) | (Z << 1) | C
    )

def byte_to_flags(ps):
    """Extract individual flags from status byte"""
    N = (ps & FLAG_N) != 0
    V = (ps & FLAG_V) != 0
    B = (ps & FLAG_B) != 0
    D = (ps & FLAG_D) != 0
    I = (ps & FLAG_I) != 0
    Z = (ps & FLAG_Z) != 0
    C = (ps & FLAG_C) != 0
    return N, V, B, D, I, Z, C

def create_initial_state():
    """Create initial CPU state after RESET"""
    return CPUState(
        A=jnp.uint8(0),
        X=jnp.uint8(0),
        Y=jnp.uint8(0),
        SP=jnp.uint8(0xFD),
        PC=jnp.uint16(0),
        PS=jnp.uint8(0x34),  # I=1, B=1
        cycles=jnp.uint32(0)
    )

# Instruction tables
# Addressing modes
ADDR_IMPLIED = 0
ADDR_IMMEDIATE = 1
ADDR_ZERO = 2
ADDR_ZERO_X = 3
ADDR_ZERO_Y = 4
ADDR_ABSOLUTE = 5
ADDR_ABSOLUTE_X = 6
ADDR_ABSOLUTE_Y = 7
ADDR_INDIRECT = 8
ADDR_INDIRECT_X = 9
ADDR_INDIRECT_Y = 10
ADDR_RELATIVE = 11

ADDRESSING_MODE_TABLE = jnp.array([
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
], dtype=jnp.uint8)

# Cycle count table (base cycles for each opcode)
CYCLE_TABLE = jnp.array([
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
], dtype=jnp.uint8)

# Instruction type classification for execute switch
INSTRUCTION_TYPE = jnp.array([
    0x10, 0x01, 0x00, 0x01, 0x02, 0x01, 0x03, 0x01,  # 0x0? BRK, ORA, ..., ASL
    0x04, 0x01, 0x03, 0x01, 0x02, 0x01, 0x03, 0x01,
    0x05, 0x01, 0x00, 0x01, 0x02, 0x01, 0x03, 0x01,  # 0x1? BPL, ORA, ..., ASL
    0x06, 0x01, 0x02, 0x01, 0x02, 0x01, 0x03, 0x01,
    0x07, 0x08, 0x00, 0x08, 0x09, 0x08, 0x0A, 0x08,  # 0x2? JSR, AND, ..., ROL
    0x0B, 0x08, 0x0A, 0x08, 0x09, 0x08, 0x0A, 0x08,
    0x05, 0x08, 0x00, 0x08, 0x02, 0x08, 0x0A, 0x08,  # 0x3? BMI, AND, ..., ROL
    0x06, 0x08, 0x02, 0x08, 0x02, 0x08, 0x0A, 0x08,
    0x0C, 0x0D, 0x00, 0x0D, 0x0E, 0x0D, 0x0F, 0x0D,  # 0x4? RTI, EOR, ..., LSR
    0x10, 0x0D, 0x0F, 0x0D, 0x11, 0x0D, 0x0F, 0x0D,
    0x05, 0x0D, 0x00, 0x0D, 0x02, 0x0D, 0x0F, 0x0D,  # 0x5? BVC, EOR, ..., LSR
    0x06, 0x0D, 0x02, 0x0D, 0x02, 0x0D, 0x0F, 0x0D,
    0x12, 0x13, 0x00, 0x13, 0x14, 0x13, 0x15, 0x13,  # 0x6? RTS, ADC, ..., ROR
    0x16, 0x13, 0x15, 0x13, 0x11, 0x13, 0x15, 0x13,
    0x05, 0x13, 0x00, 0x13, 0x02, 0x13, 0x15, 0x13,  # 0x7? BVS, ADC, ..., ROR
    0x06, 0x13, 0x02, 0x13, 0x02, 0x13, 0x15, 0x13,
    0x02, 0x17, 0x02, 0x17, 0x18, 0x17, 0x19, 0x17,  # 0x8? NOP, STA, ..., STX
    0x1A, 0x02, 0x1B, 0x02, 0x18, 0x17, 0x19, 0x17,
    0x05, 0x17, 0x00, 0x17, 0x18, 0x17, 0x19, 0x17,  # 0x9? BCC, STA, ..., STX
    0x1C, 0x17, 0x1D, 0x17, 0x1E, 0x17, 0x1F, 0x17,
    0x20, 0x21, 0x20, 0x21, 0x20, 0x21, 0x20, 0x21,  # 0xA? LDY, LDA, LDX, LAX
    0x22, 0x21, 0x23, 0x21, 0x20, 0x21, 0x20, 0x21,
    0x05, 0x21, 0x00, 0x21, 0x20, 0x21, 0x20, 0x21,  # 0xB? BCS, LDA, ..., LAX
    0x24, 0x21, 0x25, 0x21, 0x20, 0x21, 0x20, 0x21,
    0x26, 0x27, 0x02, 0x27, 0x28, 0x27, 0x29, 0x27,  # 0xC? CPY, CMP, ..., DEC
    0x2A, 0x27, 0x2B, 0x27, 0x28, 0x27, 0x29, 0x27,
    0x05, 0x27, 0x00, 0x27, 0x02, 0x27, 0x29, 0x27,  # 0xD? BNE, CMP, ..., DEC
    0x06, 0x27, 0x02, 0x27, 0x02, 0x27, 0x29, 0x27,
    0x2C, 0x2D, 0x02, 0x2D, 0x2E, 0x2D, 0x2F, 0x2D,  # 0xE? CPX, SBC, ..., INC
    0x30, 0x2D, 0x31, 0x2D, 0x2E, 0x2D, 0x2F, 0x2D,
    0x05, 0x2D, 0x00, 0x2D, 0x02, 0x2D, 0x2F, 0x2D,  # 0xF? BEQ, SBC, ..., INC
    0x06, 0x2D, 0x02, 0x2D, 0x02, 0x2D, 0x2F, 0x2D
], dtype=jnp.uint8)

# Instruction types
ITYPE_BRK = 0x10
ITYPE_ORA = 0x01
ITYPE_ASL = 0x03
ITYPE_BPL = 0x05
ITYPE_CLC = 0x06
ITYPE_JSR = 0x07
ITYPE_AND = 0x08
ITYPE_BIT = 0x09
ITYPE_ROL = 0x0A
ITYPE_BMI = 0x05
ITYPE_SEC = 0x06
ITYPE_RTI = 0x0C
ITYPE_EOR = 0x0D
ITYPE_LSR = 0x0F
ITYPE_PHP = 0x10
ITYPE_BVC = 0x05
ITYPE_CLI = 0x06
ITYPE_RTS = 0x12
ITYPE_ADC = 0x13
ITYPE_ROR = 0x15
ITYPE_PLA = 0x16
ITYPE_BVS = 0x05
ITYPE_SEI = 0x06
ITYPE_STA = 0x17
ITYPE_STY = 0x18
ITYPE_STX = 0x19
ITYPE_DEY = 0x1A
ITYPE_TXA = 0x1B
ITYPE_BCC = 0x05
ITYPE_TYA = 0x1C
ITYPE_TXS = 0x1D
ITYPE_LDY = 0x20
ITYPE_LDA = 0x21
ITYPE_LDX = 0x20
ITYPE_TAY = 0x22
ITYPE_TAX = 0x23
ITYPE_BCS = 0x05
ITYPE_CLV = 0x24
ITYPE_TSX = 0x25
ITYPE_CPY = 0x26
ITYPE_CMP = 0x27
ITYPE_INY = 0x2A
ITYPE_DEX = 0x2B
ITYPE_BNE = 0x05
ITYPE_CLD = 0x06
ITYPE_CPX = 0x2C
ITYPE_SBC = 0x2D
ITYPE_INX = 0x30
ITYPE_NOP = 0x02
ITYPE_BEQ = 0x05
ITYPE_SED = 0x06

def calculate_address(memory, state, mode, opcode, pc):
    """Calculate effective address based on addressing mode"""
    # This will be implemented with switch-like behavior
    pass

def execute_instruction(state, memory, opcode):
    """Execute a single instruction
    
    Args:
        state: Current CPUState
        memory: Memory array
        opcode: Instruction opcode
    
    Returns:
        New CPUState and updated memory
    """
    itype = INSTRUCTION_TYPE[opcode]
    addr_mode = ADDRESSING_MODE_TABLE[opcode]
    
    # Extract flags
    N = (state.PS & FLAG_N) != 0
    V = (state.PS & FLAG_V) != 0
    B = (state.PS & FLAG_B) != 0
    D = (state.PS & FLAG_D) != 0
    I = (state.PS & FLAG_I) != 0
    Z = (state.PS & FLAG_Z) != 0
    C = (state.PS & FLAG_C) != 0
    
    # TODO: Implement full instruction set
    # For now, return state unchanged
    
    return state, memory

def step_cpu(state, memory):
    """Execute one CPU instruction
    
    Args:
        state: Current CPU state
        memory: System memory (JAX array)
    
    Returns:
        Updated state and memory
    """
    # Fetch opcode
    opcode = jnp.uint8(memory[state.PC])
    
    # Get base cycles
    cycles = jnp.uint32(CYCLE_TABLE[opcode])
    
    # Execute instruction
    new_state, new_memory = execute_instruction(state, memory, opcode)
    
    # Update cycle count
    new_state = new_state.replace(cycles=new_state.cycles + cycles)
    
    return new_state, new_memory

def run_cpu(state, memory, num_cycles):
    """Run CPU for specified number of cycles
    
    Args:
        state: Initial CPU state
        memory: System memory
        num_cycles: Number of cycles to execute
    
    Returns:
        Final state and memory
    """
    def body_fun(carry, _):
        state, memory = carry
        new_state, new_memory = step_cpu(state, memory)
        return (new_state, new_memory), None
    
    (final_state, final_memory), _ = jax.lax.scan(
        body_fun, (state, memory), None, length=num_cycles
    )
    
    return final_state, final_memory
