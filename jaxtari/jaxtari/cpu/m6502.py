"""6502 fetch–decode–execute step.

P0 status: this is a *stub*. `step` reads the opcode at PC, looks up the base
cycle count and addressing mode, and advances PC by 1. It does NOT yet execute
the instruction — that lands in P1 (see PORTING_PLAN.md §5).

The intended shape:

- `step(state, memory)` is one CPU instruction (not one machine cycle); cycle
  accuracy is enforced via `state.cycles`.
- The decode is done via `cpu.tables.ADDRESSING_MODE_TABLE` and
  `cpu.tables.CYCLE_TABLE`.
- Execution will use `jax.lax.switch` over a 256-way table of per-opcode
  functions in HARD mode (see PORTING_PLAN.md §6.1).
"""

from typing import Callable, Tuple

import jax.numpy as jnp

from jaxtari.types import CPUState
from jaxtari.cpu.tables import ADDRESSING_MODE_TABLE, CYCLE_TABLE

# Memory is modelled as a plain uint8 array indexed by the 6502 16-bit address.
# The 6507 address-bus mirroring (13-bit, A13/A14/A15 ignored) and TIA/RIOT
# decoding will live in `jaxtari.bus.system` once that module lands.
Memory = jnp.ndarray  # shape (65536,), dtype uint8

PeekFn = Callable[[Memory, jnp.uint16], jnp.uint8]
PokeFn = Callable[[Memory, jnp.uint16, jnp.uint8], Memory]


def _peek_flat(memory: Memory, addr: jnp.uint16) -> jnp.uint8:
    """P0: flat-memory peek. Replaced by bus.system in P2."""
    return memory[addr]


def _poke_flat(memory: Memory, addr: jnp.uint16, value: jnp.uint8) -> Memory:
    """P0: flat-memory poke. Replaced by bus.system in P2."""
    return memory.at[addr].set(value)


def step(
    state: CPUState,
    memory: Memory,
    peek: PeekFn = _peek_flat,
    poke: PokeFn = _poke_flat,
) -> Tuple[CPUState, Memory]:
    """Execute one 6502 instruction.

    P0 stub: advances PC by 1 and bumps `cycles` by the base cycle count of
    the opcode at PC. Returns memory unchanged. Real opcode behaviour lands
    in P1a–P1f.
    """
    opcode = peek(memory, state.PC)
    base_cycles = jnp.uint64(CYCLE_TABLE[opcode])
    # TODO(P1): switch on opcode, perform addressing, execute, update flags,
    # add page-cross / branch-taken penalties.
    _ = ADDRESSING_MODE_TABLE[opcode]
    new_state = state._replace(
        PC=(state.PC + jnp.uint16(1)) & jnp.uint16(0xFFFF),
        cycles=state.cycles + base_cycles,
    )
    return new_state, memory
