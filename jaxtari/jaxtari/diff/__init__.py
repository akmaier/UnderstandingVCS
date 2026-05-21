"""Differentiability layer.

See PORTING_PLAN.md §6 for the design (HARD vs SOFT mode, soft opcode
dispatch, soft RAM addressing, ROM-as-weights, straight-through estimator).

P7 ships the standalone *primitives*: RomTensor, soft_select,
soft_memory_read, soft_branch, straight_through_round /
straight_through_clamp. Integration with the actual `step()` dispatch
(turning the CPU into a SOFT-mode emulator end-to-end) is a P7b
follow-up.
"""

from jaxtari.diff.modes import Mode, current_mode, set_mode, using_mode
from jaxtari.diff.rom_as_weights import RomTensor
from jaxtari.diff.soft_branch import soft_branch
from jaxtari.diff.soft_mem import soft_memory_read
from jaxtari.diff.soft_select import soft_select
from jaxtari.diff.soft_state import (
    SoftBus,
    SoftCPUState,
    initial_soft_bus,
    initial_soft_cpu_state,
)
from jaxtari.diff.soft_step import (
    SOFT_SUPPORTED_OPCODES,
    soft_ram_peek,
    soft_rom_peek,
    soft_run,
    soft_step,
)
from jaxtari.diff.straight_through import (
    straight_through_clamp,
    straight_through_round,
)

__all__ = [
    "Mode",
    "RomTensor",
    "SOFT_SUPPORTED_OPCODES",
    "SoftBus",
    "SoftCPUState",
    "current_mode",
    "initial_soft_bus",
    "initial_soft_cpu_state",
    "set_mode",
    "soft_branch",
    "soft_memory_read",
    "soft_ram_peek",
    "soft_rom_peek",
    "soft_run",
    "soft_select",
    "soft_step",
    "straight_through_clamp",
    "straight_through_round",
    "using_mode",
]
