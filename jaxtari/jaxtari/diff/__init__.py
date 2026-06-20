"""Differentiability layer — the paper's "new methods" (SOFT execution path).

This package implements the differentiable (SOFT) execution of the VCS
described in the paper "A Differentiable Atari VCS": the cartridge ROM
as a weight tensor (rom_as_weights), the RAM as a soft tape, control
flow as gates (soft_branch / soft_select), and the straight-through
estimator (straight_through) that joins the soft path to the bit-exact
hard path. Together these give a soft forward pass that is bit-exact
equal to the hard one at any finite temperature (Theorem 1, "Exact
forward equivalence") while exposing surrogate gradients where the bit
logic has none (Corollary 1). See paper sections "Hard and Soft
Execution" and "Soft Equals Hard", and the supplementary "Setup and
Notation" for the five primitives.

See PORTING_PLAN.md §6 for the design (HARD vs SOFT mode, soft opcode
dispatch, soft RAM addressing, ROM-as-weights, straight-through estimator).

Contents:
  - P7   primitives: RomTensor, soft_select, soft_memory_read,
         soft_branch, straight_through_round / straight_through_clamp.
  - P7b/P7c: `soft_step` / `soft_run` — a SOFT-mode `step()` covering
         the full 151-opcode NMOS set.
  - P7d: RomTensor registered as a JAX PyTree.
  - P7f: `soft_render_scanline` / `soft_render_frame` — a differentiable
         TIA playfield renderer, so `jax.grad` reaches a framebuffer
         pixel.
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
    soft_run_scan,
    soft_step,
)
from jaxtari.diff.soft_tia import (
    soft_collision_registers,
    soft_render_frame,
    soft_render_scanline,
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
    "soft_collision_registers",
    "soft_memory_read",
    "soft_ram_peek",
    "soft_render_frame",
    "soft_render_scanline",
    "soft_rom_peek",
    "soft_run",
    "soft_run_scan",
    "soft_select",
    "soft_step",
    "straight_through_clamp",
    "straight_through_round",
    "using_mode",
]
