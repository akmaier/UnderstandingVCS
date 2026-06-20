# SoftCPUState + SoftBus — parallel float-valued state for SOFT-mode execution.
#
# Paper reference: the soft state x = (registers, RAM, RIOT, TIA, frame
# buffer) on which the soft transition map Phi_S acts (paper "Soft Equals
# Hard"; supplementary "Setup and Notation"). The 128-byte RIOT RAM here
# is the paper's "RAM as a soft tape" (an NTM-style addressable tape,
# Graves et al. 2014) and `SoftBus.rom` the "ROM as a weight tensor".
# All quantities are Float32 but only ever at integer values in
# [-2^24, 2^24], so Float32 represents every VCS byte/address exactly and
# Theorem 1's forward equality holds in Float32, not merely over the
# reals (supplementary "Numerical scope").
#
# The HARD-mode `CPUState` and `BusState` are kept untouched; SOFT mode
# runs against these parallel types so neither the HARD test suite nor
# the existing CPU dispatch needs to change.
#
# `SoftCPUState` carries `A, X, Y, SP, PC, P, cycles` as Float32 scalars
# so an autodiff stack (Zygote / Enzyme) can flow gradients through the
# arithmetic. The status flags are packed in `P` as a single float
# (8-bit semantics, value computed via float arithmetic).
#
# `SoftBus` carries the 128 B RIOT RAM as `Vector{Float32}` and the ROM
# as a `Vector{Float32}` — peeks use a one-hot dot product so a gradient
# at the output flows back to the accessed byte.
#
# TIA / RIOT region writes are out of scope for P7b — soft_step!
# currently only handles instructions whose visible effect is on CPU
# registers and/or RAM.

"""
    SoftCPUState

Mutable CPU register file for SOFT-mode execution. All fields are
`Float32` so the autodiff stack can flow gradients through arithmetic
on them.

**P7c-dx mirror.** Alongside the packed `P` byte (the source of truth
for forward semantics), four float fields `P_N` / `P_Z` / `P_C` /
`P_V` mirror the same flag bits as 0.0 / 1.0 floats. These exist
purely to feed `soft_branch` in the functional branch handlers — a
caller that injects a soft Z / N / C / V (anywhere in [0, 1]) into
`SoftCPUState` for an XAI experiment gets a non-zero gradient through
both `pc_taken` and `pc_not_taken` via the sigmoid blend, while the
forward PC still comes from the packed `P` (so `soft_step!` /
existing tests stay bit-exact).

All four default to 0.0; the initial `P = 0x34` has N=Z=C=V=0 anyway
and the first flag-touching opcode re-syncs both representations.
"""
mutable struct SoftCPUState
    A::Float32
    X::Float32
    Y::Float32
    SP::Float32
    PC::Float32
    P::Float32
    cycles::Float32
    P_N::Float32
    P_Z::Float32
    P_C::Float32
    P_V::Float32
end

"""
    initial_soft_cpu_state(; pc=0xF000) -> SoftCPUState

SOFT-mode CPU state after RESET. `PC` defaults to \$F000 — the
standard cart-mapped reset entry point — so a fresh ROM at offset 0
starts executing immediately. Float-flag mirrors `P_N` / `P_Z` /
`P_C` / `P_V` start at 0.0 (matches `P = 0x34`, where N=Z=C=V=0).
"""
initial_soft_cpu_state(; pc::Real = 0xF000) =
    SoftCPUState(0f0, 0f0, 0f0, Float32(0xFD), Float32(pc), Float32(0x34), 0f0,
                 0f0, 0f0, 0f0, 0f0)

"""
    SoftCPUState(A, X, Y, SP, PC, P, cycles)

Backwards-compat positional constructor (pre-P7c-dx). Float-flag
mirrors default to zero — fine because the very first flag-touching
opcode re-syncs them via `_with_p` / `_with_p!`.
"""
SoftCPUState(A::Real, X::Real, Y::Real, SP::Real, PC::Real, P::Real,
             cycles::Real) =
    SoftCPUState(Float32(A), Float32(X), Float32(Y), Float32(SP),
                 Float32(PC), Float32(P), Float32(cycles),
                 0f0, 0f0, 0f0, 0f0)


"""
    SoftBus

SOFT-mode bus: float-valued RAM + raw ROM array + the minimal RIOT
timer state introduced in **P8-cx**. The ROM is the differentiability
target — a gradient at any output that depends on a
`soft_rom_peek(bus.rom, addr)` call flows back here, one-hot at the
accessed address.

P8-cx adds four scalar fields modelling the RIOT M6532 interval timer
— enough to get past the standard "load TIM*T → poll INTIM" boot
pattern that stalled SOFT execution of real ROMs. All four default to
inert values so existing constructions are unaffected.
"""
mutable struct SoftBus
    ram::Vector{Float32}    # 128-element
    rom::Vector{Float32}    # N-element
    # P8-cx RIOT timer (inert defaults; activated by the first TIM*T write).
    riot_intim::Float32
    riot_prescaler_shift::Float32   # 0/3/6/10 → 1/8/64/1024×
    riot_residual_cycles::Float32
    riot_expired::Float32            # latch, 1.0 = INTIM reached 0
end

"""
    SoftBus(ram, rom) — convenience constructor preserving the pre-P8-cx
    signature (RIOT timer fields default to inert).
"""
SoftBus(ram::Vector{Float32}, rom::Vector{Float32}) =
    SoftBus(ram, rom, 0f0, 0f0, 0f0, 0f0)

"""
    initial_soft_bus(rom) -> SoftBus

Build a `SoftBus` with all-zero RAM, the given ROM, and an inert RIOT
timer (first TIM*T write activates it).
"""
initial_soft_bus(rom::AbstractVector) =
    SoftBus(zeros(Float32, 128), Vector{Float32}(rom))
