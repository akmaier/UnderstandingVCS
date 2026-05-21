# SoftCPUState + SoftBus — parallel float-valued state for SOFT-mode execution.
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
"""
mutable struct SoftCPUState
    A::Float32
    X::Float32
    Y::Float32
    SP::Float32
    PC::Float32
    P::Float32
    cycles::Float32
end

"""
    initial_soft_cpu_state(; pc=0xF000) -> SoftCPUState

SOFT-mode CPU state after RESET. `PC` defaults to \$F000 — the
standard cart-mapped reset entry point — so a fresh ROM at offset 0
starts executing immediately.
"""
initial_soft_cpu_state(; pc::Real = 0xF000) =
    SoftCPUState(0f0, 0f0, 0f0, Float32(0xFD), Float32(pc), Float32(0x34), 0f0)


"""
    SoftBus

SOFT-mode bus: float-valued RAM + raw ROM array. The ROM is the
differentiability target — a gradient at any output that depends on a
`soft_rom_peek(bus.rom, addr)` call flows back here, one-hot at the
accessed address.
"""
mutable struct SoftBus
    ram::Vector{Float32}    # 128-element
    rom::Vector{Float32}    # N-element
end

"""
    initial_soft_bus(rom) -> SoftBus

Build a `SoftBus` with all-zero RAM and the given ROM (converted to
`Vector{Float32}`).
"""
initial_soft_bus(rom::AbstractVector) =
    SoftBus(zeros(Float32, 128), Vector{Float32}(rom))
