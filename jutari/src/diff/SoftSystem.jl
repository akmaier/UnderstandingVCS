# Full-system SOFT step — the one-step map Phi_S of Theorem 1 on the WHOLE
# machine state.
#
# `soft_step` alone is a differentiable CPU core over a simplified bus (one ROM
# window, TIA / RIOT folded into RAM); on a real ROM it drifts from the hard
# machine within ~1k instructions. `soft_system_step` joins it to the bit-exact
# hard console with the straight-through estimator (paper Eq. "ste"), exactly as
# Fig. "pipeline" draws it:
#
#   1. the HARD console executes the instruction (CPU, bus, TIA, RIOT, cart) —
#      this is the forward value, bit-exact by construction;
#   2. the SOFT handlers re-execute it against the same machine (`_SYS_CTX`):
#      ROM bytes through the hard cart's current bank / slice mapping, TIA / RIOT
#      / Superchip reads replayed from the hard bus as constants;
#   3. every CPU register and RAM cell is returned as STE(soft, hard) =
#      soft + sg(hard - soft): forward = hard, gradient = d(soft)/d(inputs).
#
# TIA, RIOT, cartridge and frame buffer state live in the hard console and carry
# no gradient. Gradients reach registers, RAM and ROM bytes — the quantities the
# paper differentiates.

import ..ConsoleModule: Console, console_step!

@inline _ste(soft::Real, hard::Real) = soft + _stop_gradient(Float32(hard) - soft)

# Hard values after the step, as Float32 (cycles kept modulo 2^24 so the
# integer stays exactly representable in float32).
function _hard_snapshot(con::Console)
    c = con.cpu
    return (A = Float32(c.A), X = Float32(c.X), Y = Float32(c.Y), SP = Float32(c.SP),
            PC = Float32(c.PC), P = Float32(c.P), cycles = Float32(c.cycles & 0xFFFFFF),
            ram = Float32.(con.bus.ram))
end

# Run the hard instruction, recording the values of its non-RAM, non-ROM reads.
function _hard_step_with_io_log!(con::Console)
    cart = con.bus.cart
    kind, bank, slices = cart.kind, cart.current_bank, copy(cart.slice_slots)
    saved = Bus._TRACE_BUFFER[]
    trace_enable!()
    console_step!(con)
    events = trace_take!()
    Bus._TRACE_BUFFER[] = saved
    io = Tuple{Int,Int}[]
    for (k, _, _, _, addr, v) in events
        a = Int(addr)
        k === :peek && !_is_ram_addr(a) &&
            ((a & 0x1000) == 0 || cart_rom_index(kind, bank, slices, a) < 0) &&
            push!(io, (a, v))
    end
    return _SysCtx(kind, bank, slices, io), _hard_snapshot(con)
end

"""
    soft_system_init(console) -> (SoftCPUState, SoftBus)

Soft state that mirrors a hard console at any point (e.g. a live game state
after `env_reset!` / `env_step!`). The bus carries the full ROM image, so
gradients reach every bank.
"""
function soft_system_init(con::Console)
    h = _hard_snapshot(con)
    p = Int(con.cpu.P)
    state = SoftCPUState(h.A, h.X, h.Y, h.SP, h.PC, h.P, h.cycles,
                         Float32((p >> 7) & 1), Float32((p >> 1) & 1),
                         Float32(p & 1), Float32((p >> 6) & 1))
    return state, SoftBus(copy(h.ram), Float32.(con.bus.cart.rom))
end

"""
    soft_system_step(state, bus, console) -> (SoftCPUState, SoftBus)

One instruction of the full machine in SOFT mode. Mutates `console` (the hard
machine advances one instruction); returns the soft state, whose forward value
equals the hard state bit-for-bit (Theorem 1) and whose gradient is the soft
handlers' surrogate (Corollary 1). Zygote-differentiable w.r.t. `state` and
`bus` (ROM and RAM).
"""
function soft_system_step(state::SoftCPUState, bus::SoftBus, con::Console)
    ctx, h = ChainRulesCore.ignore_derivatives(() -> _hard_step_with_io_log!(con))
    ChainRulesCore.ignore_derivatives(() -> (_SYS_CTX[] = ctx; nothing))
    s, b = soft_step(state, bus)
    ChainRulesCore.ignore_derivatives(() -> (_SYS_CTX[] = nothing; nothing))
    p = Int(h.P)
    new_state = SoftCPUState(_ste(s.A, h.A), _ste(s.X, h.X), _ste(s.Y, h.Y),
                             _ste(s.SP, h.SP), _ste(s.PC, h.PC), _ste(s.P, h.P),
                             _ste(s.cycles, h.cycles),
                             _ste(s.P_N, (p >> 7) & 1), _ste(s.P_Z, (p >> 1) & 1),
                             _ste(s.P_C, p & 1), _ste(s.P_V, (p >> 6) & 1))
    new_ram = b.ram .+ _stop_gradient(h.ram .- b.ram)
    return new_state, update_bus(b; ram = new_ram)
end

"""
    soft_system_run(state, bus, console, n_steps) -> (SoftCPUState, SoftBus)

`n_steps` full-system soft instructions.
"""
function soft_system_run(state::SoftCPUState, bus::SoftBus, con::Console, n_steps::Integer)
    for _ in 1:n_steps
        state, bus = soft_system_step(state, bus, con)
    end
    return state, bus
end

"""
    soft_handler_agreement(state, bus, console) -> Bool

Diagnostic: run one full-system step and report whether the SOFT handlers
alone (before the STE join) already produced the hard registers and RAM. When
true, the STE correction is the identity in value and the gradient is taken
through a computation that reproduces the hard step exactly.
"""
function soft_handler_agreement(state::SoftCPUState, bus::SoftBus, con::Console)
    ctx, h = _hard_step_with_io_log!(con)
    _SYS_CTX[] = ctx
    s, b = try
        soft_step(state, bus)
    finally
        _SYS_CTX[] = nothing
    end
    regs = (s.A, s.X, s.Y, s.SP, s.PC, s.P) == (h.A, h.X, h.Y, h.SP, h.PC, h.P)
    return regs && b.ram == h.ram
end
