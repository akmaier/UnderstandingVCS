"""
    Console

Top-level VCS state: CPU + Bus + the operations games need (reset to the
cart's reset vector, single-instruction step, "run until one frame
boundary").

Mutable struct holding two already-mutable inner objects; calling
`console_step!` / `run_until_frame!` mutates them in place.
"""
module ConsoleModule

using ..Types: CPUState, initial_cpu_state
using ..Bus: BusState, initial_bus, peek
using ..CPU: step

export Console, initial_console, console_reset!, console_step!, run_until_frame!

# Task #106 (partial-frame model): xitari runs at most this many
# INSTRUCTIONS per mediaSource().update() — `m6502().execute(25000)`
# in TIA::update (M6502Low.cxx:65 decrements once per instruction, NOT
# per cycle). When the budget is exhausted WITHOUT a frame end, xitari
# returns a "grey" (incomplete) frame: the frame counter is NOT advanced
# and the beam/cycle state is preserved, so the next update() continues
# the same frame. `run_until_frame!` mirrors this exactly.
const _UPDATE_INSTRUCTION_BUDGET = 25_000

mutable struct Console
    cpu::CPUState
    bus::BusState
end

"""
    initial_console(rom=nothing) -> Console

Build a console with a fresh CPU and a Bus around the given ROM (the
cart kind is auto-detected by `initial_bus`).
"""
initial_console(rom=nothing) = Console(initial_cpu_state(), initial_bus(rom))

"""
    console_reset!(console)

Reset the console: fresh CPU + TIA + RIOT + RAM (cart state preserved),
and load PC from the cart's reset vector at \$FFFC/\$FFFD.
"""
function console_reset!(console::Console)
    fresh = initial_bus()
    console.bus.ram  .= fresh.ram
    console.bus.tia   = fresh.tia
    console.bus.riot  = fresh.riot
    # Reset CPU registers.
    fresh_cpu = initial_cpu_state()
    console.cpu.A      = fresh_cpu.A
    console.cpu.X      = fresh_cpu.X
    console.cpu.Y      = fresh_cpu.Y
    console.cpu.SP     = fresh_cpu.SP
    console.cpu.P      = fresh_cpu.P
    console.cpu.cycles = fresh_cpu.cycles
    # Load PC from cart's reset vector.
    lo = UInt16(peek(console.bus, 0xFFFC))
    hi = UInt16(peek(console.bus, 0xFFFD))
    console.cpu.PC = (hi << 8) | lo
    return console
end

"""
    console_step!(console)

Execute one CPU instruction (with the usual TIA / RIOT post-step).
"""
function console_step!(console::Console)
    step(console.cpu, console.bus)
    return console
end

"""
    run_until_frame!(console)

Run one xitari `mediaSource().update()`: step the CPU until the TIA's
frame counter advances by one (a VSYNC-clear hold-gate or the poke-time
max-scanline cutoff — see `tia_poke!`), OR until the 25000-instruction
budget is exhausted.

Task #106 (partial-frame model): the budget is xitari's
`m6502().execute(25000)`. When it runs out WITHOUT a frame boundary, this
returns a *grey* frame — the frame counter has NOT advanced and the TIA's
beam/scanline/cycle state is preserved, so the next `run_until_frame!`
continues the same TIA frame (xitari leaves `myPartialFrameFlag` true and
skips `startFrame()`). This reproduces qbert's boot→step "sliver" frame:
its RESET-boot wait loop (task #52) has no TIA pokes for thousands of
scanlines, so the frame can only be sliced by this budget — exactly as in
xitari — instead of the per-step cutoff that used to live in `tia_advance!`.
"""
function run_until_frame!(console::Console)
    # Task #114: xitari's `TIA::update` calls `startFrame()` — which SWAPS the
    # double framebuffer (TIA.cxx:537-539) — at the START of an update IFF the
    # previous frame completed (`if(!myPartialFrameFlag)`). jutari's equivalent:
    # if the previous frame completed (buffer_swap_pending armed at the
    # vsync_reset_pending drain), swap now, BEFORE rendering the new frame. The
    # just-completed frame's pixels move to `framebuffer_prev`; the new frame
    # renders into the swapped-in buffer, whose un-rendered rows still hold the
    # content from two frames ago — exactly like xitari (qbert's boot→game
    # short frame shows the preserved board). A grey/partial frame does NOT set
    # the flag, so it continues the same buffer (no swap) like xitari.
    tia = console.bus.tia
    if tia.buffer_swap_pending
        tia.framebuffer, tia.framebuffer_prev = tia.framebuffer_prev, tia.framebuffer
        tia.buffer_swap_pending = false
    end
    start_frame = tia.frame
    for _ in 1:_UPDATE_INSTRUCTION_BUDGET
        step(console.cpu, console.bus)
        if console.bus.tia.frame != start_frame
            return console            # frame completed (xitari endFrame)
        end
    end
    return console                    # grey frame: execute(25000) budget hit
end

end # module
