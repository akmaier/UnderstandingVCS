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

const _FRAME_INSTRUCTION_LIMIT = 100_000

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

Step the CPU until the TIA's frame counter advances by one. Either a
software VSYNC falling edge or the 262-line scanline wrap (safety net)
counts as a frame.
"""
function run_until_frame!(console::Console)
    start_frame = console.bus.tia.frame
    for _ in 1:_FRAME_INSTRUCTION_LIMIT
        step(console.cpu, console.bus)
        if console.bus.tia.frame != start_frame
            return console
        end
    end
    error("run_until_frame! exceeded $_FRAME_INSTRUCTION_LIMIT instructions " *
          "without a frame boundary (start_frame=$start_frame).")
end

end # module
