"""
    JuTari

Differentiable Julia port of xitari. See PORTING_PLAN.md at the repo root for
scope and milestones.
"""
module JuTari

const VERSION_STRING = "0.0.1"

include("Types.jl")
include("tia/TIA.jl")
include("riot/RIOT.jl")
include("cart/Cart.jl")
include("bus/Bus.jl")
# CPU module pulls in its own includes (Tables, Addressing, ALU) — keep it
# the sole entry point for the CPU subtree to avoid double-include of Tables.
include("cpu/M6502.jl")
include("Console.jl")
include("io/IO.jl")
include("games/RomSettings.jl")
include("env/StellaEnvironment.jl")
include("diff/Modes.jl")

using .Types: CPUState, initial_cpu_state
using .TIA: TIAState, initial_tia_state
using .RIOT: RIOTState, initial_riot_state
using .Cart: CartState, make_cart
using .Bus: BusState, initial_bus
using .ConsoleModule: Console, initial_console
using .RomSettingsModule: RomSettings, GenericRomSettings
using .Env: StellaEnvironment
using .CPU.CPUTables: FLAG_N, FLAG_V, FLAG_U, FLAG_B, FLAG_D, FLAG_I, FLAG_Z, FLAG_C
using .Diff: Mode, HARD, SOFT, current_mode, set_mode!, using_mode

# Functions that collide with Base (`step`, `peek`) are intentionally NOT
# re-exported from JuTari — Julia's `using` refuses to import them when a
# Base binding of the same name exists. Use the qualified form
# `JuTari.CPU.step`, `JuTari.Bus.peek`, `JuTari.Bus.poke!`, or:
#     using JuTari.CPU: step
#     using JuTari.Bus: peek, poke!
# This also mirrors the jaxtari import paths
# `jaxtari.cpu.m6502.step` and `jaxtari.bus.peek/poke`.
export CPUState, initial_cpu_state,
       TIAState, initial_tia_state,
       RIOTState, initial_riot_state,
       CartState, make_cart,
       BusState, initial_bus,
       Console, initial_console,
       RomSettings, GenericRomSettings,
       StellaEnvironment,
       FLAG_N, FLAG_V, FLAG_U, FLAG_B, FLAG_D, FLAG_I, FLAG_Z, FLAG_C,
       Mode, HARD, SOFT, current_mode, set_mode!, using_mode

end # module
