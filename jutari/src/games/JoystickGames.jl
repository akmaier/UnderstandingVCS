"""
    JoystickGames

`RomSettings` subtypes for joystick games that need per-game
overrides — currently only Pitfall + Enduro, both because xitari
emits per-ROM `getStartingActions()` that put the agent into a known
initial pose AFTER the boot burn. Without emulating those frames our
env is 1 frame behind xitari's `resetGame()`, which compounds into
the documented 19/45 b/f RAM divergence over 300 NOOP frames
(tasks #81 / #82).

These intentionally do NOT implement score / reward / terminal — the
scoring stubs from `GenericRomSettings` are fine for conformance work
(0 reward, never terminal). A future task can port the BCD-score
detectors from `jaxtari/jaxtari/games/atari_classics.py` if needed.
"""
module JoystickGames

using ..RomSettingsModule: RomSettings
using ..ConsoleModule: Console
import ..RomSettingsModule: romsettings_starting_actions

export PitfallRomSettings, EnduroRomSettings

"""
    PitfallRomSettings

Pitfall! starts with `PLAYER_A_UP` (= action 2). Per
`xitari/games/supported/Pitfall.cpp::getStartingActions`.
"""
struct PitfallRomSettings <: RomSettings
end

# PLAYER_A_UP = 2 per xitari/ale_interface.hpp
romsettings_starting_actions(::PitfallRomSettings) = Int[2]

"""
    EnduroRomSettings

Enduro starts with `PLAYER_A_FIRE` (= action 1) — needed to launch
into the race. Per `xitari/games/supported/Enduro.cpp::getStartingActions`.
"""
struct EnduroRomSettings <: RomSettings
end

# PLAYER_A_FIRE = 1 per xitari/ale_interface.hpp
romsettings_starting_actions(::EnduroRomSettings) = Int[1]

end # module
