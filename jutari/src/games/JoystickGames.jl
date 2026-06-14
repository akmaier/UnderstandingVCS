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

export PitfallRomSettings, EnduroRomSettings,
       AirRaidRomSettings, AsterixRomSettings, BeamRiderRomSettings,
       DoubleDunkRomSettings, ElevatorActionRomSettings, GopherRomSettings,
       GravitarRomSettings, JourneyEscapeRomSettings, PrivateEyeRomSettings,
       SkiingRomSettings, UpNDownRomSettings, YarsRevengeRomSettings

# --------------------------------------------------------------------------- #
# Task #100 follow-up: 12 joystick games whose only conformance-relevant
# per-game behavior is xitari's `getStartingActions()` (a single action the
# ALE harness applies AFTER the boot burn). Without it our generic NOOP boot
# is 1 starting-action behind xitari → frame-0 RAM divergence in the 64-ROM
# sweep. Action codes per xitari/ale_interface.hpp: FIRE=1, UP=2, RIGHT=3,
# DOWN=5, UPFIRE=10. Source: each game's
# `xitari/games/supported/<Game>.cpp::getStartingActions`.
# Scoring is left to GenericRomSettings (irrelevant for conformance).
# --------------------------------------------------------------------------- #
struct AirRaidRomSettings        <: RomSettings end
struct AsterixRomSettings        <: RomSettings end
struct BeamRiderRomSettings      <: RomSettings end
struct DoubleDunkRomSettings     <: RomSettings end
struct ElevatorActionRomSettings <: RomSettings end
struct GopherRomSettings         <: RomSettings end
struct GravitarRomSettings       <: RomSettings end
struct JourneyEscapeRomSettings  <: RomSettings end
struct PrivateEyeRomSettings     <: RomSettings end
struct SkiingRomSettings         <: RomSettings end
struct UpNDownRomSettings        <: RomSettings end
struct YarsRevengeRomSettings    <: RomSettings end

romsettings_starting_actions(::AirRaidRomSettings)        = Int[1]   # FIRE
romsettings_starting_actions(::AsterixRomSettings)        = Int[1]   # FIRE
romsettings_starting_actions(::BeamRiderRomSettings)      = Int[3]   # RIGHT
romsettings_starting_actions(::DoubleDunkRomSettings)     = Int[10]  # UPFIRE
romsettings_starting_actions(::ElevatorActionRomSettings) = Int[1]   # FIRE
romsettings_starting_actions(::GopherRomSettings)         = Int[1]   # FIRE
romsettings_starting_actions(::GravitarRomSettings)       = Int[1]   # FIRE
romsettings_starting_actions(::JourneyEscapeRomSettings)  = Int[1]   # FIRE
romsettings_starting_actions(::PrivateEyeRomSettings)     = Int[2]   # UP
romsettings_starting_actions(::SkiingRomSettings)         = Int[5]   # DOWN
romsettings_starting_actions(::UpNDownRomSettings)        = Int[1]   # FIRE
romsettings_starting_actions(::YarsRevengeRomSettings)    = Int[1]   # FIRE

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
