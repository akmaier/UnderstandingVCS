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
import ..RomSettingsModule: romsettings_starting_actions, romsettings_difficulty,
    romsettings_is_legal_action, romsettings_console_switch_starts, romsettings_pal,
    romsettings_screen_height, romsettings_y_start

export PitfallRomSettings, EnduroRomSettings,
       AirRaidRomSettings, AsterixRomSettings, BeamRiderRomSettings,
       DoubleDunkRomSettings, ElevatorActionRomSettings, GopherRomSettings,
       GravitarRomSettings, JourneyEscapeRomSettings, PrivateEyeRomSettings,
       SkiingRomSettings, UpNDownRomSettings, YarsRevengeRomSettings,
       AmidarRomSettings, SurroundRomSettings,
       CarnivalRomSettings, PooyanRomSettings

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
romsettings_starting_actions(::ElevatorActionRomSettings) = fill(1, 16)  # 16× FIRE (xitari ElevatorAction.cpp loop)
romsettings_starting_actions(::GopherRomSettings)         = Int[1]   # FIRE
romsettings_starting_actions(::GravitarRomSettings)       = fill(1, 16)  # 16× FIRE (xitari Gravitar.cpp loop)
romsettings_starting_actions(::JourneyEscapeRomSettings)  = Int[1]   # FIRE
romsettings_starting_actions(::PrivateEyeRomSettings)     = Int[2]   # UP
romsettings_starting_actions(::SkiingRomSettings)         = fill(5, 16)  # 16× DOWN (xitari Skiing.cpp loop)
romsettings_starting_actions(::UpNDownRomSettings)        = Int[1]   # FIRE
romsettings_starting_actions(::YarsRevengeRomSettings)    = Int[1]   # FIRE

# Task #103 (skiing): xitari's SkiingSettings::isLegal (Skiing.cpp:96-111)
# disallows the entire FIRE family; `StellaEnvironment::act`'s
# `noopIllegalActions` converts those to NOOP before emulating a user step.
# Skiing is the ONLY supported game that overrides isLegal (base = all legal),
# so this is the only override needed. Without it, the 64-ROM sweep's shared
# breakout action stream injects FIRE at frame 20 and skiing diverges 84 b/f.
# ALE FIRE-family codes: FIRE=1, UP/RIGHT/LEFT/DOWN-FIRE=10/11/12/13,
# diagonal-FIRE=14/15/16/17.
const _SKIING_ILLEGAL = (1, 10, 11, 12, 13, 14, 15, 16, 17)
romsettings_is_legal_action(::SkiingRomSettings, a::Integer) = !(Int(a) in _SKIING_ILLEGAL)

# Task #103 (surround): surround is a PAL game whose xitari getStartingActions
# = {SELECT, RESET} (Surround.cpp:135) selects game variation 1 then starts
# it. SELECT/RESET are console switches (not joystick), so they go through
# `romsettings_console_switch_starts` (routed via `console_switches!` in
# env_reset!). PAL → 342-scanline max-frame cutoff (its 312-line frame would
# otherwise be split by the NTSC 290 cutoff → half-rate counters, the
# #103/#106 partial-frame family). SELECT=46, RESET=40.
struct SurroundRomSettings <: RomSettings end
romsettings_console_switch_starts(::SurroundRomSettings) = Int[46, 40]  # SELECT, RESET
romsettings_pal(::SurroundRomSettings) = true
romsettings_screen_height(::SurroundRomSettings) = 250   # task #110 (PAL bump 210→250)
# air_raid is also a PAL dump (its real VSYNC at scanline 286 ends the frame
# before either 290 or 342, so the threshold doesn't change its result — but
# flag it for correctness / xitari parity).
romsettings_pal(::AirRaidRomSettings) = true
romsettings_screen_height(::AirRaidRomSettings) = 250   # task #110 (PAL bump 210→250)

# Task #110 follow-up: three games whose stella.pro entry sets an EXPLICIT
# Display.Height (and, for carnival/pooyan, an explicit Display.YStart) — these
# are NTSC (their rendered content stays within scanline 262, verified by xitari
# screen dump), so NO PAL flag / colour-loss / 312-wrap; only the display crop
# window differs. journey_escape already exists above (FIRE start) — just add its
# height. carnival/pooyan are new render-only subtypes (no starting actions,
# confirmed: neither defines getStartingActions in xitari).
romsettings_screen_height(::JourneyEscapeRomSettings) = 230  # stella.pro Display.Height
struct CarnivalRomSettings <: RomSettings end
romsettings_screen_height(::CarnivalRomSettings) = 214       # stella.pro Display.Height
romsettings_y_start(::CarnivalRomSettings)       = 26        # stella.pro Display.YStart
struct PooyanRomSettings <: RomSettings end
romsettings_screen_height(::PooyanRomSettings)   = 220       # stella.pro Display.Height
romsettings_y_start(::PooyanRomSettings)         = 26        # stella.pro Display.YStart

# Task #103 (amidar): amidar's stella.pro entry overrides BOTH console
# difficulty switches to "A" (Console.LeftDifficulty/RightDifficulty = "A"),
# unlike xitari's B/B default. So SWCHB reads 0xFF, not 0x3F. amidar's frame-1
# object-sort kernel branches on the P0/Left difficulty bit (LDA SWCHB; AND
# #$40), so with jutari's default B/B it sorts the wrong way → 11 b/f. amidar
# has no getStartingActions (so default Int[]); only the difficulty differs.
struct AmidarRomSettings <: RomSettings end
romsettings_difficulty(::AmidarRomSettings) = (true, true)   # A/A → SWCHB 0xFF

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
