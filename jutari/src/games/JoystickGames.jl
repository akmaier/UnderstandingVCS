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
    romsettings_screen_height, romsettings_y_start, romsettings_hmove_blanks,
    romsettings_agent_player, romsettings_is_terminal

export PitfallRomSettings, EnduroRomSettings,
       AirRaidRomSettings, AsterixRomSettings, BeamRiderRomSettings,
       DoubleDunkRomSettings, ElevatorActionRomSettings, GopherRomSettings,
       GravitarRomSettings, JourneyEscapeRomSettings, PrivateEyeRomSettings,
       SkiingRomSettings, UpNDownRomSettings, YarsRevengeRomSettings,
       AmidarRomSettings, SurroundRomSettings,
       CarnivalRomSettings, PooyanRomSettings,
       BattleZoneRomSettings, MsPacmanRomSettings,
       PacmanRomSettings, QbertRomSettings, WizardOfWorRomSettings,
       PhoenixRomSettings

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
# wizard_of_wor: xitari/ALE drives the single-player agent on the RIGHT
# controller (P1, SWCHA low nibble) — verified by the SWCHA read on the first
# in-play joystick press (agent RIGHT clears bit 3, not bit 7). jutari's default
# P0 routing diverged once gameplay reads the stick (past the conformance window).
struct WizardOfWorRomSettings    <: RomSettings end
romsettings_agent_player(::WizardOfWorRomSettings) = 1

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

# Asterix terminal — xitari/games/supported/Asterix.cpp::step():
#   m_lives      = readRam(0xD3) & 0xF
#   death_counter = readRam(0xC7)
#   terminal = (death_counter == 0x01 && m_lives == 1)
# (xitari can't wait for lives==0 because the agent may restart on the last
# frame by holding fire.) Stateless predicate, so it lives on the existing
# immutable struct. The long-horizon run dies at action-frame 1158; xitari
# auto-resets (lives→3) while jutari (its render-only Asterix settings has no
# terminal reader → falls through to false) kept rendering the dead episode —
# the f1160 "TIA pixel diff" #127b MISCLASSIFIED as Cluster A render.
@inline _jg_ram(console::Console, addr::Integer) =
    @inbounds Int(console.bus.ram[(Int(addr) & 0x7F) + 1])
function romsettings_is_terminal(::AsterixRomSettings, console::Console)
    lives = _jg_ram(console, 0xD3) & 0xF
    death_counter = _jg_ram(console, 0xC7)
    return death_counter == 0x01 && lives == 1
end
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
# Pooyan terminal — xitari/games/supported/Pooyan.cpp::step():
#   lives_byte = readRam(0x96)
#   some_byte  = readRam(0x98)
#   terminal = (lives_byte == 0x0 && some_byte == 0x05)
# Reuses the existing render-only struct so Pooyan keeps height=220/y_start=26.
# #127b flagged pooyan as "the one genuine TIA render bug" (f1605, LOCALIZED) but
# that verdict was a TOOLING ARTIFACT: longhorizon_diff.py / frame_offset_probe.py
# hardcode H=210, while pooyan renders at H=220 (its Display.Height), so the
# misreshape drifted 10 rows/frame and fabricated a fake f1605 localized diff. At
# the correct H=220 the REAL first-div is f1532 — pooyan's game-over frame
# (trace_dump --auto-reset: done=true, lives=0 at 1532, then xitari restarts at
# 1533 lives→3) — a whole-screen terminal/auto-reset gap, identical mechanism to
# the other three. Boot RAM[0x96]=2 → terminal false through the in-window sweep. ✓
function romsettings_is_terminal(::PooyanRomSettings, console::Console)
    lives_byte = _jg_ram(console, 0x96)
    some_byte  = _jg_ram(console, 0x98)
    return lives_byte == 0x0 && some_byte == 0x05
end

# battle_zone + ms_pacman are the only two ROMs in the 64-game set whose
# stella.pro entry sets `Emulation.HmoveBlanks "NO"` (default is "YES"). With
# the comb disabled, xitari never blanks the 8px left edge on an HMOVE strobe —
# battle_zone strobes HMOVE every visible scanline (at cc 222), so jutari's
# default-on comb wrongly blanked cols 0-7 on EVERY row (1112 px). Render-only
# (no RAM effect); both are RAM bit-exact and have no getStartingActions.
struct BattleZoneRomSettings <: RomSettings end
romsettings_hmove_blanks(::BattleZoneRomSettings) = false
struct MsPacmanRomSettings <: RomSettings end
romsettings_hmove_blanks(::MsPacmanRomSettings)   = false
# Ms. Pac-Man terminal — xitari/games/supported/MsPacman.cpp::step():
#   lives_byte  = readRam(0xFB) & 0xF
#   death_timer = readRam(0xA7)
#   terminal = (lives_byte == 0 && death_timer == 0x53)
# Reuses the existing render-only struct so MsPacman keeps hmove_blanks=false.
# The long-horizon run dies at action-frame 1785 (lives_byte→0, death_timer→0x53);
# xitari auto-resets (lives→3) while jutari (no terminal reader → false) kept
# rendering the dead episode — the f1786 "low-severity cosmetic tail" #127b
# MISCLASSIFIED as Cluster A render. Boot RAM[0xFB]&0xF=2 → terminal false. ✓
function romsettings_is_terminal(::MsPacmanRomSettings, console::Console)
    lives_byte  = _jg_ram(console, 0xFB) & 0xF
    death_timer = _jg_ram(console, 0xA7)
    return lives_byte == 0 && death_timer == 0x53
end

# Task #113: three more games with an EXPLICIT non-default `Display.YStart` in
# stella.pro (default 34). jutari rendered from YStart=34, vertically OFFSETTING
# the whole frame vs xitari → every row compared against the wrong scanline (the
# screen sweep's pervasive "shifted pattern" divergence). Render-only (no RAM
# effect); all three are RAM bit-exact and NTSC (Height 210 default). up_n_down
# already has a struct (FIRE start); add its YStart. pacman/qbert are new
# render-only subtypes (no starting actions). NOTE: pacman = "Pac-Man" (distinct
# from ms_pacman). qbert keeps GenericRomSettings' partial-frame behavior (#106)
# — the YStart is render-crop only.
romsettings_y_start(::UpNDownRomSettings) = 30   # stella.pro Display.YStart
struct PacmanRomSettings <: RomSettings end
romsettings_y_start(::PacmanRomSettings)  = 33   # stella.pro Display.YStart
# Pac-Man terminal — xitari/games/supported/Pacman.cpp::step():
#   m_lives          = readRam(0x98) + 1
#   animationCounter = readRam(0xE4)
#   terminal = (m_lives == 1 && animationCounter == 0x3F)   (m_lives==1 ⇔ RAM[0x98]==0)
# Reuses the existing render-only struct so Pacman keeps y_start=33.
# The long-horizon run dies at action-frame 1770 (RAM[0x98]→0, RAM[0xE4]→0x3F);
# xitari auto-resets (lives→4) while jutari (no terminal reader → false) kept
# rendering the dead episode — the f1771 "low-severity cosmetic tail" #127b
# MISCLASSIFIED as Cluster A render. Boot RAM[0x98]=3 → terminal false. ✓
function romsettings_is_terminal(::PacmanRomSettings, console::Console)
    lives_byte        = _jg_ram(console, 0x98)
    animation_counter = _jg_ram(console, 0xE4)
    return lives_byte == 0 && animation_counter == 0x3F
end
struct QbertRomSettings <: RomSettings end
romsettings_y_start(::QbertRomSettings)   = 40   # stella.pro Display.YStart

# Phoenix terminal — xitari/games/supported/Phoenix.cpp::step():
#   state_byte = readRam(0xCC)
#   terminal = (state_byte == 0x80)
#   m_lives  = readRam(0xCB) & 0x7   (starts at 5)
# Phoenix has no stella.pro render override (default NTSC Height=210/YStart=34/
# HmoveBlanks=yes) and no getStartingActions — so this render-only struct behaves
# identically to GenericRomSettings except for the terminal reader. The
# long-horizon run dies at action-frame 1742 (RAM[0xCC]→0x80); xitari auto-resets
# (lives→5) while jutari (Generic → never terminal) kept rendering the dead
# episode — the f1743 whole-screen swap #127b flagged "Cluster A render
# (medium confidence)" is this terminal/auto-reset gap. Boot RAM[0xCC]=0 →
# terminal false through the in-window sweep. ✓
struct PhoenixRomSettings <: RomSettings end
function romsettings_is_terminal(::PhoenixRomSettings, console::Console)
    return _jg_ram(console, 0xCC) == 0x80
end

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
