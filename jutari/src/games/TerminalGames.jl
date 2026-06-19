"""
    TerminalGames

`RomSettings` subtypes that implement a real game-over reader
(`romsettings_is_terminal`) for joystick games whose comparison-video
pipeline needs to auto-reset at the end of an episode. Without these,
the games fall back to `GenericRomSettings` (always-false terminal), so
the long-horizon comparison-video pipeline never auto-resets at
game-over and jutari keeps rendering the dead/old episode while xitari's
ALE starts a fresh game — the Cluster B long-horizon divergence
(task #127b).

Each reader is a mechanistic mirror of the matching
`xitari/games/supported/<Game>.cpp::step()` — same RAM addresses, same
terminal predicate, same BCD score decode (`getDecimalScore`). This is a
front-end-only change: it touches NO emulation core, and CANNOT affect
the in-window RAM/screen sweeps (those run short NOOP/fixed streams that
never reach game-over). All four games are RAM-bit-exact through
game-over (verified in #127b), so the only divergence was the missing
auto-reset.

xitari's `getDecimalScore`/`readRam` semantics (RomUtils.cpp):
  - `readRam(sys, off) = peek((off & 0x7F) + 0x80)` — i.e. the high
    mirror of the 128 B RAM, equivalent to RAM[off & 0x7F].
  - `getDecimalScore(lo, hi, sys)` = BCD(lo, units/tens) +
    BCD(hi)*100 / *1000  (hi is the HUNDREDS+THOUSANDS byte).

These intentionally implement reward too (cheap, mirrors xitari), but
the conformance-relevant override is `romsettings_is_terminal`.
"""
module TerminalGames

using ..RomSettingsModule: RomSettings
using ..ConsoleModule: Console
import ..RomSettingsModule: romsettings_is_terminal, romsettings_get_reward,
                            romsettings_lives, romsettings_reset!

export SpaceInvadersRomSettings, RoadRunnerRomSettings,
       KangarooRomSettings, AsteroidsRomSettings

# Read RAM by 7-bit index — mirror of xitari `readRam` (mask 0x7F, index
# the 128-cell physical RAM array). jutari `bus.ram` is 1-indexed.
@inline _ram(console::Console, addr::Integer) =
    @inbounds Int(console.bus.ram[(Int(addr) & 0x7F) + 1])

# xitari getDecimalScore(lo, hi): lo holds tens(<<4)+ones, hi holds
# thousands(<<4)+hundreds. Returns the full decimal value.
@inline function _decimal_score2(console::Console, lo::Integer, hi::Integer)
    lo_b = _ram(console, lo)
    score = 10 * (lo_b >> 4) + (lo_b & 0x0F)
    hi_b = _ram(console, hi)
    score += 1000 * (hi_b >> 4) + 100 * (hi_b & 0x0F)
    return score
end

# --------------------------------------------------------------------------- #
# Space Invaders — xitari/games/supported/SpaceInvaders.cpp
#   score    = getDecimalScore(0xE8, 0xE6) (note: skips 0xE7)
#   lives    = readRam(0xC9)
#   terminal = (readRam(0x98) & 0x80) != 0  ||  lives == 0
# Boot RAM (probed): lives=3, RAM[0x98]=0x52 → terminal=false at start. ✓
# --------------------------------------------------------------------------- #
mutable struct SpaceInvadersRomSettings <: RomSettings
    score::Int
    SpaceInvadersRomSettings() = new(0)
end

romsettings_reset!(s::SpaceInvadersRomSettings) = (s.score = 0; nothing)

@inline _si_lives(console::Console) = _ram(console, 0xC9)

function romsettings_is_terminal(::SpaceInvadersRomSettings, console::Console)
    return ((_ram(console, 0x98) & 0x80) != 0) || (_si_lives(console) == 0)
end

function romsettings_get_reward(s::SpaceInvadersRomSettings, console::Console)
    # xitari getDecimalScore(0xE8, 0xE6) — 0xE8 = tens/ones, 0xE6 =
    # thousands/hundreds. (0xE7 is intentionally not read.)
    sc = _decimal_score2(console, 0xE8, 0xE6)
    r = sc - s.score
    s.score = sc
    return r
end

# xitari does NOT zero lives at game-over for SpaceInvaders (m_lives is the
# raw byte); mirror it directly.
romsettings_lives(::SpaceInvadersRomSettings, console::Console) = _si_lives(console)

# --------------------------------------------------------------------------- #
# Road Runner — xitari/games/supported/RoadRunner.cpp
#   score (×100) from 4 BCD digits at 0xC9..0xCC (0xA digit = 0)
#   lives_byte   = readRam(0xC4) & 0x7
#   y_vel        = readRam(0xB9)
#   x_vel_death  = readRam(0xBD)
#   terminal     = (lives_byte == 0 && (y_vel != 0 || x_vel_death != 0))
#   lives        = lives_byte + 1
# Boot RAM (probed): 0xC4=34 → lives_byte=2 → terminal=false. ✓
# --------------------------------------------------------------------------- #
mutable struct RoadRunnerRomSettings <: RomSettings
    score::Int
    RoadRunnerRomSettings() = new(0)
end

romsettings_reset!(s::RoadRunnerRomSettings) = (s.score = 0; nothing)

@inline function _rr_score(console::Console)
    score = 0
    mult = 1
    for digit in 0:3
        value = _ram(console, 0xC9 + digit) & 0x0F
        value == 0x0A && (value = 0)   # 0xA = '0, don't display'
        score += mult * value
        mult *= 10
    end
    return score * 100
end

function romsettings_is_terminal(::RoadRunnerRomSettings, console::Console)
    lives_byte  = _ram(console, 0xC4) & 0x7
    y_vel       = _ram(console, 0xB9)
    x_vel_death = _ram(console, 0xBD)
    return lives_byte == 0 && (y_vel != 0 || x_vel_death != 0)
end

function romsettings_get_reward(s::RoadRunnerRomSettings, console::Console)
    sc = _rr_score(console)
    r = sc - s.score
    s.score = sc
    return r
end

romsettings_lives(::RoadRunnerRomSettings, console::Console) =
    (_ram(console, 0xC4) & 0x7) + 1

# --------------------------------------------------------------------------- #
# Kangaroo — xitari/games/supported/Kangaroo.cpp
#   score (×100) = getDecimalScore(0xA8, 0xA7) * 100
#   lives_byte   = readRam(0xAD)
#   terminal     = (lives_byte == 0xFF)
#   lives        = (lives_byte & 0x7) + 1
# Boot RAM (probed): 0xAD=2 → terminal=false. ✓
# --------------------------------------------------------------------------- #
mutable struct KangarooRomSettings <: RomSettings
    score::Int
    KangarooRomSettings() = new(0)
end

romsettings_reset!(s::KangarooRomSettings) = (s.score = 0; nothing)

function romsettings_is_terminal(::KangarooRomSettings, console::Console)
    return _ram(console, 0xAD) == 0xFF
end

function romsettings_get_reward(s::KangarooRomSettings, console::Console)
    sc = _decimal_score2(console, 0xA8, 0xA7) * 100
    r = sc - s.score
    s.score = sc
    return r
end

romsettings_lives(::KangarooRomSettings, console::Console) =
    (_ram(console, 0xAD) & 0x7) + 1

# --------------------------------------------------------------------------- #
# Asteroids — xitari/games/supported/Asteroids.cpp
#   score (×10, wrap at 100000) = getDecimalScore(0x3E, 0x3D) * 10
#   byte    = readRam(0x3C)
#   lives   = (byte - (byte & 15)) >> 4   (= high nibble)
#   terminal = (lives == 0)
# NOTE: at boot/attract RAM[0x3C]==0 → lives==0 → terminal=true. xitari
# itself sees this (its trace stalls / emits one frame during attract,
# auto-resetting every iteration) — so mirroring it makes jutari match
# xitari's behaviour exactly (#127b notes this is partly an xitari-side
# artifact). The terminal predicate is a faithful port, not a special case.
# --------------------------------------------------------------------------- #
const _ASTEROIDS_WRAP_SCORE = 100000

mutable struct AsteroidsRomSettings <: RomSettings
    score::Int
    AsteroidsRomSettings() = new(0)
end

romsettings_reset!(s::AsteroidsRomSettings) = (s.score = 0; nothing)

function romsettings_is_terminal(::AsteroidsRomSettings, console::Console)
    byte = _ram(console, 0x3C)
    lives = (byte - (byte & 15)) >> 4
    return lives == 0
end

function romsettings_get_reward(s::AsteroidsRomSettings, console::Console)
    sc = _decimal_score2(console, 0x3E, 0x3D) * 10
    r = sc - s.score
    r < 0 && (r += _ASTEROIDS_WRAP_SCORE)
    s.score = sc
    return r
end

romsettings_lives(::AsteroidsRomSettings, console::Console) =
    let byte = _ram(console, 0x3C); (byte - (byte & 15)) >> 4 end

end # module
