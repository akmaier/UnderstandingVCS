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
       KangarooRomSettings, AsteroidsRomSettings,
       BerzerkRomSettings, MontezumaRevengeRomSettings, RiverRaidRomSettings

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

# xitari getDecimalScore(lo, mid, hi): three BCD bytes — lo = tens/ones,
# mid = thousands/hundreds, hi = hundred-thousands/ten-thousands
# (RomUtils.cpp:86). The top byte contributes 100000*high_nibble +
# 10000*low_nibble.
@inline function _decimal_score3(console::Console, lo::Integer, mid::Integer,
                                 hi::Integer)
    score = _decimal_score2(console, lo, mid)
    hi_b = _ram(console, hi)
    score += 100000 * (hi_b >> 4) + 10000 * (hi_b & 0x0F)
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

# --------------------------------------------------------------------------- #
# Berzerk — xitari/games/supported/Berzerk.cpp
#   score    = getDecimalScore(95, 94, 93)  (0x5F, 0x5E, 0x5D)
#   livesByte = readRam(0xDA)
#   terminal = (livesByte == 0xFF)
#   lives    = livesByte + 1
# Boot RAM: 0xDA = 2 → terminal=false at start; the long-horizon run dies at
# action-frame 579 (livesByte→0xFF), where xitari auto-resets to 3 lives but
# jutari (GenericRomSettings, never terminal) kept rendering the dead/dying
# episode — the f581 "orange maze-border 1 extra frame" the prior berzerk
# entry MISDIAGNOSED as a VSYNC frame-boundary phase. It is this terminal gap.
# --------------------------------------------------------------------------- #
mutable struct BerzerkRomSettings <: RomSettings
    score::Int
    BerzerkRomSettings() = new(0)
end

romsettings_reset!(s::BerzerkRomSettings) = (s.score = 0; nothing)

function romsettings_is_terminal(::BerzerkRomSettings, console::Console)
    return _ram(console, 0xDA) == 0xFF
end

function romsettings_get_reward(s::BerzerkRomSettings, console::Console)
    sc = _decimal_score3(console, 95, 94, 93)
    r = sc - s.score
    s.score = sc
    return r
end

romsettings_lives(::BerzerkRomSettings, console::Console) =
    _ram(console, 0xDA) + 1

# --------------------------------------------------------------------------- #
# Montezuma's Revenge — xitari/games/supported/MontezumaRevenge.cpp
#   score     = getDecimalScore(0x95, 0x94, 0x93)
#   new_lives = readRam(0xBA)
#   some_byte = readRam(0xFE)
#   terminal  = (new_lives == 0 && some_byte == 0x60)
#   lives     = (new_lives & 0x7) + 1
# Starts with 6 lives. Long-horizon run dies at action-frame 866; xitari
# auto-resets (lives→6) while jutari (Generic) kept rendering the dead
# episode — the f867 "localized TIA pixel diff" the #127b diagnosis
# MISCLASSIFIED as Cluster A render (RAM is bit-exact right up to the death;
# the divergence is the death/reset boundary).
# --------------------------------------------------------------------------- #
mutable struct MontezumaRevengeRomSettings <: RomSettings
    score::Int
    MontezumaRevengeRomSettings() = new(0)
end

romsettings_reset!(s::MontezumaRevengeRomSettings) = (s.score = 0; nothing)

function romsettings_is_terminal(::MontezumaRevengeRomSettings, console::Console)
    new_lives = _ram(console, 0xBA)
    some_byte = _ram(console, 0xFE)
    return new_lives == 0 && some_byte == 0x60
end

function romsettings_get_reward(s::MontezumaRevengeRomSettings, console::Console)
    sc = _decimal_score3(console, 0x95, 0x94, 0x93)
    r = sc - s.score
    s.score = sc
    return r
end

romsettings_lives(::MontezumaRevengeRomSettings, console::Console) =
    (_ram(console, 0xBA) & 0x7) + 1

# --------------------------------------------------------------------------- #
# River Raid — xitari/games/supported/RiverRaid.cpp
#   score: 6 BCD-ish digits at RAM 87,85,83,81,79,77 via a ram_val→digit LUT
#          (val/8 = digit; only multiples of 8 in [0,72] are valid).
#   terminal = (RAM[0xC0] == 0x58 && PREV RAM[0xC0] == 0x59)  ← STATEFUL
#   reset: prev byte (m_lives_byte) initialised to 0x58.
# xitari calls step() once per emulated step, updating m_lives_byte each time;
# jutari calls romsettings_is_terminal once per env_step! (after
# run_until_frame!), so we update the stored byte THERE — same cadence.
# Long-horizon run dies at action-frame 956 (0xC0 0x59→0x58); xitari
# auto-resets (lives→4) while jutari kept rendering the dead episode — the
# f958 "TIA pixel diff (grows over time)" #127b MISCLASSIFIED as Cluster A.
# --------------------------------------------------------------------------- #
const _RR_VAL_TO_DIGIT = let d = fill(0, 256)
    for k in 0:9
        d[8 * k + 1] = k        # 1-indexed: ram value 8k → digit k
    end
    d
end
@inline _rr_digit(v::Integer) = @inbounds _RR_VAL_TO_DIGIT[(Int(v) & 0xFF) + 1]

mutable struct RiverRaidRomSettings <: RomSettings
    score::Int
    lives_byte::Int
    RiverRaidRomSettings() = new(0, 0x58)
end

romsettings_reset!(s::RiverRaidRomSettings) = (s.score = 0; s.lives_byte = 0x58; nothing)

function romsettings_is_terminal(s::RiverRaidRomSettings, console::Console)
    byte_val = _ram(console, 0xC0)
    terminal = (byte_val == 0x58 && s.lives_byte == 0x59)
    s.lives_byte = byte_val          # mirror xitari step()'s m_lives_byte update
    return terminal
end

function romsettings_get_reward(s::RiverRaidRomSettings, console::Console)
    sc = 0
    mult = 1
    for off in (87, 85, 83, 81, 79, 77)
        sc += mult * _rr_digit(_ram(console, off))
        mult *= 10
    end
    r = sc - s.score
    s.score = sc
    return r
end

# xitari numericLives(): 0x58 → 4 (episode start), 0x59 → 1, else byte/8 + 1.
romsettings_lives(s::RiverRaidRomSettings, console::Console) =
    let b = _ram(console, 0xC0)
        b == 0x58 ? 4 : b == 0x59 ? 1 : (b ÷ 8 + 1)
    end

end # module
