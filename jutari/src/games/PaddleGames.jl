"""
    PaddleGames

Minimal RomSettings stubs for the paddle-game ROMs we ship — they
override only `romsettings_uses_paddles` so `StellaEnvironment`
auto-translates LEFT/RIGHT actions into INPT0 dump-pot
paddle-position changes (xitari's `applyActionPaddles` semantic).
The full per-game scoring rules live on the jaxtari side
(`jaxtari/games/breakout.py` etc.) — porting them to jutari is a
separate task.

Mirror of jaxtari's `BreakoutRomSettings.uses_paddles() = True` /
`PongRomSettings.uses_paddles() = True`.
"""
module PaddleGames

using ..RomSettingsModule: RomSettings
using ..ConsoleModule: Console
import ..RomSettingsModule: romsettings_uses_paddles, romsettings_swap_paddles,
                            romsettings_is_terminal, romsettings_get_reward,
                            romsettings_lives, romsettings_reset!

export BreakoutRomSettings, PongRomSettings

"""
    BreakoutRomSettings

xitari stella.pro: `Cartridge.MD5 f34f08e5…`, `Controller.Left "PADDLES"`.

Mirrors `xitari/games/supported/Breakout.cpp`:
  - score    = (RAM[77] & 0x0F)*1 + ((RAM[77] >> 4) & 0x0F)*10 + (RAM[76] & 0x0F)*100
  - lives    = RAM[57]
  - started  = first frame where lives observed == 5 (sticky)
  - terminal = started && lives == 0

The reset() resets the started/terminal/score flags; lives counter is
read fresh from RAM each call to step.
"""
mutable struct BreakoutRomSettings <: RomSettings
    score::Int
    lives::Int
    started::Bool
    terminal::Bool
    BreakoutRomSettings() = new(0, 0, false, false)
end

romsettings_uses_paddles(::BreakoutRomSettings) = true

function romsettings_reset!(s::BreakoutRomSettings)
    s.score    = 0
    s.lives    = 0
    s.started  = false
    s.terminal = false
    return nothing
end

# Read RAM by 7-bit index. The console's bus has 128 bytes of physical
# RAM in `bus.ram` (1-indexed in Julia). Don't go through `bus.peek` —
# `$39` / `$4C` / `$4D` directly are TIA read-side addresses (INPT/CXxx),
# not RAM. xitari's `readRam(addr)` ANDs with 0x7F then indexes the
# RAM array; we do the same here.
@inline _ram(console::Console, addr::Integer) =
    @inbounds console.bus.ram[(Int(addr) & 0x7F) + 1]

function _update!(s::BreakoutRomSettings, console::Console)
    # score: tens+ones at $4D, hundreds at $4C
    x = Int(_ram(console, 77))                       # $4D
    y = Int(_ram(console, 76))                       # $4C
    s.score = 1 * (x & 0x0F) + 10 * ((x & 0xF0) >> 4) + 100 * (y & 0x0F)
    # lives + terminal latch
    byte_val = Int(_ram(console, 57))                # $39
    if !s.started && byte_val == 5
        s.started = true
    end
    s.terminal = s.started && byte_val == 0
    s.lives    = byte_val
    return nothing
end

function romsettings_is_terminal(s::BreakoutRomSettings, console::Console)
    _update!(s, console)
    return s.terminal
end

function romsettings_get_reward(s::BreakoutRomSettings, console::Console)
    prev = s.score
    _update!(s, console)
    return s.score - prev
end

# xitari: `int lives() const { return isTerminal() ? 0 : m_lives; }`
function romsettings_lives(s::BreakoutRomSettings, console::Console)
    _update!(s, console)
    return s.terminal ? 0 : s.lives
end

"""
    PongRomSettings

The shipped `xitari/roms/pong.bin` is actually Video Olympics (Atari
1978, md5 60e0ea3c…). xitari stella.pro lists it as
`Controller.Left/Right "PADDLES"` with `Controller.SwapPaddles "YES"`.

Scoring + termination per `xitari/games/supported/Pong.cpp::step()`:
  - P0 ("x" / cpu) score at RAM[\$0D]   (= xitari `readRam(&system, 13)`)
  - P1 ("y" / human) score at RAM[\$0E] (= xitari `readRam(&system, 14)`)
  - reward = ΔP0 − ΔP1 (positive when the user / P0 scores).
  - terminal once either side reaches 21 (the standard Pong target).

(Earlier comments here used \$14/\$15 — those bytes hold sprite-pattern
data that briefly hits 0x82 = 130 within ~60 frames of FIRE, falsely
triggering terminal and freezing the paddle. Fixed against xitari.)
"""
mutable struct PongRomSettings <: RomSettings
    p0_prev::Int
    p1_prev::Int
    PongRomSettings() = new(0, 0)
end

romsettings_uses_paddles(::PongRomSettings) = true
# Pong / Video Olympics has `Controller.SwapPaddles "YES"` in xitari's
# stella.pro. With swap, the Paddles controller routes
# PaddleZeroResistance (the user paddle from `applyActionPaddles`) to
# Pin Five, which the TIA reads as INPT1 — so `_apply_paddle_action!`
# must update `paddle_resistance[1]` instead of `paddle_resistance[0]`.
romsettings_swap_paddles(::PongRomSettings) = true

function romsettings_reset!(s::PongRomSettings)
    s.p0_prev = 0
    s.p1_prev = 0
    return nothing
end

@inline _pong_scores(console::Console) = (
    Int(_ram(console, 0x0D)),    # P0 ("x" cpu) score, per xitari Pong.cpp:55
    Int(_ram(console, 0x0E)),    # P1 ("y" human) score, per xitari Pong.cpp:56
)

function romsettings_is_terminal(::PongRomSettings, console::Console)
    p0, p1 = _pong_scores(console)
    return max(p0, p1) >= 21
end

function romsettings_get_reward(s::PongRomSettings, console::Console)
    p0, p1 = _pong_scores(console)
    r = (p0 - s.p0_prev) - (p1 - s.p1_prev)
    s.p0_prev = p0
    s.p1_prev = p1
    return r
end

# Pong has no explicit life counter — the score is the only progress
# signal; return 0 (= "no lives indicator" — matches jaxtari).
romsettings_lives(::PongRomSettings, ::Console) = 0

end # module
