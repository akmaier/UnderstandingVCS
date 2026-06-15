"""
    RomSettings

Per-game scoring + termination rules. The `RomSettings` abstract type
is the interface front-ends use to read out reward / lives / game-over
from a running `Console`. Per-game subtypes inspect the console's RAM
or registers at game-specific addresses; that's reverse-engineering
work, deferred beyond P6.

`GenericRomSettings` is a no-op stub: zero reward, never terminal.
"""
module RomSettingsModule

using ..ConsoleModule: Console

export RomSettings, GenericRomSettings,
       romsettings_reset!, romsettings_is_terminal,
       romsettings_get_reward, romsettings_lives,
       romsettings_uses_paddles, romsettings_swap_paddles,
       romsettings_starting_actions, romsettings_difficulty,
       romsettings_is_legal_action,
       romsettings_console_switch_starts, romsettings_pal,
       romsettings_screen_height

abstract type RomSettings end

# Required interface — subtypes override these via Julia multiple dispatch.

romsettings_reset!(::RomSettings)             = nothing
romsettings_is_terminal(::RomSettings, ::Console) = false
romsettings_get_reward(::RomSettings, ::Console) = 0
romsettings_lives(::RomSettings, ::Console)      = 0
# Per-game starting actions emulated by xitari's `StellaEnvironment::reset`
# AFTER the 60-NOOP + 4-RESET boot burn and the settings.reset() call
# (only when settings.getBool("use_starting_actions") is true — default
# in xitari is `true`). Pitfall + Enduro use this to put the agent into
# a known initial pose (PLAYER_A_UP / PLAYER_A_FIRE respectively); without
# emulating those frames we diverge from xitari by 1 frame of state.
# Default empty — joystick games without a startup pose return `Int[]`.
# See `xitari/games/supported/Pitfall.cpp::getStartingActions` etc.
romsettings_starting_actions(::RomSettings)      = Int[]
# Default: joystick. Override `romsettings_uses_paddles(::MyType) =
# true` in per-game subtypes whose stella.pro entry has
# Controller.Left/Right "PADDLES" — `StellaEnvironment` reads this
# to auto-translate LEFT/RIGHT actions into INPT0 dump-pot
# paddle-position changes. Mirror of xitari's `m_use_paddles`
# auto-detection from the cart's properties.
romsettings_uses_paddles(::RomSettings)          = false
# Default: paddles not swapped (Breakout convention). Override to
# `true` on per-game subtypes whose stella.pro entry has
# `Controller.SwapPaddles "YES"` (Pong / Video Olympics being the
# headline case). With the swap, xitari's `Paddles` controller wires
# `PaddleZeroResistance` (the user-driven paddle on Pin Five via
# `applyActionPaddles`) to controller Pin Five — which the TIA then
# reads as `INPT1` (left-jack Pin Five → INPT1 per
# `xitari/emucore/TIA.cxx::peek` case 0x09). So for swapped paddle
# games, our `_apply_paddle_action!` must write the user paddle to
# `paddle_resistance[1]` (INPT1) instead of `paddle_resistance[0]`
# (INPT0).
romsettings_swap_paddles(::RomSettings)          = false
# Console difficulty switches (SWCHB bits 0x40 = P0/Left, 0x80 = P1/Right).
# Returns `(p0_difficulty_a, p1_difficulty_a)`. xitari's default properties
# (Props.cxx) set BOTH difficulties to "B" → SWCHB 0x3F, so the default here
# is `(false, false)` (B/B), matching the 58 bit-exact games. Per-game
# subtypes whose stella.pro entry overrides a difficulty to "A" return
# `true` for that switch (e.g. Amidar = A/A → SWCHB 0xFF). xitari
# Switches.cxx: a "B" property clears the bit, otherwise the bit stays set.
romsettings_difficulty(::RomSettings)            = (false, false)
# Per-game action legality — mirror of xitari's `RomSettings::isLegal`
# (RomSettings.cpp: base returns true for ALL actions). xitari's
# `StellaEnvironment::act` calls `noopIllegalActions` BEFORE emulating a
# USER step (stella_environment.cpp:189), converting any action for which
# `isLegal` is false into PLAYER_A_NOOP. Only ONE supported game overrides
# this: Skiing (Skiing.cpp:96-111) disallows the whole FIRE family. The
# default here returns `true` for every action so the filter is a no-op for
# all other games. NOTE: starting actions are emulated via xitari's
# `emulate()` which BYPASSES `noopIllegalActions`, so this filter is applied
# in `env_step!` only, NOT to `romsettings_starting_actions`.
romsettings_is_legal_action(::RomSettings, ::Integer) = true
# Per-game CONSOLE-SWITCH starting actions (SELECT=46, RESET=40), emulated
# AFTER the joystick `romsettings_starting_actions`. xitari's
# `getStartingActions` returns these as ordinary ALE actions, but they map to
# Event::ConsoleSelect/Reset (SWCHB bit1/bit0) rather than the joystick port,
# and jutari's `apply_action!` can't encode codes 46/40 — so `env_reset!`
# routes them through `console_switches!` instead. Surround is the headline
# case: getStartingActions = {SELECT, RESET} selects game variation 1 then
# starts it (Surround.cpp:135). Default empty.
romsettings_console_switch_starts(::RomSettings) = Int[]
# PAL vs NTSC. xitari auto-detects the TV format with a 60-frame probe
# (Console.cxx:199-206: PAL if ≥15 of frames 30-60 exceed 285 scanlines) and
# sets `myMaximumNumberOfScanlines` to 342 (PAL) vs 290 (NTSC) (TIA.cxx:206-211),
# which gates the max-scanlines frame cutoff (TIA.cxx:2003). For RAM
# conformance only that cutoff threshold matters (the PAL colour palette is
# render-only). Default NTSC (false). Surround + Air-Raid are PAL dumps; only
# surround's 312-line frame actually needs 342 (it would otherwise be split by
# the 290 cutoff — the #103/#106 partial-frame family).
romsettings_pal(::RomSettings) = false
# Task #110: the display HEIGHT (rows `get_screen` returns from Y_START),
# matching xitari's per-ROM `Display.Height` (Props.cxx default 210; PAL games
# that kept the default get auto-bumped to 250, e.g. surround/air_raid). NTSC
# default 210. Per-game PAL overrides return their stella.pro height.
romsettings_screen_height(::RomSettings) = 210

"""No-op RomSettings — never terminal, zero reward, joystick-only."""
mutable struct GenericRomSettings <: RomSettings
    GenericRomSettings() = new()
end

end # module
