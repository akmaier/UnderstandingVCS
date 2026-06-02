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
       romsettings_uses_paddles, romsettings_swap_paddles

abstract type RomSettings end

# Required interface — subtypes override these via Julia multiple dispatch.

romsettings_reset!(::RomSettings)             = nothing
romsettings_is_terminal(::RomSettings, ::Console) = false
romsettings_get_reward(::RomSettings, ::Console) = 0
romsettings_lives(::RomSettings, ::Console)      = 0
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

"""No-op RomSettings — never terminal, zero reward, joystick-only."""
mutable struct GenericRomSettings <: RomSettings
    GenericRomSettings() = new()
end

end # module
