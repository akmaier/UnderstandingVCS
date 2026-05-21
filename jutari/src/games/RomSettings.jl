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
       romsettings_get_reward, romsettings_lives

abstract type RomSettings end

# Required interface — subtypes override these via Julia multiple dispatch.

romsettings_reset!(::RomSettings)             = nothing
romsettings_is_terminal(::RomSettings, ::Console) = false
romsettings_get_reward(::RomSettings, ::Console) = 0
romsettings_lives(::RomSettings, ::Console)      = 0

"""No-op RomSettings — never terminal, zero reward."""
mutable struct GenericRomSettings <: RomSettings
    GenericRomSettings() = new()
end

end # module
