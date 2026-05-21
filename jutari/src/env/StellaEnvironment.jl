"""
    Env

ALE-style RL interface over a `Console`. `reset!` puts the console at
the cart's reset vector; `step!(action)` applies the action, runs one
frame, returns the per-step reward; `get_screen` / `get_ram` expose the
visible state.

Phosphor blending is intentionally absent in P6.
"""
module Env

using ..ConsoleModule: Console, console_reset!, run_until_frame!, initial_console
using ..IO: apply_action!
using ..RomSettingsModule: RomSettings, GenericRomSettings,
                           romsettings_reset!, romsettings_is_terminal,
                           romsettings_get_reward, romsettings_lives

export StellaEnvironment, env_reset!, env_step!,
       get_screen, get_ram, game_over, lives, frame_number,
       act!, getScreen, getRAM, gameOver, getEpisodeFrameNumber

"""
    StellaEnvironment

Thin one-shot wrapper around a `Console` + `RomSettings`. Lifecycle:

    env = StellaEnvironment(rom_bytes)
    env_reset!(env)
    while !game_over(env)
        reward = env_step!(env, action)
        frame  = get_screen(env)
    end
"""
mutable struct StellaEnvironment
    console::Console
    settings::RomSettings
    terminal::Bool
end

StellaEnvironment(rom, settings::RomSettings = GenericRomSettings()) =
    StellaEnvironment(initial_console(rom), settings, false)

function env_reset!(env::StellaEnvironment)
    console_reset!(env.console)
    romsettings_reset!(env.settings)
    env.terminal = false
    return env
end

function env_step!(env::StellaEnvironment, action::Integer)
    env.terminal && return 0
    apply_action!(env.console, action)
    run_until_frame!(env.console)
    reward = Int(romsettings_get_reward(env.settings, env.console))
    env.terminal = romsettings_is_terminal(env.settings, env.console)
    return reward
end

get_screen(env::StellaEnvironment) = env.console.bus.tia.framebuffer
get_ram(env::StellaEnvironment)    = env.console.bus.ram
game_over(env::StellaEnvironment)  = env.terminal
lives(env::StellaEnvironment)      = Int(romsettings_lives(env.settings, env.console))
frame_number(env::StellaEnvironment) = Int(env.console.bus.tia.frame)

# ALE-API aliases (camelCase to match the original C++ ALE).
act!(env::StellaEnvironment, action::Integer) = env_step!(env, action)
getScreen(env::StellaEnvironment)              = get_screen(env)
getRAM(env::StellaEnvironment)                 = get_ram(env)
gameOver(env::StellaEnvironment)               = game_over(env)
getEpisodeFrameNumber(env::StellaEnvironment)  = frame_number(env)

end # module
