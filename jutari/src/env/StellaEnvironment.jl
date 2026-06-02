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
using ..IO: apply_action!, console_switches!, NOOP, LEFT, RIGHT,
            LEFTFIRE, RIGHTFIRE, UPLEFT, UPRIGHT, DOWNLEFT, DOWNRIGHT,
            UPLEFTFIRE, UPRIGHTFIRE, DOWNLEFTFIRE, DOWNRIGHTFIRE
using ..RomSettingsModule: RomSettings, GenericRomSettings,
                           romsettings_reset!, romsettings_is_terminal,
                           romsettings_get_reward, romsettings_lives,
                           romsettings_uses_paddles, romsettings_swap_paddles
using ..TIA: Y_START, VISIBLE_HEIGHT, set_paddle_resistance!

export StellaEnvironment, env_reset!, env_step!,
       get_screen, get_ram, game_over, lives, frame_number,
       act!, getScreen, getRAM, gameOver, getEpisodeFrameNumber

# Task #54 — paddle-action support. Same constants as xitari's
# `applyActionPaddles` (xitari/environment/ale_state.cpp:150 and
# ale_state.hpp:43-49) so paddle motion produced by a given action
# sequence matches xitari's. PADDLE_DELTA is the per-frame step;
# PADDLE_DEFAULT puts the paddle in the middle of its range.
const _PADDLE_MIN     = 27_450
const _PADDLE_MAX     = 790_196
const _PADDLE_DELTA   = 23_000
const _PADDLE_DEFAULT = (_PADDLE_MAX - _PADDLE_MIN) ÷ 2 + _PADDLE_MIN

# Actions that move the left paddle in xitari's `applyActionPaddles`:
# LEFT-family pushes the resistance UP, RIGHT-family pushes it DOWN.
const _ACTIONS_LEFT_INC = Set(Int.([LEFT,  LEFTFIRE,  UPLEFT,  DOWNLEFT,
                                    UPLEFTFIRE,  DOWNLEFTFIRE]))
const _ACTIONS_LEFT_DEC = Set(Int.([RIGHT, RIGHTFIRE, UPRIGHT, DOWNRIGHT,
                                    UPRIGHTFIRE, DOWNRIGHTFIRE]))

"""
    StellaEnvironment

Thin one-shot wrapper around a `Console` + `RomSettings`. Lifecycle:

    env = StellaEnvironment(rom_bytes)
    env_reset!(env)
    while !game_over(env)
        reward = env_step!(env, action)
        frame  = get_screen(env)
    end

Whether LEFT/RIGHT actions drive a joystick or a paddle is decided
per-game by `romsettings_uses_paddles(settings)` (default False —
override on per-game subtypes like `BreakoutRomSettings`). Mirror of
xitari's stella.pro-driven `m_use_paddles` autodetection.
"""
mutable struct StellaEnvironment
    console::Console
    settings::RomSettings
    terminal::Bool
    left_paddle::Int
    right_paddle::Int
end

StellaEnvironment(rom, settings::RomSettings = GenericRomSettings()) =
    StellaEnvironment(initial_console(rom), settings, false,
                      _PADDLE_DEFAULT, _PADDLE_DEFAULT)

"""
    env_reset!(env; boot_noop_steps=0, boot_reset_steps=0)

Reset the console (PC ← cart reset vector). For ALE / xitari parity
(matching `ALEInterface::resetGame`), pass `boot_noop_steps=60,
boot_reset_steps=4` — xitari's reset burns 60 deterministic NOOP
frames so the cart's startup routine settles, then 4 frames with the
RESET switch held. The PXC1 conformance harness uses these values;
the default of 0 preserves the historical jutari behaviour where the
caller decides the startup convention.
"""
function env_reset!(env::StellaEnvironment;
                    boot_noop_steps::Integer = 0,
                    boot_reset_steps::Integer = 0)
    console_reset!(env.console)
    romsettings_reset!(env.settings)
    env.terminal = false
    # PXC1-x round 5: for paddle games, push the default paddle
    # resistance into the TIA BEFORE the boot-burn loop. xitari does
    # this at construction via `resetPaddles`; jaxtari does the same
    # in `StellaEnvironment.reset()`. Without the pre-boot apply, the
    # boot frames run with INPT0/INPT1 = $80 (the centred default),
    # not the dump-pot cycle-threshold path — diverging from xitari
    # by the time the boot phase finishes. Mirror of the jaxtari fix.
    env.left_paddle  = _PADDLE_DEFAULT
    env.right_paddle = _PADDLE_DEFAULT
    if romsettings_uses_paddles(env.settings)
        _apply_paddle_action!(env, Int(NOOP))
    end

    # --- Boot-burn: NOOP frames -----------------------------------------
    for _ in 1:boot_noop_steps
        apply_action!(env.console, Int(NOOP))
        run_until_frame!(env.console)
    end

    # --- Boot-burn: RESET-switch frames ---------------------------------
    if boot_reset_steps > 0
        console_switches!(env.console; reset_pressed = true)
        for _ in 1:boot_reset_steps
            apply_action!(env.console, Int(NOOP))
            run_until_frame!(env.console)
        end
        console_switches!(env.console; reset_pressed = false)
    end

    return env
end

function env_step!(env::StellaEnvironment, action::Integer)
    env.terminal && return 0
    uses_paddles = romsettings_uses_paddles(env.settings)
    if uses_paddles
        _apply_paddle_action!(env, Int(action))
    end
    # Task #65: paddle games need `paddle_mode=true` to skip the SWCHA
    # write — xitari's `applyActionPaddles` only sets paddle resistance
    # + fire event, NOT joystick direction. Without this, pong reads
    # SWCHA and sees a phantom LEFT/RIGHT bit, branches differently, and
    # the paddle stays stuck despite the resistance update.
    apply_action!(env.console, action; paddle_mode = uses_paddles)
    run_until_frame!(env.console)
    reward = Int(romsettings_get_reward(env.settings, env.console))
    env.terminal = romsettings_is_terminal(env.settings, env.console)
    return reward
end

"""
    _apply_paddle_action!(env, action)

xitari `applyActionPaddles` — translate an action into ±PADDLE_DELTA
on the left paddle (the right paddle stays put because the action
enum only encodes one player). The new position is converted to a
paddle resistance and written into the TIA so INPT0's dump-pot
cycle threshold reflects the paddle position. Mirror of the
jaxtari `StellaEnvironment._apply_paddle_action`.
"""
function _apply_paddle_action!(env::StellaEnvironment, action::Int)
    if action in _ACTIONS_LEFT_INC
        env.left_paddle += _PADDLE_DELTA
    elseif action in _ACTIONS_LEFT_DEC
        env.left_paddle -= _PADDLE_DELTA
    end
    # Clamp.
    if env.left_paddle < _PADDLE_MIN
        env.left_paddle = _PADDLE_MIN
    elseif env.left_paddle > _PADDLE_MAX
        env.left_paddle = _PADDLE_MAX
    end
    # Task #66: route paddle resistance per xitari Paddles wiring. With
    # SwapPaddles=NO (Breakout convention) the user paddle reaches INPT0
    # (Pin Nine on the left controller); with SwapPaddles=YES (Pong /
    # Video Olympics) it reaches INPT1 (Pin Five). xitari handles this
    # inside `Paddles::Paddles(jack, event, swap)` via the
    # `myPinEvents[2/3][0/1]` wiring table — see
    # `xitari/emucore/Paddles.cxx`. Without this routing fix, jutari's
    # pong paddle update lands on INPT0 which the game never reads, so
    # the on-screen paddle stays frozen at the centred default.
    if romsettings_swap_paddles(env.settings)
        set_paddle_resistance!(env.console.bus.tia, 1, env.left_paddle)
        set_paddle_resistance!(env.console.bus.tia, 0, env.right_paddle)
    else
        set_paddle_resistance!(env.console.bus.tia, 0, env.left_paddle)
        set_paddle_resistance!(env.console.bus.tia, 1, env.right_paddle)
    end
    return nothing
end

"""
    get_screen(env) -> Matrix{UInt8} of size (VISIBLE_HEIGHT, SCREEN_WIDTH)
                                            = (210, 160)

Return the visible portion of the current framebuffer — matches
xitari/ALE's `Display.YStart=34` / `Display.Height=210` default crop.
The first 34 scanlines (VSYNC + VBLANK + any score-header area outside
xitari's display window) are cropped out, so jutari videos line up
vertically with xitari videos.

The full internal framebuffer (244 rows, scanlines 0..243) is still
on `env.console.bus.tia.framebuffer` for tests / debugging that want
the uncropped view.
"""
get_screen(env::StellaEnvironment) =
    @view env.console.bus.tia.framebuffer[Y_START + 1 : Y_START + VISIBLE_HEIGHT, :]
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
