"""
    IO

ALE-style actions + joystick / console-switch helpers. Actions are
discrete integers (the canonical xitari / ALE 0..17 single-player set)
that get decoded into joystick direction bits on SWCHA and a trigger
bit on TIA INPT4.

SWCHA wiring (active-low):
  bit 7 = P0 RIGHT, bit 6 = P0 LEFT, bit 5 = P0 DOWN, bit 4 = P0 UP
  bits 0-3 mirror these for P1 (defaulted released in P6).

SWCHB carries console switches; see `console_switches!`.
"""
module IO

using ..ConsoleModule: Console
using ..RIOT: set_swcha_input!, set_swchb_input!
using ..TIA: set_trigger!

export Action, NUM_ACTIONS, apply_action!, console_switches!

# ALE-canonical Action set as an enum.
@enum Action begin
    NOOP            = 0
    FIRE            = 1
    UP              = 2
    RIGHT           = 3
    LEFT            = 4
    DOWN            = 5
    UPRIGHT         = 6
    UPLEFT          = 7
    DOWNRIGHT       = 8
    DOWNLEFT        = 9
    UPFIRE          = 10
    RIGHTFIRE       = 11
    LEFTFIRE        = 12
    DOWNFIRE        = 13
    UPRIGHTFIRE     = 14
    UPLEFTFIRE      = 15
    DOWNRIGHTFIRE   = 16
    DOWNLEFTFIRE    = 17
end

const NUM_ACTIONS = 18

const _P0_RIGHT = 0x80
const _P0_LEFT  = 0x40
const _P0_DOWN  = 0x20
const _P0_UP    = 0x10
const _P1_RIGHT = 0x08
const _P1_LEFT  = 0x04
const _P1_DOWN  = 0x02
const _P1_UP    = 0x01

const _P0_ALL_RELEASED = _P0_RIGHT | _P0_LEFT | _P0_DOWN | _P0_UP   # 0xF0
const _P1_ALL_RELEASED = _P1_RIGHT | _P1_LEFT | _P1_DOWN | _P1_UP   # 0x0F

# Direction → (P0 bit, P1 bit). Same enum drives either player; the
# `apply_action!` resolver picks the right nibble.
const _DIR_UP    = 0
const _DIR_RIGHT = 1
const _DIR_LEFT  = 2
const _DIR_DOWN  = 3

const _DIR_BIT = Dict{Tuple{Int,Int}, UInt8}(
    (0, _DIR_UP)    => UInt8(_P0_UP),    (0, _DIR_RIGHT) => UInt8(_P0_RIGHT),
    (0, _DIR_LEFT)  => UInt8(_P0_LEFT),  (0, _DIR_DOWN)  => UInt8(_P0_DOWN),
    (1, _DIR_UP)    => UInt8(_P1_UP),    (1, _DIR_RIGHT) => UInt8(_P1_RIGHT),
    (1, _DIR_LEFT)  => UInt8(_P1_LEFT),  (1, _DIR_DOWN)  => UInt8(_P1_DOWN),
)

# (direction-index tuple, fire_pressed). Direction indices are
# *player-agnostic* — `_action_bits_for_player` resolves them to the
# right SWCHA nibble.
const _ACTION_DECODE = Dict{Action, Tuple{NTuple{N, Int} where N, Bool}}(
    NOOP          => ((),                              false),
    FIRE          => ((),                              true),
    UP            => ((_DIR_UP,),                      false),
    RIGHT         => ((_DIR_RIGHT,),                   false),
    LEFT          => ((_DIR_LEFT,),                    false),
    DOWN          => ((_DIR_DOWN,),                    false),
    UPRIGHT       => ((_DIR_UP, _DIR_RIGHT),           false),
    UPLEFT        => ((_DIR_UP, _DIR_LEFT),            false),
    DOWNRIGHT     => ((_DIR_DOWN, _DIR_RIGHT),         false),
    DOWNLEFT      => ((_DIR_DOWN, _DIR_LEFT),          false),
    UPFIRE        => ((_DIR_UP,),                      true),
    RIGHTFIRE     => ((_DIR_RIGHT,),                   true),
    LEFTFIRE      => ((_DIR_LEFT,),                    true),
    DOWNFIRE      => ((_DIR_DOWN,),                    true),
    UPRIGHTFIRE   => ((_DIR_UP, _DIR_RIGHT),           true),
    UPLEFTFIRE    => ((_DIR_UP, _DIR_LEFT),            true),
    DOWNRIGHTFIRE => ((_DIR_DOWN, _DIR_RIGHT),         true),
    DOWNLEFTFIRE  => ((_DIR_DOWN, _DIR_LEFT),          true),
)

function _action_bits_for_player(action::Integer, player::Integer)
    dirs, fire = _ACTION_DECODE[Action(Int(action))]
    bits = UInt8(0)
    for d in dirs
        bits |= _DIR_BIT[(Int(player), d)]
    end
    return bits, fire
end

"""
    apply_action!(console, action; player=0)

Drive one player's joystick + fire-button inputs from `action`. The
*other* player's nibble of SWCHA is preserved across calls, so the
canonical two-player driving idiom is

    apply_action!(console, p0_action; player=0)
    apply_action!(console, p1_action; player=1)
    run_until_frame!(console)

`player` defaults to 0 so single-player code continues to work
unchanged. Does NOT advance the CPU — call `run_until_frame!` to
consume the inputs over a game frame.
"""
function apply_action!(console::Console, action::Integer; player::Integer = 0)
    (player == 0 || player == 1) || throw(ArgumentError("player must be 0 or 1, got $player"))
    pressed_bits, fire_pressed = _action_bits_for_player(action, player)

    nibble_mask = player == 0 ? UInt8(_P0_ALL_RELEASED) : UInt8(_P1_ALL_RELEASED)
    prev = console.bus.riot.swcha_in
    p_byte = (prev & ~nibble_mask) | (nibble_mask & ~pressed_bits)
    set_swcha_input!(console.bus.riot, UInt8(p_byte & 0xFF))
    set_trigger!(console.bus.tia, Int(player), fire_pressed)
    return console
end

# SWCHB bit layout
const _SWCHB_RESET           = 0x01
const _SWCHB_SELECT          = 0x02
const _SWCHB_COLOR_OR_BW     = 0x08
const _SWCHB_P0_DIFFICULTY   = 0x40
const _SWCHB_P1_DIFFICULTY   = 0x80

"""
    console_switches!(console; select_pressed=false, reset_pressed=false,
                                 color=true, p0_difficulty_a=true,
                                 p1_difficulty_a=true)

Set the four console switches on SWCHB.
"""
function console_switches!(
    console::Console;
    select_pressed::Bool = false,
    reset_pressed::Bool = false,
    color::Bool = true,
    p0_difficulty_a::Bool = false,                # task #64
    p1_difficulty_a::Bool = false,                # task #64
)
    b = 0xFF
    select_pressed && (b &= ~_SWCHB_SELECT)
    reset_pressed  && (b &= ~_SWCHB_RESET)
    color || (b &= ~_SWCHB_COLOR_OR_BW)
    p0_difficulty_a || (b &= ~_SWCHB_P0_DIFFICULTY)
    p1_difficulty_a || (b &= ~_SWCHB_P1_DIFFICULTY)
    set_swchb_input!(console.bus.riot, UInt8(b & 0xFF))
    return console
end

end # module
