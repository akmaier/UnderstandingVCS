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
const _P1_DEFAULT_HIGH_BITS = 0x0F

# (pressed_bits, fire_pressed)
const _ACTION_DECODE = Dict{Action, Tuple{UInt8, Bool}}(
    NOOP          => (UInt8(0),                              false),
    FIRE          => (UInt8(0),                              true),
    UP            => (UInt8(_P0_UP),                         false),
    RIGHT         => (UInt8(_P0_RIGHT),                      false),
    LEFT          => (UInt8(_P0_LEFT),                       false),
    DOWN          => (UInt8(_P0_DOWN),                       false),
    UPRIGHT       => (UInt8(_P0_UP | _P0_RIGHT),             false),
    UPLEFT        => (UInt8(_P0_UP | _P0_LEFT),              false),
    DOWNRIGHT     => (UInt8(_P0_DOWN | _P0_RIGHT),           false),
    DOWNLEFT      => (UInt8(_P0_DOWN | _P0_LEFT),            false),
    UPFIRE        => (UInt8(_P0_UP),                         true),
    RIGHTFIRE     => (UInt8(_P0_RIGHT),                      true),
    LEFTFIRE      => (UInt8(_P0_LEFT),                       true),
    DOWNFIRE      => (UInt8(_P0_DOWN),                       true),
    UPRIGHTFIRE   => (UInt8(_P0_UP | _P0_RIGHT),             true),
    UPLEFTFIRE    => (UInt8(_P0_UP | _P0_LEFT),              true),
    DOWNRIGHTFIRE => (UInt8(_P0_DOWN | _P0_RIGHT),           true),
    DOWNLEFTFIRE  => (UInt8(_P0_DOWN | _P0_LEFT),            true),
)

"""
    apply_action!(console, action)

Drive SWCHA + INPT4 from `action`. Does NOT advance the CPU — call
`run_until_frame!` after this to consume the inputs over a game frame.
"""
function apply_action!(console::Console, action::Integer)
    pressed_bits, fire_pressed = _ACTION_DECODE[Action(Int(action))]
    p0_byte = ((0xF0 & ~pressed_bits) | _P1_DEFAULT_HIGH_BITS) & 0xFF
    set_swcha_input!(console.bus.riot, UInt8(p0_byte))
    set_trigger!(console.bus.tia, 0, fire_pressed)
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
    p0_difficulty_a::Bool = true,
    p1_difficulty_a::Bool = true,
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
