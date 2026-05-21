"""ALE-style actions and joystick / console-switch helpers.

An "action" is a discrete integer the agent emits each frame. The 18
single-player actions map to combinations of joystick direction + fire
trigger. They follow the canonical xitari / ALE numbering so a port that
adopts our API can be a drop-in for code that talks to the original ALE.

Action → physical inputs:

  NOOP             → joystick centred, fire released
  FIRE             → joystick centred, fire pressed
  UP / DOWN / LEFT / RIGHT             → direction asserted, fire released
  UPRIGHT / UPLEFT / DOWNRIGHT / DOWNLEFT → diagonal asserted, fire released
  UPFIRE / DOWNFIRE / LEFTFIRE / RIGHTFIRE → direction + fire pressed
  UPRIGHTFIRE / UPLEFTFIRE / DOWNRIGHTFIRE / DOWNLEFTFIRE → diagonal + fire

Joystick wiring on the Atari (active-low — 0 = pressed):

  SWCHA (port A):
    bit 7 = P0 RIGHT
    bit 6 = P0 LEFT
    bit 5 = P0 DOWN
    bit 4 = P0 UP
    bit 3 = P1 RIGHT
    bit 2 = P1 LEFT
    bit 1 = P1 DOWN
    bit 0 = P1 UP

The fire buttons live in TIA INPT4 (P0) / INPT5 (P1), bit 7.

SWCHB carries the console switches. P6 exposes a single helper
`console_switches` for setting all four (SELECT, RESET, color/B&W,
P0/P1 difficulty); games inspect these via plain LDA on \$0282.
"""

from __future__ import annotations

from enum import IntEnum

from jaxtari.console import Console
from jaxtari.riot.system import (
    set_swcha_input,
    set_swchb_input,
)
from jaxtari.tia.system import set_trigger


class Action(IntEnum):
    """ALE-canonical single-player action set."""
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


NUM_ACTIONS = 18


# SWCHA bit positions for player 0's joystick (active-low).
_P0_RIGHT = 0x80
_P0_LEFT  = 0x40
_P0_DOWN  = 0x20
_P0_UP    = 0x10
# P1 mirrors these in bits 0..3 — defaulted released for P6 (single-player).
_P1_DEFAULT_HIGH_BITS = 0x0F  # all P1 directions released (high = 1)


# Decompose each Action into (joystick_bits_pressed, fire_pressed).
# joystick_bits_pressed is the OR of the active-low bit masks for
# directions that are pressed.
_ACTION_DECODE: dict[Action, tuple[int, bool]] = {
    Action.NOOP:          (0,                              False),
    Action.FIRE:          (0,                              True),
    Action.UP:            (_P0_UP,                         False),
    Action.RIGHT:         (_P0_RIGHT,                      False),
    Action.LEFT:          (_P0_LEFT,                       False),
    Action.DOWN:          (_P0_DOWN,                       False),
    Action.UPRIGHT:       (_P0_UP    | _P0_RIGHT,          False),
    Action.UPLEFT:        (_P0_UP    | _P0_LEFT,           False),
    Action.DOWNRIGHT:     (_P0_DOWN  | _P0_RIGHT,          False),
    Action.DOWNLEFT:      (_P0_DOWN  | _P0_LEFT,           False),
    Action.UPFIRE:        (_P0_UP,                         True),
    Action.RIGHTFIRE:     (_P0_RIGHT,                      True),
    Action.LEFTFIRE:      (_P0_LEFT,                       True),
    Action.DOWNFIRE:      (_P0_DOWN,                       True),
    Action.UPRIGHTFIRE:   (_P0_UP    | _P0_RIGHT,          True),
    Action.UPLEFTFIRE:    (_P0_UP    | _P0_LEFT,           True),
    Action.DOWNRIGHTFIRE: (_P0_DOWN  | _P0_RIGHT,          True),
    Action.DOWNLEFTFIRE:  (_P0_DOWN  | _P0_LEFT,           True),
}


def apply_action(console: Console, action: int) -> Console:
    """Drive the joystick + fire-button inputs from `action`.

    Returns a new `Console` whose RIOT SWCHA and TIA INPT4 reflect the
    requested state. Does NOT advance the CPU — call `run_until_frame`
    after this to actually consume the inputs over a game frame.
    """
    pressed_bits, fire_pressed = _ACTION_DECODE[Action(action)]
    # SWCHA: high nibble = P0 directions (active low). 0xFF means
    # "nothing pressed"; clear the bits that ARE pressed.
    p0_byte = (0xF0 & ~pressed_bits) | _P1_DEFAULT_HIGH_BITS
    new_riot = set_swcha_input(console.bus.riot, p0_byte)
    new_tia = set_trigger(console.bus.tia, player=0, pressed=fire_pressed)
    new_bus = console.bus._replace(riot=new_riot, tia=new_tia)
    return console._replace(bus=new_bus)


# SWCHB bit positions for the console switches (active-low for SELECT/
# RESET; the other two are level signals — see notes below).
_SWCHB_RESET            = 0x01   # 0 = pressed
_SWCHB_SELECT           = 0x02   # 0 = pressed
_SWCHB_COLOR_OR_BW      = 0x08   # 1 = colour, 0 = B&W
_SWCHB_P0_DIFFICULTY    = 0x40   # 1 = A (hard), 0 = B (easy)
_SWCHB_P1_DIFFICULTY    = 0x80   # 1 = A (hard), 0 = B (easy)


def console_switches(
    console: Console,
    *,
    select_pressed: bool = False,
    reset_pressed: bool = False,
    color: bool = True,
    p0_difficulty_a: bool = True,
    p1_difficulty_a: bool = True,
) -> Console:
    """Set all four console switches on SWCHB.

    SELECT and RESET are momentary buttons (active-low when pressed).
    Colour / B&W and the two difficulty switches are level signals that
    games sample once at boot (and occasionally re-read).
    """
    b = 0xFF
    if select_pressed:
        b &= ~_SWCHB_SELECT
    if reset_pressed:
        b &= ~_SWCHB_RESET
    if not color:
        b &= ~_SWCHB_COLOR_OR_BW
    if not p0_difficulty_a:
        b &= ~_SWCHB_P0_DIFFICULTY
    if not p1_difficulty_a:
        b &= ~_SWCHB_P1_DIFFICULTY
    new_riot = set_swchb_input(console.bus.riot, b & 0xFF)
    return console._replace(bus=console.bus._replace(riot=new_riot))
