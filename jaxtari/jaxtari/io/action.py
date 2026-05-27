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


# SWCHA bit positions for each player's joystick (active-low).
#
# Stella / xitari layout — high nibble is P0, low nibble is P1, in both
# nibbles the order is RIGHT / LEFT / DOWN / UP from high bit to low.
_P0_RIGHT = 0x80
_P0_LEFT  = 0x40
_P0_DOWN  = 0x20
_P0_UP    = 0x10
_P1_RIGHT = 0x08
_P1_LEFT  = 0x04
_P1_DOWN  = 0x02
_P1_UP    = 0x01

# Per-player bit masks for "all directions in this nibble released" (i.e.
# all 1s = pulled high = nothing pressed). Used to default the *other*
# player's nibble when only one player is being driven.
_P0_ALL_RELEASED = _P0_RIGHT | _P0_LEFT | _P0_DOWN | _P0_UP   # 0xF0
_P1_ALL_RELEASED = _P1_RIGHT | _P1_LEFT | _P1_DOWN | _P1_UP   # 0x0F


# Decompose each Action into (direction_index_set, fire_pressed). The
# direction set carries which of {UP, RIGHT, LEFT, DOWN} are pressed,
# *independent* of the player wiring — `_action_bits_for_player` then
# resolves the index to the right nibble of SWCHA so the same Action
# enum can drive either player.
_DIR_UP, _DIR_RIGHT, _DIR_LEFT, _DIR_DOWN = 0, 1, 2, 3

_ACTION_DECODE: dict[Action, tuple[tuple[int, ...], bool]] = {
    Action.NOOP:          ((),                              False),
    Action.FIRE:          ((),                              True),
    Action.UP:            ((_DIR_UP,),                      False),
    Action.RIGHT:         ((_DIR_RIGHT,),                   False),
    Action.LEFT:          ((_DIR_LEFT,),                    False),
    Action.DOWN:          ((_DIR_DOWN,),                    False),
    Action.UPRIGHT:       ((_DIR_UP, _DIR_RIGHT),           False),
    Action.UPLEFT:        ((_DIR_UP, _DIR_LEFT),            False),
    Action.DOWNRIGHT:     ((_DIR_DOWN, _DIR_RIGHT),         False),
    Action.DOWNLEFT:      ((_DIR_DOWN, _DIR_LEFT),          False),
    Action.UPFIRE:        ((_DIR_UP,),                      True),
    Action.RIGHTFIRE:     ((_DIR_RIGHT,),                   True),
    Action.LEFTFIRE:      ((_DIR_LEFT,),                    True),
    Action.DOWNFIRE:      ((_DIR_DOWN,),                    True),
    Action.UPRIGHTFIRE:   ((_DIR_UP, _DIR_RIGHT),           True),
    Action.UPLEFTFIRE:    ((_DIR_UP, _DIR_LEFT),            True),
    Action.DOWNRIGHTFIRE: ((_DIR_DOWN, _DIR_RIGHT),         True),
    Action.DOWNLEFTFIRE:  ((_DIR_DOWN, _DIR_LEFT),          True),
}

# (player, direction_index) → SWCHA active-low bit mask.
_DIR_TO_BIT: dict[tuple[int, int], int] = {
    (0, _DIR_UP): _P0_UP, (0, _DIR_RIGHT): _P0_RIGHT,
    (0, _DIR_LEFT): _P0_LEFT, (0, _DIR_DOWN): _P0_DOWN,
    (1, _DIR_UP): _P1_UP, (1, _DIR_RIGHT): _P1_RIGHT,
    (1, _DIR_LEFT): _P1_LEFT, (1, _DIR_DOWN): _P1_DOWN,
}


def _action_bits_for_player(action: int, player: int) -> tuple[int, bool]:
    """Return `(swcha_pressed_bits, fire_pressed)` for `action` routed to
    `player`. The pressed-bits are active-low — the caller clears those
    bits in SWCHA, leaving the *other* player's nibble untouched.
    """
    dirs, fire = _ACTION_DECODE[Action(action)]
    bits = 0
    for d in dirs:
        bits |= _DIR_TO_BIT[(player, d)]
    return bits, fire


def apply_action(console: Console, action: int, *, player: int = 0) -> Console:
    """Drive one player's joystick + fire-button inputs from `action`.

    Returns a new `Console` whose RIOT SWCHA and the addressed player's
    INPT trigger reflect the requested state. The *other* player's
    nibble of SWCHA is left at whatever the previous frame's
    `apply_action` set it to (default = released), so the canonical
    two-player driving idiom is

        console = apply_action(console, p0_action, player=0)
        console = apply_action(console, p1_action, player=1)
        console = run_until_frame(console)

    `player` defaults to 0 so single-player code is unchanged.
    """
    if player not in (0, 1):
        raise ValueError(f"player must be 0 or 1, got {player}")
    pressed_bits, fire_pressed = _action_bits_for_player(action, player)

    # SWCHA active-low: start from the bus's current value (so the
    # untouched-nibble of the OTHER player is preserved across calls),
    # set this player's nibble to all-released, then clear the bits
    # that ARE pressed.
    nibble_mask = _P0_ALL_RELEASED if player == 0 else _P1_ALL_RELEASED
    prev = int(console.bus.riot.swcha_in)
    p_byte = (prev & ~nibble_mask) | (nibble_mask & ~pressed_bits)
    new_riot = set_swcha_input(console.bus.riot, p_byte)
    new_tia = set_trigger(console.bus.tia, player=player, pressed=fire_pressed)
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
    p0_difficulty_a: bool = False,
    p1_difficulty_a: bool = False,
) -> Console:
    """Set all four console switches on SWCHB.

    SELECT and RESET are momentary buttons (active-low when pressed).
    Colour / B&W and the two difficulty switches are level signals that
    games sample once at boot (and occasionally re-read).

    Defaults match xitari (and the standard Atari 2600 default for any
    cartridge without explicit `Console.LeftDifficulty` / `…Right…`
    overrides): **B/B difficulty** (amateur / kids mode), COLOR TV,
    SELECT and RESET released. For Atari Breakout specifically, the
    P1 difficulty bit doubles as the paddle-size toggle — A draws a
    4-bit small paddle, B draws an 8-bit large paddle (task #64).
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
