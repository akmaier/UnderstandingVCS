"""Base `RomSettings` interface for per-game scoring + termination.

Implementations override `is_terminal`, `get_reward`, and (optionally)
`lives`. Each is called by `StellaEnvironment.step` after the frame
finishes; the settings object reads the console's RAM / RIOT / cart
state to compute its return value.

`reset` is invoked at the start of each episode so the settings can
clear any running score / lives counters.

A subclass may need to retain state between frames (e.g. last frame's
score, to compute the per-step reward delta). For that we deliberately
keep RomSettings plain mutable Python — there's no JAX tracing path
through reward computation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jaxtari.console import Console


@runtime_checkable
class RomSettings(Protocol):
    """Per-game scoring + termination rules. Plain Python, not a PyTree."""

    def reset(self) -> None:
        """Called at the start of each episode."""

    def is_terminal(self, console: Console) -> bool:
        """Return True when the game is over."""

    def get_reward(self, console: Console) -> int:
        """Return the per-step reward (score delta since the previous call)."""

    def lives(self, console: Console) -> int:
        """Return the player's remaining lives, or 0 if the game has no
        explicit life counter."""

    def uses_paddles(self) -> bool:
        """Return True when the game expects paddle controllers
        (Breakout, Pong/Video Olympics, Warlords, Casino, …) so
        `StellaEnvironment` translates LEFT/RIGHT actions into
        INPT0 dump-pot paddle-position changes. xitari reads this
        from stella.pro's `Controller.Left "PADDLES"`; jaxtari
        encodes it directly on the per-game settings class.
        Default `False` matches the majority of ROMs (joystick-only).
        """

    def swap_paddles(self) -> bool:
        """Return True when the game's stella.pro entry has
        `Controller.SwapPaddles "YES"` (Pong / Video Olympics).
        With SwapPaddles=YES, the user's (left-controller) paddle is
        wired to INPT1 instead of INPT0. xitari handles this inside
        `Paddles::Paddles(jack, event, swap)` via the
        `myPinEvents[2/3][0/1]` wiring table. Default `False`
        matches Breakout / Warlords / Casino — only Pong (Video
        Olympics) flips this. Mirrors jutari's
        `romsettings_swap_paddles`.
        """


class GenericRomSettings:
    """No-op stub: reward = 0, never terminal, no lives, joystick-only.

    Use this when you want to run a ROM without per-game scoring (e.g.
    for collecting raw screen frames or building a goldens dataset).
    """

    def reset(self) -> None:
        return None

    def is_terminal(self, console: Console) -> bool:
        return False

    def get_reward(self, console: Console) -> int:
        return 0

    def lives(self, console: Console) -> int:
        return 0

    def uses_paddles(self) -> bool:
        # Default: assume joystick. Override in per-game subclasses
        # whose stella.pro entry has Controller.Left/Right "PADDLES".
        return False

    def swap_paddles(self) -> bool:
        # Default: assume no swap. Override in per-game subclasses
        # whose stella.pro entry has Controller.SwapPaddles "YES"
        # (currently only Pong / Video Olympics).
        return False
