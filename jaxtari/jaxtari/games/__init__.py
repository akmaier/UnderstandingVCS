"""Per-game scoring + termination rules.

The `RomSettings` base class is the contract front-ends use to read out
the current score, lives, and game-over flag from the running console.
Per-game subclasses inspect RAM at specific addresses (the addresses are
game-specific reverse-engineering work; see xitari/games/supported/* for
the original lookups).

P6 shipped only `GenericRomSettings` (the no-op stub). **P6c** adds
`PongRomSettings` — score detection + 21-point terminal — as the first
real per-game scorer. Breakout / Space Invaders / Pitfall etc. land in
follow-up commits each.
"""

from jaxtari.games.pong import (
    PONG_P0_SCORE_ADDR,
    PONG_P1_SCORE_ADDR,
    PONG_TARGET_SCORE,
    PongRomSettings,
)
from jaxtari.games.rom_settings import GenericRomSettings, RomSettings

__all__ = [
    "GenericRomSettings",
    "RomSettings",
    "PongRomSettings",
    "PONG_P0_SCORE_ADDR",
    "PONG_P1_SCORE_ADDR",
    "PONG_TARGET_SCORE",
]
