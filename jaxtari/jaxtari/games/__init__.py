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

from jaxtari.games.breakout import (
    BREAKOUT_LIVES_ADDR,
    BREAKOUT_SCORE_HI_ADDR,
    BREAKOUT_SCORE_LO_ADDR,
    BreakoutRomSettings,
)
from jaxtari.games.pong import (
    PONG_P0_SCORE_ADDR,
    PONG_P1_SCORE_ADDR,
    PONG_TARGET_SCORE,
    PongRomSettings,
)
from jaxtari.games.rom_settings import GenericRomSettings, RomSettings
from jaxtari.games.space_invaders import (
    SI_LIVES_ADDR,
    SI_SCORE_HI_ADDR,
    SI_SCORE_LO_ADDR,
    SpaceInvadersRomSettings,
)

__all__ = [
    "GenericRomSettings",
    "RomSettings",
    "PongRomSettings",
    "PONG_P0_SCORE_ADDR",
    "PONG_P1_SCORE_ADDR",
    "PONG_TARGET_SCORE",
    "BreakoutRomSettings",
    "BREAKOUT_SCORE_LO_ADDR",
    "BREAKOUT_SCORE_HI_ADDR",
    "BREAKOUT_LIVES_ADDR",
    "SpaceInvadersRomSettings",
    "SI_SCORE_HI_ADDR",
    "SI_SCORE_LO_ADDR",
    "SI_LIVES_ADDR",
]
