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

from jaxtari.games.atari_classics import (
    BEAMRIDER_LIVES_ADDR,
    BEAMRIDER_SCORE_ADDRS,
    BEAMRIDER_TERM_ADDR,
    BeamriderRomSettings,
    ENDURO_DEATH_ADDR,
    ENDURO_DEATH_VALUE,
    ENDURO_LEVEL_ADDR,
    ENDURO_SCORE_HI_ADDR,
    ENDURO_SCORE_LO_ADDR,
    EnduroRomSettings,
    PITFALL_INITIAL_SCORE,
    PITFALL_LIVES_ADDR,
    PITFALL_LOGO_ADDR,
    PITFALL_SCORE_ADDRS,
    PitfallRomSettings,
    SEAQUEST_SCORE_ADDRS,
    SEAQUEST_TERM_ADDR,
    SeaquestRomSettings,
)
from jaxtari.games.breakout import (
    BREAKOUT_LIVES_ADDR,
    BREAKOUT_SCORE_HI_ADDR,
    BREAKOUT_SCORE_LO_ADDR,
    BreakoutRomSettings,
)
from jaxtari.games.more_games import (
    ASTEROIDS_LIVES_ADDR,
    ASTEROIDS_SCORE_HI_ADDR,
    ASTEROIDS_SCORE_LO_ADDR,
    ASTEROIDS_SCORE_MULT,
    ASTEROIDS_WRAP,
    AsteroidsRomSettings,
    MSPACMAN_DEATH_TIMER,
    MSPACMAN_DEATH_VALUE,
    MSPACMAN_LIVES_ADDR,
    MSPACMAN_SCORE_ADDRS,
    MsPacmanRomSettings,
    QBERT_LIVES_ADDR,
    QBERT_LIVES_END,
    QBERT_SCORE_ADDRS,
    QbertRomSettings,
)
from jaxtari.games.pong import (
    PONG_P0_SCORE_ADDR,
    PONG_P1_SCORE_ADDR,
    PONG_TARGET_SCORE,
    PongRomSettings,
)
from jaxtari.games.joystick_starts import (
    AirRaidRomSettings,
    AmidarRomSettings,
    AsterixRomSettings,
    DoubleDunkRomSettings,
    ElevatorActionRomSettings,
    GopherRomSettings,
    GravitarRomSettings,
    JourneyEscapeRomSettings,
    PrivateEyeRomSettings,
    SkiingRomSettings,
    SurroundRomSettings,
    UpNDownRomSettings,
    YarsRevengeRomSettings,
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
    # Asteroids / Qbert / Ms Pac-Man (P6c follow-on, second batch).
    "AsteroidsRomSettings",
    "ASTEROIDS_SCORE_HI_ADDR",
    "ASTEROIDS_SCORE_LO_ADDR",
    "ASTEROIDS_LIVES_ADDR",
    "ASTEROIDS_SCORE_MULT",
    "ASTEROIDS_WRAP",
    "QbertRomSettings",
    "QBERT_SCORE_ADDRS",
    "QBERT_LIVES_ADDR",
    "QBERT_LIVES_END",
    "MsPacmanRomSettings",
    "MSPACMAN_SCORE_ADDRS",
    "MSPACMAN_LIVES_ADDR",
    "MSPACMAN_DEATH_TIMER",
    "MSPACMAN_DEATH_VALUE",
    # Pitfall / BeamRider / Enduro / Seaquest (P6c third batch).
    "PitfallRomSettings",
    "PITFALL_SCORE_ADDRS",
    "PITFALL_LIVES_ADDR",
    "PITFALL_LOGO_ADDR",
    "PITFALL_INITIAL_SCORE",
    "BeamriderRomSettings",
    "BEAMRIDER_SCORE_ADDRS",
    "BEAMRIDER_LIVES_ADDR",
    "BEAMRIDER_TERM_ADDR",
    "EnduroRomSettings",
    "ENDURO_SCORE_HI_ADDR",
    "ENDURO_SCORE_LO_ADDR",
    "ENDURO_LEVEL_ADDR",
    "ENDURO_DEATH_ADDR",
    "ENDURO_DEATH_VALUE",
    "SeaquestRomSettings",
    "SEAQUEST_SCORE_ADDRS",
    "SEAQUEST_TERM_ADDR",
    # Task #101 — starting-action-only joystick games (mirror of jutari).
    "AirRaidRomSettings",
    "AmidarRomSettings",
    "AsterixRomSettings",
    "DoubleDunkRomSettings",
    "ElevatorActionRomSettings",
    "GopherRomSettings",
    "GravitarRomSettings",
    "JourneyEscapeRomSettings",
    "PrivateEyeRomSettings",
    "SkiingRomSettings",
    "SurroundRomSettings",
    "UpNDownRomSettings",
    "YarsRevengeRomSettings",
]
