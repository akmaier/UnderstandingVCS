"""Pong-specific `RomSettings` — score detection + termination.

Pong's two scores live at fixed RAM cells the cart writes into every
frame:

    RAM[$14]  player-0 (right paddle) score  — single byte, 0..21
    RAM[$15]  player-1 (left  paddle) score  — single byte, 0..21

The settings track the previous frame's score so `get_reward` can
return the per-step delta:

    reward = (p0_new - p0_prev) - (p1_new - p1_prev)

Positive when P0 (the agent's paddle, by convention) scores; negative
when the opponent scores. `is_terminal` fires once either side
reaches the standard target of 21 points.

Match these addresses against `xitari/src/games/supported/Pong.cpp` if
you need to cross-check — they're the canonical Stella-research /
ALE convention.
"""

from __future__ import annotations

from jaxtari.console import Console
from jaxtari.games.rom_settings import RomSettings


PONG_P0_SCORE_ADDR = 0x14
PONG_P1_SCORE_ADDR = 0x15
PONG_TARGET_SCORE  = 21


class PongRomSettings(RomSettings):
    """Score-aware settings for Atari 2600 Pong.

    Reward = (Δ P0 score) − (Δ P1 score) — P0 scoring a point is +1
    per frame the score increments, P1 scoring is −1.
    Terminal when either side hits 21.

    The reset / get_reward / is_terminal lifecycle is `episode_reset →
    (apply_action → run_frame → get_reward → is_terminal)*`. We keep
    the previous frame's scores so the delta can be computed on the
    fly without caching frames.
    """

    def __init__(self) -> None:
        self._p0_prev: int = 0
        self._p1_prev: int = 0

    # --- RomSettings protocol --------------------------------------------- #

    def reset(self) -> None:
        self._p0_prev = 0
        self._p1_prev = 0

    def is_terminal(self, console: Console) -> bool:
        p0, p1 = _scores(console)
        return p0 >= PONG_TARGET_SCORE or p1 >= PONG_TARGET_SCORE

    def get_reward(self, console: Console) -> int:
        p0, p1 = _scores(console)
        reward = (p0 - self._p0_prev) - (p1 - self._p1_prev)
        self._p0_prev = p0
        self._p1_prev = p1
        return int(reward)

    def lives(self, console: Console) -> int:
        # Pong has no explicit life counter — the score is the only
        # progress signal.
        return 0


def _scores(console: Console) -> tuple[int, int]:
    """Pull the (P0, P1) score bytes from the console's RAM. RAM is a
    `(128,)` uint8 array indexed by `addr & 0x7F`, so the canonical
    `$0014` / `$0015` map to indices 0x14 / 0x15 — both already inside
    the 128-byte window, but we mask for consistency with the games
    whose score addresses live in the $80-$FF mirror."""
    ram = console.bus.ram
    return (int(ram[PONG_P0_SCORE_ADDR & 0x7F]),
            int(ram[PONG_P1_SCORE_ADDR & 0x7F]))
