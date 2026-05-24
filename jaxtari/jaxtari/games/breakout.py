"""Breakout-specific `RomSettings` — score detection + lives + termination.

Breakout's RAM layout (xitari `xitari/src/games/supported/Breakout.cpp`):

    RAM[$77]  ones-digit of score (BCD nibble, 0-9)
    RAM[$76]  hundreds + tens digit (BCD-packed: high nibble = hundreds,
              low nibble = tens, 0-99). So total score = hundreds*100
              + tens*10 + ones, decoded from the two bytes.
    RAM[$39]  lives remaining (0-5)

Reward = score delta between successive frames; terminal once lives
hits 0 *and* the score has stabilised for a frame (the standard "end
of game" detector — Breakout latches lives = 0 the instant the last
ball is lost but lets the final score animation play out).

The simpler "terminal = lives == 0" approximation gives the same
end-of-episode signal for an RL agent without needing the
score-stabilisation hack — use it as the default behaviour.
"""

from __future__ import annotations

from jaxtari.console import Console
from jaxtari.games.rom_settings import RomSettings


BREAKOUT_SCORE_LO_ADDR = 0x77        # ones digit (0-9)
BREAKOUT_SCORE_HI_ADDR = 0x76        # tens (low nibble) + hundreds (high nibble)
BREAKOUT_LIVES_ADDR    = 0x39


def _bcd(byte: int) -> int:
    """Decode a BCD-packed byte to its decimal value."""
    return (byte >> 4) * 10 + (byte & 0x0F)


def _score(console: Console) -> int:
    # RAM is 128 bytes mirrored across $80-$FF — mask with 0x7F so
    # the canonical CPU-visible addresses land at the right cell.
    ram = console.bus.ram
    ones = int(ram[BREAKOUT_SCORE_LO_ADDR & 0x7F]) & 0x0F
    high = _bcd(int(ram[BREAKOUT_SCORE_HI_ADDR & 0x7F]))    # tens-of-tens + tens
    # xitari convention: high byte holds hundreds in its high nibble
    # and tens in its low nibble — so high = hundreds*10 + tens, then
    # final score = high*10 + ones.
    return high * 10 + ones


def _lives(console: Console) -> int:
    return int(console.bus.ram[BREAKOUT_LIVES_ADDR & 0x7F])


class BreakoutRomSettings(RomSettings):
    """Breakout: ΔScore reward, lives counter, terminate when lives = 0."""

    def __init__(self) -> None:
        self._prev_score: int = 0

    def reset(self) -> None:
        self._prev_score = 0

    def get_reward(self, console: Console) -> int:
        s = _score(console)
        r = s - self._prev_score
        self._prev_score = s
        return int(r)

    def is_terminal(self, console: Console) -> bool:
        return _lives(console) == 0

    def lives(self, console: Console) -> int:
        return _lives(console)
