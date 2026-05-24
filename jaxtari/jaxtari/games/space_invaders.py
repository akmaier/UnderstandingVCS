"""Space Invaders `RomSettings` — score detection + lives + termination.

xitari `xitari/src/games/supported/SpaceInvaders.cpp` lookups:

    RAM[$E8]  score, BCD-packed high byte (high nibble = thousands,
              low nibble = hundreds)
    RAM[$E9]  score, BCD-packed low byte (high nibble = tens, low
              nibble = ones)
    RAM[$C9]  lives remaining (0-3)

Reward = score delta. Terminal once lives reaches 0.
"""

from __future__ import annotations

from jaxtari.console import Console
from jaxtari.games.rom_settings import RomSettings


SI_SCORE_HI_ADDR = 0xE8     # thousands + hundreds (BCD nibbles)
SI_SCORE_LO_ADDR = 0xE9     # tens + ones (BCD nibbles)
SI_LIVES_ADDR    = 0xC9


def _bcd(byte: int) -> int:
    return (byte >> 4) * 10 + (byte & 0x0F)


def _score(console: Console) -> int:
    # The Atari 2600's 128-byte RAM is mirrored across $80-$FF; the
    # canonical addresses xitari uses fall in that high mirror, so we
    # mask with 0x7F to land in the real 128-cell array.
    ram = console.bus.ram
    high = _bcd(int(ram[SI_SCORE_HI_ADDR & 0x7F]))   # thousands*10 + hundreds
    low  = _bcd(int(ram[SI_SCORE_LO_ADDR & 0x7F]))   # tens*10 + ones
    return high * 100 + low


def _lives(console: Console) -> int:
    return int(console.bus.ram[SI_LIVES_ADDR & 0x7F])


class SpaceInvadersRomSettings(RomSettings):
    """Space Invaders: ΔScore reward, lives counter, terminate at 0 lives."""

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
