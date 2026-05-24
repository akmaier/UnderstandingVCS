"""Per-game `RomSettings` — Asteroids, Q*Bert, Ms Pac-Man.

Three more scorers in the same mould as Pong / Breakout / Space
Invaders. RAM addresses and BCD layouts come from `xitari/games/
supported/Asteroids.cpp / QBert.cpp / MsPacman.cpp` and are mirrored
here byte-for-byte. See the docstring of each class for the layout.

A small `_decode_bcd_chain(addresses)` helper packs up the "score is
spread across N bytes, each holding 2 BCD digits, concatenate to get
the decimal value" pattern these games share. Pong's "score is a
single 0..21 integer" doesn't fit that mould and stays in
`pong.py`.
"""

from __future__ import annotations

from typing import Sequence

from jaxtari.console import Console
from jaxtari.games.rom_settings import RomSettings


# --------------------------------------------------------------------------- #
# Shared BCD-chain decoder
# --------------------------------------------------------------------------- #

def _bcd_byte(b: int) -> int:
    """Decode a BCD-packed byte (e.g. 0x42 → 42)."""
    return ((b >> 4) & 0x0F) * 10 + (b & 0x0F)


def _decode_bcd_chain(console: Console, addresses: Sequence[int]) -> int:
    """Treat each address as a BCD byte (2 decimal digits). The
    addresses are ordered HIGH-byte first — `(0xE8, 0xE9)` reads
    `bcd(ram[0xE8]) * 100 + bcd(ram[0xE9])`.
    """
    ram = console.bus.ram
    total = 0
    for addr in addresses:
        total = total * 100 + _bcd_byte(int(ram[addr & 0x7F]))
    return total


# --------------------------------------------------------------------------- #
# Asteroids
# --------------------------------------------------------------------------- #

ASTEROIDS_SCORE_HI_ADDR = 0x3E       # high BCD digits
ASTEROIDS_SCORE_LO_ADDR = 0x3D       # low BCD digits
ASTEROIDS_LIVES_ADDR    = 0x3C       # high nibble = lives
ASTEROIDS_SCORE_MULT    = 10         # game displays score * 10
ASTEROIDS_WRAP          = 100000     # score wrap point


class AsteroidsRomSettings(RomSettings):
    """Asteroids — BCD-packed score across $3E / $3D, multiplied by
    10 for display. Lives count from RAM[$3C] high nibble. Terminal
    at 0 lives. Handles the 100000-score wrap (xitari does +100000
    when the delta is negative, since BCD wraps below 0).
    """

    def __init__(self) -> None:
        self._prev_score: int = 0

    def reset(self) -> None:
        self._prev_score = 0

    def get_reward(self, console: Console) -> int:
        score = _decode_bcd_chain(
            console, (ASTEROIDS_SCORE_HI_ADDR, ASTEROIDS_SCORE_LO_ADDR)
        ) * ASTEROIDS_SCORE_MULT
        reward = score - self._prev_score
        if reward < 0:
            reward += ASTEROIDS_WRAP
        self._prev_score = score
        return int(reward)

    def is_terminal(self, console: Console) -> bool:
        return self.lives(console) == 0

    def lives(self, console: Console) -> int:
        # High nibble of $3C.
        byte = int(console.bus.ram[ASTEROIDS_LIVES_ADDR & 0x7F])
        return (byte - (byte & 0x0F)) >> 4


# --------------------------------------------------------------------------- #
# Q*Bert
# --------------------------------------------------------------------------- #

QBERT_SCORE_ADDRS = (0xDB, 0xDA, 0xD9)   # HIGH-to-LOW BCD bytes
QBERT_LIVES_ADDR  = 0x88
QBERT_LIVES_END   = 0xFE                 # xitari's "death" sentinel


class QbertRomSettings(RomSettings):
    """Q*Bert — score from three BCD-packed bytes $DB/$DA/$D9 (high
    to low). Lives from RAM[$88]; xitari uses `lives_value == 0xFE`
    as the terminal sentinel.
    """

    def __init__(self) -> None:
        self._prev_score: int = 0

    def reset(self) -> None:
        self._prev_score = 0

    def get_reward(self, console: Console) -> int:
        # xitari skips score updates on the terminal frame so the
        # agent doesn't get a huge negative reward — mirror that.
        if self.is_terminal(console):
            return 0
        score = _decode_bcd_chain(console, QBERT_SCORE_ADDRS)
        reward = score - self._prev_score
        self._prev_score = score
        return int(reward)

    def is_terminal(self, console: Console) -> bool:
        return int(console.bus.ram[QBERT_LIVES_ADDR & 0x7F]) == QBERT_LIVES_END

    def lives(self, console: Console) -> int:
        # The xitari convention is "lives_value (interpreted as int8)
        # + 2". A signed-cast keeps things sensible when the byte
        # crosses 0x80.
        byte = int(console.bus.ram[QBERT_LIVES_ADDR & 0x7F])
        signed = byte if byte < 0x80 else byte - 256
        return max(0, signed + 2)


# --------------------------------------------------------------------------- #
# Ms Pac-Man
# --------------------------------------------------------------------------- #

MSPACMAN_SCORE_ADDRS  = (0xF8, 0xF9, 0xFA)    # HIGH-to-LOW BCD bytes
MSPACMAN_LIVES_ADDR   = 0xFB                  # low nibble = lives
MSPACMAN_DEATH_TIMER  = 0xA7
MSPACMAN_DEATH_VALUE  = 0x53                  # xitari's terminal-frame sentinel


class MsPacmanRomSettings(RomSettings):
    """Ms Pac-Man — score from three BCD bytes $F8/$F9/$FA (high to
    low). Lives count from low nibble of $FB. Terminal when lives = 0
    AND the death-animation timer ($A7) reads 0x53 (xitari sentinel).
    """

    def __init__(self) -> None:
        self._prev_score: int = 0

    def reset(self) -> None:
        self._prev_score = 0

    def get_reward(self, console: Console) -> int:
        score = _decode_bcd_chain(console, MSPACMAN_SCORE_ADDRS)
        reward = score - self._prev_score
        self._prev_score = score
        return int(reward)

    def is_terminal(self, console: Console) -> bool:
        ram = console.bus.ram
        lives = int(ram[MSPACMAN_LIVES_ADDR & 0x7F]) & 0x0F
        death = int(ram[MSPACMAN_DEATH_TIMER & 0x7F])
        return lives == 0 and death == MSPACMAN_DEATH_VALUE

    def lives(self, console: Console) -> int:
        return int(console.bus.ram[MSPACMAN_LIVES_ADDR & 0x7F]) & 0x0F
