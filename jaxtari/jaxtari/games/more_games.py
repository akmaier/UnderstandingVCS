"""Per-game `RomSettings` — Asteroids, Q*Bert, Ms Pac-Man, Road Runner, Kangaroo.

Scorers in the same mould as Pong / Breakout / Space Invaders. RAM
addresses and BCD layouts come from `xitari/games/supported/
Asteroids.cpp / QBert.cpp / MsPacman.cpp / RoadRunner.cpp / Kangaroo.cpp`
and are mirrored here byte-for-byte. See the docstring of each class for
the layout.

A small `_decode_bcd_chain(addresses)` helper packs up the "score is
spread across N bytes, each holding 2 BCD digits, concatenate to get
the decimal value" pattern these games share. Pong's "score is a
single 0..21 integer" doesn't fit that mould and stays in
`pong.py`.
"""

from __future__ import annotations

from typing import Sequence

from jaxtari.console import Console
from jaxtari.games.rom_settings import GenericRomSettings, RomSettings


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

    def screen_y_start(self) -> int:
        # Task #113: qbert's stella.pro Display.YStart = 40. Render-only crop;
        # qbert keeps its existing #106 partial-frame semantics from
        # Generic-style scoring above.
        return 40

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


class MsPacmanRomSettings(GenericRomSettings):
    """Ms Pac-Man — score from three BCD bytes $F8/$F9/$FA (high to
    low). Lives count from low nibble of $FB. Terminal when lives = 0
    AND the death-animation timer ($A7) reads 0x53 (xitari sentinel).

    Extends GenericRomSettings (joystick-game env defaults) so it carries
    BOTH the RL decoding (below) AND the per-game env overrides. Task #111:
    ms_pacman's stella.pro sets `Emulation.HmoveBlanks "NO"` → hmove_blanks
    False (disables the 8px HMOVE comb; render-only). This unifies the two
    formerly-separate MsPacmanRomSettings: the decoding one (here) and a
    hmove_blanks-only stub in joystick_starts.py that used to SHADOW it in
    the games package export — which silently dropped lives/score/terminal.
    """

    def __init__(self) -> None:
        self._prev_score: int = 0

    def hmove_blanks(self) -> bool:
        return False

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


# --------------------------------------------------------------------------- #
# Road Runner
# --------------------------------------------------------------------------- #

# xitari `xitari/games/supported/RoadRunner.cpp::step`:
#   score: 4 single-nibble digits at $C9..$CC (low nibble each), LOW digit
#          first (mult 1,10,100,1000); 0xA is the "blank zero" sentinel → 0;
#          the assembled value is then *100.
#   lives: ($C4 & 0x7); reported lives = that + 1.
#   terminal: lives_byte == 0 AND (y_vel ($B9) != 0 OR x_vel_death ($BD) != 0)
#             — i.e. the runner is dead AND still moving (death animation).
ROADRUNNER_SCORE_ADDRS    = (0xC9, 0xCA, 0xCB, 0xCC)  # LOW digit first
ROADRUNNER_SCORE_MULT     = 100
ROADRUNNER_BLANK_DIGIT    = 0xA                        # '0, don't display'
ROADRUNNER_LIVES_ADDR     = 0xC4                       # low 3 bits = lives
ROADRUNNER_YVEL_ADDR      = 0xB9
ROADRUNNER_XVEL_DEATH_ADDR = 0xBD


class RoadRunnerRomSettings(RomSettings):
    """Road Runner — score is four single-nibble decimal digits at
    $C9..$CC (low digit first), with 0xA meaning "blank zero"; the
    assembled value is *100. Lives = ($C4 & 0x7) + 1. Terminal when
    the lives field is 0 AND the runner is still moving in its death
    animation (y-velocity $B9 != 0 OR death x-velocity $BD != 0).
    Mirrors xitari RoadRunner.cpp byte-for-byte.
    """

    def __init__(self) -> None:
        self._prev_score: int = 0

    def reset(self) -> None:
        self._prev_score = 0

    def _score(self, console: Console) -> int:
        ram = console.bus.ram
        score = 0
        mult = 1
        for addr in ROADRUNNER_SCORE_ADDRS:
            value = int(ram[addr & 0x7F]) & 0x0F
            if value == ROADRUNNER_BLANK_DIGIT:
                value = 0
            score += mult * value
            mult *= 10
        return score * ROADRUNNER_SCORE_MULT

    def get_reward(self, console: Console) -> int:
        score = self._score(console)
        reward = score - self._prev_score
        self._prev_score = score
        return int(reward)

    def is_terminal(self, console: Console) -> bool:
        ram = console.bus.ram
        lives_byte = int(ram[ROADRUNNER_LIVES_ADDR & 0x7F]) & 0x7
        y_vel = int(ram[ROADRUNNER_YVEL_ADDR & 0x7F])
        x_vel_death = int(ram[ROADRUNNER_XVEL_DEATH_ADDR & 0x7F])
        return lives_byte == 0 and (y_vel != 0 or x_vel_death != 0)

    def lives(self, console: Console) -> int:
        return (int(console.bus.ram[ROADRUNNER_LIVES_ADDR & 0x7F]) & 0x7) + 1


# --------------------------------------------------------------------------- #
# Kangaroo
# --------------------------------------------------------------------------- #

# xitari `xitari/games/supported/Kangaroo.cpp::step`:
#   score: getDecimalScore(0xA8, 0xA7) — two BCD bytes, $A7 is the HIGH
#          byte (thousands+hundreds), $A8 the LOW byte (tens+ones); *100.
#   lives: ($AD & 0x7) + 1.
#   terminal: $AD == 0xFF.
KANGAROO_SCORE_ADDRS = (0xA7, 0xA8)   # HIGH byte first (BCD pairs)
KANGAROO_SCORE_MULT  = 100
KANGAROO_LIVES_ADDR  = 0xAD
KANGAROO_GAME_OVER   = 0xFF           # xitari's terminal sentinel


class KangarooRomSettings(RomSettings):
    """Kangaroo — score from two BCD bytes $A7 (high) / $A8 (low), *100.
    Lives = ($AD & 0x7) + 1. Terminal when $AD == 0xFF (xitari's
    game-over sentinel). Mirrors xitari Kangaroo.cpp byte-for-byte.
    """

    def __init__(self) -> None:
        self._prev_score: int = 0

    def reset(self) -> None:
        self._prev_score = 0

    def get_reward(self, console: Console) -> int:
        score = _decode_bcd_chain(console, KANGAROO_SCORE_ADDRS) * KANGAROO_SCORE_MULT
        reward = score - self._prev_score
        self._prev_score = score
        return int(reward)

    def is_terminal(self, console: Console) -> bool:
        return int(console.bus.ram[KANGAROO_LIVES_ADDR & 0x7F]) == KANGAROO_GAME_OVER

    def lives(self, console: Console) -> int:
        return (int(console.bus.ram[KANGAROO_LIVES_ADDR & 0x7F]) & 0x7) + 1
