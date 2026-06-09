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
    """Breakout: ΔScore reward, lives counter, terminate when lives = 0.

    Mirrors `xitari/games/supported/Breakout.cpp`:
      - score    = decode_score(RAM[\$76], RAM[\$77])
      - lives    = RAM[\$39]
      - started  = first observation of lives == 5 (sticky)
      - terminal = started && lives == 0

    The sticky `started` latch avoids the boot-time edge case where
    `lives` is briefly 0 (the RAM is zeroed at reset, then the cart's
    boot code initializes lives to 5). Without `started`, a naive
    `terminal = lives == 0` check would briefly return True during
    boot — surviving for one StellaEnvironment.step() call before the
    boot completes — which is enough to confuse the auto-reset loop.
    Matches jutari/src/games/PaddleGames.jl::BreakoutRomSettings.
    """

    def __init__(self) -> None:
        self._prev_score: int = 0
        self._started:    bool = False
        self._terminal:   bool = False
        self._lives_now:  int = 0

    def reset(self) -> None:
        self._prev_score = 0
        self._started    = False
        self._terminal   = False
        self._lives_now  = 0

    def _update(self, console: Console) -> None:
        byte_val = _lives(console)
        if not self._started and byte_val == 5:
            self._started = True
        self._terminal  = self._started and byte_val == 0
        self._lives_now = byte_val

    def get_reward(self, console: Console) -> int:
        s = _score(console)
        r = s - self._prev_score
        self._prev_score = s
        return int(r)

    def is_terminal(self, console: Console) -> bool:
        self._update(console)
        return self._terminal

    def lives(self, console: Console) -> int:
        self._update(console)
        # xitari: `int lives() const { return isTerminal() ? 0 : m_lives; }`
        return 0 if self._terminal else self._lives_now

    def uses_paddles(self) -> bool:
        # Breakout is a paddle game (xitari stella.pro: `Controller.Left
        # "PADDLES"`). Setting this True makes StellaEnvironment translate
        # LEFT/RIGHT actions into INPT0 dump-pot paddle-position changes.
        return True

    def starting_actions(self) -> list[int]:
        # Breakout has no per-game startup pose in xitari's
        # `getStartingActions`. Explicit override needed because
        # RomSettings is a Protocol — see task #81/#82.
        return []
