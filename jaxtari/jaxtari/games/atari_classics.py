"""Per-game `RomSettings` — Pitfall, Beamrider, Enduro, Seaquest.

Four more scorers in the same mould as Pong / Breakout / Space
Invaders / Asteroids / Q*Bert / Ms Pac-Man. RAM addresses lifted
from `xitari/games/supported/`. Where the original xitari settings
implement subtle multi-frame logic (e.g. BeamRider's "lives blink
during death animation, latch only after stability"), the simpler
single-frame reading is preserved here — the score is the RL signal
that matters; the lives counter is only used for termination + a
display value.
"""

from __future__ import annotations

from jaxtari.console import Console
from jaxtari.games.more_games import _decode_bcd_chain
from jaxtari.games.rom_settings import RomSettings


# --------------------------------------------------------------------------- #
# Pitfall
# --------------------------------------------------------------------------- #

PITFALL_SCORE_ADDRS  = (0xD7, 0xD6, 0xD5)
PITFALL_LIVES_ADDR   = 0x80          # high nibble = lives byte
PITFALL_LOGO_ADDR    = 0x9E          # nonzero → cannot control player
PITFALL_INITIAL_SCORE = 2000


class PitfallRomSettings(RomSettings):
    """Pitfall — score from 3-byte BCD chain $D7/$D6/$D5. Lives
    encoded as a 0xA / 0x8 / lower magic value in the high nibble of
    $80. Terminal when lives = 0 AND the logo-timer $9E is nonzero
    (xitari's "game-over splash showing" detector).

    The initial score in xitari is 2000 — Pitfall starts the agent
    above zero so penalties read as positive deltas downward.
    """

    def __init__(self) -> None:
        self._prev_score: int = PITFALL_INITIAL_SCORE

    def reset(self) -> None:
        self._prev_score = PITFALL_INITIAL_SCORE

    def get_reward(self, console: Console) -> int:
        score = _decode_bcd_chain(console, PITFALL_SCORE_ADDRS)
        reward = score - self._prev_score
        self._prev_score = score
        return int(reward)

    def is_terminal(self, console: Console) -> bool:
        ram = console.bus.ram
        lives_byte = int(ram[PITFALL_LIVES_ADDR & 0x7F]) >> 4
        logo_timer = int(ram[PITFALL_LOGO_ADDR & 0x7F])
        return lives_byte == 0 and logo_timer != 0

    def lives(self, console: Console) -> int:
        lives_byte = int(console.bus.ram[PITFALL_LIVES_ADDR & 0x7F])
        if lives_byte == 0xA0:
            return 3
        if lives_byte == 0x80:
            return 2
        return 1


# --------------------------------------------------------------------------- #
# BeamRider
# --------------------------------------------------------------------------- #

BEAMRIDER_SCORE_ADDRS = (0x09, 0x0A, 0x0B)
BEAMRIDER_LIVES_ADDR  = 0x85
BEAMRIDER_TERM_ADDR   = 0x05         # 0xFF or low-nibble < 0


class BeamriderRomSettings(RomSettings):
    """BeamRider — score from 3-byte BCD chain $09/$0A/$0B (low
    addresses — they live in the bottom of zero-page). Lives count
    = RAM[$85] + 1. Terminal when RAM[$05] == 0xFF or its low nibble
    indicates an end-of-game state.
    """

    def __init__(self) -> None:
        self._prev_score: int = 0

    def reset(self) -> None:
        self._prev_score = 0

    def get_reward(self, console: Console) -> int:
        score = _decode_bcd_chain(console, BEAMRIDER_SCORE_ADDRS)
        reward = score - self._prev_score
        self._prev_score = score
        return int(reward)

    def is_terminal(self, console: Console) -> bool:
        return int(console.bus.ram[BEAMRIDER_TERM_ADDR & 0x7F]) == 0xFF

    def lives(self, console: Console) -> int:
        return int(console.bus.ram[BEAMRIDER_LIVES_ADDR & 0x7F]) + 1


# --------------------------------------------------------------------------- #
# Enduro
# --------------------------------------------------------------------------- #

ENDURO_SCORE_HI_ADDR = 0xAB
ENDURO_SCORE_LO_ADDR = 0xAC
ENDURO_LEVEL_ADDR    = 0xAD
ENDURO_DEATH_ADDR    = 0xAF
ENDURO_DEATH_VALUE   = 0xFF


def _bcd(byte: int) -> int:
    return (byte >> 4) * 10 + (byte & 0x0F)


class EnduroRomSettings(RomSettings):
    """Enduro — score is "cars passed", which is 2-byte BCD at
    $AB/$AC but xitari's convention subtracts that from a level-
    dependent baseline (200 for level 1, 300 for level 2+) so the
    agent's score *increases* as cars are passed. Level 0 (waiting
    to start) yields score 0. Terminal when $AF == 0xFF.
    """

    def __init__(self) -> None:
        self._prev_score: int = 0

    def reset(self) -> None:
        self._prev_score = 0

    def get_reward(self, console: Console) -> int:
        ram = console.bus.ram
        level = int(ram[ENDURO_LEVEL_ADDR & 0x7F])
        if level == 0:
            score = 0
        else:
            cars = _bcd(int(ram[ENDURO_SCORE_HI_ADDR & 0x7F])) * 100 \
                 + _bcd(int(ram[ENDURO_SCORE_LO_ADDR & 0x7F]))
            baseline = 200 if level == 1 else 300
            score = baseline - cars
        reward = score - self._prev_score
        self._prev_score = score
        return int(reward)

    def is_terminal(self, console: Console) -> bool:
        return int(console.bus.ram[ENDURO_DEATH_ADDR & 0x7F]) == ENDURO_DEATH_VALUE

    def lives(self, console: Console) -> int:
        return 0


# --------------------------------------------------------------------------- #
# Seaquest
# --------------------------------------------------------------------------- #

SEAQUEST_SCORE_ADDRS = (0xBA, 0xB9, 0xB8)
SEAQUEST_TERM_ADDR   = 0xA3          # nonzero = terminal


class SeaquestRomSettings(RomSettings):
    """Seaquest — score from 3-byte BCD $BA/$B9/$B8. Terminal when
    RAM[$A3] != 0 (xitari sentinel).
    """

    def __init__(self) -> None:
        self._prev_score: int = 0

    def reset(self) -> None:
        self._prev_score = 0

    def get_reward(self, console: Console) -> int:
        score = _decode_bcd_chain(console, SEAQUEST_SCORE_ADDRS)
        reward = score - self._prev_score
        self._prev_score = score
        return int(reward)

    def is_terminal(self, console: Console) -> bool:
        return int(console.bus.ram[SEAQUEST_TERM_ADDR & 0x7F]) != 0

    def lives(self, console: Console) -> int:
        # Seaquest's lives counter isn't reliably exposed in the
        # xitari source; consumers should rely on `is_terminal`.
        return 0
