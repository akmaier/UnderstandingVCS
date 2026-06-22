"""Test that jaxtari pong's user paddle reaches INPT1 (= SwapPaddles=YES).

Mirror of jutari `breakout ball-death` regression: after FIRE + a few
LEFT/RIGHT presses, RAM cells $04 and $3c should move (= paddle X
positions tracking the player's input). Pre-fix this stayed stuck at
the initial 0x6e / 0x6d because jaxtari put the user paddle on INPT0
(which pong with SwapPaddles=YES doesn't read as the user paddle).
"""
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / 'jaxtari'))

import numpy as np
import pytest
from jaxtari.env.stella_environment import StellaEnvironment
from jaxtari.games.pong import PongRomSettings

_PONG_ROM = _REPO / "xitari" / "roms" / "pong.bin"


@pytest.mark.skipif(not _PONG_ROM.exists(),
                    reason="pong.bin not present (gitignored ROM); run locally")
def test_pong_paddle_moves_under_right_action():
    rom = np.fromfile(str(_PONG_ROM), dtype=np.uint8)
    env = StellaEnvironment(rom, PongRomSettings())
    env.reset(boot_noop_steps=60, boot_reset_steps=4)
    # Push 20 NOOPs + 1 FIRE + 3 NOOPs + several RIGHT actions
    actions = [0]*20 + [1] + [0]*3 + [3]*8
    rams = []
    for a in actions:
        env.step(int(a))
        rams.append(np.asarray(env.get_ram(), dtype=np.uint8).copy())
    # The first RIGHT action is actions[24]. After env.step on this action,
    # rams[24] reflects the post-step state. Pre-fix jaxtari stayed at
    # the initial 0x6e / 0x6d (paddle didn't move). Post-fix it transitions
    # to xitari/jutari's 0x68 / 0x6a — the paddle moved off centre by
    # one tick.
    ram_after_first_right = rams[24]
    print(f'After first RIGHT: $04 = 0x{int(ram_after_first_right[0x04]):02x} (expected 0x68)')
    print(f'After first RIGHT: $3c = 0x{int(ram_after_first_right[0x3c]):02x} (expected 0x6a)')
    assert int(ram_after_first_right[0x04]) == 0x68, \
        f"$04 stuck at 0x{int(ram_after_first_right[0x04]):02x} — SwapPaddles broken?"
    assert int(ram_after_first_right[0x3c]) == 0x6a, \
        f"$3c stuck at 0x{int(ram_after_first_right[0x3c]):02x} — SwapPaddles broken?"
    print('PASS — jaxtari pong SwapPaddles=YES routing works.')


if __name__ == '__main__':
    test_pong_paddle_moves_under_right_action()
