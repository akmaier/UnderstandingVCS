"""Test that jaxtari pong's user paddle reaches INPT1 (= SwapPaddles=YES).

Mirror of jutari `breakout ball-death` regression: after FIRE + a few
LEFT/RIGHT presses, RAM cells $04 and $3c should move (= paddle X
positions tracking the player's input). Pre-fix this stayed stuck at
the initial 0x6e / 0x6d because jaxtari put the user paddle on INPT0
(which pong with SwapPaddles=YES doesn't read as the user paddle).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('/Users/maier/Documents/code/UnderstandingVCS') / 'jaxtari'))

import numpy as np
from jaxtari.env.stella_environment import StellaEnvironment
from jaxtari.games.pong import PongRomSettings


def test_pong_paddle_moves_under_right_action():
    rom = np.fromfile('/Users/maier/Documents/code/UnderstandingVCS/xitari/roms/pong.bin', dtype=np.uint8)
    env = StellaEnvironment(rom, PongRomSettings())
    env.reset(boot_noop_steps=60, boot_reset_steps=4)
    # Push 20 NOOPs + 1 FIRE + 3 NOOPs + several RIGHT actions
    actions = [0]*20 + [1] + [0]*3 + [3]*8
    rams = []
    for a in actions:
        env.step(int(a))
        rams.append(np.asarray(env.get_ram(), dtype=np.uint8).copy())
    # After ~24 frames jutari/xitari have $04=0x68 (paddle moved off centre).
    # Pre-fix jaxtari stayed at 0x6e. Assert paddle has moved.
    ram24 = rams[23]   # index 23 = frame 24 (1-indexed) since the loop ran 24
    print(f'After 24 frames: $04 = 0x{int(ram24[0x04]):02x} (expected 0x68 = paddle moved)')
    print(f'After 24 frames: $3c = 0x{int(ram24[0x3c]):02x} (expected 0x6a)')
    assert int(ram24[0x04]) == 0x68, f"$04 stuck at 0x{int(ram24[0x04]):02x} — SwapPaddles broken?"
    assert int(ram24[0x3c]) == 0x6a, f"$3c stuck at 0x{int(ram24[0x3c]):02x} — SwapPaddles broken?"
    print('PASS — jaxtari pong SwapPaddles=YES routing works.')


if __name__ == '__main__':
    test_pong_paddle_moves_under_right_action()
