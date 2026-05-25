"""Run jaxtari on Breakout with a given action sequence and dump
per-frame screens as a flat `(n_frames, 210, 160)` uint8 binary
file.

Task #53 (vertical-alignment fix): output shape is `(n, 210, 160)`,
not the old `(n, 192, 160)`. `env.get_screen()` now returns the
ALE-standard `Display.YStart=34` / `Display.Height=210` crop (same as
xitari), so the per-frame screen vertically aligns with
`dump_xitari_frames.py`'s output. The video composer
(`render_breakout_compare.py`) accordingly stopped cropping
xitari to 192 lines from row 18.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from jaxtari.env.stella_environment import StellaEnvironment


def _load_actions(path: Path) -> list[int]:
    out: list[int] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            out.append(int(line))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rom', required=True, type=Path)
    p.add_argument('--actions', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--max-frames', required=True, type=int)
    args = p.parse_args(argv)

    rom = np.fromfile(args.rom, dtype=np.uint8)
    env = StellaEnvironment(rom)
    # Match the ALE / xitari boot burn so frame 1 alignment matches
    # `tools/trace_dump`'s output.
    env.reset(boot_noop_steps=60, boot_reset_steps=4)

    actions = _load_actions(args.actions)
    n = min(args.max_frames, len(actions))

    frames = np.empty((n, 210, 160), dtype=np.uint8)
    for i in range(n):
        env.step(int(actions[i]))
        frames[i] = np.asarray(env.get_screen(), dtype=np.uint8)
        if (i + 1) % 300 == 0:
            print(f"  jaxtari: {i + 1}/{n} frames", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames.tofile(args.out)
    print(f"wrote {n} frames of shape (210, 160) to {args.out}",
          file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
