"""Drive xitari (via `tools/trace_dump`) on Breakout with an action
sequence and dump the per-frame screen indices as a flat binary file.

Output: a single uint8 array of shape `(n_frames, 210, 160)` written
contiguously. The reader (`render_breakout_compare.py`) reshapes it.

This script wraps the existing `trace_dump --screen` and parses its
JSONL output; it doesn't link xitari directly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rom', required=True, type=Path)
    p.add_argument('--actions', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--max-frames', required=True, type=int)
    p.add_argument('--trace-dump',
                   default=REPO_ROOT / 'tools' / 'trace_dump', type=Path)
    args = p.parse_args(argv)

    cmd = [
        str(args.trace_dump),
        '--rom', str(args.rom),
        '--actions', str(args.actions),
        '--screen',
        '--max-frames', str(args.max_frames),
        '--repeat-last-on-exhaust',
        # auto-reset so a Breakout / Asteroids run that loses all
        # lives within the action sequence keeps producing frames
        # instead of stopping short.
        '--auto-reset',
    ]
    print(f"running: {' '.join(cmd)}", file=sys.stderr)

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    frames: list[np.ndarray] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        h, w = rec['h'], rec['w']
        screen = np.frombuffer(bytes.fromhex(rec['screen']),
                               dtype=np.uint8).reshape(h, w)
        frames.append(screen)
        if len(frames) >= args.max_frames:
            break

    arr = np.stack(frames, axis=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(args.out)
    print(f"wrote {len(frames)} frames of shape {arr.shape[1:]} to {args.out}",
          file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
