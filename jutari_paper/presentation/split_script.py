#!/usr/bin/env python3
"""Split script.md into one text file per slide for the TTS step.

Reads `## Slide N` headings and writes `build/slide-NN.txt` (zero-padded).
Lines starting with `#` are treated as headings and dropped from the
spoken text; everything else is passed through verbatim.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


HEADING = re.compile(r"^## Slide (\d+)\b.*$")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: split_script.py SCRIPT.md OUT_DIR", file=sys.stderr)
        return 2
    script = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    current_idx: int | None = None
    buf: list[str] = []
    text = script.read_text()

    def flush():
        nonlocal buf, current_idx
        if current_idx is None:
            buf = []
            return
        out = out_dir / f"slide-{current_idx:02d}.txt"
        body = "\n".join(buf).strip()
        body = re.sub(r"\n{2,}", "\n\n", body)
        out.write_text(body + "\n")
        print(f"wrote {out} ({len(body)} chars)")
        buf = []

    for line in text.splitlines():
        m = HEADING.match(line)
        if m:
            flush()
            current_idx = int(m.group(1))
            continue
        if line.startswith("#"):
            continue
        buf.append(line)
    flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
