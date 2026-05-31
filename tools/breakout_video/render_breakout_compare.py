"""Orchestrate the Breakout side-by-side comparison videos.

Two 60-second (3600-frame) videos, both driven by the *same* random
paddle action sequence:

  breakout_xitari_vs_jaxtari.mp4  — xitari | jaxtari | difference
  breakout_xitari_vs_jutari.mp4   — xitari | jutari  | difference

Steps:

  1. Generate a deterministic random action sequence.
  2. Dump xitari frames (subprocess to `tools/trace_dump`).
  3. Dump jaxtari frames (subprocess to `dump_jaxtari_frames.py`).
  4. Dump jutari frames (subprocess to `dump_jutari_frames.jl`).
  5. Build side-by-side composites, pixel difference, scale 3×.
  6. ffmpeg-encode each composite to a 60 fps MP4.

Usage:
  python3 tools/breakout_video/render_breakout_compare.py \\
      --out-dir tools/breakout_video/output

The intermediate frame dumps land in `<out-dir>/*.raw` so a re-run
that wants a different composite (different layout, scale, etc.)
can skip re-running the emulators.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# Make `tools.breakout_video` importable for the generate_actions /
# palette helpers.
sys.path.insert(0, str(REPO_ROOT))
from tools.breakout_video import (        # noqa: E402
    decode_palette,
    generate_actions,
    load_ntsc_palette,
)


FPS              = 60
N_FRAMES         = FPS * 60                              # 60-second videos
ACTION_FILE_NAME = 'breakout_random_actions.txt'
XITARI_OUT       = 'xitari_frames.raw'
JAXTARI_OUT      = 'jaxtari_frames.raw'
JUTARI_OUT       = 'jutari_frames.raw'
COMP_DIR         = 'composites'

# ROM-name-keyed output suffixes — for a non-default ROM the dumps + mp4s
# get a `<rom>_` prefix so multiple ROMs can share the same out-dir
# without clobbering each other.
def _rom_prefix(rom_path: Path) -> str:
    """`pong.bin` → `pong_`, `breakout.bin` → `breakout_`, etc. Default
    breakout uses an empty prefix to preserve the original file names."""
    stem = rom_path.stem
    return '' if stem == 'breakout' else f'{stem}_'

# Task #53 (vertical-alignment fix): jaxtari / jutari `get_screen()`
# now return the ALE-standard (210, 160) crop — `Display.YStart=34` +
# `Display.Height=210` — same as xitari. All three engines render
# the same scanline range (34..243), so the previous
# XITARI_TOP_CROP=18 workaround (which cropped xitari to 192 rows but
# kept jaxtari/jutari at scanline 0..191 — a 52-line misalignment) is
# no longer needed.
XITARI_TOP_CROP = 0
XITARI_ROWS     = 210
COLS            = 160

SCALE = 3          # 3× upscale → 480 × 576 panel, 1440 × 576 frame.


def _run(cmd: list[str], *, env=None) -> None:
    print(f"\n>>> {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True, env=env)


def dump_actions(out: Path, n: int, seed: int) -> None:
    actions = generate_actions(n, seed=seed)
    with open(out, 'w') as f:
        for a in actions:
            f.write(f"{a}\n")
    print(f"wrote {n} actions (seed={seed}) to {out}", file=sys.stderr)


def dump_xitari(rom: Path, actions: Path, out: Path, n: int) -> None:
    _run([
        sys.executable,
        str(REPO_ROOT / 'tools' / 'breakout_video' / 'dump_xitari_frames.py'),
        '--rom', str(rom),
        '--actions', str(actions),
        '--out', str(out),
        '--max-frames', str(n),
    ])


def dump_jaxtari(rom: Path, actions: Path, out: Path, n: int) -> None:
    _run([
        str(REPO_ROOT / 'jaxtari' / '.venv' / 'bin' / 'python3'),
        str(REPO_ROOT / 'tools' / 'breakout_video' / 'dump_jaxtari_frames.py'),
        '--rom', str(rom),
        '--actions', str(actions),
        '--out', str(out),
        '--max-frames', str(n),
    ])


def dump_jutari(rom: Path, actions: Path, out: Path, n: int) -> None:
    _run([
        'julia',
        '--project=' + str(REPO_ROOT / 'jutari'),
        str(REPO_ROOT / 'tools' / 'breakout_video' / 'dump_jutari_frames.jl'),
        '--rom', str(rom),
        '--actions', str(actions),
        '--out', str(out),
        '--max-frames', str(n),
    ])


# --------------------------------------------------------------------------- #
# Compositor
# --------------------------------------------------------------------------- #

def _load_raw(path: Path, n: int, h: int, w: int) -> np.ndarray:
    """Load a raw frame dump as a `(n_frames, h, w)` array. The
    declared `n` is a *maximum*; if the file is shorter (e.g. xitari
    stopped early when the game declared `done=true`), we use whatever
    is there. The caller is responsible for aligning frame counts
    across emulators."""
    buf = np.fromfile(path, dtype=np.uint8)
    frame_bytes = h * w
    if buf.size % frame_bytes != 0:
        raise RuntimeError(
            f"{path}: {buf.size} bytes is not a whole multiple of "
            f"{frame_bytes} ({h}*{w}) — corrupted dump?")
    actual = buf.size // frame_bytes
    if actual < n:
        print(f"  warning: {path.name} has {actual} frames, less than the "
              f"requested {n} — the emulator ended the episode early.",
              file=sys.stderr)
    return buf[: actual * frame_bytes].reshape(actual, h, w)


def _composite_frames(left: np.ndarray, right: np.ndarray,
                      palette: np.ndarray) -> np.ndarray:
    """left + right are `(n, 192, 160)` palette indices. Returns
    `(n, 192, 160 * 3, 3)` uint8 RGB — three panels stacked
    horizontally:
        | LEFT (xitari) | RIGHT (jaxtari/jutari) | DIFF |
    The diff highlights pixels where the two palettes diverge: a
    differing pixel is drawn as bright magenta on a black background,
    so visual regressions pop. Identical pixels stay black.
    """
    n = left.shape[0]
    h, w = left.shape[1], left.shape[2]
    rgb_left  = decode_palette(left,  palette)             # (n, h, w, 3)
    rgb_right = decode_palette(right, palette)             # (n, h, w, 3)
    diff_mask = (left != right)[..., None]                 # (n, h, w, 1)
    magenta = np.array([255, 0, 255], dtype=np.uint8)
    rgb_diff = (diff_mask * magenta).astype(np.uint8)
    return np.concatenate([rgb_left, rgb_right, rgb_diff], axis=2)


def _upscale(frames: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbour scale-up via numpy repeat."""
    return frames.repeat(scale, axis=1).repeat(scale, axis=2)


# Tiny 5×7 bitmap font, just the chars we need for "XITARI / JAXTARI /
# JUTARI / DIFFERENCE / |". Each glyph is a 5×7 string of 0/1 columns.
_FONT_5x7: dict[str, list[str]] = {
    'X': ['10001', '10001', '01010', '00100', '01010', '10001', '10001'],
    'I': ['11111', '00100', '00100', '00100', '00100', '00100', '11111'],
    'T': ['11111', '00100', '00100', '00100', '00100', '00100', '00100'],
    'A': ['01110', '10001', '10001', '11111', '10001', '10001', '10001'],
    'R': ['11110', '10001', '10001', '11110', '10100', '10010', '10001'],
    'J': ['00001', '00001', '00001', '00001', '00001', '10001', '01110'],
    'U': ['10001', '10001', '10001', '10001', '10001', '10001', '01110'],
    'D': ['11110', '10001', '10001', '10001', '10001', '10001', '11110'],
    'F': ['11111', '10000', '10000', '11110', '10000', '10000', '10000'],
    'E': ['11111', '10000', '10000', '11110', '10000', '10000', '11111'],
    'N': ['10001', '11001', '10101', '10011', '10001', '10001', '10001'],
    'C': ['01110', '10001', '10000', '10000', '10000', '10001', '01110'],
    ' ': ['00000'] * 7,
}


def _draw_label(canvas: np.ndarray, text: str, x: int, y: int,
                scale: int = 2, color=(255, 255, 255)) -> None:
    """Render `text` into `canvas` (an HxWx3 RGB array) at (x, y).
    Uses the inline 5×7 bitmap font scaled `scale`× — keeps the
    pipeline dependency-free (no PIL / freetype). Mutates `canvas`
    in place."""
    cx = x
    for ch in text.upper():
        if ch not in _FONT_5x7:
            cx += (5 + 1) * scale
            continue
        glyph = _FONT_5x7[ch]
        glyph_w = len(glyph[0]) * scale
        for r, row in enumerate(glyph):
            for c, bit in enumerate(row):
                if bit == '1':
                    canvas[y + r * scale: y + (r + 1) * scale,
                           cx + c * scale: cx + (c + 1) * scale] = color
        cx += (len(glyph[0]) + 1) * scale


def encode_video(left: np.ndarray, right: np.ndarray,
                 palette: np.ndarray, out_path: Path,
                 left_label: str, right_label: str) -> None:
    """Build the side-by-side composite, burn in the three panel
    labels via a tiny bitmap font (no PIL / freetype dependency),
    upscale, and stream into ffmpeg."""
    comp = _composite_frames(left, right, palette)        # (n, h, w*3, 3)

    # Add a 14-pixel header band at the top with the three panel
    # labels, then upscale. Header colour is dark grey so the white
    # text is readable.
    n, h, w3, _ = comp.shape
    band_h = 14
    header = np.full((n, band_h, w3, 3), 24, dtype=np.uint8)
    panel_w = w3 // 3
    label_y = 4
    # Draw labels into the first frame's header, then broadcast it
    # over all frames (much cheaper than per-frame text rendering).
    one_header = header[0].copy()
    # Centre each label inside its panel column.
    def _x_for(label: str, panel_idx: int) -> int:
        glyph_w = (5 + 1) * 1                              # scale=1
        text_w = len(label) * glyph_w
        return panel_idx * panel_w + (panel_w - text_w) // 2
    _draw_label(one_header, left_label,
                _x_for(left_label, 0), label_y, scale=1)
    _draw_label(one_header, right_label,
                _x_for(right_label, 1), label_y, scale=1)
    _draw_label(one_header, 'DIFFERENCE',
                _x_for('DIFFERENCE', 2), label_y, scale=1)
    header[:] = one_header[None]
    comp_with_header = np.concatenate([header, comp], axis=1)

    comp_scaled = _upscale(comp_with_header, SCALE)
    n2, height, width, _ = comp_scaled.shape
    assert n2 == n

    print(f"\nencoding {out_path.name}: {n} frames @ {width}×{height} (scale={SCALE})",
          file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}',
        '-r', str(FPS),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'medium',
        '-crf', '20',
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.stdin.write(comp_scaled.tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg returned {rc}")
    print(f"wrote {out_path}", file=sys.stderr)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rom', default=REPO_ROOT / 'xitari' / 'roms' / 'breakout.bin',
                   type=Path,
                   help='ROM to render. Defaults to breakout for backwards-compat. '
                        'Pong, etc., supported as long as the dumpers know the '
                        'RomSettings (see _SETTINGS_BY_BASENAME).')
    p.add_argument('--out-dir', default=REPO_ROOT / 'tools' / 'breakout_video' / 'output',
                   type=Path)
    p.add_argument('--n-frames', default=N_FRAMES, type=int)
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--skip-xitari',  action='store_true')
    p.add_argument('--skip-jaxtari', action='store_true')
    p.add_argument('--skip-jutari',  action='store_true')
    p.add_argument('--skip-encode',  action='store_true')
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = _rom_prefix(args.rom)
    actions_path = args.out_dir / (prefix + ACTION_FILE_NAME if prefix else ACTION_FILE_NAME)
    xitari_path  = args.out_dir / (prefix + XITARI_OUT)
    jaxtari_path = args.out_dir / (prefix + JAXTARI_OUT)
    jutari_path  = args.out_dir / (prefix + JUTARI_OUT)

    if not actions_path.exists():
        dump_actions(actions_path, args.n_frames, args.seed)
    else:
        print(f"reusing actions file: {actions_path}", file=sys.stderr)

    if not args.skip_xitari:
        dump_xitari(args.rom, actions_path, xitari_path, args.n_frames)
    if not args.skip_jaxtari:
        dump_jaxtari(args.rom, actions_path, jaxtari_path, args.n_frames)
    if not args.skip_jutari:
        dump_jutari(args.rom, actions_path, jutari_path, args.n_frames)

    if args.skip_encode:
        return 0

    palette = load_ntsc_palette()
    # Load whichever dumps are present; encode pairs only when both
    # emulator inputs are available. Lets the caller run jutari at
    # full 60 s while jaxtari (much slower per frame) only renders a
    # shorter clip.
    if not xitari_path.exists():
        print(f"xitari frames missing ({xitari_path}); skipping encode.",
              file=sys.stderr)
        return 0
    xitari_full = _load_raw(xitari_path, args.n_frames, 210, COLS)
    xitari      = xitari_full[:, XITARI_TOP_CROP:XITARI_TOP_CROP + XITARI_ROWS, :]
    jaxtari     = (_load_raw(jaxtari_path, args.n_frames, XITARI_ROWS, COLS)
                   if jaxtari_path.exists() else None)
    jutari      = (_load_raw(jutari_path, args.n_frames, XITARI_ROWS, COLS)
                   if jutari_path.exists() else None)

    def _align(*arrays):
        n = min(a.shape[0] for a in arrays)
        if n != args.n_frames:
            print(f"  aligning emulators to {n} common frames", file=sys.stderr)
        return tuple(a[:n] for a in arrays)

    rom_stem = args.rom.stem  # 'breakout', 'pong', etc.
    if jaxtari is not None:
        xi, jx = _align(xitari, jaxtari)
        encode_video(
            xi, jx, palette,
            args.out_dir / f'{rom_stem}_xitari_vs_jaxtari.mp4',
            left_label='xitari', right_label='jaxtari',
        )
    if jutari is not None:
        xi, jt = _align(xitari, jutari)
        encode_video(
            xi, jt, palette,
            args.out_dir / f'{rom_stem}_xitari_vs_jutari.mp4',
            left_label='xitari', right_label='jutari',
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
