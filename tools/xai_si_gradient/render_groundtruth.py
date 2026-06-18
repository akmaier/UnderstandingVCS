#!/usr/bin/env python3
"""Render the HARD ground-truth joystick-RIGHT saliency on the real Space
Invaders 35 s scene: the 35 s frame next to an |RIGHT-NOOP| finite-difference
heat overlay (what a faithful d screen / d right looks like). Run with system
python3 (needs numpy + PIL). Output: out/gt_saliency.png
"""
import os
import sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))  # tools/ on path
from breakout_video import decode_palette, load_ntsc_palette  # noqa: E402

SCALE = 3


def load_raw(name, dtype, shape):
    return np.fromfile(os.path.join(OUT, name), dtype=dtype).reshape(shape)


def main():
    h, w = (int(x) for x in open(os.path.join(OUT, "gt_base.shape")).read().split())
    base = load_raw("gt_base.raw", np.uint8, (h, w))
    sal = load_raw("gt_sal.raw", np.float32, (h, w))

    pal = load_ntsc_palette()
    rgb = decode_palette(base, pal).astype(np.float32)            # (h,w,3)

    # heat overlay: orange where the cannon moved, intensity = visit count
    s = sal / (sal.max() + 1e-6)
    heat = np.zeros_like(rgb)
    heat[..., 0] = 255 * s          # R
    heat[..., 1] = 140 * s          # G  -> orange
    dim = rgb * 0.45
    over = np.where(s[..., None] > 0, 0.25 * dim + 0.75 * heat, rgb)

    gap = np.full((h, 6, 3), 255, np.float32)
    panel = np.concatenate([rgb, gap, over], axis=1)
    panel = np.clip(panel, 0, 255).astype(np.uint8)
    img = Image.fromarray(panel).resize((panel.shape[1] * SCALE, h * SCALE),
                                        Image.NEAREST)
    img.save(os.path.join(OUT, "gt_saliency.png"))
    print("wrote out/gt_saliency.png   sal nz=%d max=%d  cannon rows %s" %
          ((sal > 0).sum(), int(sal.max()),
           str((int(np.where(sal.sum(1) > 0)[0].min()),
                int(np.where(sal.sum(1) > 0)[0].max())))))


if __name__ == "__main__":
    main()
