#!/usr/bin/env python3
"""Compose the hard/soft-exact/relaxed divergence MP4 (4-B) from the frame
streams dumped by dump_divergence_frames.jl.

Panels (left->right): HARD | SOFT-exact | RELAXED (alpha,T) | DIFF.
HARD and SOFT-exact are byte-identical (Theorem 1: the executed soft step equals
the hard step), so the first two panels stay locked together while RELAXED
diverges; DIFF magenta-highlights RELAXED-vs-EXACT pixel differences.

    python3 make_divergence_video.py [--fps 20] [--label "alpha=5.5, T=0.15"]
"""
import os, sys, subprocess, argparse
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "video_out")
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))   # tools/ on path
from breakout_video import decode_palette, load_ntsc_palette     # noqa: E402

SCALE = 3


def _load(name):
    n, h, w = (int(x) for x in open(os.path.join(OUT, f"{name}_frames.shape")).read().split())
    a = np.fromfile(os.path.join(OUT, f"{name}_frames.raw"), dtype=np.uint8)
    return a.reshape(n, h, w)


def _header(width, labels, band_h=16):
    """Static dark header band with one label centred over each panel."""
    img = Image.new("RGB", (width, band_h), (24, 24, 24))
    d = ImageDraw.Draw(img)
    pw = width // len(labels)
    for i, t in enumerate(labels):
        w_t = d.textlength(t)
        d.text((i * pw + (pw - w_t) / 2, 3), t, fill=(235, 235, 235))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--label", default="alpha=5.5, T=0.15")
    ap.add_argument("--out", default=os.path.join(OUT, "divergence_si.mp4"))
    args = ap.parse_args()

    exact, relaxed = _load("exact"), _load("relaxed")
    n = min(len(exact), len(relaxed)); exact, relaxed = exact[:n], relaxed[:n]
    pal = load_ntsc_palette()
    rgb_e = decode_palette(exact, pal)                     # (n,h,w,3)
    rgb_r = decode_palette(relaxed, pal)

    # 3 panels: HARD | SOFT-STE (= HARD, Theorem 1) | SOFT (relaxed). No diff.
    panels = np.concatenate([rgb_e, rgb_e, rgb_r], axis=2)         # (n,h,3w,3)
    h, w4 = panels.shape[1], panels.shape[2]
    band = _header(w4, ["HARD", "SOFT-STE", f"SOFT {args.label}"])
    band = np.broadcast_to(band, (n, band.shape[0], w4, 3))
    frames = np.concatenate([band, panels], axis=1)               # (n, band+h, 4w, 3)
    frames = frames.repeat(SCALE, axis=1).repeat(SCALE, axis=2)
    H, W = frames.shape[1], frames.shape[2]

    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium",
           "-crf", "20", args.out]
    p = subprocess.run(cmd, input=np.ascontiguousarray(frames, np.uint8).tobytes(),
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[-800:])
    print(f"wrote {args.out}  ({n} frames @ {W}x{H}, {args.fps} fps)")


if __name__ == "__main__":
    main()
