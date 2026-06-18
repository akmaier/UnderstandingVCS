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

    # Manifest: one relaxed stream per line "soft<i> alpha T".
    softs = []
    mpath = os.path.join(OUT, "manifest.txt")
    if os.path.exists(mpath):
        for ln in open(mpath):
            name, a, t = ln.split(); softs.append((name, float(a), float(t)))
    else:
        softs = [("relaxed", 0.0, 0.0)]            # back-compat single stream

    pal = load_ntsc_palette()
    exact = _load("exact")
    streams = [exact] + [_load(name) for name, _, _ in softs]
    n = min(len(s) for s in streams)
    rgb = [decode_palette(s[:n], pal) for s in streams]    # [exact, soft0, soft1, ...]

    # Panels: HARD | SOFT-STE (both = exact, Theorem 1) | one SOFT per setting.
    panels = np.concatenate([rgb[0], rgb[0]] + rgb[1:], axis=2)
    h, w4 = panels.shape[1], panels.shape[2]
    labels = ["HARD", "SOFT-STE"] + [f"SOFT a={a:g} T={t:g}" for _, a, t in softs]
    band = _header(w4, labels)
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
