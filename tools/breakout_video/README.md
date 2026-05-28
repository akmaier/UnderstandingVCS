# Breakout side-by-side comparison videos

Two MP4s comparing emulator outputs on a Breakout run with a
deterministic random paddle-motion action sequence (seed 42):

| file                                        | duration | layout                  |
|---------------------------------------------|----------|-------------------------|
| `output/breakout_xitari_vs_jaxtari.mp4`     | 10 s     | xitari \| jaxtari \| diff |
| `output/breakout_xitari_vs_jutari.mp4`      | 10 s     | xitari \| jutari  \| diff |
| `output_60s/breakout_xitari_vs_jutari.mp4`  | 60 s     | xitari \| jutari  \| diff |

The "diff" panel highlights every pixel where the two emulators
produce a different palette index in bright magenta; identical
pixels stay black.

**P3i-g (write-cycle threading) fixed the big rendering gaps.**
Earlier clips showed a thick magenta bar where the right grey
wall was pulled ~12 px inward and a grey wall segment embedded in
the middle of the brick rows. Root cause: Breakout draws its
walls + bricks by writing PF0/PF1/PF2 *mid-scanline* (racing the
beam), and our TIA was advanced once per CPU instruction, so those
writes landed at the instruction-START color clock — ~12 px early.
The fix threads sub-instruction CPU cycles into the TIA so a TIA
*write* flushes the exact number of color-clock cycles before it
applies (`Bus.pending_tia_cycles`), and the indexed zero-page
addressing modes now count their cycle-3 dummy read so `STA zp,X`
to PF*/COLU* flushes 4 cycles, not 3. The grey side walls now sit
flush at columns 0-7 / 152-159 in every panel, matching xitari
exactly, and the brick rows are continuous.

What residual magenta remains is the documented **PXC1 ~5-byte
RAM divergence** surfacing as two small artifacts:

  * a **1-PF-pixel center seam** (a 4 px gap at the x≈76-79
    playfield half-boundary on the brick rows) — a sub-pixel
    beam-racing detail that needs full per-cycle bus accuracy to
    close; and
  * a **~10 px paddle X offset** — the paddle position the ROM
    computes from its INPT0 dump-pot poll differs by a few RAM
    bytes (read-path timing), independent of sprite-render timing.

Vertical alignment, formerly listed as a gap, was the
`Display.YStart=34` / `Display.Height=210` crop that xitari
applies but jaxtari/jutari were not — that's **fixed as of
task #53**, so the scores, brick rainbow, and paddle all sit at
the same Y position in every panel now.

## Why is the jaxtari video only 10 s?

`StellaEnvironment.step` in jaxtari is currently **~2.5 s per
frame** (measured), so a 3600-frame 60 s video would take ~2.5
hours to render. The full 60 s xitari-vs-jaxtari render is kicked
off in the background and will replace the 10 s clip if the host
session has time to complete it. The 10 s clip is honest, complete,
and shows the comparison clearly.

The per-frame cost is Python overhead in JAX's eager-trace mode —
each scanline cycle goes through a Python-level dispatcher with a
fresh JAX trace. JIT-compiling `run_until_frame` would close this
gap; that's a real refactor, out of scope for this delivery.

`jutari`'s `env_step!` runs ~5 ms / frame so the 60 s side-by-side
with xitari ships in full.

## Reproducing

```bash
# Quick smoke test (60 frames):
python3 tools/breakout_video/render_breakout_compare.py \
    --out-dir /tmp/breakout_smoke --n-frames 60 --seed 42

# 10-second videos for both ports (jaxtari included):
python3 tools/breakout_video/render_breakout_compare.py \
    --out-dir tools/breakout_video/output --n-frames 600 --seed 42

# 60-second xitari-vs-jutari only (jaxtari skipped):
python3 tools/breakout_video/render_breakout_compare.py \
    --out-dir tools/breakout_video/output_60s --n-frames 3600 --seed 42 \
    --skip-jaxtari
```

The script orchestrates three subprocesses (xitari trace_dump,
jaxtari frame dumper, jutari frame dumper), then composites the
panels in pure NumPy with a tiny 5×7 bitmap font for the labels
and streams the result into `ffmpeg`. No PIL / freetype dependency
beyond NumPy + a working `ffmpeg` on PATH.

## Frame dumps (`*.raw`)

The intermediate `xitari_frames.raw` / `jaxtari_frames.raw` /
`jutari_frames.raw` are gitignored (each is ~20 MB and easy to
regenerate). Reuse them by re-running with `--skip-xitari
--skip-jaxtari --skip-jutari` to only redo the compositing step.
