# make_video — anonymous supplementary video

Turns the beamer deck (`presentation.tex`), the narration (`script.md`), the
comparison clips (`../../tools/comparison_videos/output/`), and the relaxation
animations (`../paper/figures/*.gif`) into one ~5-minute MP4
(`presentation.mp4`) with a **synthetic, anonymous** voice-over.

This reuses the `EmpiricalBoundsKnownOperator/presentation` pipeline
(beamer→PDF→PNG slides; `split_script.py` + `tts_chatterbox.py` Chatterbox TTS;
`ffmpeg` mux+concat) and extends it with a **segment manifest** (`segments.tsv`)
so gameplay/animation clips sit on the same timeline as the slides.

## One command

```bash
bash build_video.sh                 # default synthetic voice (anonymous)
TTS_VENV=/path/to/venv bash build_video.sh   # override venv location
```

Prereqs (already present on the build machine): `ffmpeg`, `ffprobe`,
`pdftoppm`, a LaTeX with `latexmk`, and the Chatterbox venv at
`~/venvs/chatterbox` (`pip install chatterbox-tts torch torchaudio soundfile`).

## What it does (steps in `build_video.sh`)

0. **Lint** `script.md` for Chatterbox triggers (sentences > 20 words, `;`/`:`/dashes).
1. Compile `presentation.tex` → `presentation.pdf`.
2. Render beamer pages → `build/page-NN.png`.
3. **Segment 5 clip** — one game large (`space_invaders_xitari_vs_jutari`), 16:9.
4. **Segment 6 clip** — stacked grid, jutari comparison on top, jaxtari on the
   bottom, cycling Space Invaders → Seaquest → Enduro (10 s each), 16:9.
5. **Segment 8 clip** — the two relaxation GIFs side by side, looped.
6. Split `script.md` (`## Slide N`) → `build/slide-NN.txt`; synthesise
   `build/slide-NN.wav` with Chatterbox (default voice, per-sentence chunking).
7. **Mux** each segment per `segments.tsv`: a slide is its PNG under its narration;
   a clip is the footage with the narration laid on top, **looped/trimmed to the
   narration length** so audio and motion stay locked (continuous voice-over over
   all motion). Every segment is normalised to 1920×1080, 30 fps, yuv420p, AAC.
8. **Concatenate** and **strip all metadata** (`-map_metadata -1`) for anonymity.

## Anonymity (double-blind / AAAI)

- Title slide author is `Anonymous Submission`; no names, affiliations, repo URLs,
  or host names anywhere on screen or in narration.
- The voice is Chatterbox's **synthetic default** (not cloned from any author).
  AAAI forbids "identifiable voices" in supplementary video.
- The final MP4 has its container metadata stripped.

## Editing

- **Narration:** edit `script.md` (keep the TTS rules: spell out symbols, short
  sentences, no `—`/`;`/`:`). Re-run `build_video.sh`.
- **Timeline / clips:** edit `segments.tsv` and the clip variables at the top of
  `build_video.sh` (`GRID_GAMES`, `GRID_SECS`, `ONE_LARGE`).
- **Slides:** edit `presentation.tex`.

`build/` is git-ignored; `presentation.mp4` and all sources are committed.
