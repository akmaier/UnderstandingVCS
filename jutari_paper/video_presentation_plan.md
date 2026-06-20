# Supplementary Video — Plan (for discussion before implementation)

A ~5-minute narrated video for the AAAI-27 supplement that walks through the
paper and supplementary material and **shows off the bit-exact comparison
videos**. Built with the same reproducible pipeline as the previous paper
(`EmpiricalBoundsKnownOperator/presentation`), extended to embed real gameplay
clips, with a **fully anonymous synthetic voice-over** for double-blind review.

> **Status: PLAN AGREED — ready to build.** Decisions locked in §8. Nothing is
> built yet beyond a layout mock; awaiting final go-ahead.

### Decisions (locked)

- **Length:** ~5:00.
- **Final MP4:** committed to the repo (under `jutari_paper/presentation/`).
- **Segment 5 (centerpiece) = two phases:** (a) **one game large** — a single
  `XITARI | JUTARI | DIFFERENCE` clip full-width; then (b) **switch to a stacked
  grid**: top row = `XITARI | JUTARI | DIFFERENCE`, bottom row =
  `XITARI | JAXTARI | DIFFERENCE`, composited to **16:9 (1920×1080)**. Both
  DIFFERENCE panels are black → "two independent ports, both bit-exact" in one
  shot. Layout confirmed via a rendered mock.
- **jaxtari shown:** yes — it is the bottom row of the grid (side-by-side with jutari).
- **Voice:** Chatterbox built-in synthetic voice. **AAAI explicitly forbids
  "identifiable voices" in supplementary video**, so a synthetic voice is required,
  not merely preferred.
- **Continuous narration over all motion:** every clip *and* animation segment
  carries synchronized voice-over — there is no silent gameplay or silent
  animation. Each motion segment is trimmed/looped to exactly its narration
  length so audio and motion stay locked. The game-montage and the
  relaxation-animation segments are scripted as a **guided walkthrough of what is
  on screen** ("watch the difference panel stay black as the invaders descend"),
  not generic captions.

### AAAI-26/27 supplementary rules (verified)

- Supplementary material is **one `.zip` uploaded to OpenReview** (code + multimedia together).
- **Anonymization is mandatory:** AAAI urges authors to ensure a submitted video
  "avoids images of the authors, **identifiable voices**, university or lab logos,
  recognizable campus scenes, etc." → drives every anonymity choice here.
- **No published hard size/codec limit**; OpenReview enforces a max, and if data
  exceeds it authors upload a representative subset. → keep the MP4 modest (target
  well under ~50 MB, H.264).
- Supplementary deadline is one week after the paper deadline (was Aug 4 for AAAI-26).

---

## 1. Goals & hard constraints

- **Duration ~5:00** (target 4:45–5:15). Roughly 10 segments.
- **Audience:** AAAI reviewers + the wider community after acceptance. Self-contained: someone who has not read the paper should follow it.
- **Anonymity (non-negotiable — double-blind):**
  - Title slide author = **"Anonymous"**; no names, affiliations, emails, repo URL, or cluster/host names anywhere on screen or in narration.
  - **Voice = Chatterbox built-in default (synthetic) voice — NOT cloned from the author.** This is the key anonymity choice: no real human voice.
  - Strip all container metadata from the final MP4 (`-map_metadata -1`).
  - Narration says "we"/"the authors", never a name.
- **Reproducible & versioned:** everything (deck, script, build scripts, manifest) committed under `jutari_paper/presentation/`; one command rebuilds the MP4. The large `build/` intermediates and the final `.mp4` get a local `.gitignore` decision (see §6).

## 2. What we have (verified)

- **139 comparison videos** in `tools/comparison_videos/output/` — `<game>_xitari_vs_{jutari,jaxtari}.mp4`.
  - Format: **1440×672, 60 fps, no audio**; jutari clips are 30 s, jaxtari clips are 10 s.
  - Layout: **3 panels — `XITARI | <PORT> | DIFFERENCE`**. The DIFFERENCE panel is **solid black** because the frames are pixel-identical. *This is the visual money shot.*
- **Paper figures** in `jutari_paper/paper/figures/`: `fig_pipeline.pdf`, `fig_architecture.pdf`, `gpu_throughput.{pdf,png}`, `si_joystick_gradient.pdf`, `fig_xai_joystick.pdf`, `fig_relax_heatmap.pdf`, `fig_temp_heatmap.pdf`, `fig_timeline.pdf`, and two **animations**: `fig_alpha_anim.gif`, `fig_temp_anim.gif` (usable as short motion clips for the relaxation segment).
- **Tooling present:** `ffmpeg`, `ffprobe`, `pdftoppm` (Homebrew); **Chatterbox venv already exists** at `~/venvs/chatterbox`. No new installs needed.
- **Reusable scripts** from the previous paper: `split_script.py`, `tts_chatterbox.py`, and the `make_video.sh`/`make_video.md` pipeline (we adapt the driver to handle video clips, not just static slides).

## 3. Pipeline (reuse + one extension)

Identical to the previous paper for **static slides**:
`presentation.tex` → `pdflatex` → `presentation.pdf` → `pdftoppm` PNGs;
`script.md` → `split_script.py` → per-segment `.txt` → `tts_chatterbox.py`
(Chatterbox default voice) → per-segment `.wav`; `ffmpeg` muxes image+audio.

**New piece — a segment manifest** so segments can be *either* a beamer slide
*or* a gameplay clip, in one ordered timeline:

```
# segments.tsv  — columns: id  type  source  script_section
01  slide  3                                 S1     # beamer frame 1 (title)
05  clip   space_invaders_xitari_vs_jutari   S5a    # gameplay clip
...
```

- **slide segment:** PNG (from the deck) shown for the length of its narration (as before).
- **clip segment:** the comparison `.mp4` is the visuals; the TTS `.wav` is laid **on top** (clips have no audio). The clip is scaled+padded to 1920×1080 and **trimmed or slowed to exactly the narration length** (`setpts`/`-t`), so audio and video stay in lockstep. A thin caption bar (game name + "difference = 0") can be burned in via `drawtext`.
- Final step unchanged: per-segment MP4s → `concat` → `presentation.mp4`, then a metadata-strip pass.

All Chatterbox authoring rules from last time carry over: spell out every symbol
in English (no LaTeX read aloud), sentences < 20 words, no em-dashes/semicolons.

## 4. Storyboard (~5:00, 10 segments)

| # | Segment | Visual | ~sec | Narration gist |
|---|---|---|---|---|
| 1 | **Title** | Beamer title, "Anonymous", subtitle "AAAI-27 supplementary video" | 15 | One-line pitch: a fully-known yet differentiable complex system for XAI ground truth. |
| 2 | **The XAI ground-truth gap** | Beamer bullets (known-but-trivial vs complex-but-unknown) | 35 | Why explanation needs ground truth; today's dichotomy. |
| 3 | **Idea: the Atari VCS** | `fig_architecture.pdf` (VCS block diagram) | 30 | A real 1977 computer: complex, fully specified, and we make it differentiable. |
| 4 | **Two bit-exact ports** | `tab:ports` rendered as a slide (jutari/jaxtari, 64/64) | 30 | Built twice (Julia + JAX); both match xitari bit-for-bit on all 64 games. |
| 5 | **★ Conformance showcase** | **(a) one game large** (`XITARI\|JUTARI\|DIFFERENCE`, full-width, ~18 s) → **(b) stacked grid**: top `XITARI\|JUTARI\|DIFFERENCE`, bottom `XITARI\|JAXTARI\|DIFFERENCE`, 16:9, cycling 2–3 games (~42 s) | 60 | "Left: reference. Middle: our port. Right: the difference — solid black, every frame. And both ports agree: jutari on top, jaxtari below, both pixel-identical to xitari." The showcase. |
| 6 | **Soft equals Hard** | `fig_pipeline.pdf` + the `fig_alpha_anim`/`fig_temp_anim` GIFs as motion | 40 | ROM as weights, RAM as soft tape, branches as gates; forward bit-exact, surrogate gradients. |
| 7 | **GPU throughput** | `gpu_throughput.png` | 25 | jutari fastest per-env on CPU; jaxtari vmap-batches to ~3M env-steps/s on a commodity GPU. |
| 8 | **XAI proof of concept** | `si_joystick_gradient.pdf` (real Space Invaders gradient) | 35 | Gradients of a pixel w.r.t. ROM/inputs on a real ROM — attribution scored against truth. |
| 9 | **In the supplement** | Beamer bullets + a peek of `tab:games` / `fig_relax_heatmap` | 25 | Proofs, relaxation study, per-game ROM-hash/exactness table, and these comparison videos. |
| 10 | **Conclusion + release** | Beamer closing slide | 20 | The XAI testbed; full code released under the MIT license on acceptance. "Thank you." |

Total ≈ 4:55. Segment 5 is the centerpiece "cool videos" moment; segments 3/4/6/7/8/9 track the paper's own figure set so the video and PDF reinforce each other.

**Montage game pick (segment 5):** visually distinct, instantly recognizable, clean black-diff: **Space Invaders, Pong, Seaquest, Enduro** (jutari, 30 s sources trimmed to ~14 s each → ~56 s). Alternative: a 2×2 grid playing simultaneously (denser, ~25 s) — see §8 Q3.

## 5. File layout (new)

```
jutari_paper/
  video_presentation_plan.md      # this file
  presentation/
    presentation.tex              # beamer deck (anonymous), slides for segments 1–4,6,9,10
    script.md                     # per-segment narration (## Segment N)
    segments.tsv                  # ordered manifest: slide vs clip
    split_script.py               # reused (verbatim)
    tts_chatterbox.py             # reused (verbatim; default voice)
    build_video.sh                # adapted driver: handles slide + clip segments, strips metadata
    make_video.md                 # short doc (adapted)
    build/                        # intermediates (PNG/WAV/MP4) — gitignored
    presentation.mp4              # final (tracked or released-on-accept — see §8 Q4)
```

## 6. Build & verification steps (once approved)

1. Scaffold `jutari_paper/presentation/`; copy `split_script.py` + `tts_chatterbox.py`; write `presentation.tex`, `script.md`, `segments.tsv`, `build_video.sh`.
2. `pdflatex presentation.tex`; render slides to PNG.
3. Pre-process the 3–4 montage clips: trim to target length, scale+pad to 1920×1080, optional caption bar.
4. `split_script.py` + `tts_chatterbox.py` → per-segment WAVs (Chatterbox default voice).
5. `build_video.sh`: mux each segment (slide→image+audio; clip→video+overlaid audio), concat, then `ffmpeg -map_metadata -1` to scrub metadata.
6. **QA gate:** total duration 4:45–5:15; audio intelligible (spot-check the symbol-heavy segments 6–8); **no identifying info** on any frame or in metadata (`ffprobe` the output); file size sane for AAAI supplement (target < ~50 MB; re-encode/CRF-tune if needed).
7. Commit deck+script+scripts+manifest; decide on committing the MP4 (§8 Q4). Rebase-before-push as usual.

## 7. Risks / notes

- **Chatterbox on symbol-heavy lines** can stutter; the per-sentence chunking + short-sentence rule from last time mitigates it. Segments 6–8 need careful "spell it out" wording.
- **Clip ↔ narration sync:** we set each clip's duration to its narration length, so they can't drift. Slowing a 30 s clip to ~14 s means *trimming* (take the liveliest window), not slow-motion.
- **File size:** four 1080p clip segments + slides; H.264 CRF ~23 should land well under 50 MB. Will verify against AAAI's supplementary-material size/format limits (**Q1**).
- **Anonymity regressions:** the title slide, any burned-in captions, and the MP4 metadata are the three leak points; all three are explicitly handled.

## 8. Decisions (resolved)

1. **AAAI limits** — verified (see §1): single anonymized ZIP to OpenReview; no hard size cap; keep MP4 modest.
2. **Length** — **~5:00** (10 segments as storyboarded).
3. **Segment 5** — **two-phase**: one game large, then a stacked jutari-top / jaxtari-bottom grid at 16:9 (layout mock confirmed). Candidate games: Space Invaders (lead), then cycle e.g. Seaquest / Enduro / Pong.
4. **MP4** — **committed** to `jutari_paper/presentation/`.
5. **Voice** — Chatterbox synthetic default (required by AAAI's no-identifiable-voices rule). Pacing tuned via `--cfg-weight` if needed.
6. **jaxtari** — **shown**, as the bottom row of the segment-5 grid.

### Remaining before build
- Final go-ahead from the author.
- Pick the exact games for the grid cycle (default: Space Invaders → Seaquest → Enduro).
