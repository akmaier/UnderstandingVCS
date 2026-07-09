# Real-ROM Space Invaders joystick-gradient XAI experiment

Scripts: `si_joystick_gradient.jl` + `si_joystick_fig.py`. Figure: `out/si_joystick_gradient.png` / `.pdf`.

This reproduces the main paper's synthetic-scene joystick-saliency result (Fig.
`fig_xai_joystick`) **on the real Space Invaders ROM**, on the bit-exact jutari
substrate. It uses the paper's documented solution — the **sub-pixel bilinear sampler**
(`tri(t)=max(0,1−|t|)`, spatial-transformer style) over the sprite position — *not* the
soft CPU rollout (see the substrate-limit note below for why the soft path cannot run a
real frame).

## Scene facts (established empirically; a few were corrected by the user)

- **The video and the env run at 30 fps: seconds = frame / 30.** So **35 s = env frame
  1050** (with `boot_noop_steps=60, boot_reset_steps=4`). (An earlier assumption of 60 fps
  → frame 2100 was wrong; the user corrected it.)
- `get_screen` is 210×160 palette indices; `get_ram` is the 128-byte RAM indexed by offset
  `$00–$7F` (= CPU `$80–$FF`).
- **Player-ship X = RAM offset `$1C`** (array index 29, CPU `$9C`). Holding RIGHT raises it;
  LEFT/NOOP hold it flat.
- **Frame 1050 (35 s) is the clean classic-colour scene**: black background, yellow
  invaders, brown shields, green ground — matches the reference movie. The cannon is
  present, **colour index 196**, rows 186–198, cols 34–41 at player-X = 35.
- A screen **palette change happens at frame ~2103 (≈70 s)**: background 0→182, cannon→114
  (the "teal/inverted" look). This is **beyond** the 60 s / 1800-frame divergence video, so
  the video correctly never shows it, and it is **not** a jutari regression (the HARD render
  is byte-identical to the reference at 35 s). Whether it is faithful SI at 70 s is
  UNVERIFIED — conformance only covers the 60-frame window; do not call it a bug without an
  xitari comparison.

## Method

Take the real 35 s scene, extract the real player-cannon footprint (colour 114, 48 px,
origin row 186 / col 34), drive its horizontal position `px` with a continuous joystick
(`joy_x(j) = px0 + STEP·(right − left)`, bypassing the controller→CPU path as the paper
does), and differentiate `screen ← sampler ← px ← joystick`.

## Results (all three soft variants: SOFT-STE, relaxed α=6/T=0.14, α=5.5/T=0.145)

- **Forward ∂screen/∂RIGHT**: nonzero on the cannon's leading/trailing edges (max |val| 34)
  — the recovered position gradient.
- **Naive ∂screen/∂RIGHT** through the integer sprite index: **0** (vanishes) — this is
  exactly why the sampler is needed.
- **Inverse ∂(move-cannon-right)/∂joystick = (up 0, down 0, left −35.7, right +35.7)** —
  recovers "push RIGHT"; up/down vanish.
- **All three variants are bit-identical (max |Δ| = 0)** — Theorem 1 is forward-exact, and
  the sampler's position gradient is independent of the α/T relaxation knobs, so the
  joystick saliency is robust across STE and both relaxed settings.

## Substrate limit (why the soft CPU path can't be used here)

The SOFT `_bus_read` folds the whole address space into 128 cells via `addr & 0x7F` (so TIA
`$06` aliases RAM `$86`). Promoting a real state and running `soft_run` therefore diverges
and is joystick-insensitive: the soft path is a **single-step-exact differentiable CPU
core**, not a full-frame emulator. Use the differentiable bilinear sampler for the
position gradient.
