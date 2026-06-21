# Ground-truth attribution oracle (spec)

The object every interpretability method is scored against. **Subject: the VCS
itself** (no agents). Two independent constructions; agreement between them is itself
a reported result.

## (1) Intervention oracle (causal, exact)
For a chosen VCS output `y` (a pixel, the score, a game event, a future state) and a
candidate cause `u` (a RAM byte, register, ROM byte, opcode, joystick input, or — for
mechanistic interp — an internal state variable):
- Replay deterministically to the target state, **intervene** on `u` (occlude / set
  to a baseline / replace / resample), continue, and record Δ`y`.
- True causal-effect map = {Δ`y`(u)} over all `u`. *Exact* (the emulator is
  deterministic + bit-exact); no model of the world is assumed.
- Batched on GPU via the **SOFT-STE exact-forward** path (bit-identical to HARD).

## (2) Gradient oracle (differential)
- `∂y/∂u` through the differentiable substrate (jutari/Zygote, jaxtari/JAX).
  Exact-forward; gradient is the SOFT relaxation's (Paper-1 Corollary).
- Use raw gradient and Integrated Gradients (a path integral to a baseline state).

## Cross-check
Report correlation between (1) and (2); flag where they disagree (non-smooth /
saturated points) — disagreement is informative, not a bug.

## Known semantics (tiers)
- **T1** causal (exact, all games) and **T2** hardware roles / data-flow (exact) —
  intrinsic, no labels.
- **T3** game-concept labels (ball-x, lives, score) — partial; sourced from
  OCAtari/AtariARI and **verified by intervention**; needed only for semantic metrics.

## Subjects of `y`
- A pixel / the score / a game event / a future state — true data-flow known (T1/T2).
- The same oracle serves Phases A–D (it is the shared measuring instrument).

## Entry point (pilot)
`oracle_pong.{jl,py}` — intervention map + IG map for one VCS output (e.g., the score
pixel) on Pong; assert (1)↔(2) correlation; reuse `tools/xai_si_gradient/`.

## Outputs
`out/oracle_<game>_<output>.npz`: `delta_y[u]` (intervention), `grad_y[u]`,
`ig_y[u]`, metadata (game, frame, output target, baseline).
