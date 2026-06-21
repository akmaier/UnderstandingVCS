# Ground-truth attribution oracle (spec)

The object every interpretability method is scored against. Two independent
constructions; agreement between them is itself a reported result.

## (1) Intervention oracle (causal, exact)
For a chosen output `y` (a pixel, the score, an agent's chosen action, a Q-value)
and a candidate input unit `u` (a screen pixel/region, a detected object, a RAM
byte, a register/ROM byte):
- Replay deterministically to the target state, **intervene** on `u` (occlude /
  set to a baseline / replace), continue, and record Δ`y`.
- True causal-effect map = {Δ`y`(u)} over all `u`. This is *exact* (the emulator
  is deterministic + bit-exact); no model of the world is assumed.
- Batched on GPU via the **SOFT-STE exact-forward** path (bit-identical to HARD).

## (2) Gradient oracle (differential)
- `∂y/∂u` through the differentiable substrate (jutari/Zygote, jaxtari/JAX), and
  for agents through `emulator ∘ agent` end-to-end. Exact-forward; gradient is the
  SOFT relaxation's (Paper-1 Corollary).
- Use raw gradient and Integrated Gradients (a path integral to a baseline state).

## Cross-check
Report correlation between (1) and (2); flag where they disagree (non-smooth /
saturated points) — disagreement is informative, not a bug.

## Subjects of `y`
- Phase A: emulator outputs (next pixel, RAM cell, register) — true data-flow known.
- Phase B: agent action/Q — the "which input pixels/objects matter" ground truth.

## Entry point (pilot)
`oracle_pong.{jl,py}` — pixel-occlusion intervention map + IG map for one
agent/state on Pong; assert (1)↔(2) correlation; reuse `tools/xai_si_gradient/`.

## Outputs
`out/oracle_<game>_<state>.npz`: `delta_y[u]` (intervention), `grad_y[u]`,
`ig_y[u]`, metadata (game, frame, target, baseline).
