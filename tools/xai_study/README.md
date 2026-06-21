# xai_study — experiment harness for Paper 2 (ground-truth XAI benchmark)

Experiments for the XAI study planned in `jutari_paper/../xai_paper/xai_paper_plan.md`.
The thesis: our bit-exact + differentiable Atari VCS lets us compute the **true
causal attribution** for any output, so every interpretability method (classic and
mechanistic) can be **scored against ground truth**, not eyeballed.

Two subjects, one microscope:
- **Phase A** (`phaseA_kording/`): the emulator as Jonas & Kording's "model
  organism" — replay the neuroscience battery on our architectural state, scored.
- **Phase B** (`phaseB_agents/`): DQN agents — audit modern XAI against the true
  causal pixel/object attribution from the oracle below.

The shared foundation is the **ground-truth attribution oracle**
(`ground_truth/`): the object every method is scored against.

## Reuse from Paper 1
- jutari (Julia/Zygote) + jaxtari (JAX) differentiable emulators, bit-exact.
- `tools/xai_si_gradient/` — real-ROM screen↔action gradients (the seed of the oracle).
- The IG-on-ROM experiment (Paper-1 P8); the conformance harness.
- **Key enabler:** SOFT-STE forward is *bit-exact to HARD* and GPU-batchable →
  exact interventions and end-to-end gradients run batched on GPU.

## Compute
Pilots run locally (M1 Max). Full sweeps go to the LME cluster — see
`xai_paper_plan.md` §4.6. Reuse the Paper-1 `tools/cluster/*.sbatch` pattern,
`/cluster/maier`, and the jaxtari GPU venv.

## Status
Specs only (this folder is the *preparation*). Pilot entry points are named in
each subdir's README; implement those first.
