# xai_study — experiment harness for Paper 2 (ground-truth XAI benchmark)

Experiments for the XAI study planned in `jutari_paper/../xai_paper/xai_paper_plan.md`.
The thesis: our bit-exact + differentiable Atari VCS lets us compute the **true
causal attribution** for any output, so every interpretability method (classic and
mechanistic) can be **scored against ground truth**, not eyeballed.

**Three traditions of understanding, one ground truth:**
- **Phase A** (`phaseA_kording/`): *mechanistic / neuroscience* — the emulator as
  Jonas & Kording's "model organism"; replay the neuroscience battery on our
  architectural state, scored against the known mechanism.
- **Phase B** (`phaseB_agents/`): *attributional / XAI* on DQN agents — **B1**
  attribution/saliency and **B2** mechanistic-interpretability (activation/
  attribution patching, SAEs, circuits, probing) — both scored vs the oracle.
- **Phase D** (`phaseD_behavioral/`): *behavioral / psychology* — probe the agent
  as a psychology participant (Binz & Schulz 2023; Shiffrin & Mitchell 2023) and
  test whether the inferred account matches the true mechanism.

The shared foundation is the **ground-truth attribution oracle**
(`ground_truth/`): the object every method is scored against. "Interpretable" is
operationalized as *ground-truth recovery* (Barbiero et al. 2025).

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
