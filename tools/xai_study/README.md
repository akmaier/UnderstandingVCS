# xai_study — experiment harness for Paper 2 (ground-truth interpretability benchmark)

Experiments for the XAI study planned in `../xai_paper/xai_paper_plan.md`.
**Subject: the Atari VCS itself** (chip + program + game logic) — *not* learned
agents. The thesis: our bit-exact + differentiable VCS lets us compute the **true
causal structure** of any output, so every interpretability method can be **scored
against ground truth**, not eyeballed. Every phase runs on the current substrate.

**The interpretability toolkit, applied to one fully-known system:**
- **Phase A** (`phaseA_kording/`): *neuroscience / mechanistic* — replay the Jonas &
  Kording battery on the VCS state, scored against the known mechanism.
- **Phase B** (`phaseB_attribution/`): *attribution / XAI* — explain a VCS output
  (pixel / score / event) from its inputs/state; score vs the oracle.
- **Phase C** (`phaseC_mechanistic/`): *mechanistic interpretability* — activation/
  attribution patching, SAEs, circuits, probing on the VCS state, scored vs the
  *known* circuit.
- **Phase D** (`phaseD_behavioral/`): *behavioral / psychology* — probe the game's own
  decision logic (e.g., a built-in opponent AI) as a participant; test whether the
  inferred account matches the true code.

The shared foundation is the **ground-truth attribution oracle** (`ground_truth/`):
the object every method is scored against. "Interpretable" is operationalized as
*ground-truth recovery* (Barbiero et al. 2025).

## Reuse from Paper 1
- jutari (Julia/Zygote) + jaxtari (JAX) differentiable emulators, bit-exact.
- `tools/xai_si_gradient/` — real-ROM screen↔input gradients (the seed of the oracle).
- The IG-on-ROM experiment (Paper-1 P8 — already attribution on the VCS); the
  conformance harness; the state-dump tooling.
- **Key enabler:** SOFT-STE forward is *bit-exact to HARD* and GPU-batchable →
  exact interventions and gradients run batched on GPU.

## Compute
Pilots run locally (M1 Max). Full sweeps + SAE training go to the LME cluster — see
`xai_paper_plan.md` §4.6. Reuse the Paper-1 `tools/cluster/*.sbatch` pattern,
`/cluster/maier`, and the jaxtari GPU venv.

## Status
Specs only (this folder is the *preparation*). Pilot entry points are named in each
subdir's README; implement those first.
