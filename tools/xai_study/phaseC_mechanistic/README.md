# Phase C — mechanistic interpretability on the VCS (known-circuit testbed) (spec)

**Subject: the VCS** (no agents). The VCS **state trajectory is the "activations"**
and the program's **data-flow is the "circuit"** — both known — so the
mechanistic-interpretability toolkit can be scored against a *known* circuit for the
first time. Ground truth = T1/T2 (always) + T3 (where labeled).

## Methods
- **Activation patching / causal tracing:** clamp/resample RAM cells / registers /
  TIA state between a clean and a corrupted run; score recovered important components
  vs the exact intervention effect (oracle) and the true data-flow.
- **Attribution patching (+ edge AP):** gradient approximation of patching; score the
  approximation error vs true patching.
- **Sparse autoencoders / dictionary learning:** train an SAE on the recorded state
  trajectory; score features vs known variables (T2 hardware signals always; T3 game
  variables where labeled) — feature↔variable matching + causal use.
- **Circuit discovery + causal scrubbing:** recover the circuit for a behavior
  (ball-bounce, opponent-AI); validate by scrubbing vs the true disassembled routine.
- **Linear probing:** decode concepts from state; contrast decodable vs *used*.

## Infra to build (substrate already supports the rest)
state-trajectory recorder (reuse Paper-1 dumps), patch (clamp/resample) hooks on
state, SAE training, circuit search. No agents, no NN required.

## Metrics (vs ground truth)
recovered-effect vs exact-patch agreement; feature↔known-variable matching;
scrubbing-preserved performance vs the true routine; circuit P/R vs true data-flow.

## Pilot (local)
`pilot_patch_sae.{py,jl}`: activation patching + one SAE on the VCS state for one
game; patched-effect vs exact-patch; SAE-feature vs known-variable matching.

## Scale-out (cluster, GPU)
patch sweeps over state × outputs × games; SAE training on recorded state
trajectories. See `xai_paper/xai_2_interpretability/experiment_design.md` §8 (compute).

Outputs: `out/mech_<method>_<game>.*` → faithfulness vs the known circuit, feeding
the cross-tradition comparison and Phase E (semantic recovery).
