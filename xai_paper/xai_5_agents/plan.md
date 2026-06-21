# P5 — Paper plan (outline only; the learned-agent capstone)

> **Status: OUTLINE.** Designed *after* P2–P4 deliver the validated toolkit + the shared
> oracle. This file is a placeholder so the program arc is visible. Program map →
> [`../general_paper_plan.md`](../general_paper_plan.md); shared oracle/T3/triad →
> [`../xai_2_interpretability/experiment_design.md`](../xai_2_interpretability/experiment_design.md) §0–§3.

## P5 in one line
*From known software to learned policy.* Take the interpretability toolkit **validated
against ground truth in P2–P4** and apply it to **learned DQN agents** playing on the
VCS — the deliberate **hard case**, where ground truth becomes only **partial** and the
operational-semantics layer is itself distributed/unknown. This is the original
"interpret real RL agents" goal, now done with a methodology that has been *calibrated on
a system where we knew the answer*.

## Why it is the capstone (and why last)
- P2–P4 establish *which methods recover truth* on a fully-known system (the necessary-
  condition screen). P5 asks the payoff question: **do the methods that passed the screen
  transfer to a neural policy, where we can no longer see the whole truth?**
- It directly answers the reviewer's representativeness objection by *crossing the gap*:
  the VCS supplies what ground truth it can (exact input→action interventions via
  `emulator ∘ agent`; the game's true variables as candidate concepts), while the
  agent's internal semantics are genuinely unknown — the real interpretability setting.

## What changes vs P2–P4 (the new subject)
- **Subject:** a trained DQN agent (Mnih-2015 / Atari Model Zoo, Such et al. 2019),
  studied as `emulator ∘ agent`. (This is the *only* paper that uses learned agents.)
- **Ground truth becomes partial:** T1/T2 still hold for the *environment*; the agent's
  internal causal structure is obtainable by **activation intervention** (we hold the
  weights), but its *semantics* (does a feature mean "ball-x"?) is now an empirical
  question, not given — exactly what P2–P4 taught us to test.
- **New substrate need to pilot:** long-horizon end-to-end `emulator ∘ agent` gradients
  (credit through dynamics) — flagged as the one capability that may stress the substrate
  (P2 experiment_design §3); pilot before committing.

## Likely structure (to be filled after P2–P4)
- Reuse the F∧S∧M triad, the oracle, and the metrics from P2.
- Re-run the *passing* methods (causal/gradient/mechanistic) on the agent; test whether
  faithfulness transfers; measure where partial ground truth still discriminates.
- Behavioral probing of the agent (P3 methods) with the game's true variables as the
  candidate decision variables; mechanistic recovery of the agent's circuit (P4/Phase-C
  methods) with the environment's known structure as a partial anchor.

## Target
**Nature Machine Intelligence / Nature** — the capstone that closes the arc from
fully-known software to learned policy.

## Open questions (defer)
- Which agents (zoo vs a small set we train); how much GPU.
- How to report "partial ground truth" honestly (what is anchored vs inferred).
- Whether P5 is one paper or splits by method family — decide after P2–P4 results.
