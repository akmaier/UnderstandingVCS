# P5 — Submission compliance checklist — *Nature Machine Intelligence / Nature* (DEFERRED)

> **Deferred by design.** P5 is an outline (`plan.md`); its venue family is Nature
> Portfolio (same as P2), so this checklist will be **derived from
> [`../xai_2_interpretability/document_check.md`](../xai_2_interpretability/document_check.md)**
> once the paper is scoped after P2–P4. Do not fill in until then.

When P5 activates, copy P2's Nature/NMI checklist and add the P5-specific items:
- [ ] **Agent provenance & training** statement (zoo agents vs trained; seeds, configs,
      compute) — P5 is the only paper using learned agents.
- [ ] **Partial-ground-truth disclosure:** state clearly which claims rest on exact
      ground truth (environment, input→action interventions) vs inferred agent
      semantics — the honesty guard for the capstone.
- [ ] **Long-horizon `emulator ∘ agent` gradients** validated (the substrate-stress
      item flagged in P2 experiment_design §3) before any gradient-through-rollout claim.
- [ ] Reuse the shared oracle / triad (cite P2) rather than re-deriving.
