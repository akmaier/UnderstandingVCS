---
id: P2-3M-RUNNER
title: Build multi-mode experiment runner for gradient XAI methods
epic: 3M (Three Modes)
status: todo
sprint:
owner:
where: local
depends_on: [P2-3M-METHODS]
file_scope:
  - tools/xai_study/phaseB_attribution/multi_mode_runner.jl
  - tools/xai_study/phaseB_attribution/out/multimode_ste_*
  - tools/xai_study/phaseB_attribution/out/multimode_soft_*
estimate: L
spec_ref: plan.md
---

## Goal
Build a Julia script `multi_mode_runner.jl` that runs gradient-based attribution methods (vanilla saliency, grad×input, SmoothGrad, Integrated Gradients) across the 6 core games in both SOFT-STE mode and SOFT mode (with an alpha/T grid). Uses the existing jutari infrastructure: `JuTari.Diff` modes, `set_mode!()`, `set_relax!()`, `using_relax()`. The script should:

- Accept command-line args: --games (comma-separated or "all"), --modes (ste, soft, both), --alpha-grid, --temp-grid
- For SOFT-STE: run each method in SOFT mode (which uses STE by default) 
- For SOFT: run each method with relaxation on=true and sweep alpha ∈ [2, 6, 10, 20], T ∈ [0.05, 0.1, 0.14, 0.2, 0.5]
- Score against the intervention oracle (always HARD mode)
- Write results to `out/multimode_ste_*` and `out/multimode_soft_*` in the SPEC results schema

## Definition of Done
- [ ] runner script exists and accepts CLI args
- [ ] runs one method on one game in one mode successfully
- [ ] alpha/T grid sweep works for SOFT mode
- [ ] output written in SPEC schema
- [ ] committed + pushed to main; status: done

## Notes / handoff
- Reuse existing method implementations from the individual .jl files
- The oracle is always in HARD mode (intervention oracle)
- SOFT-STE mode = `set_mode!(SOFT)` with relaxation OFF. Default alpha=10, T=0.1 are the STE defaults
- SOFT mode = `using_relax(on=true, alpha=X, temperature=Y)` with `set_mode!(SOFT)`
