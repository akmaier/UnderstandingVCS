---
id: P2-3M-METHODS
title: Update methods section for three Paper-1 execution modes
epic: 3M (Three Modes)
status: in-sprint
sprint: 18
owner: agent-1
where: local
depends_on: []
file_scope:
  - paper/sections/03_methods.tex
estimate: M
spec_ref: plan.md
---

## Goal
Update `03_methods.tex` to properly describe all three Paper-1 execution modes: HARD (bit-exact), SOFT-STE (straight-through estimator, forward-exact at any T), and SOFT (fully relaxed with alpha/T). Describe Paper 1 as a technical companion with mathematical proofs. Describe three companion gradient constructions (not one). The partial edit already committed is a starting point — verify and complete it.

## Definition of Done
- [ ] Paper 1 described as technical companion with proofs, not just "built it"
- [ ] All three modes clearly described: HARD, SOFT-STE, SOFT
- [ ] Soft mode alpha/T parameters, temperature-limit error bound stated
- [ ] Three companion gradient constructions described (intervention oracle + STE gradient + soft gradient)
- [ ] Prop.~zero scoped correctly against the three modes
- [ ] No stale "two constructions" / "two exceptions" language
- [ ] Paper compiles cleanly with no undefined refs
- [ ] committed + pushed to main; status: done

## Notes / handoff
- Current partial edit is in place. Read it and complete/review.
- Key changes: lines 40-69 (substrate intro), lines 112-152 (oracle / proposition)
- The bilinear sampler is a special case of the soft relaxation, not a separate mode
