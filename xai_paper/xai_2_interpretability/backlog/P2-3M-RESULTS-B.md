---
id: P2-3M-RESULTS-B
title: Update Results B with STE and soft-mode findings
epic: 3M (Three Modes)
status: todo
sprint:
owner:
where: local
depends_on: [P2-3M-STE, P2-3M-SOFT]
file_scope:
  - paper/sections/05_results_B.tex
estimate: M
spec_ref: plan.md
---

## Goal
Update `05_results_B.tex` to include the SOFT-STE and soft-mode experimental results. Show that SOFT-STE mode does not fix position outputs (STE gradient is content-only). Show that soft mode with appropriate alpha/T provides non-zero gradients on position outputs, and report the alpha/T faithfulness landscape. Include supplement-reference for the full grid.

## Definition of Done
- [ ] SOFT-STE results described: no change on position regime
- [ ] Soft mode results described: alpha/T landscape, where gradients become faithful
- [ ] Supplement reference for full grid
- [ ] Paper compiles cleanly with no undefined refs
- [ ] committed + pushed to main; status: done
