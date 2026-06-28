---
id: P2-R7-S07-compare
title: Present the sampler-on result + reframe the comparison to the universal gap
epic: R7 (universal semantic gap)
status: todo
sprint: 16
owner:
where: local
depends_on: [P2-R7-EXP-sampler]
file_scope:
  - xai_paper/xai_2_interpretability/paper/sections/07_results_compare.tex
estimate: M
spec_ref: plan.md (storyline steps 4-6)
---

## Goal
Add a subsection presenting the keystone sampler-on result (faithfulness of the gradient methods
rises off the provable zero on the position regime when the bilinear sampler is on, while semantic
recovery stays 0) and reframe the section so the headline is the **universal gap**, not the
faithful-vs-plausible contrast. Keep the faithfulness leaderboard + danger zone as the way-station;
make the destination "even universally faithful methods recover wiring, not meaning." Numbers come
ONLY from the committed `sampler_faithfulness.csv` + the existing records (number_audit §1). Embed
`fig7` if appropriate (caption owned here). Maier voice; honest regime/robustness caveats retained.

## Definition of Done
- [ ] new subsection presents sampler-on numbers from `tools/xai_study/compare/out/sampler_faithfulness.csv` (every value traced)
- [ ] section reframed: universal gap is the headline; faithful-vs-plausible is the way-station
- [ ] paper compiles: `latexmk -pdf` exit 0, 0 undefined; if fig7 embedded, it resolves
- [ ] no fabricated numbers; nothing outside file_scope changed
- [ ] committed + pushed (rebase-before-push); primary pulled ff-only; `status: done`

## Notes / handoff
- HARD dependency on P2-R7-EXP-sampler (needs its CSV). Do not write sampler numbers before the
  experiment has produced them.
- **Fig 4 caption coupling with P2-R7-FIG-plane:** that item restores the clean dashed-curve plane
  and REMOVES the per-method CI whiskers. When it lands, this item's Fig 4 caption must drop the
  "95% bootstrap CI whiskers" sentence and instead lean on the dashed offset curve + the unreached
  oracle ceiling as the visual of the gap (uncertainty still reported in Fig 2/3 + the supplement).
