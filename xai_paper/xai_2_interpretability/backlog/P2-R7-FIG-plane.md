---
id: P2-R7-FIG-plane
title: Restore the clean dashed-curve faithfulness-vs-plausibility plane (Fig 4), declutter the error bars
epic: R7 (universal semantic gap)
status: done
sprint: 15
owner:
where: local
depends_on: []
file_scope:
  - xai_paper/xai_2_interpretability/paper/figures/fig2_faithfulness_vs_plausibility.py
  - xai_paper/xai_2_interpretability/paper/figures/fig2_faithfulness_vs_plausibility.pdf
estimate: M
spec_ref: plan.md (Display items; storyline — the offset + none-reach-oracle is the visual of the gap)
---

## Goal
Bring back the clean, legible version of the faithfulness-vs-plausibility plane (paper **Fig 4**,
file `fig2_faithfulness_vs_plausibility.py`). The current version (R2 redraw, commit `7f49d9b`) is
**too busy** — the per-method CI whiskers + the key-results table crowd the panel. The earlier
version (commit **`8b919764`**) is the one to restore: a clean scatter with the **dashed offset
curve** that shows the faithful-vs-plausible separation, and the **oracle ceiling star at (1,1)**
that visibly **no method reaches** — which is exactly our storyline (the gap, made a picture).

This is NOT a blind revert. Take `8b919764`'s composition (dashed offset curve + oracle-ceiling
star, NO CI whiskers, NO in-panel results table) and **re-apply the legitimate R2/R6 fixes** that
the old version predates:
- "§1 oracle"/"Section 1" → **"intervention oracle (§3.2)"** in all drawn text.
- Remove any **drawn data-path microtext** (tools/xai_study/... ) from the figure body.
- Y axis = **"plausibility proxy"** (never "human plausibility"); count = **"30 methods + oracle =
  31 rows"** if a count is shown.
- 8 pt min font; Okabe-Ito colour-blind-safe palette; embedded fonts; vector PDF.
- Label only ~6–8 anchor points (the danger-zone names + the oracle) to keep it readable; no
  overlap/clipping; legend outside the data field.

**Uncertainty is NOT lost:** the R-UNC 95% bootstrap CIs stay reported in the Phase-A/B battery
figures (Fig 2/3) and the supplement + leaderboard_ci — they are simply removed from THIS plane so
the offset curve and the unreached oracle read cleanly (reviewer P1#16 still satisfied elsewhere).

**Optional enhancement (only if it stays clean):** with the sampler-on result (P2-R7-EXP-sampler),
add faint arrows showing the gradient methods sliding RIGHT (faithfulness up) but NOT up toward the
oracle — i.e. faithful yet still short of understanding. Skip if it re-clutters; the clean plane is
the priority.

## Definition of Done
- [x] `fig2_faithfulness_vs_plausibility.py` regenerated as the dashed-curve clean layout (no CI
      whiskers, no in-panel table); dashed offset curve + oracle-ceiling star both present
- [x] R2/R6 labeling fixes applied (§3.2 oracle, plausibility proxy, count, no drawn source paths,
      8 pt, Okabe-Ito); ≤~8 labeled points (6 method anchors + oracle); render-checked (no overlap/clipping)
- [x] every plotted value traces to the committed leaderboard.json / faithful_demo.json
- [x] PDF regenerated; paper still compiles (`latexmk -pdf` exit 0, 0 undefined refs/cites, 50 pp)
- [x] committed + pushed (rebase-before-push); primary pulled ff-only; `status: done`

## Notes / handoff
- **Caption coupling:** the Fig 4 caption lives in `07_results_compare.tex` (owned by
  `P2-R7-S07-compare`). When the whiskers are removed, that item MUST drop the "95% bootstrap CI
  whiskers" sentence from the caption and lean on the dashed-curve / unreached-oracle reading. Note
  recorded in both items.
- Recover the base composition with: `git show 8b919764:xai_paper/xai_2_interpretability/paper/figures/fig2_faithfulness_vs_plausibility.py`.
