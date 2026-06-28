# Sprint 16 — R7-2: reframe to the universal gap (consume the keystone numbers)

## Goal
Write the reframe now that the experiment has produced real numbers. Two waves; disjoint sections.
Canonical sampler numbers (from `tools/xai_study/compare/out/sampler_faithfulness.csv`, R7-1):
- naive faithfulness = **0.000** for all gradient methods on the position regime (Prop. prop:zero).
- **sampler ON: 0.791 on pong** (vanilla saliency / grad×input / smoothgrad / expected gradients),
  **0.529 on qbert** (vanilla saliency / grad×input / smoothgrad / integrated gradients).
- **semantic_recovery = 0** for every method in both conditions.
- mechanism: the restored gradient concentrates on the position RAM byte (e.g. `ram[54]` on pong) —
  it finds the right cause, but names no meaning.

### Wave A (results + methods; all consume the CSV)
- **P2-R7-S07-compare** (`07_results_compare.tex`) — add the sampler-on subsection (faithfulness
  rises off zero, semantics stays 0), reframe the headline to the universal gap; **embed `fig7`**
  (caption owned here); **drop the "95% bootstrap CI whiskers" sentence from the Fig 4 caption**
  (FIG-plane removed them — R7-1 handoff).
- **P2-R7-S08-discussion** (`08_discussion.tex`) — the positive thesis: faithfulness made universal
  still recovers only mechanism; meaning's reference is behavior (docs + data, IEEE-610.12).
- **P2-R7-S03-methods** (`03_methods.tex`) — document the sampler-on protocol + the semantic_recovery
  measure, matching what the runner actually did (ram-byte position cause, Pearson vs oracle).

### Wave B
- **P2-R7-S01-intro** (`01_intro.tex`) — reframe the intro to the universal gap + behavior-as-reference
  (may cite the sampler result qualitatively).

## DoD verification (SM — strict)
Every sampler number traces to sampler_faithfulness.csv; no fabrication; paper recompiles (latexmk
exit 0, 0 undefined); fig7 embedded resolves; Fig 4 caption no longer mentions CI whiskers.

## Review
_(to be filled at the R7-2 barrier)_
