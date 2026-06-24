---
id: P2-N-ABSTRACT
title: Replace abstract with condensed version from most important findings
epic: RV (Revision)
status: in-sprint
sprint: 17
owner: agent-1
where: local
depends_on: [P2-N-SENT]
file_scope:
  - paper/sections/00_abstract.tex
estimate: S
spec_ref: plan.md (Abstract section)
---

## Goal
Replace the current abstract with a new version that is condensed from the most
important findings. The abstract must cover: (1) the faithfulness audit and the
confirmed gap (causal methods faithful, saliency plausible-but-wrong), (2) the
semantic gap finding — even perfect faithfulness does not yield understanding,
including the oracle-sampler experiment, (3) the key insight that the fully
differentiable substrate does not carry semantics — understanding requires
behaviour as reference in documentation and data. Keep to ~150-200 words,
Nature Machine Intelligence style.

## Definition of Done
- [ ] new abstract written (~150-200 words)
- [ ] covers: faithfulness audit, semantic gap, oracle-sampler result, behaviour-as-reference insight
- [ ] paper compiles cleanly with no undefined refs
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
- The current abstract is in `paper/sections/00_abstract.tex`. Replace its content entirely.
- Use condensed, punchy prose. Maier voice per STYLE.md exactly.
- Run the STYLE.md §7 self-check before marking done.
- The new plan.md has the latest abstract sketch — use it as the template.
- Abstract must be ~150-200 words, Nature Machine Intelligence format, flowing prose
  (no labels, no \paragraph{}, plain topic sentences per STYLE.md).
