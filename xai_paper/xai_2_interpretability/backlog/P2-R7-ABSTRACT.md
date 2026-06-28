---
id: P2-R7-ABSTRACT
title: Replace the abstract with a condensed version from the most important findings (FINAL step)
epic: R7 (universal semantic gap)
status: done
sprint: 17
owner:
where: local
depends_on: [P2-R7-EXP-sampler, P2-R7-S01-intro, P2-R7-S03-methods, P2-R7-S07-compare, P2-R7-S08-discussion, P2-R7-SENT]
file_scope:
  - xai_paper/xai_2_interpretability/paper/sections/00_abstract.tex
estimate: S
spec_ref: plan.md (Abstract draft sketch); STYLE.md
---

## Goal
The VERY LAST step. Once the experiment + reframe + sentence pass have landed and the findings are
settled, replace the abstract with a fresh, condensed version drawn from the *most important*
findings: the universal semantic gap (faithfulness is not the dividing line — proven by the
sampler-on result), that meaning is not recoverable from mechanism and its reference is behavior
(documentation + data, IEEE-610.12), with the canonical numbers and the honest position-regime/
robustness caveat. ~150-200 words, citation-free (NMI unreferenced-abstract rule), Maier voice —
NO glib closers (e.g. not "Explainable AI does not understand a thing"; that was DeepSeek's). State
the result with restraint; let the numbers carry it.

## Definition of Done
- [ ] abstract rewritten from the settled findings; ~150-200 words; citation-free; restrained voice
- [ ] every number matches number_audit §1 + the committed records (incl. the sampler-on result)
- [ ] paper compiles: `latexmk -pdf` exit 0, 0 undefined
- [ ] nothing outside file_scope changed
- [ ] committed + pushed (rebase-before-push); primary pulled ff-only; `status: done`

## Notes / handoff
- HARD dep on everything else — it condenses the final findings, so it runs last.
