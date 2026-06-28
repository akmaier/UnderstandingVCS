# Sprint 17 — R7-3: polish (sentence split, then final abstract)

## Goal
The last two items, strictly sequential (each owns broad scope, so neither runs concurrently).

- **R7-3a · P2-R7-SENT** — one copy-edit pass splitting over-long sentences across all sections +
  supplement, in Maier's voice. **Meaning-preserving only** — change NO numbers, claims, citations,
  or labels. Owns all sections, so it runs ALONE. (It may touch 00_abstract, but R7-3b replaces the
  abstract wholesale next, so 00 effort is optional — focus on the body + supplement.)
- **R7-3b · P2-R7-ABSTRACT** — replace 00_abstract.tex with a fresh ~150-200-word condensation of the
  settled findings: the universal semantic gap (faithfulness is not the dividing line — the sampler-on
  result: 0.000→0.791 pong while semantics stays 0), meaning's reference is behavior/docs/data
  (IEEE-610.12), canonical numbers + the honest position-regime/robustness caveat. Citation-free
  (NMI rule), restrained Maier voice — NO glib closers. Runs after SENT.

## DoD verification (SM — strict)
- SENT: `git diff` shows ONLY sentence-structure edits (spot-check: no number/claim/citation deltas);
  paper compiles (latexmk exit 0, 0 undefined, 0 multiply-defined); page count reported.
- ABSTRACT: ~150-200 words; citation-free; every number matches number_audit §1 + sampler CSV; voice
  restrained (grep no "does not understand a thing"-style closers); paper compiles.

## Review
_(to be filled at the R7-3 barrier — then ALL 9 P2-R7-* done → rebuild, final grep, CronDelete heartbeat)_
