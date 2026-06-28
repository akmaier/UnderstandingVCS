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

## Review — R7-3 closed 2026-06-24 (2/2; R7 COMPLETE, 8/8 items done)

- **P2-R7-SENT** — sentence-split pass, **verified meaning-preserving**: the multiset of decimals,
  `\cite` keys, and `\label`/`\ref` across all 9 body sections is byte-identical before/after (only
  sentence structure changed). No numeric/citation/label drift.
- **P2-R7-ABSTRACT** (`3c1b01ba`) — abstract rewritten: 188 words, citation-free (NMI rule), no glib
  closer; leads with the sampler-on result (0.000→0.791 Pong, semantics 0) and "meaning… its
  reference is behaviour, fixed for software by its documentation and data". Numbers match
  number_audit §1 + the sampler CSV.

**Final state:** `latexmk` exit 0, **0 undefined, 0 multiply-defined, 54 pp**. Stale-string grep
clean (only the intended S6 "we do not measure human plausibility" disclaimer). fig7 embedded once
(S07). The keystone experiment is real and SM-verified (R7-1).

**R7 complete — 8/8 done.** Heartbeat `6e9dadad` retired. Remaining for the PO: the author-only
items in `reviews/resubmission_checklist.md` (Zenodo DOI, optional Code Ocean capsule, funding line,
cover letter / point-by-point response).
