# Sprint 14 — Revision R6: abstract + integration (finale)

## Goal
The last content item (abstract) + the SM integration: wire the supplement into main.tex, recompile
the whole, finish the residual cleanups, drive the final consistency grep to zero, and run the
reviewer's resubmission checklist. After R6 all 20 P2-R* items are done.

## Planning
### Content item (1 worker)
- **P2-R-S00-abstract** — `sections/00_abstract.tex`. number_audit §2 [00]: ACDC "sufficiency 0.44"
  add "on Breakout" (00:16-17); SAE "matched 1.0, F=0.04" add "on Pong" (00:17-18); "1.000 versus
  0.000" add "position regime" (00:12-13); "human plausibility" (00:13) → "plausibility proxy";
  "We remove the excuse" (00:7) → cool it (e.g. "Here the ground truth is known"). Reframe per
  plan.md + STYLE.md: lead with the semantic-gap thesis (faithful attribution recovers the causal
  wiring, not the meaning); canonical numbers (30 methods + oracle = 31 rows; family means with the
  position-regime caveat that the all-regimes gap is not robust). Depends on settled numbers (done).

### SM integration (Scrum Master, AFTER the abstract lands — sequential, no file races)
1. Residual cleanups: `06:11` "We remove that excuse" → soften; `07:5` delete the leftover comment;
   `S3` per-game coverage "all 30" → "all 30 (+oracle)" or a clarifying column note.
2. Wire `\input{supplement/supplement}` into `main.tex` at the correct place (after the bibliography
   / before \end{document}, matching the sn-jnl SI convention the SUPP fragment expects).
3. Full recompile (latexmk) — main + SI; require exit 0, 0 undefined, and the SI figures resolve.
4. Final stale-string grep == 0 (except the S6 legit disclaimer).
5. Resubmission checklist against `reviews/review_improvement_instructions.txt` — record pass/defer.
6. Close R6, regen board (all 20 P2-R* done), update memory, **CronDelete the heartbeat**, hand the
   user the revised package.

## Contract
number_audit §1 canonical numbers; STYLE.md voice; never fabricate; Siming Bayer 2nd author.

## Review — R6 closed 2026-06-24 (abstract + integration + QA; ALL 20 P2-R* done)

- **S00-abstract** (`3c8abb2`): reframed to the semantic-gap thesis; canonical numbers + the
  not-robust caveat; ACDC "on Breakout" / SAE "on Pong" / position-regime labels; "human
  plausibility"→proxy; "remove the excuse" cooled.
- **Integration** (`301a431`): `\input{supplement/supplement}` wired into main.tex → 46 pp; residual
  cleanups (06 "remove that excuse", 07 stale comment, S3 "all 30"→"30 + oracle").
- **QA verification** (read-only adversarial pass vs all 3 reviewer docs): **51 ADDRESSED / 11
  PARTIAL / 0 NOT DONE / 7 author-only**. Found 2 must-fix self-inconsistencies →
- **R6-fix** (`bc9dd30` + `04c049d`): (1) delivered the promised plausibility **rubric + proxy-
  sensitivity** (S7 + plausibility_sensitivity.py/csv; Spearman F-vs-proxy −0.64, robust −0.79..−0.40
  over 14 perturbations) and made §3.4 true; (2) supplement cross-refs → `\ref{}` (S4 Fig 4, S6
  Fig 5/6 un-swapped, prop:zero), §8/§6 "first" overclaim softened to match §2.

**Final build:** latexmk exit 0, **50 pp** (main + SI S1–S7 + both full SI figures), **0 undefined**;
stale-string grep clean (only the intended S6 disclaimer). Resubmission checklist:
`reviews/resubmission_checklist.md`. **The revision is complete; remaining items are author-only**
(Zenodo DOI, optional Code Ocean capsule, optional human study, funding line, visual figure eyeball,
cover letter).
