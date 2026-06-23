# P2 resubmission checklist (2026-06-24)

Adversarial point-by-point verification of the revised paper vs the three reviewer docs, after the
full revision (R1–R6 + the number audit + R3-fix + R6-fix). Build: **50 pp (main + SI), 0 undefined
cites/refs**, supplement S1–S7 + both full SI figures integrated, stale-string grep clean.

## Tally vs `review_improvement_instructions.txt` (P0–P7) + `review_final.txt`
- **ADDRESSED: 53** (51 at QA + the 2 closed below)
- **PARTIAL: 9** (documented below — none blocking; mostly "more controls/tables would be nice")
- **NOT DONE: 0**
- **AUTHOR-ONLY: 7** (cannot be done without you; see end)

## Closed in R6-fix (the two defects QA would not waive)
1. **Plausibility rubric + proxy-sensitivity now real** (was: §3.4 promised them, neither existed —
   the reviewer's most-emphasized weakness). Delivered `tools/xai_study/compare/plausibility_sensitivity.py`
   + `out/plausibility_sensitivity.csv` + supplement **S7**: the per-tradition proxy rubric (criteria +
   sourced rationale, values from the records) and a sensitivity sweep. Baseline Spearman(F, proxy)
   = **−0.64**; across 14 perturbations (jitter σ∈{.05,.1,.2}, rank-compression, rank-jitter,
   leave-one-tradition-out) ρ ∈ **[−0.79, −0.40]**, danger-zone count ∈ [6,15], the saliency-more-
   plausible-yet-less-faithful anchor holds 13/13 → **"plausible ≠ faithful" survives any proxy
   assignment.** §3.4 reworded to point to S7 and state this outcome (no overclaim).
2. **Supplement cross-refs + "first" overclaim.** All hardcoded supplement figure/section numbers →
   `\ref{}` (S4 faithfulness→Fig 4; S6 representativeness→Fig 6 / taxonomy→Fig 5, was backwards;
   "§5" for the gradient-zero result → `Prop.~\ref{prop:zero}`). §8/§6 "first ground-truth
   calibration" softened to match the toned-down §2.

## Major-weakness status (review_final)
MW1 reviewable artifact ✓ · MW2 cite Paper 1 (arXiv:2606.22447) ✓ · MW3 formal F/S/M ✓ ·
MW4 plausibility proxy + rubric + sensitivity ✓ (R6-fix) · MW5 MIB/SAEBench/M4 ✓ ·
MW6 scope narrowed (necessary-condition screen, not a proven NN floor) ✓ · Correctness (count,
oracle §3.2, jutari/Julia, scoped prop:zero) ✓ · Reproducibility bundle ✓.

## Remaining PARTIAL (non-blocking; candidate strengthenings, not corrections)
- P1-13 coverage table collapses the 58 non-core games into one summary row (not 64 explicit rows).
- P1-22 SAE: extra controls (random-feature / RAM-cell / matched-but-randomized) beyond the NMF/PCA
  contrast not added; dict hyperparameters not fully tabulated in S1.
- P1-23 probing: probe class/regularization/label-balance not fully tabulated (acc 0.89 in §6 vs the
  Breakout single-cell 0.72 in §7 — both correct, distinct; a reader may double-take).
- P3-34 the causal-high/gradient-low/inversion claim is still restated in several sections.
- P4-39/40/41 figure visual-overlap fixes were applied in the `.py` but only the author can confirm
  by eye on the rendered PDF.

## AUTHOR-ONLY (I will not fabricate these — they are yours to do)
1. **Mint the Zenodo DOI** for the versioned snapshot (endmatter says "will be archived").
2. **Optional Code Ocean capsule** that regenerates the leaderboard + figures from the committed
   records (no ROMs needed) — strongest reproducibility answer; partnership is active.
3. **Real human/expert plausibility study** — only if you want to convert the proxy to a measurement;
   the rubric + sensitivity (S7) now stands in for it and is defensible without it.
4. Confirm the **funding/acknowledgement** line (currently "no specific external funding").
5. Visually eyeball the rendered figures (overlap/clipping) before submission.
6/7. Final read-through for voice + the cover letter / point-by-point response to reviewers.

## Verdict
**Resubmission-ready modulo the author-only steps above.** The scientific core — formal F/S/M, the
scoped `prop:zero`, the honest "gap robust only on the position regime" reporting, the reviewable
artifact bundle + per-number provenance (`number_audit.md`), the full literature integration, and the
delivered plausibility rubric + sensitivity — is verifiably in place.
