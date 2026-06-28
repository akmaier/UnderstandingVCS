# Sprint 18 — Clarity pass (apply the 227-finding style audit, whole paper)

## Goal
Rewrite the flagged sentences from `reviews/style_audit.md` (227 findings) under the CORRECTED
STYLE.md §6b (clarity first: ≤25-word sentences, no aphoristic refrains / no "the X, not the Y"
cadence / no halving or journey metaphors, non-native-readable). **Meaning-preserving only:** no
change to any number, claim, citation, or label. One agent per section, 3 per wave.

## Waves (one section per agent; disjoint files; ≤3 workers)
- Wave 1: 01_intro, 02_related, 03_methods
- Wave 2: 04_results_A, 05_results_B, 06_results_C
- Wave 3: 07_results_compare, 08_discussion, 09_endmatter
(Abstract 00 already done. Supplement S1–S7 deferred — mostly tables; do only if a later pass needs it.)

## Per-agent contract
- Read `reviews/style_audit.md` for YOUR section's findings + apply the rewrites; also fix any other
  sentence that fails the corrected STYLE.md §6b. **Go further than the audit where it kept a refrain**
  (the new §6b bans "the wiring, not the meaning" and all "X, not Y" slogans — rephrase plainly).
- MEANING-PRESERVING: change NO number, NO claim, NO `\cite`, NO `\label`/`\ref`, NO figure/table.
- Fix the 2 meaning-inverting bugs first (04_results_A "No longer lag…"; the 08_discussion one).

## DoD verification (SM — strict, the SENT method)
For each section: the multiset of decimals + `\cite` + `\label`/`\ref` is byte-identical before/after
(I will diff it); `latexmk` exit 0, 0 undefined, 0 multiply-defined; sentences read plainly. Any
section that changed a number/cite/label is reverted and re-run.

## Review — closed 2026-06-24 (9/9 sections; VERIFIED meaning-preserving; build clean)

All nine sections rewritten under the corrected STYLE.md (commits dc64c0d3, 3704d4e3, 4f3aa297,
a87b1a9b, … 57306e8e). Insertions per section: 01=94, 02=43, 03=108, 04=71, 05=54, 06=77, 07=78,
08=155 (heaviest), 09=23 — the line growth is from splitting long sentences into short ones.

**SM verification (whole paper, PRE 03f58d5d vs HEAD):**
- **Meaning-preserved.** The multiset of decimals, `\cite`, `\label`/`\ref`, `$int$`, and `ram[..]`
  is byte-identical across all 9 sections, with ONE benign exception: `04_results_A` restates A7's
  matched-component fraction `$0.60$` as "about $60$ percent" (same value; the data sentences keep
  `$0.60$`). No claim, citation, label, or figure changed. The 2 meaning-inverting bugs are fixed.
- **Refrains removed from prose:** "wiring, not the meaning" 2→0; "smaller/easier/wrong half" 1→0;
  "breathe" 1→0; "way-station/destination" now 0 (the last was a non-rendered planning comment, tidied).
- **Build:** `latexmk` exit 0, 0 undefined, 0 multiply-defined, 54 pp.

Root cause already fixed (STYLE.md §6b → clarity-first). Open for the PO: the lone decimal-vs-percent
restatement in 04 (keep "60 percent" for plainness, or normalize to `$0.60$`).
