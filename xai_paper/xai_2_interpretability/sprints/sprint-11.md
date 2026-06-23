# Sprint 11 — Revision R3: supplement + P0 core (metrics + count + reviewable artifact)

## Goal
The substantive P0 backbone the reviewers demanded: formalize the metrics, set the canonical count,
and ship a reviewable supplement (per-method protocols + schema/coverage/applicability/claims tables
+ number-provenance + "does not claim" box). Three disjoint owners.

## Planning (≤3 local workers; pairwise-disjoint file_scopes)
- **P2-R-S03-methods** — `sections/03_methods.tex` (the canonical home of the metrics): give
  **formal definitions** of Faithfulness, Sufficiency, Minimality — reconcile "F/S/M as boolean
  predicates" vs "reported as continuous": define each as a continuous score in [0,1] with the
  boolean obtained by a stated threshold; faithfulness = the causal-effect agreement score (state
  precisely when it's correlation vs F1 vs causal-effect, and which is used where). State the
  "provably zero" gradient claim as a precise proposition tied to the companion theorem
  (arXiv:2606.22447) — prove it here or cite the exact section. Apply the **Fig 1 caption handoff**
  (reviews/r2_caption_handoffs.md). (improvement: metrics formalization; "provably zero")
- **P2-R-S07-compare** — `sections/07_results_compare.tex`: set the **canonical count** ("30
  interpretability methods + the oracle positive control = 31 rows") and make every count in this
  file consistent; remove the "verified:false"/raw-record-string leakage; apply the **Fig 4 + Fig 5
  caption handoffs**; `\cite` the new related keys where compare references them; reconcile the
  `sparse_autoencoder` F discrepancy (leaderboard.json 0.1947 vs leaderboard_ci.csv) by stating the
  canonical aggregation. (improvement: count consistency; verified:false; Fig captions)
- **P2-R-SUPP** — NEW `paper/supplement/*.tex` (S1 per-method protocols; S2 benchmark schema; S3
  coverage + applicability; S4 claims-and-evidence; S5 number-provenance; S6 "what this paper does
  NOT claim" box; supplement.tex wrapper). REFERENCE the F/S/M definitions in §3 (do NOT re-derive
  them — S03 owns them); use the canonical count. Every number in S5 → record file + key + script +
  command (consistent with REPRODUCIBILITY §6.5). Include the full Fig 5/6 `_full.pdf` as SI figures.
  The supplement must compile as a fragment; **main.tex `\input{supplement}` wiring is deferred to
  R6 integration** (do NOT edit main.tex). (improvement P0#2,#4; P1#13,#14,#21; P3#35,#36)

## Cross-item consistency contract (all three MUST honor)
- Canonical count = "30 methods + oracle = 31 rows" (S07 authoritative; S03/SUPP match).
- F/S/M definitions live in §3 (S03 authoritative; SUPP references "§3").
- Plausibility = "plausibility proxy"; oracle = "intervention oracle (§3.2)".
- Never fabricate; every reported number traces to a committed record. Recompile-clean before done.

## Review — R3 closed 2026-06-24 (3/3 + audit + R3-fix; recompiles 35 pp, 0 undefined)

- **S03-methods** (`84e086b`): F/S/M formalized (continuous [0,1] + threshold predicate); `prop:zero`
  proved inline (cited `maier2025vcs`); Fig 1 caption handoff. **+ R3-fix `55509be`:** M4 rewrote the
  F definition to the committed estimators (raw Pearson for Phase B, clip-at-0 for Phase A) — no 0.5
  floor, so the position-regime collapse-to-0.00 is now consistent with `prop:zero`; prop hypothesis
  tightened (g non-constant per piece; "zero wherever defined").
- **S07-compare** (`b443a59`): canonical count, `verified:false` removed, robustness framing, SAE
  reconciliation, Fig 2/6 captions, benchmark cites. **+ R3-fix `1fa4ef9`:** U1–U4 + M7 — ACDC
  "on Breakout", SAE "on Pong", cell-84 "present-but-unused", Fig 6 caption = family-mean F±CI (not
  "leaderboard F=1.0"), per-record provenance in headers.
- **SUPP** (`aa3208f`): S1–S6 supplement (protocols, schema, coverage, claims-evidence, provenance,
  not-claimed box) + full SI figures; standalone-compiles 11 pp. **+ R3-fix `ae18e60`:** M6 (S5
  0.6808→0.6811, 0.3869→0.3873, family-row provenance) + S4 regime + robustness-caveat labels.

**The number audit** (`a050bb5`, `reviews/number_audit.md`) is the spine: 119 findings, **no fatal
error**, `prop:zero` SOUND, F/S/M consistent. 7 R3-owned fixes applied above; the remaining §2 fixes
are routed to R4 (02/04/05/06 + fig3.py), R5 (01/08), R6 (00) and folded into those sprint plans.

**Watch:** page count 31→33→35 (M4 + supplement detail). R6 must run a real **length/tightening
pass** (push detail into the SI; the sn-jnl print form is shorter, but trim anyway). **Next:** R4.
