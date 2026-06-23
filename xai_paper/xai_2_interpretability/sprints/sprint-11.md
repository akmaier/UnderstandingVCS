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

## Review
_(to be filled at the R3 barrier)_
