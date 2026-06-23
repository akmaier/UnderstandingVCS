---
id: P2-R-SUPP
title: New Supplementary Information — per-method protocols, schemas, claims/provenance, no-claim box
epic: RV (Revision — supplement)
status: done
sprint: 11
owner:
where: local
depends_on: [P2-R-UNC]
file_scope:
  - paper/supplement/supplement.tex
  - paper/supplement/S1_per_method_protocols.tex
  - paper/supplement/S2_benchmark_schema.tex
  - paper/supplement/S3_coverage_applicability.tex
  - paper/supplement/S4_claims_evidence.tex
  - paper/supplement/S5_number_provenance.tex
  - paper/supplement/S6_not_claimed.tex
estimate: L
spec_ref: ../SPEC.md#e9-reproducibility--submission-prep-sprint-7-local-sm-led
---

## Goal
Create the Supplementary Information the main paper promises but does not ship
(improvement_instructions P0#2, P0#4; P1#13, P1#14, P1#21; P3#35, P3#36). New, self-contained
files under `paper/supplement/` (disjoint from every section + figure item). Contents:
1. **S1 per-method protocol table (P0#2):** one subsection/row per method with exact input/output
   objects, candidate-cause set, output regime, baseline/reference choice, #samples or
   interventions, optimization settings, thresholds, compute budget, random seeds, applicability
   conditions, failure/N-A criteria, metric formula, aggregation rule.
2. **S2 benchmark schema table (P0#4):** row id, method, family, phase, game, output type, output
   regime, candidate-cause universe, metric, #records, applicability flag, artifact file path;
   plus the glossary distinguishing method / method row / per-game record / output record /
   family mean / oracle positive control / N-A method, and the single authoritative method count.
3. **S3 coverage + applicability (P1#13, P1#14):** the 64-game coverage table (Paper-1 bit-exact
   status, RAM/screen horizon, T1/T2/T3 availability, core-game flag, methods run, output regimes,
   #records, excluded methods + reasons) AND the per-method applicability table (raw VCS?
   learned policy? needs gradients/layers/attention/labels? uses oracle-like interventions?
   receives supplied hypothesis?).
4. **S4 claims-and-evidence table (P3#35):** claim, evidence source, games, methods, metric,
   artifact, limitations — for every headline claim.
5. **S5 number-provenance table (P1#21):** every headline number → artifact file, JSON key/path,
   generating script, reproduce command, commit hash; pull the CIs from P2-R-UNC's
   `leaderboard_ci.csv`.
6. **S6 "What this paper does not claim" box (P3#36):** does not recover program semantics from
   scratch; does not evaluate learned agents; does not prove NN interpretability impossibility;
   does not measure human plausibility; does not redistribute ROMs.

## Definition of Done
- [x] `paper/supplement/supplement.tex` compiles standalone (sn-jnl tester driver, 3-pass pdflatex: 11 pp, 0 errors, 0 overfull boxes, 0 undefined refs/cites; both SI figures resolve) and as an `\input` target
- [x] S5 numbers + CIs match `tools/xai_study/compare/out/leaderboard_ci.csv` / `leaderboard.json` / `faithful_demo.json` exactly (and REPRODUCIBILITY §6.5); SAE two-aggregation mismatch reconciled in S5
- [x] the single method count in S2/supplement is the canonical "30 interpretability methods + the oracle positive control = 31 rows" (matches S07/leaderboard.json.n_methods=31)
- [x] nothing outside `file_scope` changed (only `paper/supplement/*` + this item); tester scratch files removed
- [x] committed + pushed to main (rebase-before-push)
- [x] `status: done`

## Notes / handoff
The main `paper/main.tex` `\input{supplement/...}` wiring is an SM-only Review write (main.tex
is shared) — record a handoff line for the SM to add it. Section items will reference
"Supplement Table Sx"; coordinate the table labels with S07/S03 via the revision_plan mapping.
Depends on P2-R-UNC for the provenance/CI numbers.
