# Sprint 12 — Revision R4: results + literature (audit fixes + caption handoffs + honest framing)

## Goal
Land the results-section revisions: the number_audit §2 fixes tagged R4, the R2 caption handoffs for
Figs 2/3 (paper Fig 2 = 04, Fig 3 = 05), the honest regime-dependent-gap disclosure, and the
related-work positioning the reviewer asked for. Four disjoint owners.

## Planning (≤3 local workers; two waves; pairwise-disjoint file_scopes)
### Wave A (results trio)
- **P2-R-S04-resultsA** — `sections/04_results_A.tex` **+ `paper/figures/fig3_phaseA_battery.py`**
  (owns both so the A7 wording stays consistent). Audit §2: **M1** A2 "25%"→"40%" (prose + Fig.3
  caption); **M2** A7 "NMF edges out PCA" is FALSE → "NMF = PCA (both mediocre)" in BOTH 04.tex and
  fig3.py:428-433 (NMF only leads on the recon-error composite); CI-vs-SD label "0.49 ± 0.24 (95%
  CI)". Apply the **Fig 2 caption handoff** (per-metric A1–A8 keys, "intervention oracle (§3.2)",
  95% bootstrap CIs, panel (b) is a table). (Note: paper Fig 2 = fig3_phaseA_battery.)
- **P2-R-S05-resultsB** — `sections/05_results_B.tex`. Audit §2: **M3** Phase-B family-mean labeling
  (0.681/0.412 are cross-phase families shown in a Phase-B table — qualify or use Phase-B-only
  0.393/0.216; gradient 0.294 cross-phase / 0.298 Phase-B-only); **M5** ROM-scramble "~95–127"→
  "~69–127"; extremal/on_dist_cf "position outputs 0.346/0.350" are all-regime (position 0.202/0.139)
  — fix the regime label. Apply the **Fig 3 caption handoff** (drop \S1→"intervention oracle (§3.2)";
  "human-plausibility proxy"→"plausibility proxy"; uncertainty sentence; 22 Phase-B+C rows ⊂ 31).
- **P2-R-S06-resultsC** — `sections/06_results_C.tex`. Audit §2: ACDC Breakout recap (06:115-116) add
  "on Breakout"; "25 such cells" (06:93-95) reword (25 = all not-causally-used superset, 6 =
  decodable-and-not-used). Disclose the **regime-dependent gap** (UNC: gap robust on position but
  collapses with CI crossing 0 under content-only / excl-oracle-like) where the mechanistic results
  are summarized; keep ACDC/SAE family-vs-exemplar straight per number_audit §1.

### Wave B (literature)
- **P2-R-S02-related** — `sections/02_related.tex`. Add and position **MIB 2025, SAEBench 2025, M4
  2023** (\cite{mib2025,saebench2025,m4_2023} — already in references.bib); reorganize related work
  per the reviewer (ground-truth/known-answer benchmarks vs our fully-known-machine setting);
  tone-down any overclaim. No experimental numbers introduced.

## Contract (all R4 items)
number_audit §1 canonical numbers are authoritative; ACDC F=1.0/S=0.44 = "on Breakout", SAE F=0.04 =
"on Pong", 1.0-vs-0.0 = position regime, all-regimes gap not robust. STYLE.md voice (no run-in
\paragraph headings; Maier register). Never fabricate. Recompile-clean (latexmk exit 0, 0 undefined)
before done.

## Review
_(to be filled at the R4 barrier)_
