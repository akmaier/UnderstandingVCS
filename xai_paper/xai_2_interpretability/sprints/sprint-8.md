# Sprint 8 — E9 submission prep (PO said: run all of E9, hand back the package)

## Goal
Turn the maximal v1 (37 pp / ~19.5k words) into a submission-ready NMI Article package:
condense to length, pass document_check, verify every reference is real (no hallucinated
citations), build the reproducibility bundle, and produce a clean final build + checklist.
Preserve the thesis (faithfulness ≠ understanding), every committed number, all citations,
and the honesty contract. Do NOT invent or alter any result.

## Stages (chained autonomously; SM builds at each barrier)
1. **Condense** (per-section, disjoint files, ≤3 workers/wave) — target ~10k words total:
   - 00_abstract 289→~190 · 01_intro 1966→~1000 · 02_related 1911→~800 · 03_methods 3334→~1700
   - 04_results_A 1620→~850 · 05_results_B 2687→~1200 · 06_results_C 2354→~1100
   - 07_results_compare 2143→~1400 (the contribution — cut least) · 08_discussion 2667→~1200
   - 09_endmatter keep (+ E9-1 statements). Rule: state each headline number ONCE (intro asserts,
     results detail); cut verbosity, keep substance + all \cite/\ref/\label + the honesty contract.
2. **Compliance** (parallel, on the condensed draft):
   - `E9-1` document_check — NMI required statements (data/code availability, author contributions,
     competing interests, ethics), limitations, Reporting-Summary checklist → document_check report
     + endmatter edits.
   - `E9-2` reference no-hallucination pass — **WebSearch-verify all 95 references** (authors/year/
     venue/title); fix/flag any wrong or fabricated → verification report + references.bib fixes.
   - `E9-3` reproducibility bundle — seeds, configs, the packaged benchmark (E6-2), ROM SHA-256
     hashes, commit pins → repro bundle/README.
3. **Final build (`E9-4`, SM):** assemble, coherence/transition pass, `latexmk` build, page/word
   count vs the NMI limit, the final package + checklist. Then hand back to PO.

## Review
_(to be filled at the Sprint-8 barrier — the submission package)_
