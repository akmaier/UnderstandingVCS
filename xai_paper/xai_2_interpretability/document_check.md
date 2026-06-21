# P2 — Submission compliance checklist — *Nature* / *Nature Machine Intelligence* (Article)

Verify the finished manuscript against the Nature Portfolio author requirements. Mark
each `[ ]` → `[x]` and record where it is satisfied. Same checklist serves *Nature* and
*Nature Machine Intelligence* (shared template; per-journal limits differ — confirm the
**journal-specific numbers** on the live guidelines before submission).

Sources: Nature Portfolio "Formatting guide" + "Editorial policies"; *NMI* "Submission
guidelines / Content types". Template (`sn-jnl.cls`, `sn-nature.bst`) is in `paper/`.

> ⚠️ Numbers marked **(confirm)** vary by journal/article-type — check the live
> guidelines for the chosen target before submitting.

## A. Submission logistics
- [ ] Target journal chosen (NMI primary; *Nature* only if the discrepancy is striking — see `plan.md` "Journal").
- [ ] Article type chosen (Article vs Analysis) and matches the journal's content types.
- [ ] Submitted via the journal's manuscript system (snapp / editorial manager).
- [ ] Cover letter (significance, fit, why this journal).
- [ ] Suggested + opposed reviewers prepared.
- [ ] ORCID for the corresponding author; all authors' affiliations correct.
- [ ] Preprint policy ok — Nature Portfolio permits preprints (arXiv); link in the cover letter if posted.

## B. Format & template
- [ ] Built from the Springer Nature template: `\documentclass[sn-nature]{sn-jnl}`.
- [ ] `sn-jnl.cls` and `sn-nature.bst` unmodified and included for submission.
- [ ] Single-column "content-first" output compiles clean (pdfLaTeX + BibTeX).
- [ ] Line numbers for review (`lineno`) if requested.
- [ ] Fonts embedded; figures vector (PDF/EPS) or ≥300 dpi raster.

## C. Structure & length
- [ ] **Title** concise, no jargon/acronyms; within the title limit **(confirm)**.
- [ ] **Abstract** unreferenced, ~150–200 words **(confirm; NMI Article)**.
- [ ] **Main text** within the Article word limit **(confirm; ~3,000–5,000 *Nature*, longer NMI)**; Introduction → Results → Discussion.
- [ ] **Methods** after references, no length limit, sufficient to reproduce.
- [ ] **References** within cap **(confirm; ~50–70)**; numbered style via `sn-nature.bst`.
- [ ] Main **display items** within cap **(confirm; ~5–8)**.
- [ ] **Extended Data** within allowance **(confirm; ≤10)**.
- [ ] **Supplementary Information** for the rest (full per-method results, oracle proofs, all games).

## D. Required statements / policy sections
- [ ] **Data availability** (ROM provenance — ROMs not redistributed; hashes + AutoROM, as in Paper 1).
- [ ] **Code availability** (jutari/jaxtari + the benchmark; license; repo or "on publication").
- [ ] **Author contributions** (CRediT-style).
- [ ] **Competing interests**.
- [ ] **Acknowledgements** (funding, compute).
- [ ] **Reporting Summary** completed and submitted.
- [ ] **Ethics / dual-use / broader impact** (interpretability-method dual-use, stated honestly; no human-subjects data; ROMs not redistributed).
- [ ] **Statistics & reproducibility**: every quantitative claim states n, the test, and uncertainty; faithfulness scores reported with variation across seeds/games.

## E. Figures & display items
- [ ] Each main figure self-contained with a full legend; panels labelled a, b, c…
- [ ] Colour-blind-safe palettes; no info by colour alone.
- [ ] Figure source data where required (Nature "Source Data").
- [ ] Extended Data called out in the main text; SI items numbered and referenced.

## F. Content coverage vs `plan.md`
- [ ] **Intro:** the ground-truth gap; Jonas & Kording framing; the differentiable-VCS opportunity (Paper 1, cited); **prior ground-truth benchmarks acknowledged** — Tracr (Lindner et al., NeurIPS 2023, arXiv:2301.05062), InterpBench (Gupta et al., NeurIPS 2024 D&B, arXiv:2407.14494), BIM (Yang & Kim 2019, arXiv:1907.09701) — *citations verified real* — with the real-artifact delta argued (they build truth into the model; we recover it).
- [ ] **Representativeness / necessary-condition screen** stated up front (the VCS↔NN failure-mode map; "fail here ⇒ don't trust on an NN").
- [ ] **Results — Phase A (neuroscience):** Kording battery reproduced **and scored** against the known mechanism (calibration baseline; delta over J&K argued).
- [ ] **Results — Phase B (attribution/XAI):** attribution of a VCS output to its inputs/state, scored vs the intervention oracle; the N/A finding for NN-specific methods.
- [ ] **Results — Phase C (mechanistic):** patching/SAEs/circuits/probing on the VCS state, scored against the *known* circuit/data-flow/variables.
- [ ] **Results — cross-tradition comparison + faithful-method demonstration:** A–C on the shared faithfulness-vs-plausibility axes.
- [ ] **Discussion:** shared toolkit; Kording quantified; mech-interp validated on a known circuit; representativeness; directions — with **companion papers named** (P3 behavioral, P4 recovery, P5 agents).
- [ ] **Benchmark artifact** (scoring suite + ground-truth oracle + metrics) described and released.
- [ ] Every claim has a backing experiment; all referenced works real (run the Paper-1 reference-verification pass on the new bib — incl. Tracr/InterpBench/BIM and the Barbiero *contrast* cite).

## G. Pre-submission gate
- [ ] `pdflatex`+`bibtex` clean: no undefined refs/citations, no overfull-hbox blockers.
- [ ] Length, abstract, reference, display-item limits within the **confirmed** caps.
- [ ] Reporting Summary completed; Data + Code availability present and accurate.
- [ ] Author contributions + competing interests present.
- [ ] References verified real (no hallucinations) and Nature-styled.
- [ ] Reproducibility: code + benchmark bundle prepared; seeds and configs recorded.
- [ ] Cover letter + reviewer suggestions ready; ORCID set.
