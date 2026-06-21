# Submission compliance checklist — *Nature* / *Nature Machine Intelligence* (Article)

Verify the finished manuscript against the Nature Portfolio author requirements.
Mark each `[ ]` → `[x]` and record where it is satisfied. Same checklist serves
*Nature* and *Nature Machine Intelligence* (shared template; per-journal limits
differ — confirm the **journal-specific numbers** on the live guidelines, which
sit behind an author-login redirect, before submission).

Sources: Nature Portfolio "Formatting guide" and "Editorial policies"; *Nature
Machine Intelligence* "Submission guidelines / Content types". Springer Nature
LaTeX template (`sn-jnl.cls`, `sn-nature.bst`) is in `paper/`.

> ⚠️ Numbers marked **(confirm)** vary by journal/article-type and must be checked
> against the live guidelines for the chosen target before submitting.

## A. Submission logistics
- [ ] Target journal chosen (NMI primary; *Nature* only if Phase B is striking — see plan §7).
- [ ] Article type chosen (Article vs Analysis) and matches the journal's content types.
- [ ] Submitted via the journal's manuscript system (editorial manager / snapp).
- [ ] Cover letter (significance, fit, why this journal).
- [ ] Suggested + opposed reviewers prepared.
- [ ] ORCID for the corresponding author; all authors' affiliations correct.
- [ ] Preprint policy ok — Nature Portfolio permits preprints (arXiv); link the preprint in the cover letter if posted.

## B. Format & template
- [ ] Built from the Springer Nature template: `\documentclass[sn-nature]{sn-jnl}` (Nature Portfolio numbered style).
- [ ] `sn-jnl.cls` and `sn-nature.bst` unmodified and included for submission.
- [ ] Single-column "content-first" template output compiles clean (pdfLaTeX + BibTeX).
- [ ] Line numbers for review (`lineno` option) if requested by the journal.
- [ ] All fonts embedded; figures vector (PDF/EPS) or ≥300 dpi raster.

## C. Structure & length
- [ ] **Title** concise, no jargon/acronyms; within the title length limit **(confirm)**.
- [ ] **Abstract** unreferenced, ~150–200 words **(confirm; NMI Article)**.
- [ ] **Main text** within the Article word limit **(confirm; ~3,000–5,000 for *Nature*, longer for NMI)**; Introduction → Results → Discussion (Nature has no rigid IMRaD).
- [ ] **Methods** after the references, no length limit, sufficient to reproduce.
- [ ] **References** within the Article cap **(confirm; ~50–70)**; Nature numbered style via `sn-nature.bst`.
- [ ] Main **display items** within the cap **(confirm; ~5–8 figures/tables)**.
- [ ] **Extended Data** figures/tables within allowance **(confirm; ≤10)**.
- [ ] **Supplementary Information** for everything that doesn't fit (full per-method results, proofs of the attribution oracle, all games).

## D. Required statements / policy sections
- [ ] **Data availability** statement (datasets / ROMs provenance — ROMs not redistributed; hashes + AutoROM, as in Paper 1).
- [ ] **Code availability** statement (jutari/jaxtari + the benchmark; license; repository or "on publication").
- [ ] **Author contributions** statement (CRediT-style).
- [ ] **Competing interests** statement.
- [ ] **Acknowledgements** (funding, compute).
- [ ] **Reporting Summary** — the Nature Portfolio Reporting Summary is required; complete and submit it.
- [ ] **Ethics / dual-use / broader impact** as applicable (agentic-coding + interpretability dual-use, stated honestly; no human-subjects data).
- [ ] **Statistics & reproducibility**: every quantitative claim states n, the test, and uncertainty; faithfulness scores reported with variation across seeds/games.

## E. Figures & display items
- [ ] Each main figure is self-contained with a full legend; panels labelled a, b, c…
- [ ] Colour-blind-safe palettes; no information conveyed by colour alone.
- [ ] Figure source data provided where required (Nature "Source Data").
- [ ] Extended Data items called out in the main text; SI items numbered and referenced.

## F. Content coverage vs `xai_paper_plan.md`
- [ ] **Intro:** the ground-truth gap in interpretability; Jonas & Kording framing; the differentiable-VCS opportunity (builds on Paper 1, cited).
- [ ] **Results — Phase A:** Kording battery (connectomics, lesions, tuning, correlations, LFP, Granger, dim-reduction) reproduced **and scored** against the known mechanism.
- [ ] **Results — Phase B:** modern deep-RL XAI (saliency, Grad-CAM/++, IG, occlusion, attention, counterfactual, XDQN) on DQN agents, scored against the true intervention/gradient attribution oracle.
- [ ] **Results — discrepancy:** quantified failure modes; a faithful-attribution demonstration (Phase C2).
- [ ] **Discussion:** XAI ↔ neuroscience shared toolkit; Kording's lesson made measurable; mechanistic-interpretability validation; where it goes beyond (§5).
- [ ] **Benchmark artifact (C1)** described and released.
- [ ] Every claim has a backing experiment; all referenced works real (run the Paper-1 reference-verification pass on the new bib).

## G. Pre-submission gate
- [ ] `pdflatex`+`bibtex` clean: no undefined refs/citations, no overfull-hbox blockers.
- [ ] Length, abstract, reference, and display-item limits all within the **confirmed** journal caps.
- [ ] Reporting Summary completed; Data + Code availability statements present and accurate.
- [ ] All statements (author contributions, competing interests) present.
- [ ] References verified real (no hallucinations) and Nature-styled.
- [ ] Reproducibility: code + benchmark bundle prepared; seeds and configs recorded.
- [ ] Cover letter + reviewer suggestions ready; ORCID set.
