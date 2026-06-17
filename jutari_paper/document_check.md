# AAAI-27 submission compliance checklist

Use this to verify the finished paper satisfies **every** AAAI-27 call-for-papers and
author-kit requirement. Mark each `[ ]` → `[x]` and fill the "where" column when the paper
exists. Sources: AAAI-27 CfP (`https://aaai.org/conference/aaai/aaai-27/`) and AuthorKit27
(`https://aaai.org/authorkit27/`, extracted in `aaai27_authorkit/`).

> How to run the check (once a draft exists): compile `AnonymousSubmission2027.tex`, confirm
> it is ≤ 7 content pages (AAAI-27 main track), grep the source against each item below, and record the
> section/figure that satisfies it. Re-run before each submission deadline.

## A. Submission logistics & deadlines
- [ ] OpenReview author registration done (site opens **2026-06-17**).
- [ ] Paper registered on OpenReview (site opens **2026-06-24**).
- [ ] **Abstract submitted by 2026-07-21** (11:59 PM UTC-12).
- [ ] **Full paper submitted by 2026-07-28** (11:59 PM UTC-12).
- [ ] **Supplementary material + code submitted by 2026-07-31** (11:59 PM UTC-12).
- [ ] Track confirmed = AAAI-27 main technical track (deadlines are track-specific).

## B. Format & template
- [ ] Built from **`AnonymousSubmission2027.tex`** (double-blind variant), not CameraReady.
- [ ] `\usepackage[submission]{aaai2027}` present; `aaai2027.sty` unmodified.
- [ ] Bibliography uses **`aaai2027.bst`**.
- [ ] **Two-column** AAAI layout; letterpaper 8.5×11in; `\frenchspacing`; default fonts/margins (no manual margin/spacing hacks — kit forbids it).
- [ ] **≤ 7 content pages** (AAAI-27 main track); references **unlimited** (separate from the 7). Proofs + effort figure live in a separate supplementary PDF (supplementary.tex).
- [ ] Title, author block anonymized; no `\author` identity, no acknowledgements, no self-identifying links/repos in the submission build.
- [ ] No page numbers / no `\thanks` / no copyright block in the submission variant (handled by `[submission]`).
- [ ] PDF embeds all fonts; figures are vector/`.pdf` where possible (see `Figures/`).

## C. Anonymity (double-blind)
- [ ] No author names/affiliations anywhere (incl. PDF metadata).
- [ ] Self-citations phrased in third person ("Maier et al. show…", not "we previously showed").
- [ ] Code/data links anonymized (anonymous repo or "available upon publication").
- [ ] No identifying institution names in figures, paths, or screenshots.

## D. Required statements / sections
- [ ] **Reproducibility Checklist** completed (`ReproducibilityChecklist.tex`) and `\input` before `\end{document}` OR submitted separately per track instructions — see section E.
- [ ] **Ethics statement** — include if human subjects / sensitive data (likely **N/A** here; state so).
- [ ] **Broader-impact statement** — include if required for the category; cover dual-use of agentic coding + XAI honestly.
- [ ] Clear **abstract** + contributions list.

## E. Reproducibility Checklist items (from `ReproducibilityChecklist.tex`)
Answer each in the checklist with yes/partial/no/NA, and ensure the **paper text actually backs the answer**.

**1. General paper structure**
- [ ] 1.1 Conceptual outline / pseudocode of introduced AI methods (soft/hard model; conformance harness). → target: yes
- [ ] 1.2 Opinions/hypotheses/speculation clearly delineated from facts/results. → yes
- [ ] 1.3 Well-marked pedagogical references for less-familiar readers (VCS, AD, XAI). → yes

**2. Theoretical contributions** (we DO make one: soft→hard convergence + error bound → answer 2 = yes)
- [ ] 2.1 Assumptions/restrictions stated clearly & formally.
- [ ] 2.2 Novel claims stated formally (theorem/lemma for soft→hard convergence + error bound).
- [ ] 2.3 Proofs of all novel claims included (appendix ok).
- [ ] 2.4 Proof sketches/intuitions for complex results.
- [ ] 2.5 Citations to theoretical tools used.
- [ ] 2.6 Theoretical claims demonstrated empirically (soft→hard convergence experiment).
- [ ] 2.7 Experimental code to test/disprove claims included.

**3. Dataset usage** (Atari ROMs / ALE → answer 3 = yes)
- [ ] 3.1 Motivation for the selected datasets (Atari VCS as ground truth; ALE game set).
- [ ] 3.2 Novel datasets in a data appendix (N/A — no novel dataset; the *artifacts* are the ports).
- [ ] 3.3 Novel datasets public w/ research license (NA) — but **the ports/code will be public**.
- [ ] 3.4 Existing datasets cited (ALE/Bellemare 2013; Stella/xitari; ROM provenance).
- [ ] 3.5 Existing datasets publicly available (note ROM licensing carefully).
- [ ] 3.6 Non-public datasets justified (address ROM availability honestly).

**4. Computational experiments** (we have many → answer 4 = yes)
- [ ] 4.1 Number/range of (hyper-)parameters tried + selection criterion (soft temperature, frame counts, etc.).
- [ ] 4.2 Pre-processing code in appendix.
- [ ] 4.3 All experiment source code in a code appendix.
- [ ] 4.4 Source code public upon publication (research license).
- [ ] 4.5 New-method code commented with references back to the paper steps.
- [ ] 4.6 Randomness/seed method described (NOOP-reset randomization; ALE seeds).
- [ ] 4.7 Computing infrastructure stated (CPU/GPU models, memory, OS, Julia/JAX/xitari versions).
- [ ] 4.8 Evaluation metrics formally described + motivated (bit-exactness, pixel-exactness, wall-time, runtimes).
- [ ] 4.9 Number of runs per reported result stated.
- [ ] 4.10 Analysis beyond single-number summaries (variation/confidence/distribution).
- [ ] 4.11 Statistical significance tests where claiming improvement (e.g., Wilcoxon) — apply where relevant.
- [ ] 4.12 All final (hyper-)parameters listed.

## F. Content coverage vs. `paper_plan.md`
- [ ] Introduction storyline (neuroscience↔XAI; Jonas & Kording; ground-truth gap) present.
- [ ] SOTA: XAI review w/ "no ground-truth system" thesis; known-operator learning; Julia vs JAX; VCS-as-ground-truth + DQN lineage + **Figure 1 (VCS architecture)**.
- [ ] Methods: jutari/jaxtari↔xitari port; AI-assisted engineering; soft/hard + equations + proof + error analysis; evaluation methodology.
- [ ] Results: AI-vs-human effort estimate; conformance table (bit + pixel exactness, all evaluated ROMs); soft-vs-hard; first gradient analysis.
- [ ] Discussion: coding speed-up → new paradigms; lessons from logs; Fable + Kimi-1T anecdotes (cited).
- [ ] Summary: "new age of XAI".
- [ ] References complete; every 🔍 in `paper_plan.md` resolved.

## G. Pre-submission gate
- [ ] `pdflatex`+`bibtex` clean (no missing refs/citations, no overfull-hbox blockers).
- [ ] Page count ≤ 7 (content) verified on the compiled PDF; references on their own page(s).
- [ ] Anonymity self-audit (section C) passed.
- [ ] Reproducibility checklist answers all consistent with paper text.
- [ ] Code/supplementary bundle prepared + anonymized.
