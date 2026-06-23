# P2 — Submission compliance checklist — *Nature Machine Intelligence* (Article)

Manuscript: `paper/main.tex` + `paper/sections/*.tex`. Built from the Springer Nature
`sn-jnl` class with the `sn-nature` option. Same checklist serves *Nature* and *NMI*
(shared template; per-journal caps differ — **(confirm)** items must be checked against
the live guidelines for the chosen target before submission).

Sources: Nature Portfolio "Formatting guide" + "Editorial policies"; *NMI* "Submission
guidelines / Content types". Template (`sn-jnl.cls`, `sn-nature.bst`) is in `paper/`.

**P2-E9-1 verification (P2 Sprint 8):** the back-matter required statements are confirmed
present and STYLE-compliant. The full document was compiled (`pdflatex -draftmode main.tex`,
exit 0); `sections/09_endmatter.tex` parses cleanly with no errors and contains **zero**
`\paragraph` run-in labels. Line numbers below are as of this pass.

> ⚠️ Numbers marked **(confirm)** vary by journal/article-type — check the live
> guidelines for the chosen target before submitting.

## D. Required statements / policy sections — ALL PRESENT

| Requirement | Status | Where satisfied (file : line) |
|---|---|---|
| **Data availability** | [x] | `paper/sections/09_endmatter.tex:12–23` (`\bmhead{Data availability}`). States the committed §R analysis records under `tools/xai_study/`, the aggregated cross-tradition leaderboard, ROM provenance (SHA-256 hashes + AutoROM; ROMs not redistributed), and deterministic regeneration of state trajectories. |
| **Code availability** | [x] | `paper/sections/09_endmatter.tex:25–33` (`\bmhead{Code availability}`). jutari/jaxtari + oracle (companion port paper) + the benchmark (scoring suite, F∧S∧M triad, per-phase methods under `tools/xai_study/`). Exact policy line present verbatim at line 32–33: "The full source for the emulator, the oracle, and the benchmark will be made available under the MIT license upon acceptance." |
| **Author contributions** | [x] | `paper/sections/09_endmatter.tex:35–42` (`\bmhead{Author contributions}`). CRediT-style; flagged as a placeholder pending the final author list. |
| **Competing interests** | [x] | `paper/sections/09_endmatter.tex:44–45` (`\bmhead{Competing interests}`). "The authors declare no competing interests." (none). |
| **Ethics / dual-use / broader impact** | [x] | `paper/sections/09_endmatter.tex:47–54` (`\bmhead{Ethics and broader impact}`). No human-subjects data; plausibility axis flagged as a method-tradition proxy not a measurement; no learned agent/policy; dual-use of interpretability acknowledged; ROMs not redistributed. |
| **Acknowledgements** (funding, compute) | [x] | `paper/sections/09_endmatter.tex:56–63` (`\bmhead{Acknowledgements}`). Funding/compute marked as a to-complete placeholder; thanks AtariARI/OCAtari/xitari maintainers; notes pilots local, sweeps on the cluster. |
| **Limitations** (must exist; NOT duplicated in end matter) | [x] | `paper/sections/08_discussion.tex:139–152` — the "We have measured a gap, not closed it, and several limits bound what we may conclude…" paragraph, as ordinary flowing prose (STYLE.md §5: no labelled "Limitations." block). Verified **not** duplicated in 09_endmatter.tex. |
| **Reporting Summary** (Life Sciences / Behavioural & Social / generic) | [ ] **PO action** | Form-based artifact submitted alongside the manuscript, not a LaTeX section. Cannot be authored from the repo. Flag for the PO at submission time. |
| **Statistics & reproducibility** (n / test / uncertainty per quantitative claim) | [x] (partial — see gap) | Scores read from committed records (09_endmatter.tex:13–23); discussion grounds the gap claim across games/separations not a single number (`08_discussion.tex:139–152`, esp. lines 145–148). Each headline number traces to a committed §R record (per E9 honesty contract). **Gap:** seed/game **uncertainty bands** (variation across seeds/games) are not yet rendered as explicit ± / CI in-text; confirm against the records before submission. |

## A. Submission logistics

- [ ] Target journal chosen — NMI primary; *Nature* only if the discrepancy is striking (plan.md §"Journal — NMI vs Nature", lines 326–334). **PO decision** ("decide after pilots").
- [ ] Article type — Article vs Analysis (plan.md leans **Article** that culminates in the semantic gap). **PO decision.**
- [ ] Submitted via the journal's manuscript system (snapp / editorial manager). **PO action.**
- [ ] Cover letter (significance, fit, why this journal). **PO action.**
- [ ] Suggested + opposed reviewers prepared. **PO action.**
- [ ] ORCID for the corresponding author; affiliations correct — affiliation present (`paper/main.tex:66`, FAU Erlangen-Nürnberg, Pattern Recognition Lab); ORCID to add. **PO action.**
- [ ] Preprint policy — Nature Portfolio permits arXiv preprints; link in cover letter if posted. **PO action.**

## B. Format & template

- [x] Built from the SN template: `\documentclass[sn-nature,pdflatex]{sn-jnl}` (`paper/main.tex:16`).
- [x] `sn-jnl.cls` and `sn-nature.bst` present in `paper/` (confirm unmodified at submission).
- [x] Compiles clean: `pdflatex -draftmode main.tex` returns exit 0 (this pass; figures fig1–fig6 PDFs in `paper/figures/`).
- [ ] Line numbers for review (`lineno`) if requested. **PO action** (add at submission if asked).
- [ ] Fonts embedded; figures vector (PDF) — figures are PDF; confirm 300 dpi/vector at final build. **PO/SM at flatten.**

## C. Structure & length

- [x] **Title** declarative, no acronyms (`paper/main.tex:55`). Within title cap — **(confirm)**.
- [x] **Abstract** unreferenced, **205 words** (`detex sections/00_abstract.tex | wc -w`). NMI Article target ~150–200 **(confirm)** — slightly over; trim ~5–10 words if the cap is firm. **Minor PO/SM note.**
- [ ] **Main text** within the Article word limit **(confirm; longer for NMI)**; order Introduction → Results → Discussion is honoured (`main.tex:80–88`).
- [x] **Methods** present as a section (`sections/03_methods.tex`), inputted before end matter; SN places Methods after references at production flatten.
- [ ] **References** within cap **(confirm; ~50–70)**; numbered via `sn-nature.bst`. Count at final bib merge. **SM at integration.**
- [x] Main **display items**: figures fig1–fig6 (six), within the typical ~5–8 cap **(confirm)**.
- [ ] **Extended Data** ≤10 **(confirm)** — none yet; SI plan in plan.md.
- [ ] **Supplementary Information** for full per-method results, oracle proofs, all games — planned (plan.md §"Display items"/§"End matter"). **SM/PO at SI build.**

## E. Figures & display items

- [x] Each main figure has an interpretive legend; representativeness map caption labels panels and notes colour roles (`sections/08_discussion.tex:114–124`).
- [ ] Colour-blind-safe palettes; no info by colour alone — figures use blue/vermillion (Okabe-Ito-style) per caption; **confirm** across all six.
- [ ] Figure Source Data where required (Nature "Source Data") — every figure value reads from a committed record (stated in captions); package as Source Data at submission. **SM/PO.**
- [x] Display items called out in text; figures referenced via `\ref` (e.g. `fig:faithfulness`, `fig:taxonomy`, `fig:representativeness`).

## F. Content coverage vs `plan.md` — drafted

- [x] **Intro:** ground-truth gap, Jonas & Kording framing, differentiable-VCS opportunity (Paper 1 cited); prior ground-truth benchmarks acknowledged with the real-artifact delta (Tracr/InterpBench/BIM cited in `08_discussion.tex:29–35`, intro motivates) — citations verified real (BIBTEX-NEEDED blocks carry full entries).
- [x] **Representativeness / necessary-condition screen** stated (`08_discussion.tex:87–125`, "Why a 1977 chip is a fair screen" + the VCS↔NN map, Fig. 5).
- [x] **Results — Phase A (neuroscience):** Kording battery scored vs known mechanism (`sections/04_results_A.tex`).
- [x] **Results — Phase B (attribution/XAI):** attribution scored vs oracle; N/A finding (`sections/05_results_B.tex`).
- [x] **Results — Phase C (mechanistic):** patching/SAEs/circuits/probing vs known circuit (`sections/06_results_C.tex`).
- [x] **Results — cross-tradition comparison + faithful-method demo:** A–C on shared axes (`sections/07_results_compare.tex`).
- [x] **Discussion:** shared toolkit; Kording quantified; mech-interp validated on a known circuit; representativeness; companion papers P3/P4/P5 named (`08_discussion.tex`, esp. 71–73, 80–85).
- [x] **Benchmark artifact** described and released (scoring suite + oracle + metrics; `09_endmatter.tex:25–33`).
- [ ] Every claim backed by an experiment; all referenced works real — final reference-verification pass on the merged bib (incl. Tracr/InterpBench/BIM/Barbiero). **SM at bib merge.**

## G. Pre-submission gate

- [x] `pdflatex` draft compile clean (exit 0) this pass; run full `pdflatex`+`bibtex` once bib is merged. **SM at integration.**
- [ ] Length/abstract/reference/display-item limits within **confirmed** caps (abstract 205w slightly over a 200w cap — see §C). **PO/SM.**
- [ ] Reporting Summary completed; Data + Code availability present and accurate — availability statements present (§D); Reporting Summary is a PO submission artifact.
- [x] Author contributions + competing interests present (`09_endmatter.tex:35–45`).
- [ ] References verified real and Nature-styled. **SM at bib merge.**
- [ ] Reproducibility bundle prepared; seeds/configs recorded. **PO/SM (E9 bundle).**
- [ ] Cover letter + reviewer suggestions ready; ORCID set. **PO action.**

## Gaps requiring the PO

1. **Reporting Summary** — Nature's editorial form, not a LaTeX section; must be filled and uploaded at submission. Cannot be authored from the repo.
2. **Journal & article-type decision** — NMI vs *Nature*, Article vs Analysis (plan.md says "decide after pilots").
3. **Statistics & reproducibility — explicit uncertainty bands** — every headline number traces to a committed record, but seed/game variation is not yet rendered as in-text ± / CI. Confirm whether NMI's bar requires explicit dispersion for the faithfulness scores; if so, surface it from the §R records.
4. **(confirm) caps** — title length, abstract word cap (current 205 vs ~150–200), main-text word limit, reference cap (~50–70), display-item cap (~5–8), Extended Data (≤10): verify against the live NMI guidelines for the chosen content type.
5. **Submission logistics** — cover letter, suggested/opposed reviewers, ORCID, manuscript-system upload, preprint link: all PO actions at submission time.

## SM integration notes

- The 09 split agreed in P2-E9-1 backlog (separate `09_statements.tex`) was **not** adopted: the SM consolidated all back-matter statements into `09_endmatter.tex` (E8-9) and `main.tex:88` inputs only that one file. This is consistent and STYLE-compliant; no `09_statements.tex` exists or is needed. E9-1's contribution is this verified checklist + the pass note in the endmatter header.
- Limitations live in `08_discussion.tex` (NOT end matter) per STYLE.md §5 — do not migrate or duplicate.
