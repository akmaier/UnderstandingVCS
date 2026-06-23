# Sprint 7 — Write the Nature/NMI draft (E8) [PO-approved, with the semantics reframe]

## Goal
Draft Paper 2 in Andreas Maier's voice (STYLE.md), centered on the reframed thesis
(plan.md, commit `17c73c4`): **faithful attribution recovers the causal wiring, not the
semantics — "a faithful explanation is not yet an understanding."** The semantics/understanding
question is explicit in the **title, introduction, results-comparison, and discussion**. Honour
the honesty contract: Paper 2 *measures faithfulness + defines/measures the semantic gap*;
semantic recovery proper is Paper 4; T3 labels are external (verified, not discovered).

## Planning
Per-section files under `paper/sections/` (SCRUM §6, disjoint); `paper/main.tex` +
`references.bib` are SM-only (skeleton + Review merge). Figures already at `paper/figures/`
(fig1–6). Every claim cites a committed number (leaderboard.md / §R records) or a figure.

- **Wave 0 (skeleton, blocks the rest):** `E8-0` — `paper/main.tex` (NMI/Nature template,
  reframed title, `\input` the 10 section stubs), empty `sections/NN_*.tex`, `references.bib`.
- **Wave 1 (substance):** `E8-4` methods (03), `E8-5` results-A (04), `E8-6` results-B (05).
- **Wave 2 (substance):** `E8-7` results-C (06), `E8-8` results-comparison (07, the semantic-gap
  section — the new contribution).
- **Wave 3 (frame):** `E8-2` introduction (01), `E8-3` related-work (02), `E8-9` discussion +
  end-matter (08/09).
- **Wave 4:** `E8-1` abstract (00) — last, summarizes the assembled draft.

Each writer: read plan.md (thesis + honesty contract — do not drift), experiment_design.md
(numbers), STYLE.md (voice), the leaderboard + figures; write ONE section `.tex`; cite real
numbers + `\ref` figures; follow the honesty contract.

## Review (SM)
Assemble `main.tex`, merge `references.bib`, build the PDF, check it compiles + page/limit;
then the **E9 gate** (PAUSE for PO): document_check, reference no-hallucination pass,
reproducibility bundle, final build.

## Review — closed 2026-06-23: the NMI draft is WRITTEN and BUILDS (E8 10/10)

- **E8-0 skeleton** (sn-jnl / sn-nature, reframed title + provisional authors) + **E8-1..9** all
  sections in Maier's voice, centered on *faithful attribution recovers the wiring, not the
  semantics*, every quantitative claim citing a committed number or one of the 6 figures, honesty
  contract honoured.
- **SM integration:** merged 59 section BibTeX entries → `references.bib` (95 total); fixed a
  **corrupted `sn-nature.bst`** (`format.in.ed.booktitle` popped an empty stack on every
  `@inproceedings`) + 3 related-work cross-ref typos.
- **Build:** `latexmk` exits 0 → `paper/main.pdf`: **37 pages, 0 undefined refs/cites, all 6
  figures embedded** (commit `301cc3d`).
- **Length:** ~19.5k words / 37 pp — a complete but MAXIMAL v1; **needs condensing** to NMI Article
  length in E9 (abstract 289 w → ~200).

**Next:** E9 (submission prep) — **PO-gated**: E9-1 document_check, E9-2 reference no-hallucination
pass, E9-3 reproducibility bundle, E9-4 condense + final page/limit build. **PAUSED for PO.**
