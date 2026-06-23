---
id: P2-R-REF
title: Fix references.bib — Paper 1 arXiv + code URL, MIB/SAEBench/M4, Fig-6 refs, [35] title
epic: RV (Revision — references)
status: todo
sprint: 9
owner:
where: local
depends_on: []
file_scope:
  - paper/references.bib
estimate: M
spec_ref: ../SPEC.md#e8-nature-paper-draft-writing-sprint-6-local-one-section-per-item
---

## Goal
Repair and extend the bibliography (improvement_instructions P0#3; P2#26,#27,#28,#30; P3#46;
final-review §"Correctness and clarity"). Specifically:
1. **Paper 1 (P0#3):** replace the placeholder `@misc{maier2025vcs}` (currently title "A
   Bit-Exact, Differentiable Atari 2600 (Paper 1)", year 2025, author "Bayer, Stefan") with the
   real arXiv entry: title **"A Differentiable Atari VCS: A Complex, Fully Known Ground Truth for
   Explainable AI"**, `eprint = {2606.22447}`, `archivePrefix = {arXiv}`, submitted 2026-06-21,
   authors corrected (Maier, Bayer **Siming**, Krauss). Keep the citation key `maier2025vcs` (so
   no section needs re-keying) OR add an alias note. Add a `note`/`howpublished` with the code
   URL **https://github.com/akmaier/UnderstandingVCS**.
2. **MIB 2025 (P2#26):** add the Mechanistic Interpretability Benchmark entry (key `mib2025`).
3. **SAEBench 2025 (P2#27):** add the SAEBench entry (key `saebench2025`).
4. **M4 2023 (P2#28):** add the M4 faithfulness benchmark entry (key `m4_2023`).
5. **Figure-6 missing refs (P2#30):** add bib entries for every work named in Fig. 6:
   Balduzzi 2017, Jain & Wallace 2019, Morcos et al. 2018, Leavitt & Morcos 2020,
   Elhage et al. 2022, Kindermans et al. 2019, Sturmfels et al. 2020.
6. **[35] / Yang–Kim title (P2#28, final-review):** verify and correct the title/acronym of the
   currently-mis-titled reference (the BAM/BIM Yang & Kim entry).
Use WebSearch to verify each new entry's exact title/authors/venue/year before committing.

## Definition of Done
- [ ] every new key resolves: `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main` produces **0 undefined citations** once the section items cite the new keys
- [ ] `maier2025vcs` shows the arXiv:2606.22447 title + the github URL in the rendered bib
- [ ] all seven Fig-6 works + MIB/SAEBench/M4 present with verified metadata
- [ ] the previously mis-titled Yang–Kim/[35] entry corrected
- [ ] nothing outside `file_scope` changed; Paper-1 gates untouched/green
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
references.bib is normally SM-only at Review, but in the revision it is given its OWN item so
all bib work is disjoint and parallel-safe. The section items (S02-related, S07-compare,
S08-discussion) will `\cite{}` these new keys — REF must land BEFORE or in the same sprint as
those section items so their compiles resolve. Do NOT edit any `sections/*.tex` here.
