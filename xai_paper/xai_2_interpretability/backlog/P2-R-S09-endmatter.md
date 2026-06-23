---
id: P2-R-S09-endmatter
title: Revise endmatter — code review-time availability, [33]/[34] as archived artifacts, Paper-1 cite, contributions
epic: RV (Revision — section)
status: todo
sprint: 13
owner:
where: local
depends_on: [P2-R-REPRO, P2-R-REF]
file_scope:
  - paper/sections/09_endmatter.tex
estimate: S
spec_ref: ../SPEC.md#e9-reproducibility--submission-prep-sprint-7-local-sm-led
---

## Goal
Revise the endmatter per improvement_instructions P0#1, P3#37 (and final-review checklist:
review-time code availability; [33]/[34] as archived artifacts; complete author contributions/
acknowledgements). Apply IN THIS FILE:
- **Code availability (P0#1):** lines 41–43 "will be released ... under the MIT license upon
  acceptance" → review-time availability: "available for review at [anonymized link]" (or
  venue-compatible), and cite the companion emulator code https://github.com/akmaier/UnderstandingVCS
  via arXiv:2606.22447. Point to REPRODUCIBILITY.md.
- **Paper-1 cite (P0#3, P3#37):** line 38 `\cite{maier2025vcs}` now resolves to the arXiv entry
  (fixed in P2-R-REF); ensure the companion-paper roadmap is brief and does not lean on
  unpublished Papers 4/5 (line 9 mentions Paper 4 — keep as a forward pointer only).
- **[33]/[34] artifacts (final-review):** ensure the records cited as [33]/[34] (leaderboardP2 etc.)
  read as archived, reviewable artifacts (point to the repro bundle), not opaque internal cites.
- **Author contributions + acknowledgements (final-review):** complete them.

## Definition of Done
- [ ] no "released ... upon acceptance" for the XAI suite remains; review-time availability + github/arXiv pointers present
- [ ] Paper-1 cite resolves to arXiv:2606.22447; roadmap brief; author contributions + acknowledgements complete
- [ ] records cite the reviewable bundle (REPRODUCIBILITY.md)
- [ ] `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main` clean
- [ ] nothing outside `file_scope` changed; Paper-1 gates untouched/green
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
Depends on P2-R-REPRO (the doc + bundle it points to) and P2-R-REF (Paper-1 arXiv key). The
anonymized review link itself is a PO decision (double-blind logistics) — see revision_plan §PO;
this item writes the wording with a placeholder the PO fills.
