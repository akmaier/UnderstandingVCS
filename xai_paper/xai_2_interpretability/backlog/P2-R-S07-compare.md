---
id: P2-R-S07-compare
title: Revise leaderboard section — canonical count, exploratory framing, proxy, "verified:false", Fig2+Fig6 captions
epic: RV (Revision — section)
status: todo
sprint: 11
owner:
where: local
depends_on: [P2-R-UNC, P2-R-F2, P2-R-F6, P2-R-REF]
file_scope:
  - paper/sections/07_results_compare.tex
estimate: L
spec_ref: ../SPEC.md#e8-nature-paper-draft-writing-sprint-6-local-one-section-per-item
---

## Goal
Revise the cross-tradition leaderboard section per improvement_instructions P3#51 (and P0#4,#9,#11,
#12; P1#17,#24; P3#33,#34; P4#42,#43,#44). This section owns the **canonical method count** the
rest of the paper inherits. Apply IN THIS FILE:
- **Canonical method count (P0#4):** line 28 "31 method rows" — fix to ONE consistent count
  (decide: "N methods" vs "N method rows including the oracle positive control") and use it as the
  source of truth that S00/S05/SUPP/F2/F4 all match.
- **Exploratory framing + robustness (P3#51, P1#17):** make the leaderboard explicitly exploratory
  unless metrics are commensurable; provide alternate rankings + robustness (read from P2-R-UNC's
  aggregation_robustness.csv); do not present the plausibility proxy as a measured human result.
- **Plausibility proxy (P0#9):** any "plausibility" framing here labeled as a proxy, not a
  measurement (line 55 already hedges — make it explicit and consistent).
- **Remove "verified:false" (P1#24, P4#44):** line 165 `\texttt{verified:false}` → describe the
  provenance in prose without the implementation token.
- **Semantic-gap precision + NN scope (P0#11,#12):** operationalize the gap as three separations
  (faithful-not-sufficient, named-not-causal, decodable-not-used) with formal diagnostics; keep
  the NN map as analogy/evidence-transfer, not proof.
- **Tone (P3#33,#34):** soften combative phrasing; move repeated claims into tables.
- **Fig. 2 caption (P4#39):** Fig.2 .py fixed by P2-R-F2 — this item owns the Fig.2 caption (lines
  49–66): shorten, key-findings as a table not callouts, legends-outside note, "plausibility
  proxy", oracle at §3.2.
- **Fig. 6 caption (P4#43):** Fig.6 .py fixed by P2-R-F6 — this item owns the Fig.6 caption (lines
  173–191): shorten, label NN counterparts as "analogue" not proof, ensure cited works resolve
  (depends_on P2-R-REF), no "Section 1 oracle".

## Definition of Done
- [ ] ONE canonical method count established here; no "verified:false"; no stale "Section 1 oracle" in this file
- [ ] leaderboard framed exploratory with alternate rankings/robustness from aggregation_robustness.csv; proxy clearly not a measurement
- [ ] semantic gap = three named separations with diagnostics; NN map labeled analogy
- [ ] Fig.2 + Fig.6 captions shortened, proxy/analogy wording, Fig-6 cites resolve (depends P2-R-REF)
- [ ] `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main` clean, 0 undefined cites
- [ ] nothing outside `file_scope` changed; Paper-1 gates untouched/green
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
Pivotal item: it sets the canonical method count for S00/S05/SUPP/F2/F4. Depends on P2-R-UNC
(robustness), P2-R-F2/F6 (figure consistency), P2-R-REF (Fig-6 cites). This file embeds BOTH Fig.2
and Fig.6 (figure number ≠ section order) — owns both their captions.
