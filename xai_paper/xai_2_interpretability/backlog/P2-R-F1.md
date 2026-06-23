---
id: P2-R-F1
title: Redraw Fig 1 (platform/oracle) — jutari(Julia), plausibility-proxy, split/declutter, fonts
epic: RV (Revision — figure)
status: todo
sprint:
owner:
where: local
depends_on: []
file_scope:
  - paper/figures/fig1_platform_oracle.py
estimate: M
spec_ref: ../SPEC.md#e7-figures-sprint-5-local-one-figure-per-item
---

## Goal
Fix Figure 1 per figure_detail_pass "Fig. 1" + improvement_instructions P4#38. Edit ONLY the .py
(regenerate fig1_platform_oracle.pdf). Apply:
- **jutari label (BLOCKER):** line 155 "jutari (JAX) · jaxtari (JAX)" → "jutari (Julia) · jaxtari
  (JAX)" (or "jutari (Julia/Zygote) · jaxtari (JAX/XLA)").
- **plausibility proxy (BLOCKER):** lines 28–29, 286 "human-plausibility"/"plausibility" → label
  the Y axis / reporting axis "plausibility proxy"; remove any "human" framing.
- **Declutter / split:** split into three clean columns (Substrate / Oracle / Scoring) or two
  panels (1a substrate+oracle, 1b scoring triad); remove most in-box prose; short labels only
  ("Exact re-run", "do(u:=v')", "Δy", "F: causes", "S: behavior", "M: parsimony"); make
  "oracle" vs "method under test" visually distinct.
- **Discrete-index caveat** as a separate small callout under the oracle, not nested in the
  gradient panel.
- **Remove source-path microtext:** line 323 (oracle: tools/xai_study/...) and any other
  provenance string → drop from figure body (provenance goes to SUPP).
- **Fonts:** minimum 8 pt for all in-figure text; legend outside or replaced by panel headers.
- **Global style guide:** colorblind-safe palette, no text boxes over data, legends outside axes,
  embedded fonts in the exported PDF.

## Definition of Done
- [ ] regenerates: `python paper/figures/fig1_platform_oracle.py` → updated fig1_platform_oracle.pdf
- [ ] grep clean in the .py: no "(JAX)" for jutari, no "human plausibility", no source paths in the drawn figure
- [ ] all in-figure text ≥ 8 pt; oracle vs method-under-test visually distinct; PDF has embedded fonts
- [ ] nothing outside `file_scope` changed; Paper-1 gates untouched/green
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
CAPTION RULE: figure items edit ONLY the .py; the Fig.1 caption lives in sections/03_methods.tex
and is owned by P2-R-S03-methods. HANDOFF to S03: shorten the Fig.1 caption, say "plausibility
proxy" and "jutari (Julia)", move precision/recall/sign/magnitude detail to caption/SUPP.
