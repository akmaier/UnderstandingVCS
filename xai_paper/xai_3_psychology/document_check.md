# P3 — Submission compliance checklist — *PNAS* (Research Article)

> Stub — confirm every number against the live PNAS author guidelines before
> submission. Alternative venues (NMI, *Cognition*, CogSci) have different rules; if the
> target changes, replace this file.

## A. Logistics
- [ ] PNAS submission route chosen (Direct Submission vs Contributed/Prearranged Editor).
- [ ] Significance Statement written (**≤120 words**, plain language) **(confirm)**.
- [ ] Suggested editors/reviewers; ORCIDs; correct affiliations.
- [ ] Preprint allowed (PNAS permits bioRxiv/arXiv) — link if posted.

## B. Format & length
- [ ] PNAS LaTeX template (`pnas-new` / Overleaf) — **fetch into `paper/` (not yet present)**.
- [ ] Main text within the PNAS length cap **(confirm; ~6 pages / word+display budget)**.
- [ ] Abstract within limit **(confirm; ~250 words)**.
- [ ] References within style/cap; PNAS numbered style.
- [ ] Figures: vector or ≥300 dpi; colour-blind-safe.

## C. Required statements
- [ ] **Data availability** (ROM provenance — not redistributed; the behavioral-probe data + harness).
- [ ] **Code availability** (the probing harness; reuse of jutari/jaxtari + the P2 oracle, cited).
- [ ] **Author contributions** (PNAS author-contribution categories).
- [ ] **Competing interests**.
- [ ] **No human-subjects research** (the "participant" is a program) — state explicitly to avoid an IRB query.
- [ ] **Statistics & reproducibility:** n, tests, uncertainty; psychometric fits with CIs; generalization to held-out stimuli reported.

## D. Content coverage vs `plan.md`
- [ ] **Significance + Intro:** the behavioral tradition; the no-ground-truth problem (Shiffrin & Mitchell; Binz & Schulz); the known-machine opportunity.
- [ ] **Results — Phase D:** the named behavioral methods (psychometric / constant-stimuli / SDT / reverse-correlation / drift-diffusion / cognitive battery), each scored vs the true code; the **trustworthy-vs-mirage** map; the externally-manifest-only boundary.
- [ ] **Discussion:** implications for psychology-of-AI on LLMs; behavioral probing as the **semantic-grounding bridge** to documentation recovery (P4); limits.
- [ ] **Methods:** the VCS substrate; the shared ground-truth oracle (cite P2 §1); the controlled-stimulus protocol; the true-driver ground truth.
- [ ] Every claim backed; all references real (run the no-hallucination pass — incl. Wichmann & Hill, Green & Swets, Ahumada, Murray, Ratcliff, Binz & Schulz, Shiffrin & Mitchell).
