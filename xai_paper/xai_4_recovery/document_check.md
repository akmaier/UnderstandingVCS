# P4 — Submission compliance checklist — *ICSE / FSE / ASE* (Research Track)

> Stub — confirm against the chosen venue's live CfP (ICSE/FSE/ASE differ in length,
> double-blind rules, and artifact tracks). If the target becomes USENIX Security/NDSS
> or a journal (TSE/EMSE), replace this file.

## A. Logistics
- [ ] Target venue + track chosen (ICSE/FSE/ASE Research Track; or Security; or journal).
- [ ] **Double-blind** prepared if required (ICSE/FSE are double-blind): no author names,
      anonymized self-citations, anonymized repo/artifact link.
- [ ] Page limit met **(confirm; ~10–12 pp + references, venue-specific)**.
- [ ] ACM/IEEE template (`acmart` / IEEEtran) — **fetch into `paper/` (not yet present)**.
- [ ] Submission system (HotCRP) + topic/area selected.

## B. Artifact (SE venues weight this heavily)
- [ ] **Artifact Evaluation** package planned: the benchmark (ROM-derived tasks +
      ground-truth design) + the recovery harness (invariant mining, L*, decompilation
      glue), reproducible; aim for Available/Functional/Reusable badges.
- [ ] ROMs **not redistributed** — ship hashes + AutoROM instructions (as in Paper 1);
      ensure the artifact is runnable without shipping copyrighted ROMs.

## C. Content coverage vs `plan.md`
- [ ] **Intro:** software = documentation+programs (IEEE 610.12); the stripped-ROM
      problem; reimplementation (AtariARI/OCAtari/JAXAtari) is circular; we score genuine
      recovery.
- [ ] **Related work:** Chikofsky & Cross (RE/design recovery); Biggerstaff (concept
      assignment); Ammons (spec mining); Ernst (Daikon); Angluin/Vaandrager (model
      learning); TIE/Howard/REWARDS + DEBIN/DIRE/DIRTY; decompilation (Ghidra;
      LLM4Decompile; Pearce).
- [ ] **Approach/Eval — Phase E:** E1 data-dictionary, E2 routine/concept, E3 spec/
      behavior (with the **measured residual equivalence gap**, not "exact"), E4
      redocumentation (with the **recovery-% denominator defined**).
- [ ] **Threats to validity:** under-determination (design recovery needs an external
      anchor); the conformance horizon; representativeness (a 6502 vs modern software);
      the membership-vs-equivalence oracle distinction stated honestly.
- [ ] **Discussion:** what RE can/can't recover (structure/behavior high; intent/names
      low); the behavioral anchor (P3); reusability of the benchmark.

## D. Statements & gate
- [ ] Data/Code availability (artifact link; jutari/jaxtari + P2 oracle cited).
- [ ] Reproducibility: seeds, configs, query budgets for the model-learning recorded.
- [ ] All references real (no-hallucination pass — verify TIE 2011, Howard 2011,
      REWARDS 2010, DIRE 2019, DIRTY 2022, DEBIN 2018, Nero 2020, Daikon 2007,
      Ammons 2002, Angluin 1987, Vaandrager 2017, Chikofsky & Cross 1990, Biggerstaff
      1993/94, Pearce 2022, LLM4Decompile 2024).
- [ ] Clean build; figures/tables within limits.
