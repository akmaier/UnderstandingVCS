# P4 — Experiment design (Phase E: semantic / design recovery)

> **Role of this file:** the experiments and how each is scored. Storyline → `plan.md`.
> **The correctness triad (§0), the ground-truth oracle (§1), the T3 procedure (§2),
> and the substrate audit (§3) are SHARED in
> [`../xai_2_interpretability/experiment_design.md`](../xai_2_interpretability/experiment_design.md)
> §0–§3 — reused, not repeated.** Subject: the VCS ROMs as software (no learned agents).
> Citations: verify before bib.

## Substrate feasibility (delta vs the shared §3 audit)
Phase E needs: **trace logging** (for invariant mining), **membership + equivalence
queries** (for model learning), and the **ROM binary** (for decompilation/var recovery) —
all available now. **T3 is the *target* of E** (E is, in part, T3 recovery), so E is
scored on the verified-T3 subset (shared §2) and *extends* it. Stay within the
conformance horizon (shared §3).

> **Oracle nuance (do not overclaim):** the bit-exact emulator is an **exact membership
> oracle** (run the input, observe the output) and a **high-coverage conformance-testing
> equivalence oracle** (sample inputs / bounded check against the reference) — *not* an
> exact equivalence oracle. Learned models are therefore exact over the **explored input
> space / chosen finite abstraction**, PAC-style otherwise; the residual equivalence gap
> is a **measured** quantity, not zero.

## Phase E — methods

| Analysis (named method) | Finding (output) | Measured score | Needs T3? |
|---|---|---|---|
| **E1** Variable / data-dictionary recovery — binary type/var recovery (TIE, Lee et al. 2011; Howard, Slowinska et al. 2011; REWARDS, Lin et al. 2010) + neural name/type recovery (DEBIN, He et al. 2018; DIRE, Lacomis et al. 2019; DIRTY, Chen et al. 2022) + **P3 behavioral grounding** | candidate RAM/register → concept/type map (a "data dictionary") | precision/recall + type-accuracy vs the *true* data dictionary; fraction grounded *observationally* | **This IS T3 recovery** — scored vs held-out T3 |
| **E2** Routine / concept recovery — concept assignment (Biggerstaff et al. 1993/94); neural function naming (Nero, David et al. 2020) + **P2 Phase-C circuits** | "this routine = ball-bounce / opponent-AI" labels | match to the true disassembled routine; behavior-equivalence under scrubbing | naming = T3; structure = T1/T2 |
| **E3** Specification / behavior recovery — dynamic invariant detection (Daikon, Ernst et al. 2007); specification mining (Ammons et al. 2002); **active automata / model learning** (Angluin 1987 L*; Vaandrager 2017) | inferred invariants / state machine / behavioral spec | held-out behavioral prediction vs true rules; invariant P/R; learned-automaton equivalence over the chosen abstraction **+ the measured residual equivalence gap** | **No** for the behavioral spec (vs true rules = T1/T2); naming states/vars = T3 |
| **E4** Redocumentation / decompilation & gap-to-bar — decompilation (Ghidra/Hex-Rays; LLM4Decompile 2024; Pearce et al. 2022) assembling E1–E3 | human-readable design doc | **fraction of the IEEE "software" recovered** (documentation %) vs the true design — *with the denominator defined concretely* | naming = T3 |

## Method matrix — expected (a prediction to test)
| Method | Applies? | Expected | Why |
|---|---|---|---|
| Active automata / model learning (E3) | ✓ | **Succeed over the chosen finite abstraction** | exact membership oracle + conformance EQ oracle; report the residual gap |
| Dynamic invariant detection — Daikon (E3) | ✓ | **Partial→Succeed** | recovers true invariants from traces; coverage-limited to what traces exercise |
| Binary type/variable recovery — TIE/DIRTY (E1) | ✓ | **Partial** | recovers types/structure; *names* are guesses (no intent in the binary) |
| Decompilation (+LLM) (E4) | ✓ | **Partial** | recovers code structure, not intent; redocumentation hallucination risk |
| Concept assignment / feature location (E2) | ✓ | **Partial / hard** | the AI-hard mapping; needs the P3 behavioral anchor |

**Ideal:** the recovered documentation == the true design (the IEEE "software" made
whole). **Right when:** recovered dictionary/routines/spec match truth (F); predict
held-out behavior on the exact re-run (S); design-level (M). **Best case:** a measured
**recovery rate toward the bar** showing (i) behavioral grounding is *necessary* to cross
from operational to design semantics, and (ii) current methods fall short of full
recovery — with our substrate as the first *non-circular* yardstick (vs reimplementation).

## Pilot (local)
`pilot_recovery`: Daikon-style invariant mining + L* automaton learning on one routine
(Pong's ball-bounce or the CPU-opponent tracker); score the recovered spec/automaton vs
the true routine; report the residual equivalence gap. Reuse the shared oracle (P2 §1)
and `tools/xai_study/phaseE_recovery/`.

## Scale-out (cluster)
trace logging for invariant mining; query-driven automata learning; decompilation ×
games — many short deterministic re-runs / queries (CPU array).
