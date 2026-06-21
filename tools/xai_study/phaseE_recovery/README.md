# Phase E — semantic recovery (design recovery): reconstructing the documentation (spec)

**The constructive goal.** Combine A–D (especially Phase-D behavioral grounding) with
the **software-reverse-engineering** toolkit to reconstruct the program's
documentation/design, and **score the recovered documentation against the truth we
hold**. The bar (from the IEEE 610.12 definition of *software* = programs + procedures +
**documentation** + data): understanding the VCS program means recovering its
documentation/design, not just describing behavior — i.e., **reverse engineering /
design recovery** (Chikofsky & Cross 1990), which needs an *external anchor* supplied by
our behavioral probing + exact oracle. Framed honestly as **measuring the gap to the
bar**, not "solving recovery." (See `xai_paper/experiment_design.md` §8 and the
Discussion in `xai_paper/xai_paper_plan.md`.)

## Sub-experiments (named methods → output → score)
- **E1 Variable / data-dictionary recovery** — binary type/var recovery (TIE, Lee 2011;
  Howard, Slowinska 2011; REWARDS, Lin 2010) + neural name/type recovery (DEBIN, He 2018;
  DIRE, Lacomis 2019; DIRTY, Chen 2022) + Phase-D grounding → candidate RAM→concept/type
  map → P/R + type-accuracy vs the true data dictionary (T3). *This IS T3 recovery.*
- **E2 Routine / concept recovery** — concept assignment (Biggerstaff 1993/94); neural
  function naming (Nero, David 2020) + Phase-C circuits → routine labels → match to the
  true disassembled routine; behavior-equivalence under scrubbing.
- **E3 Specification / behavior recovery** — dynamic invariant detection (Daikon, Ernst
  2007); specification mining (Ammons 2002); **active automata / model learning**
  (Angluin 1987 L*; Vaandrager 2017) → invariants / state machine / behavioral spec →
  held-out behavioral prediction vs true rules; **learned-automaton equivalence is
  *exact*** (the bit-exact emulator is a minimally-adequate teacher answering membership
  **and** equivalence queries).
- **E4 Redocumentation / gap-to-bar** — decompilation (Ghidra/Hex-Rays; LLM4Decompile;
  Pearce 2022) assembling E1–E3 → a human-readable design doc → **fraction of the IEEE
  "software" recovered** (documentation %) vs the true design.

## Why uniquely possible here
We hold the artifact (ROM) *and* an independent bit-exact ground truth, so we can score
genuine **recovery** — unlike the field's workaround of **reimplementation** (AtariARI
hand-reads disassemblies; OCAtari hand-codes extractors; JAXAtari fully reimplements),
which re-authors documentation and is circular as a ground truth.

## Pilot (local)
`pilot_recovery.{py,jl}`: Daikon-style invariant mining + L* automaton learning on one
routine (e.g., Pong's ball-bounce or the CPU-opponent tracker); score the recovered
spec/automaton vs the true routine (equivalence exact via the reference).

## Scale-out (cluster)
trace logging for invariant mining; query-driven automata learning; decompilation ×
games — many short deterministic re-runs / queries.

Outputs: `out/recovery_<method>_<game>.*` + the **recovery-rate** table (gap to the
documentation bar), feeding the cross-tradition comparison (Results reporting).
