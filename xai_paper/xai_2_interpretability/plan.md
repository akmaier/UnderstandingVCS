# P2 — Paper plan (storyline & Nature structure)

> **Role of this file:** the *storyline* + the Nature-Portfolio structure — what we
> write the paper from. Experiments, oracle, T3, scoring tables, compute → `experiment_
> design.md`. The program map + shared invariants → [`../general_paper_plan.md`](../general_paper_plan.md);
> doc rules → [`../README.md`](../README.md).

## P2 in one line
*Do our interpretability methods recover what is true?* — a **ground-truth faithfulness
benchmark** for the interpretability toolkit, run on the fully-known, bit-exact,
differentiable Atari VCS. **Phases A (neuroscience battery) + B (attribution/XAI) + C
(mechanistic interpretability).** Behavioral probing (D) and semantic recovery (E) are
**companion papers** (P3, P4); learned agents are P5.

## Metadata
- **Working title:** *Do our methods recover what is true? A fully-known, differentiable
  model organism for evaluating interpretability.*
- **Authors (provisional):** A. Maier, S. Bayer, P. Krauss (+ collaborators).
- **Target:** *Nature Machine Intelligence* (primary), written to *Nature*'s bar
  (shared template); decide after pilots. Article (or Analysis).

## Subject & scope
The subject is the **Atari VCS itself** — chip, program, game logic — studied with the
interpretability toolkit and scored against the substrate's ground truth. **No learned
agents** (that is P5). Ground truth is tiered (T1 causal exact · T2 hardware exact · T3
game-concept partial); "interpretable" is operationalized as **ground-truth recovery**
(our definition; Barbiero et al. 2025 propose a *different* notion, inference-
equivariance — cited as contrast, not as the source of ours).

## The storyline (the throughline that must hold)
1. **The gap.** Attribution/XAI, mechanistic interpretability, and neuroscience methods
   all *find structure*, but on real systems there is **no ground truth**, so none can
   be *validated* — Jonas & Kording's warning, unanswerable in the wild.
2. **The opportunity.** Paper 1 built a complex system that is fully known, bit-exact,
   and differentiable, so the *true* causal structure of any output is computable.
3. **The move.** Turn the toolkit — neuroscience/mechanistic, attribution/XAI,
   mechanistic interpretability — on this one system and **score every method against
   ground truth**.
4. **The payoff.** A measured map of which methods recover truth vs produce plausible-
   but-wrong accounts; a reusable ground-truth benchmark; concrete directions.

## Why a 1977 chip is a fair test (the representativeness argument — front and centre)
The obvious objection: *"you study deterministic hand-coded software to judge methods
built for learned, distributed, stochastic neural nets — why does a score here predict
anything about a transformer?"* Our answer, stated explicitly in the Introduction and
Discussion:
- **It is a *necessary-condition screen*, not a sufficiency predictor.** A method that
  fails *here* — perfect ground truth, full observability, exact interventions, **no
  learning** — has no business being trusted on a neural network. Passing here is
  necessary, not sufficient.
- **The VCS *does* exhibit the failure modes that make interpretation hard:** aliased /
  polysemantic RAM cells reused across routines (superposition-like), race-the-beam
  timing (distributed temporal computation), floating-bus reads, and registers whose
  meaning is context-dependent. We map each VCS property to the NN difficulty it does /
  does not capture (a Discussion display item).
- **What it does *not* capture** (learned features, scale, stochasticity) we state
  plainly and defer to P5 (learned agents).

## Contributions (vs Paper 1 and vs Jonas & Kording)
1. A **ground-truth interpretability benchmark** on one fully-known, differentiable,
   *real* complex system — and the necessary-condition-screen framing for using it.
2. A **quantitative re-run of Jonas & Kording** (Phase A) — the neuroscience battery
   scored against the known mechanism (calibration baseline; see §novelty).
3. A **faithfulness audit of attribution/XAI** (Phase B) on the VCS's own computation,
   including the finding that several popular methods (Grad-CAM, attention) do not even
   *apply* to a non-neural system.
4. **Mechanistic interpretability validated on a *known* circuit** (Phase C) — the first
   ground-truth calibration of patching/SAEs/circuits *on a real artifact* (vs prior
   synthetic/compiled-network benchmarks; see §novelty).
5. A **formal, operationalized notion of interpretability** = ground-truth recovery
   (ours; Barbiero 2025 as contrast).
6. **The discrepancy + concrete directions** for faithful, ground-truth-validated
   interpretability (pointing to the companion papers P3 behavioral, P4 recovery).

## Novelty / related work (must be argued, not assumed — verify cites before bib)
Prior work *does* score interpretability against ground truth, on **synthetic or
compiled** systems: **Tracr** (Lindner et al. 2023, compiled transformers with circuits
by construction), **InterpBench** (Gupta et al. 2024, semi-synthetic transformers with
known circuits), **BIM** (Yang & Kim 2019, ground-truth attribution benchmark), and
synthetic-attribution benchmarks. **Our delta:** we score against a *real, deployed,
hand-authored, complex* artifact (the shipped ROM on real silicon semantics) whose
ground truth is **independent of the artifact** (not a toy built to be interpretable,
not a program compiled into weights). State this explicitly; do not claim an unqualified
"first."

## The single subject — the VCS (framing)
`output(t) = VCS(ROM, inputs[0..t], initial_state)`, three observable/intervenable
faces: **inputs** (ROM, joystick, switches — *we* drive them), **internal state** (RAM,
CPU/TIA/RIOT registers, opcodes — the "activations"), **outputs** (pixel, score, event —
the "predictions to explain"). One subject carries A, B, and C.

---

# Paper structure (Nature Portfolio)

### Abstract (~150–200 words) — draft sketch
Interpretability methods find structure but cannot be validated without ground truth. We
make the Atari VCS — fully known, bit-exact, differentiable — a model organism and score
neuroscience, attribution, and mechanistic-interpretability methods against the true
causal structure. We find [headline], release a ground-truth benchmark, and frame it as
a necessary-condition screen for trusting these methods on neural networks.

### Introduction
The arc (gap → opportunity → move → payoff); the representativeness/necessary-condition
framing up front; cite Jonas & Kording, the XAI/mech-interp literature, the prior
ground-truth benchmarks (§novelty), and Paper 1. Operationalize "interpretable" as
ground-truth recovery.

### Results — one subsection per phase (the *what*; the *how* is in `experiment_design.md`)
- **A — neuroscience battery.** The Kording battery scored: classical methods find rich
  structure yet score low against the known mechanism, *quantifying* his lesson and
  setting the low-F baseline.
- **B — attribution / XAI.** Attribution of a VCS output to its causes scored: popular
  saliency plausible-but-unfaithful; causal/gradient methods faithful; several popular
  methods don't even apply (NN-specific).
- **C — mechanistic interpretability.** The mech-interp toolkit on a *known* circuit:
  the first ground-truth calibration on a real artifact — what patching/SAEs/circuits
  recover, where they break.
- **Cross-tradition comparison + a faithful-method demonstration** — the headline figure
  placing A–C on the shared faithfulness-vs-plausibility axes (reporting).

### Discussion
- **Many traditions, one ground truth** (A–C validated on a complex system); **Kording's
  lesson made measurable**; **mech-interp calibrated on a known circuit**.
- **The shared toolkit** (tuning curves ↔ feature viz; lesions ↔ ablations; connectomics
  ↔ circuits; probing ↔ dim-reduction; Granger ↔ causal attribution).
- **Representativeness** (the necessary-condition screen; the VCS↔NN failure-mode map).
- **What interpretability needs** (causal over correlational; ground-truth-validated
  benchmarking; sieve methods on known systems first) — and **the two open frontiers we
  pursue in companion papers:** behavioral probing as the semantic-grounding bridge
  (P3), and recovering the *documentation/design* as the real bar for "understanding
  software" (P4; the IEEE-software / reverse-engineering framing lives there).
- **Beyond the brain:** differentiability tells us which *gradient* explanations are
  trustworthy (Krauss co-author fit). **Toward P5:** the learned agent is the hard case.

### Methods
Summarize the emulator, the ground-truth oracle, and the metrics; full setups in
`experiment_design.md`.

### Display items (~6 figures)
(1) the VCS platform & ground-truth oracle; (2) A–C scored on the shared faithfulness-
vs-plausibility axes (headline); (3) Kording battery scored; (4) attribution vs
mechanistic-interp faithfulness; (5) the VCS↔NN failure-mode / representativeness map;
(6) failure taxonomy / what to do instead.

### End matter
Data/Code availability, Author contributions, Competing interests, Reporting Summary,
Acknowledgements — tracked in [`document_check.md`](document_check.md).

## Journal — NMI vs Nature
**Primary: NMI** (interpretability/XAI, AI↔neuroscience, benchmarks, methods-critique;
the "discrepancy + directions" framing is NMI's Analysis type or a regular Article).
**Nature** only if the discrepancy is dramatic and broadly framed; same template, so
shoot-for-Nature-first at near-zero switching cost. Lean **Article**. Decide after pilots.

## Open storyline questions
- Lead with the *benchmark* (Article) or the *critique* (Analysis)? Decide after pilots.
- **Granularity vs Kording:** architectural level leads; the Visual6502 transistor track
  is a parallel head-to-head we may feature or cut (an experiment-design decision — and a
  novelty risk, since transistor-level *is* J&K's level).
