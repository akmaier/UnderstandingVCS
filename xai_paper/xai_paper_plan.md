# Paper 2 — Paper plan (storyline & Nature structure)

> **Role of this file:** the *storyline* and the *Nature-Portfolio structure* — how to
> tell the story so it is sound. **This is what we write the paper from.** All
> experimental setups, method lists, scoring tables, the oracle, the T3 procedure, the
> compute plan, and the substrate audit live in
> [`experiment_design.md`](experiment_design.md). See [`README.md`](README.md) for the
> document rules.

## Metadata
- **Working title:** *Do our methods recover what is true? A fully-known,
  differentiable model organism for evaluating interpretability.*
- **Authors (provisional):** A. Maier, S. Bayer, P. Krauss (+ collaborators);
  Krauss's neuroscience/neuroprosthetics background anchors the discussion bridge.
- **Target:** *Nature Machine Intelligence* (primary), written to *Nature*'s bar
  (shared template); decide after the pilots. Article (or Analysis).

## Subject & scope (one sentence)
The subject is the **Atari VCS itself** — the chip, the game program, and the game's
own decision logic — studied with the full interpretability toolkit and scored against
the substrate's ground truth. **No learned agents** (a natural Paper 3); this keeps
the study self-contained on the current substrate.

## The storyline (the throughline that must hold)
1. **The gap.** Attribution/XAI, mechanistic interpretability, and the behavioral
   "psychology of AI" all *find structure*, but on real systems there is **no ground
   truth**, so none can be *validated* — Jonas & Kording's warning, and Shiffrin &
   Mitchell's caution, are unanswerable in the wild.
2. **The opportunity.** Paper 1 built a complex system that is **fully known,
   bit-exact, and differentiable**, so the *true* causal structure of any output is
   computable.
3. **The move.** Turn the whole toolkit — neuroscience/mechanistic, attribution/XAI,
   mechanistic interpretability, behavioral — on this *one* system and **score every
   method against ground truth**.
4. **The payoff.** A measured map of which methods recover truth and which produce
   plausible-but-wrong accounts; a reusable ground-truth benchmark; and concrete
   directions for a faithful science of understanding complex systems.

"Interpretable" is operationalized (Barbiero et al. 2025) as **ground-truth recovery**
— making "is this explanation correct?" measurable rather than a matter of taste.

## Contributions (what's new vs Paper 1 and vs Jonas & Kording)
1. A **ground-truth interpretability benchmark** on one fully-known, differentiable,
   complex system.
2. A **quantitative re-run of Jonas & Kording** — the neuroscience battery scored
   against the known mechanism.
3. A **faithfulness audit of attribution/XAI** on the VCS's own computation, including
   the finding that several popular methods (Grad-CAM, attention) do not even *apply*
   to a non-neural system.
4. **Mechanistic interpretability validated on a *known* circuit** — the first
   ground-truth calibration of patching/SAEs/circuits.
5. A **ground-truth test of behavioral probing** on the game's own decision logic —
   Shiffrin & Mitchell's caution made measurable.
6. A **formal, operationalized notion of interpretability** (Barbiero) = ground-truth
   recovery.
7. **The discrepancy + concrete directions** for faithful, ground-truth-validated
   interpretability.

## The single subject — the VCS (framing)
The object of study is the **running Atari VCS**, a deterministic function
`output(t) = VCS(ROM, inputs[0..t], initial_state)` with three fully observable,
intervenable faces: **inputs** (ROM, joystick, switches — *we* drive them), **internal
state** (RAM, CPU/TIA/RIOT registers, opcodes — the "activations"), and **outputs** (a
pixel, the score, a game event — the "predictions to explain"). The game's *own
decision logic* (e.g., Pong's CPU opponent) is a genuine hand-coded decision-maker we
can study behaviorally without any learned agent. One subject carries every tradition.

---

# Paper structure (Nature Portfolio) — the outline we write into

### Abstract (~150–200 words, unreferenced) — draft sketch
Interpretability methods find structure but cannot be validated without ground truth.
We make the Atari VCS — a complex system that is fully known, bit-exact, and
differentiable — a model organism: we apply neuroscience/mechanistic, attribution,
mechanistic-interpretability, and behavioral methods to it and score each against the
true causal structure. We find [headline result], release a ground-truth benchmark,
and argue [direction].

### Introduction
The arc above (gap → opportunity → move → payoff). Frame the three/four traditions;
cite Jonas & Kording, the XAI/mech-interp/behavioral literatures, and Paper 1.
Operationalize "interpretable" as ground-truth recovery.

### Results — one subsection per phase (the *what*; the *how* is in `experiment_design.md`)
- **A — neuroscience battery.** The Kording battery scored: classical methods find
  rich structure yet score low against the known mechanism, *quantifying* his lesson.
- **B — attribution / XAI.** Attribution of a VCS output to its causes scored:
  popular saliency is plausible-but-unfaithful, causal/gradient methods are faithful,
  and several popular methods do not even apply (NN-specific).
- **C — mechanistic interpretability.** The mech-interp toolkit on a *known* circuit:
  the first ground-truth calibration — what patching/SAEs/circuits recover, where they
  break.
- **D — behavioral / psychology.** Probing the game's own decision logic: which
  behavioral inferences match the true code and which are "right for the wrong reasons."
- **Cross-tradition discrepancy + a faithful-method demonstration** (the headline
  figure; see Phase E, deferred).

### Discussion
- **Many traditions, one ground truth** — the VCS is the first place all can be
  *validated* on a complex system.
- **Kording's lesson, made measurable**; **mech-interp finally calibrated** on a known
  circuit; **the behavioral question answered** on a known mechanism.
- **The shared toolkit** (tuning curves ↔ feature viz; lesions ↔ ablations;
  connectomics ↔ circuits; probing ↔ dim-reduction; Granger ↔ causal attribution).
- **Beyond the brain:** differentiability tells us which *gradient* explanations are
  trustworthy. (Co-author fit: Krauss.)

### Methods
Summarize the emulator, the ground-truth oracle, and the metrics; the full setups are
in `experiment_design.md`. (Nature Methods has no length limit.)

### Display items (~6 figures) — the figure story
(1) the VCS platform & ground-truth oracle; (2) the traditions scored side by side
(headline); (3) Kording battery scored; (4) attribution vs mechanistic-interp
faithfulness; (5) the behavioral verdict vs the true code; (6) failure taxonomy / what
to do instead.

### End matter
Data availability, Code availability, Author contributions, Competing interests,
Reporting Summary, Acknowledgements — tracked in [`document_check.md`](document_check.md).

---

## Phase E (synthesis) — deferred by design
The cross-tradition synthesis and the released benchmark are designed **once results
are in**; **semantics (T3) will be central there.** We do not specify it in advance
(see `experiment_design.md` §10 for the placeholder).

## Journal — *Nature Machine Intelligence* vs *Nature*
**Primary: NMI.** It explicitly covers interpretability/XAI, the AI↔neuroscience
interface, benchmarks, and methodological critique; Jonas & Kording was itself a
methods-critique and the successor lives at NMI. The "discrepancy + directions" framing
is NMI's **Analysis** type (or a regular Article). **Nature is justified** only if the
discrepancy is dramatic and broadly framed ("widely used interpretability methods fail
a ground-truth test"). Because **Nature and NMI share the template**, write to Nature's
bar and decide after the pilots at near-zero switching cost. Lean **Article**.

## Open storyline questions
- **Headline framing:** lead with the *benchmark* (Article) or the *critique*
  (Analysis)? Decide after pilots.
- **Phase D placement:** full fourth pillar, or a focused capstone? (Depends on a
  cognitive-science co-author.)
- **Granularity vs Kording:** architectural level leads; the Visual6502 transistor
  netlist is a parallel head-to-head we may feature or cut (an experiment-design
  decision — see that doc).
