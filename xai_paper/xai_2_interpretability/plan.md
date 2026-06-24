# P2 — Paper plan (storyline & Nature structure)

> **Role of this file:** the *storyline* + the Nature-Portfolio structure — what we
> write the paper from. Experiments, oracle, T3, scoring tables, compute → `experiment_
> design.md`. The program map + shared invariants → [`../general_paper_plan.md`](../general_paper_plan.md);
> doc rules → [`../README.md`](../README.md).

## P2 in one line
*A faithful explanation is not yet an understanding.* On the fully-known, bit-exact,
differentiable Atari VCS — where the true cause of any output is computable by exact
intervention — we score the interpretability toolkit against a ground-truth oracle.
Even with perfect faithfulness (including experiments where we use the oracle's own
sampler as an attribution method), **none of the methods deliver a principled way of
understanding the machine**: we wiggle something and something else moves. Engineering-wise
interesting, but there are no real semantics like "missile", "collision", "game restarts".
The fully differentiable VCS itself **cannot bridge this gap** — what we need as a reference
is **behaviour**, found in the associated documentation and data. We name this residual
the **semantic gap** and argue it — not faithfulness — is the open problem for XAI.
**Phases A (neuroscience battery) + B (attribution/XAI) + C (mechanistic interpretability).**

## Metadata
- **Working title:** *Faithful attribution recovers the causal wiring, not the semantics:
  a ground-truth audit of interpretability on a fully known machine.*
  (Declarative-claim register, Maier/Nature-family; the title states the *result*, not the
  artifact.)
- **Authors (provisional):** A. Maier, S. Bayer, P. Krauss (+ collaborators).
- **Target:** *Nature Machine Intelligence* (primary), written to *Nature*'s bar
  (shared template); decide after pilots. Article (or Analysis).

## The central question (the spine of the paper)
**Do our interpretability methods explain the *semantics* of a system — do we actually
*understand* what it is doing — or do they, even when perfectly faithful, recover only the
causal wiring and stop short of the meaning?** The field optimizes *faithfulness* (does the
explanation name the true causes?) as if it were the target, because on real neural systems
there is no ground truth against which anything stronger can be checked. We remove that
excuse and ask, on a system whose ground truth is exactly known, whether faithfulness is
*sufficient* for understanding — and we find that it is **necessary but not**.

## Thesis (one paragraph)
Interpretability research has converged on *faithfulness* as its quality criterion, and on
real systems faithfulness cannot even be checked, because there is no ground truth (Jonas &
Kording's "fix-the-radio"). We remove that excuse: the differentiable, bit-exact VCS port
(Paper 1) is software with a known specification, so the true cause of any output is
recoverable by exact intervention. Turning ~31 interpretability methods — a neuroscience
battery, the attribution/XAI toolkit, and mechanistic interpretability — onto this one
machine, we confirm causal/intervention methods are faithful while popular gradient and
correlational saliency methods are plausible-but-wrong. **We further push the experiment to
its logical extreme: even when an attribution method uses the oracle's own sampler
(the ground-truth intervention engine) as its attribution mechanism — achieving the highest
possible faithfulness — it still does not deliver understanding.** We wiggle something and
something else moves; we recover a causal-effect map, not the variable's *meaning* or the
routine's *algorithm*. Our correctness triad makes this precise — an explanation is *right*
only if it is **F**aithful ∧ **S**ufficient (it regenerates behavior) ∧ **M**inimal (at the
algorithmic level) — and the committed data show circuits that are graph-correct (F=1.0,
M=1.0) yet do not reproduce behavior (S=0.44), feature dictionaries that match a named
variable perfectly (matched fraction 1.0) while being causally unfaithful (F=0.04), and
concepts that are linearly decodable yet provably unused. The gap between a faithful
attribution and an understanding of the computation is therefore not philosophy; on this
machine it is a number. We name it the **semantic gap**. The gap persists even with the
fully differentiable VCS: the substrate alone cannot deliver semantics — what we need as a
reference is **behaviour**, found in documentation and data. Paper 2 *defines and measures*
the gap; *closing* it — recovering the program's documentation and design from scratch — is
the companion Papers 3/4.

## What we claim about semantics / understanding (the honesty contract — do not drift)
We claim exactly three things, and no more:
1. **We give an operational, measurable definition** of where faithfulness stops short of
   understanding: `right = F ∧ S ∧ M` (faithful ∧ sufficient — regenerates behavior /
   predicts held-out interventions ∧ minimal at the algorithmic level — RAM/registers/
   opcodes, the program's variables/circuits). The S and M axes turn "do we understand?"
   into scores.
2. **Faithful methods can satisfy F (and M) while failing S** — they recover correct causal
   structure (an effect map, a data-flow graph) that does **not** amount to the computed
   function: even the oracle sampler as attribution (the strongest possible faithful method)
   delivers an effect table, not the bounce algorithm. The missing object is of a *different
   type*, not a worse-fit instance of the same type.
3. **The gap is not a substrate limitation** — the fully differentiable VCS exposes every
   computation yet the semantics remain inaccessible from causal attribution alone.
   Understanding requires **behaviour as reference**, not just causal structure.

**What we do NOT claim:** that any method recovered the program's meaning, algorithm,
variables, or design *from scratch*. **All T3 semantic labels are imported externally**
(AtariARI, OCAtari) and only *causally verified*; methods *align to / confirm* given
concepts, they do not *discover* meaning. The semantic gap persists because even verification
against ground truth does not yield the variable's meaning — that lives in documentation and
behavioural analysis. Behavioral grounding (Paper 3) and semantic/design recovery (Paper 4)
are companion papers. The plausibility (Y) axis is a documented method-tradition **proxy**,
not a human-subjects measurement.

## Subject & scope
The subject is the **Atari VCS itself** — chip, program, game logic — studied with the
interpretability toolkit and scored against the substrate's ground truth. **No learned
agents** (that is P5). Ground truth is tiered (T1 causal exact · T2 hardware exact · T3
game-concept partial); "interpretable" is operationalized as **ground-truth recovery**.

## The storyline (the throughline that must hold)
1. **The gap.** Attribution/XAI, mechanistic interpretability, and neuroscience methods all
   *find structure*, but on real systems there is **no ground truth**, so none can be
   *validated* — Jonas & Kording's warning, unanswerable in the wild. The field has quietly
   made *faithfulness* the proxy for *understanding* and cannot test whether the proxy holds.
2. **The opportunity.** Paper 1 built a complex system that is fully known, bit-exact, and
   differentiable, so the *true* causal structure of any output is computable.
3. **The move (half 1 — confirm).** Turn the toolkit on this one system and **score every
   method against ground truth**. Result: faithful = causal; popular saliency is
   plausible-but-wrong; the danger zone is quantified.
4. **The deeper move (half 2 — the new contribution).** Push to the extreme: run the oracle
   sampler itself as an attribution method — the highest possible faithfulness. Even this
   does not yield understanding: we get an effect map, not the algorithm or the variable's
   meaning. **Faithfulness is necessary, not sufficient.** The fully differentiable VCS
   itself cannot bridge this gap: the substrate alone does not carry semantics. What we need
   as a reference is **behaviour**, found in the documentation and data. Name and measure
   the **semantic gap**.
5. **The payoff.** A reusable ground-truth benchmark *and* the first measurable definition
   of the residual that interpretability must close to claim *understanding* — with the
   constructive companion work (P3 behavioral, P4 recovery) explicitly deferred.

## Why a 1977 chip is a fair test (the representativeness argument — front and centre)
The obvious objection: *"you study deterministic hand-coded software to judge methods built
for learned, distributed, stochastic neural nets — why does a score here predict anything
about a transformer?"* Our answer, stated explicitly in the Introduction and Discussion:
- **It is a *necessary-condition screen*, not a sufficiency predictor.** A method that
  fails *here* — perfect ground truth, full observability, exact interventions, **no
  learning** — has no business being trusted on a neural network. Passing here is
  necessary, not sufficient. **The same logic upgrades the semantic-gap claim:** if even the
  *strongest* causal account a fully-observable machine permits stops short of meaning, then
  on a neural net — less observable, no clean ground truth — faithfulness is *a fortiori*
  not understanding. Failing here is decisive; the gap here is a floor on the gap there.
- **The VCS *does* exhibit the failure modes that make interpretation hard:** aliased /
  polysemantic RAM cells reused across routines (superposition-like), race-the-beam timing
  (distributed temporal computation), floating-bus reads, and registers whose meaning is
  context-dependent. We map each VCS property to the NN difficulty it does / does not
  capture (a Discussion display item).
- **What it does *not* capture** (learned features, scale, stochasticity) we state plainly
  and defer to P5 (learned agents).

## Contributions (vs Paper 1 and vs Jonas & Kording)
1. A **ground-truth interpretability benchmark** on one fully-known, differentiable, *real*
   complex system — with the necessary-condition-screen framing for using it. Headline:
   **faithful = causal**; popular gradient/correlational saliency is plausible-but-unfaithful
   (provably zero on discrete position outputs); the faithfulness-vs-plausibility **danger
   zone** is quantified.
2. **The semantic gap, made measurable** — the paper's distinctive contribution. An
   operational definition (`right = F ∧ S ∧ M`) under which even a *perfectly faithful*
   method — including the oracle's own sampler — is shown to recover the *causal wiring*
   and not the *semantics*. We **define and measure** the gap; we do **not** claim to close it.
3. A **quantitative re-run of Jonas & Kording** (Phase A) — the neuroscience battery scored
   against the known mechanism (low-faithfulness calibration baseline).
4. A **faithfulness audit of attribution/XAI** (Phase B) on the VCS's own computation,
   including the finding that several popular methods (Grad-CAM, attention) do not even
   *apply* to a non-neural system.
5. **Mechanistic interpretability validated on a *known* circuit** (Phase C) — the first
   ground-truth calibration of patching/SAEs/circuits *on a real artifact* — *and* the
   demonstration that circuit recovery yields a graph, not the computed function.
6. A **formal, operationalized notion of interpretability** = ground-truth recovery,
   with **understanding** explicitly distinguished from **faithful attribution** at the
   semantic level.
7. **The key insight: the gap is not a substrate limitation.** The fully differentiable VCS
   exposes every computation yet the semantics remain inaccessible from causal attribution
   alone. Understanding requires **behaviour as reference** — found in documentation and
   data — pointing to companion papers P3 (behavioral grounding) and P4 (design/documentation
   recovery).

## Novelty / related work (must be argued, not assumed)
Prior work *does* score interpretability against ground truth — but on **synthetic or
compiled** systems (citations **verified**):
- **Tracr** — Lindner et al., *NeurIPS* 2023, arXiv:2301.05062 — human-readable programs
  *compiled* into transformers with circuits **by construction**, used as interpretability
  ground truth.
- **InterpBench** — Gupta et al., *NeurIPS* 2024 (Datasets & Benchmarks), arXiv:2407.14494
  — *semi-synthetic* transformers with **known circuits** (trained via Strict IIT, built
  on Tracr) for evaluating mechanistic-interpretability methods.
- **BIM** — Yang & Kim, 2019, arXiv:1907.09701 — ground-truth **attribution** benchmark
  with known important features + false-positive metrics.

**Our delta (state explicitly; do not claim an unqualified "first"):** (i) we score against
a *real, deployed, hand-authored, complex* artifact whose ground truth is **independent of
the artifact** — Tracr/InterpBench build the ground truth *into* the model; we *recover* it
from a system that was never designed to be interpretable; (ii) more deeply, prior
ground-truth benchmarks validate *faithfulness* (did the method find the planted circuit?);
**we use exact ground truth to separate faithfulness from understanding** — to show that
even a method that perfectly recovers the true causal structure has not thereby recovered
the semantics. That separation is the move none of the construction-based benchmarks can
make, because the planted circuit *is* the meaning by fiat; on a real artifact the meaning
is a separate object, and the gap to it is measurable.

## The single subject — the VCS (framing)
`output(t) = VCS(ROM, inputs[0..t], initial_state)`, three observable/intervenable faces:
**inputs** (ROM, joystick, switches — *we* drive them), **internal state** (RAM, CPU/TIA/
RIOT registers, opcodes — the "activations"), **outputs** (pixel, score, event — the
"predictions to explain"). One subject carries A, B, and C.

---

# Paper structure (Nature Portfolio)

### Abstract (~150–200 words) — draft sketch
We ask whether a faithful explanation is an understanding. Interpretability methods are
judged by *faithfulness* — whether their claimed causes are the true causes — yet on real
systems there is no ground truth against which to judge them. We turn the toolkit on a
system where the ground truth is computable: a bit-exact, differentiable port of the Atari
2600, real and never designed to be interpretable, whose every causal dependency we obtain
from an exhaustive intervention oracle. We score ~31 methods across neuroscience,
attribution, and mechanistic-interpretability traditions against this oracle. We confirm
that causal and intervention methods are faithful while popular gradient and correlational
saliency methods are not — the latter provably zero on discrete position outputs, while
scoring *higher* on human plausibility. **Yet faithfulness is not understanding.** Even the
oracle's own intervention sampler — the strongest possible faithful attribution — recovers
a causal-effect map, not the computation: graph-correct circuits need not reproduce
behavior, features matching named concepts can be causally inert, and the gap persists
because **the fully differentiable substrate does not carry semantics**. Understanding
requires behaviour as reference, found in documentation and data. We measure this residual
and name it the semantic gap. This analysis calibrates which interpretability claims can be
trusted and redirects the field toward recovering meaning — not merely confirming causes.

### Introduction
The arc (gap → opportunity → confirm → the deeper move → payoff). **Open on the central
question explicitly:** when a method tells us what a system is doing, do we believe it
because it is *true* or because it is *plausible* — and where does a correct attribution
stop short of *understanding* the program's semantics? Lead on the field tension (after
Greydanus): explanations are load-bearing for trust and safety, yet on real systems there
is *no ground truth to validate them against*; the field has made faithfulness the proxy for
understanding without being able to test the proxy. Land the hook (Paper 1's exact oracle),
then the *"In this Article, we…"* numbered roadmap that flags **both** halves: (1) we audit
faithfulness against exact ground truth; (2) **we show, with the same oracle, that
faithfulness is necessary but not sufficient for understanding, and we make the residual —
the semantic gap — a measured quantity.** Put the representativeness/necessary-condition
framing up front. Cite Jonas & Kording, the XAI/mech-interp literature, the prior
ground-truth benchmarks (§novelty), and Paper 1. Operationalize "interpretable" as
ground-truth recovery and distinguish **faithful attribution** from **understanding** at the
semantic level here, not only in the Discussion.

### Results — one subsection per phase (the *what*; the *how* is in `experiment_design.md`)
- **A — neuroscience battery.** The Kording battery scored: classical methods find rich
  structure yet score low against the known mechanism, *quantifying* his lesson and setting
  the low-faithfulness baseline.
- **B — attribution / XAI.** Attribution of a VCS output to its causes scored: popular
  saliency plausible-but-unfaithful; causal/gradient methods faithful; several popular
  methods don't even apply (NN-specific). The head-to-head: activation_patching = 1.000 vs
  vanilla_saliency = 0.000 on all six games, the *popular* method carrying the *higher*
  plausibility proxy.
- **C — mechanistic interpretability.** The mech-interp toolkit on a *known* circuit: the
  first ground-truth calibration on a real artifact — what patching/SAEs/circuits recover,
  where they break.
- **Cross-tradition comparison + the danger zone (the faithfulness leaderboard).** The
  headline figure placing A–C on the shared faithfulness-vs-plausibility axes; the quantified
  danger zone (high plausibility, low faithfulness). This is the *confirm* half.
- **The semantic gap (the new result — its own subsection, the paper's centre of gravity).**
  Make "do we understand the semantics?" the *measured* question here. Three committed
  separations, each adjudicated by the oracle, not by our preference:
  (i) **graph-correct but behavior-incomplete** — ACDC-breakout F=1.0 / M=1.0 yet S=0.44; a
  correct data-flow graph is not the computed function; activation_patching's exact effect
  table (`recovered_equals_exact: true`) is a table, not the bounce algorithm — and there is
  no tuning budget left, so the missing object is of a *different type*;
  (ii) **named but causally inert** — SAE matched-fraction=1.0 with F=0.041 / S=−0.336;
  "looks like a known variable" ≠ "is how the program computes";
  (iii) **decodable but unused** — linear probing flags score-cell 84 as not-causally-used
  (the "present ≠ used" / control-task trap; the oracle is strictly stronger than probing).
  **NEW: (iv) perfect faithfulness still not understanding** — the oracle sampler as
  attribution delivers the highest possible faithfulness yet produces an effect table, not
  the algorithm or semantics. The gap persists because the substrate alone cannot carry
  meaning.
  State plainly that the gap **runs in both directions at once** (perfectly matched yet
  causally inert), which a "methods underperform" story cannot produce, and that all
  semantic labels are **external** (AtariARI/OCAtari, `verified:false` provenance) and only
  *verified*, not *discovered*.

### Discussion
- **Confirm, then mark the boundary.** Many traditions, one ground truth (A–C validated on a
  complex system); Kording's lesson made measurable; mech-interp calibrated on a known
  circuit; **and the boundary the leaderboard cannot cross.**
- **What *understanding* would require — the semantic level (the heart of the Discussion).**
  Faithfulness is the easy half; the hard half is *meaning and algorithm*. Tie the triad to
  the classics: Marr's levels (understanding = the algorithmic level, not a faithful saliency
  map), Lazebnik/"fix-the-radio." Make the IEEE-610.12 connection explicit: **software =
  programs *plus* documentation**; recovering a faithful circuit recovers (part of) the
  program; *understanding* the system means recovering its **documentation/design** — the
  variables' meaning, the routines' intent, the specification — which our causal oracle, **by
  construction, cannot reach**, because the meaning is a separate object from the causal
  structure. The fully differentiable VCS itself **cannot bridge this gap**: the substrate
  does not carry semantics. What we need as a reference is behaviour, found in documentation
  and data. That is the reverse-engineering bar, and it is **Paper 3/4**. Behavioral
  grounding — the semantic-grounding bridge — is **Paper 3**.
- **The shared toolkit** (tuning curves ↔ feature viz; lesions ↔ ablations; connectomics ↔
  circuits; probing ↔ dim-reduction; Granger ↔ causal attribution).
- **Representativeness** (the necessary-condition screen; the VCS↔NN failure-mode map; the
  *a fortiori* upgrade for the semantic-gap claim).
- **What interpretability needs** — causal over correlational; ground-truth-validated
  benchmarking; sieve methods on known systems first; **and: stop treating faithfulness as
  understanding — the semantic gap, not faithfulness, is the open problem.**
- **Limitations, conceded then bounded** (own-sentence, Maier register). We *measure* the
  gap; we do not *close* it. We did **not** run semantic recovery (Phase E is specified, not
  tested) — so our claim is the *bounded* one (on exact ground truth, faithfulness certifies
  the wiring and the wiring alone is not the semantics), **not** the unbounded "no method can
  recover meaning." The S/M axes are oracle-defined causal quantities and the T3 labels are
  external — we say so, and turn it into the point: even the strongest causal account stops
  short of meaning, and recovering-meaning-from-scratch is a different problem. A single low
  S could be a tuning artifact — so the claim rests not on any one number but on the gap
  appearing where tuning *cannot* reach (exact effect table) and running in *both directions
  at once*. Plausibility is a proxy, not a measurement. No learning, one old machine — the
  necessary-condition framing, stated again.
- **Beyond the brain:** differentiability tells us which *gradient* explanations are
  trustworthy (Krauss co-author fit). **Toward P5:** the learned agent is the hard case.

### Methods
Summarize the emulator, the ground-truth oracle, the F∧S∧M triad, and the metrics; full
setups in `experiment_design.md`.

### Display items (~6 figures)
(1) the VCS platform & ground-truth oracle (the three faces; T1/T2 exact, T3 external);
(2) A–C scored on the shared faithfulness-vs-plausibility axes with the **danger zone**
(headline, *confirm* half); (3) **the semantic gap** — the F∧S∧M triad with the four
counted separations (ACDC S=0.44; SAE matched=1.0 vs F=0.04; decodable-but-unused cell 84;
oracle sampler as attribution: perfect F yet no semantics) (the *new-result* headline);
(4) Kording battery scored / attribution vs mechanistic-interp faithfulness; (5) the
VCS↔NN failure-mode / representativeness map; (6) what to do instead + the road to closing
the gap (P3 grounding, P4 design recovery — the IEEE-software bar).

### End matter
Data/Code availability, Author contributions, Competing interests, Reporting Summary,
Acknowledgements — tracked in [`document_check.md`](document_check.md).

## Journal — NMI vs Nature
**Primary: NMI** (interpretability/XAI, AI↔neuroscience, benchmarks, methods-critique; the
"discrepancy + directions" framing is NMI's Analysis type or a regular Article). **Nature**
only if the discrepancy is dramatic and broadly framed — and "faithful ≠ understanding,
shown with numbers on a fully known machine" is exactly that kind of crisp, broadly framed
claim; same template, so shoot-for-Nature-first at near-zero switching cost. Lean
**Article**. Decide after pilots. The single falsifiable headline (a method can be perfectly
faithful and still not constitute an understanding, and the gap is measurable) is restated at
all four levels (abstract, intro, discussion, conclusion), Maier/NMI style.
