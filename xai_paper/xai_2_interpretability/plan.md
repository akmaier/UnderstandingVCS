# P2 — Paper plan (storyline & Nature structure)

> **Role of this file:** the *storyline* + the Nature-Portfolio structure — what we
> write the paper from. Experiments, oracle, T3, scoring tables, compute → `experiment_
> design.md`. The program map + shared invariants → [`../general_paper_plan.md`](../general_paper_plan.md);
> doc rules → [`../README.md`](../README.md).

## P2 in one line
*Explainable AI does not understand a thing.* On the fully-known, bit-exact,
differentiable Atari VCS — where the true cause of any output is computable by exact
intervention — we score the interpretability toolkit and find: **faithfulness is not the
dividing line.** Every method only tells us "wiggle X, Y moves": causal/mechanical
relations, engineering-useful, but not semantics. None of them — not even on the fully
differentiable VCS — recover "missile", "collision", "game restarts". Meaning is not
recoverable from the mechanism at all; the reference for it is **behaviour**, which for
software lives in the documentation and data (and in our study, that is exactly what
T3/OCAtari/AtariARI is — imported, never discovered). The **semantic gap** is universal:
interpretability recovers mechanism; meaning requires an external behavioural reference.

## Metadata
- **Working title:** *Faithful attribution recovers the causal wiring, not the semantics:
  a ground-truth audit of interpretability on a fully known machine.*
  (Declarative-claim register, Maier/Nature-family; the title states the *result*, not the
  artifact. Toward the end we update this to reference the IEEE definition of software as
  code, documentation, and data.)
- **Authors (provisional):** A. Maier, S. Bayer, P. Krauss (+ collaborators).
- **Target:** *Nature Machine Intelligence* (primary), written to *Nature*'s bar
  (shared template); decide after pilots. Article (or Analysis).

## The central question (the spine of the paper)
**Does explainable AI actually understand anything?** The field optimizes *faithfulness*
(does the explanation name the true causes?) as if it were the target, because on real
neural systems there is no ground truth against which anything stronger can be checked.
We remove that excuse and ask, on a system whose ground truth is exactly known, whether
faithfulness is *sufficient* for understanding — and we find that it is **not**:
faithfulness is not the dividing line. Even if every method were faithful — and we
prove this by turning on the bilinear sampler so gradient methods become faithful too —
we still do not understand the machine. Meaning is not recoverable from the mechanism;
the reference for it is behaviour, which for software lives in documentation and data.

## Thesis (one paragraph)
Interpretability research has converged on *faithfulness* as its quality criterion, and on
real systems faithfulness cannot even be checked (Jonas & Kording's "fix-the-radio"). We
remove that excuse: the differentiable, bit-exact VCS port (Paper 1) is software with a
known specification, so the true cause of any output is recoverable by exact intervention.
Turning ~31 interpretability methods — a neuroscience battery, the attribution/XAI toolkit,
and mechanistic interpretability — onto this one machine, we first confirm that causal/
intervention methods are faithful while popular gradient and correlational saliency methods
are plausible-but-wrong. **But faithfulness is not the dividing line.** We then turn the
bilinear sampler on — Paper 1's construction that interpolates the discrete index boundary
to manufacture a usable position gradient — so that gradient methods now become faithful
too. The result is uniform: every method, whether faithful or not, delivers the same kind
of thing: "wiggle X, Y moves." Causal/mechanical relations, engineering-useful, but no
semantics. None recover "missile", "collision", "game restarts". None deliver an accurate
and interpretable model of the actual processing in the microcontroller. No objects, no
missiles, no collisions, no description of behaviour at all. The semantic gap is universal
— it is not a methods-underperformance problem. Meaning is not recoverable from the
mechanism at all; the reference for it is **behaviour**, which for software lives in
documentation and data (in our study, that is exactly what T3/OCAtari/AtariARI is —
imported, never discovered). We name this residual the **semantic gap**. Neural nets, ROMs,
differentiable Ataris and XAI — none deliver the tools to open the black box yet. None are
even close to delivering an accurate and interpretable model of the actual processing in the
microcontroller. Explainable AI does not understand a thing.

## What we claim (the honesty contract — do not drift)
1. **We give an operational, measurable definition** of where faithfulness stops short of
   understanding: `right = F ∧ S ∧ M` (faithful ∧ sufficient ∧ minimal at the algorithmic
   level).
2. **Faithfulness is not the dividing line.** Even when we make gradient methods faithful
   (via the bilinear sampler), every method produces causal/mechanical relations — an effect
   map, a data-flow graph — and nothing more. None yield semantics.
3. **The gap is universal and fundamental.** The fully differentiable VCS exposes every
   computation yet the semantics remain inaccessible from mechanism alone. Meaning is not
   recoverable from the mechanism; the reference for it is behaviour, which for software
   lives in documentation and data. IEEE-610.12: **software = code + documentation + data**.
   Interpretability methods recover code-level mechanism; the documentation and data (the
   semantics) are a separate object.

**What we do NOT claim:** that any method recovered the program's meaning, algorithm, or
design *from scratch*. **All T3 semantic labels are imported externally** (AtariARI, OCAtari)
and only *causally verified*; methods *align to / confirm* given concepts, they do not
*discover* meaning. The semantic gap is universal because even verification against ground
truth does not yield the variable's meaning — that lives in documentation and behavioural
analysis. Behavioral grounding (Paper 3) and semantic/design recovery (Paper 4) are companion
papers.

## Subject & scope
The subject is the **Atari VCS itself** — chip, program, game logic — studied with the
interpretability toolkit and scored against the substrate's ground truth. **No learned
agents** (that is P5). Ground truth is tiered (T1 causal exact · T2 hardware exact · T3
game-concept partial); "interpretable" is operationalized as **ground-truth recovery**.

## The storyline (the throughline that must hold)
1. **The gap.** All interpretability methods *find structure*, but on real systems there is
   **no ground truth**, so none can be *validated* — Jonas & Kording's warning. The field
   has made *faithfulness* the proxy for *understanding* and cannot test whether the proxy
   holds.
2. **The opportunity.** Paper 1 built a complex system that is fully known, bit-exact, and
   differentiable, so the *true* causal structure of any output is computable.
3. **The move (half 1 — confirm with naive gradients).** Score every method against ground
   truth. Result: faithful = causal; naive-gradient saliency is plausible-but-wrong (provably
   zero on discrete position outputs: Prop.~zero). The danger zone is quantified.
4. **The deeper move (half 2 — the universal gap).** Now turn on the bilinear sampler —
   Paper 1's construction that interpolates the index boundary to manufacture a usable
   position gradient — so gradient methods become faithful too. The result is uniform: every
   method gives causal/mechanical relations only. None deliver semantics. **Faithfulness is
   not the dividing line.** The gap is not about methods underperforming; it is about what
   kind of thing mechanism can deliver. Meaning is not recoverable from the mechanism at all;
   the reference for it is behaviour, found in documentation and data.
5. **The payoff.** Name and measure the semantic gap. IEEE-610.12: software = code +
   documentation + data. Interpretability methods recover code-level mechanism; the
   documentation and data (the semantics) are a separate object — unreachable from mechanism
   alone. None of this is even close to delivering an accurate and interpretable model of the
   actual processing in the microcontroller. Explainable AI does not understand a thing.

## Why a 1977 chip is a fair test (the representativeness argument)
The obvious objection: *"you study deterministic hand-coded software to judge methods built
for learned, distributed, stochastic neural nets — why does a score here predict anything
about a transformer?"* Our answer:
- **It is a *necessary-condition screen*, not a sufficiency predictor.** A method that
  fails *here* — perfect ground truth, full observability, exact interventions, **no
  learning** — has no business being trusted on a neural network. The same logic upgrades
  the semantic-gap claim: if even the *strongest* causal account a fully-observable machine
  permits stops short of meaning, then on a neural net — less observable, no clean ground
  truth — the gap is *a fortiori* wider.
- **The VCS *does* exhibit the failure modes that make interpretation hard:** aliased /
  polysemantic RAM cells reused across routines, race-the-beam timing, floating-bus reads,
  registers with context-dependent meaning.
- **What it does *not* capture** (learned features, scale, stochasticity) we state plainly
  and defer to P5.

## Contributions
1. A **ground-truth interpretability benchmark** on one fully-known, differentiable, *real*
   complex system. Headline: faithful = causal; naive-gradient saliency is plausible-but-
   unfaithful (provably zero on discrete position).
2. **The semantic gap, made universal** — the paper's distinctive contribution. Faithfulness
   is not the dividing line: even with the bilinear sampler making gradient methods faithful,
   every method recovers mechanism only, not semantics. Meaning is not recoverable from
   mechanism; the reference is behaviour, in documentation and data. IEEE-610.12: software =
   code + documentation + data. We **define and measure** the gap; we do **not** claim to
   close it.
3. A **quantitative re-run of Jonas & Kording** (Phase A).
4. A **faithfulness audit of attribution/XAI** (Phase B), including the finding that several
   popular methods (Grad-CAM, attention) do not even *apply* to a non-neural system.
5. **Mechanistic interpretability validated on a *known* circuit** (Phase C).
6. **The key result: explainable AI does not understand a thing.** Neural nets, ROMs,
   differentiable Ataris and XAI — none deliver the tools to open the black box. None are
   close to delivering an accurate and interpretable model of the actual processing.

## Novelty / related work
Prior work scores interpretability against ground truth — on **synthetic or compiled** systems:
Tracr, InterpBench, BIM. **Our delta:** (i) we score against a *real, deployed, hand-authored*
artifact whose ground truth is independent of the artifact; (ii) we use exact ground truth to
show that **faithfulness is not the dividing line** — even perfectly faithful methods deliver
only mechanism, not semantics. That separation is the move none of the construction-based
benchmarks can make, because the planted circuit *is* the meaning by fiat; on a real artifact
the meaning is a separate object.

---

# Paper structure (Nature Portfolio)

### Abstract (~150–200 words)
We ask whether explainable AI understands anything. Interpretability methods are judged by
*faithfulness* — whether their claimed causes are the true causes — yet on real systems there
is no ground truth against which to judge them. We turn the toolkit on a system where the
ground truth is computable: a bit-exact, differentiable port of the Atari 2600, whose every
causal dependency we obtain from an exhaustive intervention oracle. We score ~31 methods
across neuroscience, attribution, and mechanistic-interpretability traditions. Causal and
intervention methods are faithful while naive-gradient saliency is not — provably zero on
discrete position outputs. **Yet faithfulness is not the dividing line.** We turn on the
bilinear sampler — Paper 1's interpolation construction — so gradient methods become faithful
too. The result is uniform: every method delivers "wiggle X, Y moves": causal/mechanical
relations, not semantics. None recover "missile", "collision", or "game restarts". Meaning
is not recoverable from the mechanism at all; the reference for it is behaviour, which for
software lives in documentation and data (IEEE-610.12). We measure this residual and name it
the semantic gap. Explainable AI does not understand a thing.

### Introduction
Open on the central question: does explainable AI actually understand anything? The field
tension: explanations are load-bearing for trust and safety, yet on real systems there is
no ground truth to validate them against. The field has made faithfulness the proxy for
understanding. Land the hook (Paper 1's exact oracle). Numbered roadmap: (1) audit
faithfulness against ground truth; (2) turn on the bilinear sampler so gradient methods
become faithful too — and show the gap is universal; (3) argue from IEEE-610.12 that meaning
is a separate object from mechanism, unreachable from causal attribution alone.
Representativeness/necessary-condition framing up front.

### Results
- **A — neuroscience battery.** Kording battery scored: classical methods find rich structure
  yet score low against the known mechanism.
- **B — attribution / XAI (naive gradients).** Popular saliency plausible-but-unfaithful;
  causal methods faithful. Vanilla_saliency = 0.000 on position regime. The danger zone.
- **C — mechanistic interpretability.** First ground-truth calibration on a real artifact.
- **Cross-tradition comparison + the danger zone.** Headline figure: faithfulness vs
  plausibility.
- **The semantic gap — the universal result.** Turn the bilinear sampler on: gradient
  methods now become faithful too. The result is uniform across all methods: every method
  delivers causal/mechanical relations, no semantics. Three demonstrations:
  (i) **graph-correct but behavior-incomplete** — ACDC F=1.0/M=1.0, S=0.44;
  (ii) **named but causally inert** — SAE matched=1.0, F=0.041, S=−0.336;
  (iii) **decodable but unused** — linear probing flags cell 84 as not causally used.
  (iv) **bilinear sampler makes gradient methods faithful — still no semantics.**
  The conclusion: faithfulness is not the dividing line. The gap is universal. No objects,
  no missiles, no collisions, no description of behaviour at all. Explainable AI does not
  understand a thing.

### Discussion
- **Confirm, then mark the boundary.** Faithfulness is not the dividing line. Even with the
  bilinear sampler making every method faithful, none deliver semantics.
- **IEEE-610.12: software = code + documentation + data.** Interpretability recovers code-
  level mechanism. Documentation and data (the semantics) are a separate object —
  unreachable from mechanism alone. The fully differentiable VCS cannot bridge this gap.
- **What understanding requires.** Behaviour as reference. Marr's algorithmic level. The
  reverse-engineering bar. Paper 3 (behavioral grounding), Paper 4 (design recovery).
- **Representativeness** (necessary-condition screen; *a fortiori* upgrade for the gap).
- **The headline:** Neural nets, ROMs, differentiable Ataris and XAI — none deliver the
  tools to open the black box yet. None are even close to delivering an accurate and
  interpretable model of the actual processing in the microcontroller. Explainable AI does
  not understand a thing.

### Methods
Emulator, ground-truth oracle, F∧S∧M triad, metrics, bilinear sampler construction.

### Display items (~6 figures)
(1) VCS platform & oracle; (2) faithfulness-vs-plausibility danger zone; (3) semantic gap
with bilinear-sampler result; (4) Kording battery; (5) VCS↔NN failure-mode map;
(6) IEEE-610.12: code vs documentation vs data — what interpretability recovers and what
it cannot reach.

### End matter
Data/Code availability, Author contributions, Competing interests, Reporting Summary,
Acknowledgements — tracked in `document_check.md`.

## Journal — NMI vs Nature
**Primary: NMI.** The claim "explainable AI does not understand a thing" is deliberately
provocative and broadly framed — Nature if the editors agree, NMI otherwise.
