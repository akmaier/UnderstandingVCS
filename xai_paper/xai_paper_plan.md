# Paper 2 — Plan: a fully-known, differentiable model organism for explainable AI

**Working title:** *Scoring Explainable AI against ground truth: a differentiable Atari VCS as a model organism for interpretability.*

**One-line thesis.** Jonas & Kording (2017) showed that neuroscience analysis
methods, applied to a fully-known microprocessor, *find structure but do not
yield understanding* — yet they could only judge this qualitatively. We turn
that thought experiment into a **quantitative, falsifiable benchmark**: our
Atari VCS port (Paper 1) is not only fully specified but **bit-exact and
end-to-end differentiable**, so for the first time we can compute the *true*
causal attribution for any output and **score** each interpretability method
(neuroscience-style *and* modern XAI) against it. We replicate the Kording
battery, extend it to today's deep-RL XAI toolkit on real DQN agents, measure
where popular methods diverge from ground truth, and from that argue what new
directions interpretability actually needs.

> **Status: PLAN.** Template fetched into `paper/` (Springer Nature, `sn-nature`
> style — same base for *Nature* and *Nature Machine Intelligence*). Journal
> recommendation in §7. Discuss before we start building experiments.

---

## 1. What Paper 1 gives us (the platform)

The differentiable VCS (jutari + jaxtari, 64/64 bit-exact vs xitari) already
exposes exactly the machinery this study needs:

- **Full ground truth.** Every architectural state variable has a *known
  semantic role*: RAM bytes (RIOT 128 B), CPU registers (A, X, Y, S, P, PC),
  TIA/RIOT registers, opcodes, ROM bytes, the framebuffer. We know the true
  data-flow (which instruction reads/writes which cell) and the true I/O map.
- **Exact interventions.** We can knock out / clamp / perturb any state variable
  or ROM byte and re-run deterministically (the "lesion" knob, but clean).
- **Differentiability.** `peek` one-hot reads, `soft_select` dispatch,
  `soft_branch`, the straight-through estimator, and the relaxed read give us
  gradients of any output (a pixel, the score, an agent's action/value) w.r.t.
  any input (ROM byte, RAM cell, register, joystick action) — *the* ingredient
  Kording did not have.
- **Reusable tooling:** `tools/xai_si_gradient/` (real-ROM screen↔action
  gradient), the IG-on-ROM experiment (P8), the conformance harness, the
  comparison-video renderer.

This is the crucial upgrade over Jonas & Kording, who used the Visual6502
*transistor netlist* and could only say results "felt unsatisfying." We work at
the **architectural level** (registers/RAM/opcodes) — the level at which
"understanding" is actually defined (Marr's algorithmic level) — *and* we can put
a number on every method's faithfulness.

## 2. Contributions (what's new vs Paper 1 and vs Jonas & Kording)

1. **A ground-truth interpretability benchmark.** A complex, fully-known,
   differentiable system where the *true* causal attribution is computable, so
   any interpretability method can be scored, not just eyeballed.
2. **A quantitative re-run of Jonas & Kording.** Each neuroscience method
   (connectomics, lesions, tuning curves, correlations, LFP, Granger causality,
   dimensionality reduction) reproduced on our state variables, each scored
   against the known mechanism — converting their qualitative critique into
   measured faithfulness.
3. **A faithfulness audit of modern deep-RL XAI.** Saliency, Grad-CAM/++,
   Integrated Gradients, perturbation/occlusion, attention, and counterfactual
   methods applied to DQN agents, scored against the *true* causal pixel/object
   attribution obtained by differentiating (and intervening on) the exact
   emulator∘agent pipeline.
4. **Evidence of the discrepancy + concrete new directions:** where popular
   methods fail, why, and what a faithful, ground-truth-validated interpretability
   program should look like (gradient/known-operator/causal-intervention
   attribution; ground-truth-scored XAI benchmarking).

## 3. Two subjects under the same microscope

- **(A) The emulator as "model organism"** — the Jonas & Kording subject, now
  fully known *and* differentiable. We probe the running VCS itself.
- **(B) DQN agents playing on the emulator** — the *actual* XAI use case. The
  differentiable emulator is what makes the agent's input attribution have a
  **ground truth**: the true causal effect of each pixel/object on the agent's
  action is computable by exact intervention and by end-to-end gradients through
  `emulator ∘ agent`. This is the bridge from "fully-known chip" to "real XAI."

## 4. Experiment plan

### Phase A — Replicate Jonas & Kording, with ground-truth scoring

Map each neuroscience method onto our architectural state; for each, report both
the Kording-style qualitative result **and** a quantitative agreement with the
known mechanism. Subjects: the three games they used (Donkey Kong, Space
Invaders, Pitfall) plus several of our 64 for generality.

| # | Kording method | Our analog (on VCS state) | Ground-truth score |
|---|---|---|---|
| A1 | Connectomics / netlist graph | Program/data-flow graph: which RAM cells, registers each opcode reads/writes | recovered graph vs the *true* read/write graph (precision/recall) |
| A2 | Single-transistor lesions | Clamp each RAM bit / register bit / ROM byte / opcode; does the game still boot/run? | "lesion specificity" vs the byte's *true* role (we know it) |
| A3 | Tuning curves | Toggle-rate of each RAM/register bit vs pixel luminance / game variable | spurious-tuning rate vs true functional role |
| A4 | Spike-word / pairwise correlations | Correlations across RAM/register bits | reproduce weak-pairwise/strong-global; relate to true coupling |
| A5 | Local field potentials | Gaussian-pooled toggle activity over memory/TIA regions; power spectra | show frame/scanline "rhythms" are epiphenomena (we know the clocks) |
| A6 | Granger causality | Between CPU / TIA / RIOT subsystem activity | inferred vs true data-flow (false-edge rate) |
| A7 | Dimensionality reduction (NMF/PCA) | Over full state-trajectory tensor | recovered components vs known signals (clock, R/W, vsync) |
| A8 | Whole-"brain" recording | Full RAM+register state map over time | descriptive baseline |

**Punchline of Phase A:** the same "interesting but unfaithful" structure
Kording found, now *quantified* — most methods score poorly against the known
mechanism even with unlimited, noiseless, fully-observed data.

### Phase B — Modern XAI on DQN agents, scored against true causal attribution

1. **Agents.** Obtain/representative DQN agents (Mnih 2015; the Atari Model Zoo,
   Such et al. 2019) and run them on jutari/jaxtari.
2. **Ground-truth attribution.** For a state, compute the *true* causal influence
   of each input pixel / detected object / RAM byte on the agent's chosen action
   and Q-values, via (a) **exact interventions** on the emulator (occlude/replace
   a pixel or object and measure the deterministic effect) and (b) **end-to-end
   gradients** through `emulator ∘ agent` (Paper 1's differentiability makes this
   exact-forward). Cross-validate (a) vs (b).
3. **Methods under test.** Greydanus saliency, Grad-CAM (Selvaraju) and Grad-CAM++
   (Chattopadhyay), Integrated Gradients, occlusion/perturbation, attention
   (Nikulin free-lunch), counterfactual (Atrey), and an interpretable-by-design
   baseline (XDQN, Kontogiannis). Survey scope grounded in our XRL-survey notes
   (Qing 2022, Vouros 2023, Cheng 2025, Saulières 2025).
4. **Faithfulness metrics vs ground truth.** Correlation with the true causal
   map; deletion/insertion AUC measured on the *true* emulator (not a proxy);
   pointing-game / object-hit rate against true causal objects.
5. **Result:** quantify the discrepancy — which methods track the true causes,
   which produce plausible-but-wrong maps (sharpening Atrey's "exploratory not
   explanatory" into a measured claim with real ground truth).

### Phase C — Beyond: a benchmark and better directions

- **C1 Ground-truth XAI benchmark.** Package the scoring suite as a reusable
  benchmark (tasks, true-attribution oracle, faithfulness metrics) — a
  "leaderboard with real ground truth."
- **C2 Faithful attribution methods.** Show that gradient- / known-operator- /
  causal-intervention-based attribution (native to our differentiable substrate)
  is measurably more faithful, and propose how such ideas transfer to opaque
  models that *lack* ground truth.
- **C3 Directions.** From the discrepancy, articulate what interpretability needs:
  ground-truth-validated benchmarking, causal rather than correlational
  attribution, and method development that is *sieved* on known systems first.

## 5. Discussion — XAI ↔ neuroscience, and beyond (a core section)

- **The shared toolkit.** Neuroscience and XAI use the *same* methods under
  different names: tuning curves ↔ feature visualization; lesions ↔ ablations;
  connectomics ↔ circuit/mechanistic analysis; dimensionality reduction ↔
  probing; Granger ↔ causal-attribution. The VCS is a common benchmark for both.
- **Kording's lesson, made measurable.** "Finds structure ≠ understanding" was a
  warning; with ground truth it becomes a *test* a method passes or fails.
- **Beyond the microprocessor.** From "could a neuroscientist understand a
  microprocessor?" to "can an interpretability method recover known ground
  truth?" — a falsifiable program for both fields, and a validation harness for
  the rising field of **mechanistic interpretability** of neural networks.
- **Where it goes beyond neuroscience.** Unlike a brain, our system is
  differentiable, so it also tells us which *gradient-based* explanations are
  trustworthy — directly relevant to interpreting the deep networks neuroscience
  now uses as brain models. (Co-author fit: P. Krauss bridges pattern
  recognition and neuroscience/neuroprosthetics.)

## 6. Paper structure (Nature Portfolio format)

`paper/main.tex` is wired to `\documentclass[sn-nature]{sn-jnl}` (template files
fetched). Nature-style layout:

- **Title; Abstract** (~150–200 words, unreferenced).
- **Main text** (Nature has no rigid IMRaD): Introduction → Results
  (Phase A scoring; Phase B agent-XAI audit; the discrepancy; a faithful-method
  demonstration) → Discussion (§5).
- **Main display items** (~5–6 figures): (1) the platform & ground-truth oracle;
  (2) Kording battery scored; (3) agent-XAI faithfulness vs ground truth;
  (4) discrepancy map / failure taxonomy; (5) a faithful-attribution result;
  (6) the XAI↔neuroscience method correspondence.
- **Methods** (after references, no length limit): emulator, true-attribution
  oracle, each analysis, agents, metrics.
- **Extended Data** (additional figures), **Supplementary Information** (full
  per-method results, proofs of the attribution oracle, all games).
- **End matter:** Data availability, Code availability, Author contributions,
  Competing interests, Reporting Summary, Acknowledgements (see `document_check.md`).

## 7. Journal: *Nature Machine Intelligence* vs *Nature*

**Recommendation: target *Nature Machine Intelligence* (primary).** Reasoning:

- **Scope fit.** NMI explicitly covers interpretability/XAI, the AI↔neuroscience
  interface, benchmarks, and methodological critique — this paper is squarely in
  its remit. Jonas & Kording itself was a methods-critique (PLOS Comp Biol); the
  XAI successor lives at NMI.
- **The "discrepancy + new directions" framing** you want is exactly NMI's
  **Analysis** article type (or a regular Article). NMI rewards "an important step
  ahead" conceptual contributions, not only SOTA numbers.
- **Odds.** Realistic, high-fit home; *Nature* would demand broad
  general-science significance and tends to read a benchmark/critique as
  specialized.

**When *Nature* is justified:** if the discrepancy result is dramatic and broadly
framed — "widely used interpretability methods fail a ground-truth test, and here
is the testbed that reframes the field." Because **Nature and NMI share this exact
template**, we can write it broadly and *shoot for Nature first* at essentially
zero switching cost (downgrade to NMI on rejection without reformatting). My
suggestion: **write to Nature's bar, submit to Nature first only if Phase B shows
a striking, clearly-communicated failure of popular methods; otherwise NMI.**
Decide after Phase B results are in.

Article-type note: a standard **Article** if the benchmark + experiments lead; an
**Analysis** if the cross-method audit/critique leads. I lean **Article**.

## 8. Risks / open questions

- **Granularity vs Kording.** We work at the architectural level, not transistors
  (our sim is functional/cycle-accurate, not a netlist). This is a *feature*
  (the level where understanding is defined) but reviewers may want a direct
  transistor-level head-to-head — optional stretch: import the Visual6502 netlist
  for A1/A2 parity. Decision needed.
- **Agent training cost.** Use existing zoo agents where possible (jaxtari GPU
  rollouts help); training from scratch is a time sink.
- **Defining "the true causal map."** Intervention-based vs gradient-based ground
  truth can disagree at non-smooth points; we report both and treat agreement as
  part of the result.
- **Scope control.** Phase A + B alone is a strong paper; C2 (new methods) can be
  a follow-up if it bloats.

## 9. Phasing / next actions (after sign-off)

1. Lock journal target + article type + author list.
2. Phase A pilot: lesion + tuning-curve + dim-reduction on Space Invaders, with
   ground-truth scoring (reuses Paper-1 tooling) — proves the scoring concept.
3. Phase B pilot: one DQN agent + Integrated Gradients vs the intervention oracle
   on one game — proves the ground-truth attribution pipeline.
4. Expand to the method matrix and multiple games; build the benchmark (C1).
5. Draft `main.tex` against the structure in §6.

Author fit (provisional, TBD): A. Maier, S. Bayer, P. Krauss (+ collaborators);
Krauss's neuroscience/neuroprosthetics background anchors the §5 bridge.
