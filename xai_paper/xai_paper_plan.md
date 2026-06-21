# Paper 2 — Plan: the Atari VCS as a fully-known, differentiable model organism for interpretability

**Working title:** *Do our methods recover what is true? A fully-known,
differentiable model organism for evaluating interpretability.*

**Scope (read this first).** The **subject of study is the Atari VCS itself** — the
chip, the game program, and the game's own decision logic — *not* learned RL agents.
We take the entire modern interpretability toolkit and turn it on the VCS:
neuroscience/mechanistic methods (Jonas & Kording), attribution/XAI, mechanistic
interpretability (activation/attribution patching, sparse autoencoders, circuits,
probing), and behavioral/psychology-style probing. The VCS is bit-exact and
end-to-end differentiable (Paper 1), so for any output we can compute the *true*
causal structure and **score every method against ground truth**. Learned DQN
agents are explicitly **out of scope** (a natural Paper 3); they are not needed and
removing them makes this study entirely **self-contained on the current substrate**.

**One-line thesis.** There are several traditions for understanding a complex
opaque system — **mechanistic/neuroscience** (Kording 2017), **attributional/XAI**,
**mechanistic interpretability**, and **behavioral/psychology** (Binz & Schulz 2023;
Shiffrin & Mitchell 2023). Each "finds structure," but none can be *validated*
without ground truth — exactly Kording's warning and Shiffrin & Mitchell's caution.
The Atari VCS is **fully specified, bit-exact, and differentiable**, so we apply all
of these methods to *one* system and, for the first time, **score them against
ground truth** on a system complex enough to matter. We operationalize "interpretable"
(Barbiero et al. 2025) as *ground-truth recovery*, replicate the Kording battery,
audit attribution/XAI and the mechanistic-interpretability toolkit on the VCS's own
computation, test behavioral probing on the game's own decision logic, measure where
each diverges from truth, and argue what the science of understanding complex systems
actually needs.

> **Status: PLAN — scope corrected (this revision): the VCS is the subject in every
> phase; agents removed.** Template fetched into `paper/` (Springer Nature,
> `sn-nature`). Decisions taken:
> 1. **Conduct the experiments; decide scope later.** Architectural state (our
>    level) leads; a transistor-level head-to-head via the Visual6502 netlist runs
>    as a parallel track and we cut whichever the pilots show is weaker (§4, §8).
> 2. **Compute:** pilots local; full state-sweeps and SAE training on the LME
>    cluster (§4.6). Short answer: cluster for the full runs, but **every phase is
>    evaluable on the current substrate** — see §3.1.
> 3. **Method coverage reviewed** (§4.4–4.5): our Paper-1 literature is strong on
>    classic XRL but misses the 2023–25 mechanistic-interpretability wave
>    (activation/attribution patching, causal scrubbing, SAEs, probing, circuits) —
>    added, applied here to the VCS, where they are most ground-truth-relevant.
> Detailed per-phase design (measure / ideal / right / best case) in
> [`experiment_design.md`](experiment_design.md). Scaffolding: `tools/xai_study/`.

---

## 1. What Paper 1 gives us (the platform)

The differentiable VCS (jutari + jaxtari, 64/64 bit-exact vs xitari) already
exposes exactly the machinery this study needs:

- **Ground truth — in tiers (do not overclaim).**
  - **T1 causal/mechanistic (exact, all 64 games):** the exact causal effect of any
    state variable / input on any output, by intervention + gradient — *by
    construction*; the backbone; no labels needed.
  - **T2 hardware semantics (exact, all 64):** the role of every CPU/TIA/RIOT
    register, opcode and flag, and the true read/write data-flow (documented chip;
    the same silicon for every cartridge).
  - **T3 game-concept semantics (partial):** which RAM byte = ball-x / lives /
    score is set by the *program*, undocumented, and **not** free. Known only for a
    handful of addresses today; extendable on demand — and *verifiable by
    intervention* (perturb a byte → watch the object move on the exact framebuffer).
    Needed only for the *semantic-level* scoring (§3.1).
- **Exact interventions.** Knock out / clamp / perturb any RAM cell, register, ROM
  byte, opcode, or input and re-run deterministically (the "lesion" knob, but clean).
- **Differentiability.** `peek` one-hot reads, `soft_select` dispatch, `soft_branch`,
  the straight-through estimator, and the relaxed read give gradients of any output
  (a pixel, the score, a game event) w.r.t. any input (ROM byte, RAM cell, register,
  joystick action) — *the* ingredient Kording did not have.
- **Reusable tooling:** `tools/xai_si_gradient/` (real-ROM screen↔input gradient),
  the IG-on-ROM experiment — **P8**: the Paper-1 milestone where a SOFT program
  executes, the differentiable TIA renders a pixel, and three attribution methods
  (plain gradient, occlusion, smart-baseline Integrated Gradients) correctly trace
  that pixel to the ROM byte / TIA register that explains it, with the recorded
  finding that *naive zero-baseline IG collapses on discrete opcode bytes* — the
  conformance harness, the comparison-video renderer.

This is the crucial upgrade over Jonas & Kording, who used the Visual6502
*transistor netlist* and could only say results "felt unsatisfying." We work at the
**architectural level** (registers/RAM/opcodes) — where "understanding" is actually
defined (Marr's algorithmic level) — *and* we can put a number on every method's
faithfulness.

## 2. Contributions (what's new vs Paper 1 and vs Jonas & Kording)

1. **A ground-truth interpretability benchmark on one fully-known, differentiable,
   complex system.** The *true* causal structure of every output is computable, so
   any interpretability method can be scored, not eyeballed.
2. **A quantitative re-run of Jonas & Kording.** Each neuroscience method
   (connectomics, lesions, tuning curves, correlations, LFP, Granger causality,
   dimensionality reduction) reproduced on the VCS state and scored against the
   known mechanism — converting their qualitative critique into measured faithfulness.
3. **A faithfulness audit of attribution / XAI on the VCS's own computation.**
   Saliency, Integrated Gradients (cf. P8), occlusion/perturbation, SHAP/LIME, and
   counterfactual methods explaining a VCS output (a pixel, the score, a game event)
   from its inputs/state — scored against the *exact* causal attribution. Includes
   the finding that several popular methods (Grad-CAM, attention) are
   *NN-architecture-specific* and do not even apply to a non-neural computation.
4. **Mechanistic interpretability validated on a *known circuit*.** Activation /
   attribution patching, sparse autoencoders, circuit discovery and probing applied
   to the VCS state trajectory and scored against the true data-flow and known
   variables — the first validation of the mechanistic-interpretability toolkit on a
   complex system whose circuits are actually known.
5. **A ground-truth test of the *behavioral/psychology* tradition.** Probe the
   game's own decision logic (e.g., a game's built-in opponent AI) as a "participant"
   — controlled-stimulus / psychophysics experiments — and ask whether the inferred
   behavioral account matches the known code. This puts Shiffrin & Mitchell's
   caution to a *measurable* test (a first).
6. **A formal, operationalized notion of interpretability.** Adopt Barbiero et al.'s
   (2025) definition and operationalize "an explanation is correct" as *recovering
   the ground-truth causal structure*; interpretable-by-design as a comparison arm.
7. **Evidence of the discrepancy + concrete new directions:** where each tradition
   fails, why, and what a faithful, ground-truth-validated science of understanding
   complex systems should look like (causal/known-operator attribution;
   ground-truth-scored benchmarking; which behavioral inferences are trustworthy).

## 3. One subject under many microscopes — the VCS

The single object of study is the **running Atari VCS**, a deterministic function

> `output(t) = VCS(ROM, inputs[0..t], initial_state)`

with three faces, all fully observable and intervenable:

- **Inputs** — ROM bytes, joystick/paddle actions, console switches. *We drive the
  inputs ourselves* (scripted, random, or replay traces); no agent is required to
  make a game run.
- **Internal state ("the activations")** — RAM (128 B), CPU registers (A,X,Y,S,P,PC),
  TIA/RIOT registers, and the executing opcode stream.
- **Outputs ("the predictions to explain")** — a framebuffer pixel, the score, a
  game event (a bounce, a life lost), or a future state.

This one subject supports every tradition: attribute an *output* to inputs/state
(XAI); treat the *state trajectory* as activations to patch/decompose (mechanistic
interp); run the *neuroscience battery* on the state (Kording); and probe the
*game's own decision logic* behaviorally (psychology). Because all of T1/T2 (and,
where labeled, T3) are known, each method has a ground truth to be scored against.

The game's **own decision logic** is a genuine, hand-coded decision-maker we can
study behaviorally without any learned agent — e.g., Pong's CPU opponent (where does
it move the paddle given the ball?), Space Invaders' alien movement/fire logic,
pursuit/evasion AIs. These are the "participants" of Phase D, and their true
mechanism is in the ROM.

### 3.1 Which phases run on the current substrate (the answer to "what substrate is required")

Every phase below is evaluable on the **current** jutari/jaxtari substrate — because
the subject *is* the substrate. The only non-substrate ingredient ever needed is
**T3 game-concept labels**, and only for the *semantic-level* sub-metrics; those
labels are derivable/verifiable with the substrate's own interventions. There is **no
phase that needs a different or extended substrate**, with one optional exception:
long-horizon end-to-end gradients (credit through many steps of dynamics), which the
core single-step attribution metrics do not require. The detailed per-phase
substrate audit is in §10.

### 3.2 How we obtain T3 (game-concept labels), and what the T3 papers do

T3 = the map from RAM bytes to game concepts (ball-x, lives, score). Two public
sources supply *candidate* labels; our substrate then makes them *causal* and extends
them — the step neither source can take.

- **AtariARI** (Anand et al. 2019, NeurIPS — the *Atari Annotated RAM Interface*).
  Labels for **22 games**, obtained by reading commented disassemblies / source
  (Engelhardt & Jentzsch; CPUWIZ) to locate the RAM bytes holding sprite positions,
  room/score/lives; shipped as a gym wrapper that emits a state label per frame.
  **What they do with it:** use it as a *probing benchmark* — score whether a learned
  representation *linearly separates* the labeled variables (F1). Known caveat (their
  issue tracker): raw RAM (x,y) is **not aligned** to the rendered position (the game
  applies offsets at render time).
- **OCAtari** (Delfosse et al. 2023). Object-centric state for **40+ games** via two
  modes: a **RAM mode** that maps RAM to object properties *with the rendering offsets
  applied* (fixing AtariARI's misalignment), and a **Vision mode** that extracts an
  object list from the rendered RGB frame by per-object color filters; the two
  cross-validate. HackAtari (Delfosse 2024) builds controlled variants on top.

**Our procedure (the substrate's unique lever).**
1. **Import** OCAtari/AtariARI labels as *candidates* for the labeled subset.
2. **Verify causally by intervention:** set the candidate byte, re-render, and confirm
   the object appears at the predicted place on the **bit-exact framebuffer** — this
   upgrades a *correlational* label to a *verified causal* one and auto-corrects the
   rendering offset.
3. **Discover new labels** where no source exists: (a) correlate each RAM byte's
   time-series with an object position tracked from the framebuffer over rollouts; and
   (b) sweep interventions — perturb each byte, detect which object moved (vision) — to
   build the byte↔concept map; then verify as in (2).
4. **Coverage:** ~22 (AtariARI) → 40+ (OCAtari) of our 64, extended by (3).

**Why this matters for the thesis.** AtariARI is itself a *probing* benchmark, and
probing shows only that information is *present*, not *used* (the control-task
critique, Hewitt & Liang 2019 — exactly our A3/Phase-C trap). Our intervention test is
strictly stronger: it shows the byte is *causally* the variable. T3 is needed only for
the *semantic-level* metrics; T1/T2 carry the bulk and need no labels.

## 4. Experiment plan

> Detailed per-phase design — what is measured, the ideal explanation, when an
> explanation is *right* (the F∧S∧M correctness triad), and the best-case outcome —
> is in [`experiment_design.md`](experiment_design.md). Below is the overview.

### Phase A — Replicate Jonas & Kording, with ground-truth scoring

Map each neuroscience method onto the VCS state; for each, report both the
Kording-style qualitative result **and** a quantitative agreement with the known
mechanism. Subjects: the three games they used (Donkey Kong, Space Invaders,
Pitfall) plus several of our 64 for generality.

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

**Punchline of Phase A:** the same "interesting but unfaithful" structure Kording
found, now *quantified* — most methods score poorly against the known mechanism even
with unlimited, noiseless, fully-observed data.

### Phase B — Attribution / XAI on the VCS's input→output computation

Treat the VCS as the function to explain: attribute a chosen **output** (a pixel,
the score, a game event) to its **causes** (ROM bytes, RAM cells, registers, joystick
inputs).

1. **Ground-truth attribution.** The *true* causal influence of each candidate cause
   on the output, via (a) **exact interventions** (occlude/clamp/replace and measure
   the deterministic Δoutput) and (b) **gradients** through the differentiable
   substrate. Cross-validate (a) vs (b).
2. **Methods under test — named** (those that apply to a non-NN computation; verify
   each citation before it enters the bib):
   - *Gradient / backprop saliency:* vanilla gradient (Simonyan et al. 2014),
     Gradient×Input + DeepLIFT (Shrikumar et al. 2017), Guided Backprop
     (Springenberg et al. 2015), SmoothGrad (Smilkov et al. 2017), **Integrated
     Gradients** (Sundararajan et al. 2017 — our P8), Expected Gradients
     (Erion et al. 2021).
   - *Perturbation / occlusion on the true VCS:* occlusion (Zeiler & Fergus 2014),
     prediction-difference (Zintgraf et al. 2017), meaningful/extremal perturbations
     (Fong & Vedaldi 2017; Fong et al. 2019), RISE (Petsiuk et al. 2018), Greydanus
     perturbation saliency (2018), object occlusion (Iyer/Anderson 2018).
   - *Model-agnostic / game-theoretic:* LIME (Ribeiro et al. 2016), KernelSHAP
     (Lundberg & Lee 2017), Shapley-value sampling (Štrumbelj & Kononenko 2014).
   - *Counterfactual:* on-distribution counterfactual states via exact state-set +
     re-render (cf. Olson et al. 2021; Atrey et al. 2020 — but *valid* here).
   - *Controls:* sanity checks via model/data randomization (Adebayo et al. 2018) to
     flag methods that ignore the system. Survey scope from our XRL notes (Qing 2022,
     Vouros 2023, Cheng 2025, Saulières 2025).
3. **Explicit N/A finding.** Grad-CAM/Grad-CAM++ (Selvaraju et al. 2017;
   Chattopadhyay et al. 2018 — need conv feature maps), attention rollout (Abnar &
   Zuidema 2020), and policy-distillation surrogates (VIPER, Bastani et al. 2018) are
   *NN-architecture-specific* and do not transfer to the VCS — a measured statement
   about the narrowness of popular XAI.
4. **Faithfulness metrics vs ground truth.** Correlation with the true causal map;
   deletion/insertion AUC measured on the *true* VCS (not a proxy); precision@k /
   pointing-game / object-hit vs the true causal top-k.
5. **Result:** which methods track the true causes vs produce plausible-but-wrong
   maps (a measured, ground-truthed sharpening of Atrey 2020 — now about the system,
   not an agent).

### Phase C — Mechanistic interpretability on the VCS (the known-circuit testbed)

The modern successor to Kording's connectomics. The VCS **state trajectory is the
"activations"** and the program's **data-flow is the "circuit"** — both known.

Methods are named below (verify each citation before it enters the bib):

1. **Activation patching / causal tracing.** Clamp/resample RAM cells / registers /
   TIA state between a clean and a corrupted run; score recovered components against
   the exact intervention effect (the oracle) and the true data-flow. Methods: causal
   mediation analysis (Vig et al. 2020), causal tracing / ROME (Meng et al. 2022),
   interchange interventions & causal abstraction (Geiger et al. 2021, 2023),
   distributed alignment search (Geiger et al. 2024).
2. **Attribution patching + automatic circuit discovery.** attribution patching
   (Nanda 2023), edge attribution patching (Syed et al. 2023), path patching / the
   IOI circuit (Wang et al. 2022; Goldowsky-Dill et al. 2023), ACDC (Conmy et al.
   2023); score against true patching and the true data-flow.
3. **Sparse autoencoders / dictionary learning.** Train an SAE on the state
   trajectory; score features against **known variables** (T2 hardware signals
   always; T3 game variables where labeled) — feature↔variable matching + causal use
   (does patching the feature move the output as predicted?). Methods: SAEs
   (Cunningham et al. 2023; Bricken et al. 2023; Templeton et al. 2024) plus classical
   NMF/PCA dictionaries.
4. **Circuit validation by causal scrubbing** (Chan et al. 2022): resample the
   hypothesised-irrelevant parts; behavior preserved? — against the true disassembled
   routine (the ball-bounce or opponent-AI kernel).
5. **Probing & its controls.** linear probes (Alain & Bengio 2017) with control
   tasks / selectivity (Hewitt & Liang 2019); logit/tuned-lens readouts adapted to the
   VCS (nostalgebraist 2020; Belrose et al. 2023). Contrast "decodable" vs "actually
   used" — the probing/tuning-curve trap (cf. A3).

**Punchline of Phase C:** the first time the mechanistic-interpretability toolkit is
scored against a *known* circuit in a complex system — a calibration and validation
the mech-interp community currently lacks.

### Phase D — Behavioral / psychology probing of the game's own logic

The behavioral tradition (Binz & Schulz 2023; Shiffrin & Mitchell 2023): infer a
system's "cognition" from behavior. Their open worry — does behavioral probing reveal
the true mechanism or just plausible correspondences? — is *unanswerable for LLMs*
(no ground truth). **We answer it on a known decision-maker: the game's own logic.**

- **Subjects:** a game's built-in decision logic (e.g., Pong's CPU opponent, Space
  Invaders' alien/fire logic), whose mechanism is in the ROM.
- **Probes:** controlled-stimulus / psychophysics experiments — vary one factor of
  the game situation (object position/distance, timing, distractors) by setting state
  and re-rendering, then read the program's response.
- **Methods — named** (verify each citation before it enters the bib):
  psychometric-function fitting (Wichmann & Hill 2001), method of constant stimuli /
  adaptive staircases, signal-detection theory (Green & Swets 1966), reverse
  correlation / classification images (Ahumada 1971; Murray 2011), drift-diffusion /
  sequential-sampling models for "frames-to-respond" (Ratcliff 1978), ideal-observer
  analysis, and cognitive-task batteries in the style of Binz & Schulz (2023);
  Shiffrin & Mitchell (2023) frame the caveats we test against.
- **The test:** does the inferred behavioral account match the *true* code (the
  Phase-A/C ground truth)? Quantify "right for the wrong reasons."
- **Contribution:** the first ground-truthed verdict on behavioral-probing
  methodology — which inferences are trustworthy, which are mirages.

### Phase E — Synthesis: a benchmark and better directions

- **E1 Ground-truth interpretability benchmark.** Package the scoring suite (tasks,
  true-attribution oracle, faithfulness metrics) as a reusable benchmark — a
  "leaderboard with real ground truth."
- **E2 Faithful methods.** Show that causal-intervention / gradient / known-operator
  attribution (native to our differentiable substrate) is measurably more faithful,
  and propose how such ideas transfer to opaque systems that *lack* ground truth.
- **E3 Directions.** From the discrepancy: ground-truth-validated benchmarking,
  causal rather than correlational attribution, and method development *sieved* on
  known systems first.

## 4.4 Method matrix — where we expect success vs failure

Expected **punchline**: methods that are *causal / intervention-grounded* track our
ground truth; methods that are *correlational, off-manifold, architecture-specific,
or spatially coarse* produce plausible-but-wrong explanations — the exact analogue of
Kording's lesson that correlational methods mislead even with unlimited data.

| Method (category) | In our lit? | Applies to VCS? | Expected vs ground truth | Why |
|---|---|---|---|---|
| **Activation patching / causal tracing** (causal, mechanistic) | ✗ add | ✓ (state) | **Succeed** | causal by construction; directly comparable to our intervention oracle |
| **Causal scrubbing** (hypothesis test) | ✗ add | ✓ | **Succeed** | we *have* ground-truth hypotheses to scrub against — ideal testbed |
| **Integrated Gradients** (axiomatic gradient) | ~partly (P8) | ✓ | **Succeed/partial** | completeness axiom + exact-forward gradients; baseline-sensitive |
| **Occlusion/perturbation on the true VCS** | ✓ (Greydanus, Iyer/Anderson) | ✓ | **Succeed/partial** | faithful when the perturbation is a valid intervention; Gaussian-blur variants go off-distribution |
| **Counterfactual states** (Olson; Atrey) | ✓ | ✓ (on-distribution here!) | **Partial→Succeed** | our substrate makes counterfactuals valid (set state, re-render) — removes Atrey's off-manifold objection |
| **Attribution patching / edge AP** (gradient approx of patching) | ✗ add | ✓ | **Partial** | cheap approximation; we score the approximation error against true patching |
| **Sparse autoencoders / dictionary learning** | ~1 note | ✓ (state) | **Partial** | recovers features; score vs known variables; faithfulness-to-computation debated |
| **Circuit / mechanistic analysis** | ~1 note | ✓ | **Partial→Succeed** | the goal state is exactly our ground truth; measure recovered circuit vs true data-flow |
| **Linear probing** (concept readout) | ✗ add | ✓ | **Partial / misleading** | detects info is *present*, not *used* — the tuning-curve trap (A3) |
| **SHAP / LIME / Shapley** (model-agnostic) | ✓ (surveys) | ✓ | **Partial/Fail** | baseline- and sampling-dependent; approximations diverge; costly |
| **Vanilla gradient / raw saliency** | ✓ | ✓ | **Fail** | noisy, gradient-shattering, non-causal |
| **Grad-CAM / Grad-CAM++** (CAM) | ✓ | **✗ N/A** | — | needs conv feature maps; does not apply to a non-NN computation (a finding) |
| **Attention maps** (Mott, Nikulin) | ✓ | **✗ N/A** | — | needs attention layers; not present in the VCS (a finding) |
| **Surrogate trees / policy distillation** (VIPER, XDQN) | ✓ | **✗ N/A** | — | need a learned policy; out of scope (agents) |

Headline expected result: **causal/gradient/mechanistic methods grounded in real
interventions pass the ground-truth test; correlational and architecture-specific
methods fail or don't even apply** — a measured, ground-truthed sharpening of the
Kording lesson and of Atrey 2020.

## 4.5 Newest methods we must add (gap from the Paper-1 literature)

Our Paper-1 notes (RL-centric XRL surveys) **miss the 2023–25 mechanistic-
interpretability wave**, which is the most relevant to a *causal ground-truth*
benchmark. Add to the literature/bib and the matrix (verify each entry to Paper-1's
no-hallucination standard before it enters the bib):

- **Activation patching / causal tracing** — Meng et al. 2022 (ROME); Heimersheim & Nanda 2024.
- **Attribution patching** — Nanda 2023; **Edge Attribution Patching**, Syed et al. 2023.
- **Causal scrubbing** — Chan et al. 2022 (Redwood).
- **Sparse autoencoders for interpretability** — Cunningham et al. 2023; Bricken et al. 2023 ("Towards Monosemanticity").
- **Unifying theory** — Geiger et al. 2023/25, "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability".
- **Field reviews** — Bereska & Gavves 2024 (mech-interp for AI safety).
- **Behavioral / psychology tradition (Phase D)** — Binz & Schulz 2023 (PNAS);
  Shiffrin & Mitchell 2023 (PNAS) [in `papers/`].
- **Formal definition of interpretability** — Barbiero et al. 2025, "Foundations of
  Interpretable Models" (arXiv:2508.00545) — definitional anchor + interpretable-by-design.
- **Atari semantics resources (T3 labels)** — OCAtari (Delfosse et al. 2023, 40+
  games, RAM+vision) and AtariARI (Anand et al. 2019, ~22 games) — sources for
  game-concept labels, *verified/extended by our interventions*.

These newest methods are also the paper's *positive* story: they are exactly the
causal tools we expect to pass, so testing them on ground truth is a contribution to
the mechanistic-interpretability community.

## 4.6 Compute plan — do we need the cluster?

**Pilots local; full runs on the cluster.** Key enabler from Paper 1: the
**SOFT-STE forward is bit-exact to HARD and GPU-batchable** (`vmap`, millions of
env-steps/s), so the exact intervention oracle and the gradients run *batched on
GPU* rather than the slow CPU-only HARD path. (No agents → no agent training/zoo.)

| Workload | Where | Why |
|---|---|---|
| Phase-A pilot (1 game: lesion + tuning + dim-reduction, scored) | **local** (M1 Max) | small; reuses Paper-1 tooling |
| Phase-B pilot (IG vs intervention oracle on one output, 1 game) | **local** | proves the oracle pipeline (extends P8) |
| Phase-C pilot (activation patching + 1 SAE on the VCS state, 1 game) | **local** | proves the mech-interp-on-VCS pipeline |
| Phase-D pilot (one psychophysics probe of a game's opponent logic) | **local** | proves the behavioral pipeline |
| Phase-A/B full lesion + occlusion sweeps (RAM × ROM × opcodes × outputs × games) | **cluster (CPU array / GPU)** | embarrassingly parallel; batched SOFT-STE exact-forward on GPU |
| Batched gradients / IG over outputs × games | **cluster (GPU)** | jit+vmap, GPU-bound |
| SAE training on VCS state trajectories | **cluster (GPU)** | standard DL training on recorded state |

Reuse the Paper-1 cluster setup verbatim: LME Slurm, repo on `/cluster/maier`, the
`tools/cluster/*.sbatch` pattern + the jaxtari GPU venv (jax[cuda12]).

## 5. Discussion — the traditions of understanding, scored against truth

- **Many traditions, one ground truth.** Neuroscience/mechanistic (Phase A),
  attribution/XAI (B), mechanistic interpretability (C), and behavioral/psychology
  (D) are the ways the community tries to understand opaque systems. Each "finds
  structure"; the VCS is the first place all can be *validated* against known truth
  on a complex system.
- **What "interpretable" means.** We adopt Barbiero et al.'s (2025) definition and
  operationalize it as *ground-truth recovery* — making "is this explanation
  correct?" measurable rather than a matter of taste.
- **The shared toolkit.** Neuroscience and XAI use the *same* methods under different
  names: tuning curves ↔ feature visualization; lesions ↔ ablations; connectomics ↔
  circuit/mechanistic analysis; dimensionality reduction ↔ probing; Granger ↔
  causal-attribution. The VCS is a common benchmark for all.
- **Kording's lesson, made measurable.** "Finds structure ≠ understanding" was a
  warning; with ground truth it becomes a *test* a method passes or fails.
- **Mechanistic interpretability, finally calibrated.** Mech-interp seeks "circuits"
  in neural nets; here we run the same algorithms on a system whose circuits are
  *known*, giving the field its first ground-truth calibration — directly relevant to
  interpreting the deep networks neuroscience now uses as brain models. (Co-author
  fit: P. Krauss bridges pattern recognition and neuroscience/neuroprosthetics.)
- **And the behavioral question.** Shiffrin & Mitchell ask whether behavioral probing
  reveals a system's true mechanism — unanswerable for LLMs (no ground truth),
  answerable here on the game's known decision logic. We give behavioral methodology
  its first ground-truthed verdict.
- **Where it goes beyond the brain.** Unlike a brain, our system is differentiable,
  so it also tells us which *gradient-based* explanations are trustworthy.

## 6. Paper structure (Nature Portfolio format)

`paper/main.tex` is wired to `\documentclass[sn-nature]{sn-jnl}` (template fetched).

- **Title; Abstract** (~150–200 words, unreferenced).
- **Main text** (Nature has no rigid IMRaD): Introduction → Results (Phase A Kording
  scoring; Phase B attribution audit; Phase C mechanistic-interp on the known
  circuit; Phase D behavioral probing; the cross-tradition discrepancy; a
  faithful-method demonstration) → Discussion (§5).
- **Main display items** (~6 figures): (1) the VCS platform & ground-truth oracle;
  (2) the traditions scored side by side (headline); (3) Kording battery scored;
  (4) attribution vs mechanistic-interp faithfulness on the VCS; (5) behavioral-probe
  verdict vs the true code; (6) failure taxonomy / what to do instead.
- **Methods** (after references, no length limit): emulator, true-attribution oracle,
  each analysis, metrics.
- **Extended Data**, **Supplementary Information** (full per-method results, oracle
  proofs, all games).
- **End matter:** Data availability, Code availability, Author contributions,
  Competing interests, Reporting Summary, Acknowledgements (see `document_check.md`).

## 7. Journal: *Nature Machine Intelligence* vs *Nature*

**Recommendation: target *Nature Machine Intelligence* (primary).**

- **Scope fit.** NMI explicitly covers interpretability/XAI, the AI↔neuroscience
  interface, benchmarks, and methodological critique. Jonas & Kording itself was a
  methods-critique (PLOS Comp Biol); the successor lives at NMI.
- **The "discrepancy + new directions" framing** is exactly NMI's **Analysis**
  article type (or a regular Article).
- **Odds.** Realistic, high-fit home; *Nature* would demand broad general-science
  significance and may read a benchmark/critique as specialized.

**When *Nature* is justified:** if the discrepancy is dramatic and broadly framed —
"widely used interpretability methods fail a ground-truth test, and here is the
testbed that reframes the field." Because **Nature and NMI share this template**, we
write broadly and *shoot for Nature first* at near-zero switching cost (downgrade to
NMI on rejection without reformatting). Decide after the pilots/Phase-B–C results.

Article-type: a standard **Article** if the benchmark + experiments lead; an
**Analysis** if the cross-method audit/critique leads. I lean **Article**.

## 8. Risks / open questions

- **Granularity vs Kording.** *Decided: run both, cut later.* Architectural level
  (where "understanding" is defined) leads; the Visual6502 transistor netlist runs as
  a parallel A1/A2 track for a direct head-to-head. Pilots decide which we feature.
- **T3 label coverage.** Semantic-level metrics need game-concept labels; we have
  few today. Mitigation: source from OCAtari/AtariARI, verify/extend by intervention,
  and lean on T1/T2 (which need no labels) for the bulk of the scoring.
- **Defining "the true causal map."** Intervention-based vs gradient-based ground
  truth can disagree at non-smooth points; we report both and treat agreement as part
  of the result.
- **Scope control.** Phases A–C alone are a strong paper; D and E can be staged.

## 9. Phasing / next actions

**Prepared:** scaffolding at `tools/xai_study/` — specs + harness stubs for the
ground-truth oracle and each phase; method matrix and compute plan locked above.

Immediate pilots (all run **locally**, all on the current substrate):
1. **Phase-A pilot** — lesion + tuning-curve + dim-reduction on Space Invaders, with
   ground-truth scoring (reuses Paper-1 tooling).
2. **Phase-B pilot** — IG vs the exact intervention oracle for one output (extends P8).
3. **Phase-C pilot** — activation patching + one SAE on the VCS state, scored vs the
   true data-flow / known variables.
4. **Phase-D pilot** — one psychophysics probe of a game's opponent logic vs the code.

Then: stand up the cluster sweeps (§4.6); expand the method matrix (§4.4) incl. the
newest causal methods (§4.5) and more games; build the benchmark (E1); draft
`main.tex` (§6). Lock journal/article-type after the pilots (§7).

Author fit (provisional): A. Maier, S. Bayer, P. Krauss (+ collaborators); Krauss's
neuroscience/neuroprosthetics background anchors the §5 bridge.

## 10. Per-phase substrate audit (which phases run on the current substrate)

This is the audit, re-done after the scope fix. Substrate = jutari/jaxtari as in
Paper 1 (exact forward, exact interventions on any state, full observability,
gradients of any output w.r.t. any input, GPU-batched SOFT-STE). The subject is the
VCS in every phase, so the substrate is *both* instrument and subject throughout.

| Phase / method | Substrate capability required | In current substrate? | Other ingredient | Evaluable now? |
|---|---|---|---|---|
| **A1** connectomics | per-instruction R/W observability; true data-flow | ✅ | — | **Yes** |
| **A2** lesions | exact state intervention + deterministic re-run | ✅ | — | **Yes** |
| **A3** tuning | full state+pixel recording | ✅ | T3 only for game-concept scoring | **Yes** (T1/T2); semantic part needs T3 |
| **A4–A8** corr/LFP/Granger/dim-red/whole-state | full state recording; known clocks & data-flow | ✅ | — | **Yes** |
| **B** attribution/XAI | exact intervention + gradient of an output w.r.t. inputs/state | ✅ (P8 already does this) | T3 only for object/concept-level scoring | **Yes** |
| **C** mechanistic interp | clamp/resample state between runs; record state trajectory; gradients | ✅ | SAE training code (standard); T3 for semantic checks | **Yes** |
| **D** behavioral | set state + re-render controlled stimuli; read program response | ✅ | T3 / vision to define the situation variable | **Yes** |
| **E** synthesis/benchmark | aggregates A–D | — | depends on A–D | **Yes** (follows) |

**Cross-cutting (neither is a new substrate):**
- **T3 game-concept labels** — external knowledge, *not* a substrate capability, but
  derivable/verifiable with the substrate's interventions. Gates only the
  semantic-level sub-metrics (A3-semantic, B object/concept attribution, C
  SAE-feature↔variable, D situation variables). Everything else scores against
  T1/T2, which are intrinsic.
- **Long-horizon end-to-end gradients** (credit through many steps of dynamics) — the
  one item that could stress the substrate (BPTT / memory / the short conformance
  window). Single-step attribution — what the core metrics use — is already covered.

**Bottom line:** with the subject corrected to the VCS, **all phases are evaluable on
the current substrate.** No phase requires a different or extended substrate; the only
optional "more substrate" item is long-horizon gradients, which the planned
experiments do not need.
