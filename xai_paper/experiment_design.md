# Experiment design — Paper 2 (per-phase: measure, ideal, correct, best case)

Companion to `xai_paper_plan.md`. **Subject in every phase: the Atari VCS itself**
(chip + program + game logic) — *not* learned agents. For every phase we answer the
same four questions:
1. **What can be done & measured?**
2. **What is the ideal explanation of what we observe?**
3. **When is the explanation *right*?** (the correctness criterion)
4. **What is the best-case outcome of the analysis?**

Everything hangs on one definition (§0) and one oracle (§1); the phases (§2–§6)
instantiate them. A compact master table is in §7.

---

## 0. When is an explanation "right"? — the correctness triad

We make "correct" measurable (operationalizing Barbiero et al. 2025 via our ground
truth, in Marr's terms, and Lazebnik's "fix-the-radio" sufficiency). An explanation
`Ê` of how output `y` arises is **right** iff it is:

- **(F) Faithful** — the causes/structure it claims are the *true* causes. Measured
  against the oracle (§1): claimed dependencies match the intervention-verified
  dependencies (precision/recall, sign, magnitude).
- **(S) Sufficient** — it *predicts behavior under interventions it did not see* and
  could regenerate the behavior. Measured by **held-out interchange accuracy**
  (causal-abstraction / causal-scrubbing style): apply a new intervention, predict
  `y` from `Ê`, compare to the true `y`.
- **(M) Minimal & at the right level** — parsimonious, no spurious parts, stated at
  the algorithmic level (registers/RAM/opcodes; the program's variables, circuits,
  and decision logic), matching the known module decomposition.

`right = F ∧ S ∧ M`. Each is a number on our system. This triad is the spine of every
phase; "interesting structure" that is not F∧S∧M is the failure mode Kording and
Shiffrin–Mitchell warned about.

**Two reporting axes (the headline plot):** *faithfulness* (X, vs ground truth) vs
*human-plausibility* (Y, does it look right). The danger zone — high plausibility,
low faithfulness — is where we predict the popular methods land.

---

## 1. The ground-truth oracle (shared measurement instrument)

For any output `y` (a pixel, the score, a game event, a future state) and candidate
cause `u` (ROM byte, RAM cell, register, opcode, joystick input, or — for mechanistic
interp — an internal state variable):

- **Causal effect (exact):** intervene on `u` (occlude / clamp / resample), re-run
  deterministically, record Δ`y`. Bit-exact; no world-model assumed. The *true causal
  map* is `{Δy(u)}`.
- **Differential effect:** `∂y/∂u` through the differentiable substrate.
- **Known semantics — in tiers (do not overclaim):**
  - **T1 causal/mechanistic** (exact, all 64 games, by construction);
  - **T2 hardware semantics** (exact, all 64: register/opcode roles, data-flow);
  - **T3 game-concept semantics** (which RAM byte = ball-x / lives / score) —
    *partial*, set by the program, undocumented; sourced from OCAtari/AtariARI and
    **verified/extended by our own interventions** (perturb byte → object moves).
    Needed only for *semantic-level* metrics.

The oracle is itself validated (intervention vs gradient agreement); disagreement
flags non-smooth points and is reported, not hidden. **Everything below is scored
against this oracle.**

Running example throughout: **Pong** — outputs to explain: the score, the ball-pixel;
known variables: ball position, paddle positions, score (RAM); behavioral subject:
the **CPU opponent** (its paddle-tracking logic, in the ROM).

---

## 2. Phase A — neuroscience / mechanistic battery on the VCS (Kording, quantified)

Subject: the running VCS. Each classical method emits a "finding"; we score it.

**(1) Measure** — per analysis, the finding and its score vs the known mechanism:

| Analysis | Finding (what the method outputs) | Measured score |
|---|---|---|
| A1 connectomics | recovered dependency graph over state vars | precision/recall + graph-edit-distance vs the *true* read/write graph |
| A2 lesions | per-unit importance (does the game still run?) | rank-correlation of lesion-importance with the unit's *true* role; #units flagged "specific" that are actually generic |
| A3 tuning curves | per-unit tuning to luminance/game var | fraction of strongly-tuned units whose tuning ≠ their true role (spurious-tuning rate) |
| A4 correlations | pairwise/global correlation structure | weak-pairwise/strong-global reproduced; vs true coupling |
| A5 LFP | regional power spectra / "rhythms" | are peaks the known clocks (frame/scanline)? %-variance that is epiphenomenal |
| A6 Granger | inferred subsystem causality | false-edge / missed-edge rate vs true data-flow |
| A7 dim-reduction | latent components | matched-component fraction vs known signals (clock, R/W, vsync) |
| A8 whole-state | descriptive map | baseline only |

**(2) Ideal explanation** — the register-transfer-level account: *"frame-1 of this
game is produced by the display kernel reading RAM cells X through opcodes Y, driving
TIA registers Z; module M implements function f."* (Known exactly.)

**(3) When right** — F: recovered graph/role = true graph/role. S: the recovered
structure predicts held-out lesions/interventions. M: module level (registers/adders),
not transistors or epiphenomena. A lesion result is "right" only if the unit's *true
role* explains the broken behavior — not merely "game X broke."

**(4) Best case** — a *quantified* Kording: every classical method scores **low on
F∧S** despite rich structure (e.g., "lesion-importance explains <X%> of true role;
tuning is spurious for <Y%> of tuned units"), while the *causal* operators
(intervention/patching, §4) score **high** — proving the gap is the **method**, not
the data or the system. Plus the Visual6502 transistor-level head-to-head (parallel
track).

---

## 3. Phase B — attribution / XAI on the VCS's input→output computation

Subject: the VCS as a function. Explain a chosen **output** `y` (pixel / score /
event) from its **causes** (ROM, RAM, registers, inputs). Each method emits an
attribution map over those causes.

**(1) Measure** — vs the oracle's true causal map:
- correlation (Spearman) with the true Δ`y` map;
- **deletion/insertion AUC measured on the true VCS** (remove top-attributed causes,
  measure the *actual* effect — not a proxy);
- precision@k / pointing-game / object-hit vs the true causal top-k;
- **plausibility** (does it highlight human-sensible causes) for the X-vs-Y plot.
- **N/A check:** Grad-CAM / attention / policy-surrogates need NN structure absent in
  the VCS — record as "does not apply" (a finding about popular XAI).

**(2) Ideal explanation** — the minimal set of causes whose change actually moves `y`,
with correct sign/magnitude: the *true causal saliency* (e.g., for the Pong score: the
ball-position and paddle RAM cells and the scoring opcode, not the background).

**(3) When right** — F: top-k attributed = true causal top-k. S: deleting them on the
real VCS changes `y` as predicted. M: sparse, at the variable/opcode level. Explicitly
*not* "the map looks reasonable" unless the highlighted cause is the true one.

**(4) Best case** — a **faithfulness leaderboard with real ground truth**: vanilla
saliency / model-agnostic methods sit low on F while plausible; IG /
occlusion-on-the-true-VCS / on-distribution counterfactuals approach the ceiling; the
NN-specific methods don't even apply. Deliverable claim: "*plausible ≠ faithful*, and
here is the gap, in numbers — on the system itself."

## 4. Phase C — mechanistic interpretability on the VCS (the known-circuit testbed)

The VCS **state trajectory is the "activations"**, the program's **data-flow is the
"circuit"** — both known. Ground truth = T1/T2 (always) + T3 (where labeled).

**(1) Measure**
- **Activation patching / causal tracing:** patch RAM cells / registers / TIA state
  clean↔corrupted; score = patched effect vs the **exact** intervention effect (the
  oracle) and recovery of the truly-important components (P/R).
- **Attribution patching (+ edge AP):** gradient approximation; score = approximation
  error vs true patching, and important-edge recovery.
- **Sparse autoencoders:** train on the state trajectory; score = **feature ↔ known
  variable** matching (probing accuracy / MI for a hardware signal or, where labeled,
  a game variable) *and* causal use (does patching the feature move `y` as predicted?).
- **Circuits + causal scrubbing:** recover the circuit for a behavior (ball-bounce,
  opponent-AI); resample the hypothesised-irrelevant parts — behavior preserved? Score
  = scrubbing-preserved performance + match to the true disassembled routine.
- **Linear probing:** decode concepts from state; contrast decodable vs *used*.

**(2) Ideal explanation** — a circuit/feature-level account of *how the program
computes `y`*: e.g., "RAM cell c = ball-x; the bounce routine compares c to the wall
constant and negates the velocity cell" — verified against the disassembly.

**(3) When right** — F: recovered features/circuit = the true data-flow
(patching-verified). S: the recovered circuit alone reproduces the behavior (causal
scrubbing passes). M: minimal, monosemantic features matched to known variables.

**(4) Best case** — the **first validation of the mechanistic-interpretability toolkit
against a *known* circuit in a complex system**: (a) patching recovers the true
data-flow at a measured rate; (b) an SAE calibration — features map to known variables
at a measured rate with a measured monosemanticity ceiling. A contribution *to*
mech-interp, not only a critique. (Caveat to test: some computations may not decompose
into human-clean features — itself a reportable finding.)

## 5. Phase D — behavioral / psychology probing of the game's own logic

Subject: a game's built-in decision logic (e.g., Pong's CPU opponent), as a
"participant." Mechanism is in the ROM, so we can check.

**(1) Measure** — psychophysics-style: vary one controlled factor of the situation
(object position/distance, timing, distractor) by setting state and re-rendering; read
the program's response; fit a behavioral account (psychometric curve, inferred
decision variable, bias, reaction-time analogue). Compare the **inferred** decision
variable to the **true** driver (the code / oracle). Key metric:
**"right-for-the-wrong-reasons" rate** + held-out generalization to novel stimuli.

**(2) Ideal explanation** — a behavioral law over the program's true decision
variables (e.g., "the opponent moves toward `sign(ball_y − paddle_y)` with a 1-frame
lag and a dead-zone").

**(3) When right** — F: inferred variable = true driver (verified against the code).
S: the law predicts behavior on **novel** stimuli — directly answering Shiffrin &
Mitchell's "right for the wrong reasons" / context-dependence concern. M: simplest law
that holds.

**(4) Best case** — the **first ground-truthed verdict on behavioral-probing
methodology**: a measured split into *trustworthy* (match truth + generalize) vs
*mirage* (fit but wrong driver), with the conditions that predict each.

## 6. Phase E — synthesis (the cross-tradition result)

**(1) Measure** — put every method (A neuroscience, B attribution, C mechanistic, D
behavioral) on the two shared axes (faithfulness/sufficiency vs plausibility) with one
common protocol.

**(2/3) Ideal/right** — a single comparable F∧S∧M score per method.

**(4) Best case — the paper's headline:**
- **One figure, all traditions:** neuroscience, attribution, mechanistic, behavioral,
  each scored against one ground truth on a system complex enough to matter. The
  "plausible-but-wrong" quadrant is the *popular* methods; the causal/mechanistic
  methods occupy the faithful region.
- **A benchmark artifact** (tasks + oracle + metrics) others can run.
- **A directions claim:** the science of understanding complex systems needs
  ground-truth validation, causal (not correlational) attribution, and method
  development *sieved on known systems first* — with the VCS as that sieve.

---

## 7. Master table

| Phase | Measure (output → score) | Ideal explanation | Right when (F∧S∧M) | Best-case outcome |
|---|---|---|---|---|
| **A** VCS / neuroscience | finding (graph, lesion map, tuning, components) → agreement with known mechanism | register-transfer account of the frame | recovered structure = true graph/role; predicts held-out lesions; module-level | quantified Kording: classical low-F, causal high-F — gap is the method |
| **B** VCS / attribution | attribution map for an output → corr + deletion/insertion AUC on true VCS + precision@k | minimal true-causal inputs/state for that output | top-k = true causal top-k; deletion behaves as predicted; sparse | leaderboard: popular methods plausible-but-unfaithful or N/A; IG/occlusion/causal faithful |
| **C** VCS / mechanistic | patch/SAE/circuit on state → vs exact patch + feature↔known-var + scrubbing | circuit/feature account of how the program computes the output | recovered = true data-flow; circuit reproduces behavior; monosemantic | first validation of mech-interp on a *known* circuit + SAE calibration |
| **D** game-logic / behavioral | psychometric fit → inferred driver vs true code + generalization | behavioral law over the program's true decision variables | inferred var = true driver; predicts novel stimuli; simplest | first ground-truthed verdict: trustworthy vs mirage behavioral inferences |
| **E** synthesis | all methods on faithfulness-vs-plausibility | one comparable F∧S∧M score | — | cross-tradition headline figure + benchmark + directions |

> Outcomes in §2–§6 are *hypotheses* (our predictions), to be confirmed by the
> experiments — the pilots (`tools/xai_study/*`) test the riskiest links first: the
> oracle, and one method per tradition. **Every phase runs on the current substrate**
> (the subject is the VCS); see `xai_paper_plan.md` §10.
