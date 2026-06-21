# Experiment design — Paper 2 (per-phase: measure, ideal, correct, best case)

Companion to `xai_paper_plan.md`. For every phase we answer the same four
questions you asked:
1. **What can be done & measured?**
2. **What is the ideal explanation of what we observe?**
3. **When is the explanation *right*?** (the correctness criterion)
4. **What is the best-case outcome of the analysis?**

Everything hangs on one definition (§0) and one oracle (§1); the phases (§2–§5)
instantiate them. A compact master table is in §7.

---

## 0. When is an explanation "right"? — the correctness triad

We make "correct" measurable (operationalizing Barbiero et al. 2025 via our ground
truth, in Marr's terms, and Lazebnik's "fix-the-radio" sufficiency). An explanation
`Ê` of how output `y` arises is **right** iff it is:

- **(F) Faithful** — the causes/structure it claims are the *true* causes.
  Measured against the oracle (§1): the claimed dependencies match the
  intervention-verified dependencies (precision/recall, sign, magnitude).
- **(S) Sufficient** — it *predicts behavior under interventions it did not see*
  and could regenerate the behavior. Measured by **held-out interchange accuracy**
  (causal-abstraction / causal-scrubbing style): apply a new intervention, predict
  `y` from `Ê`, compare to the true `y`.
- **(M) Minimal & at the right level** — parsimonious, no spurious parts, stated at
  the algorithmic level (registers/RAM/opcodes for the chip; features/circuits/
  decision-variables for the agent), matching the known module decomposition.

`right = F ∧ S ∧ M`. Each is a number on our system. This triad is the spine of
every phase; "interesting structure" that is not F∧S∧M is the failure mode Kording
and Shiffrin–Mitchell warned about.

**Two reporting axes (the headline plot):** *faithfulness* (X, vs ground truth) vs
*human-plausibility* (Y, does it look right). The danger zone — high plausibility,
low faithfulness — is where we predict the popular methods land.

---

## 1. The ground-truth oracle (shared measurement instrument)

For any output `y` and candidate cause `u` (pixel, object, RAM byte, register,
ROM byte, opcode, or an internal agent activation/feature):

- **Causal effect (exact):** intervene on `u` (occlude / clamp / resample),
  re-run deterministically, record Δ`y`. Bit-exact; no world-model assumed. The
  *true causal map* is `{Δy(u)}`.
- **Differential effect:** `∂y/∂u` through the differentiable substrate (and
  through `emulator ∘ agent` end-to-end).
- **Known semantics:** the true role of every chip state variable (from the spec)
  and the true game variables (ball-x, paddle-x, lives, score — known RAM
  addresses from Paper 1).

The oracle is itself validated (intervention vs gradient agreement); disagreement
flags non-smooth points and is reported, not hidden. **Everything below is scored
against this oracle.**

Running example used throughout: **Pong** — true decision variables are
ball position, paddle position, ball velocity (known RAM); a competent agent's
action must causally depend on the ball–paddle relative position.

---

## 2. Phase A — mechanistic / neuroscience on the chip (Kording, quantified)

Subject: the running VCS. Each classical method emits a "finding"; we score it.

**(1) Measure** — per analysis, the finding and its score vs the known mechanism:

| Analysis | Finding (what the method outputs) | Measured score |
|---|---|---|
| A1 connectomics | recovered dependency graph over state vars | precision/recall + graph-edit-distance vs the *true* read/write graph |
| A2 lesions | per-unit importance (does the game still run?) | rank-correlation of lesion-importance with the unit's *true* functional role; #units flagged "specific" that are actually generic |
| A3 tuning curves | per-unit tuning to luminance/game var | fraction of strongly-tuned units whose tuning ≠ their true role (spurious-tuning rate) |
| A4 correlations | pairwise/global correlation structure | weak-pairwise/strong-global reproduced; vs true coupling |
| A5 LFP | regional power spectra / "rhythms" | are peaks the known clocks (frame/scanline)? %-variance that is epiphenomenal |
| A6 Granger | inferred subsystem causality | false-edge / missed-edge rate vs true data-flow |
| A7 dim-reduction | latent components | matched-component fraction vs known signals (clock, R/W, vsync); interpretable-variance |
| A8 whole-state | descriptive map | baseline only |

**(2) Ideal explanation** — the register-transfer-level account: *"frame-1 of this
game is produced by the display kernel reading RAM cells X through opcodes Y,
driving TIA registers Z; module M implements function f."* (Known exactly.)

**(3) When right** — F: recovered graph/role = true graph/role. S: the recovered
structure predicts held-out lesions/interventions. M: it lands on the module level
(registers/adders), not transistors or epiphenomena. A lesion result is "right"
only if the unit's *true role* explains the broken behavior — not merely "game X
broke."

**(4) Best case** — a *quantified* Kording: every classical method scores **low on
F∧S** despite finding rich structure (e.g., "lesion-importance explains <X%> of
true functional role; tuning is spurious for <Y%> of tuned units"), while the same
*causal* operators (intervention/patching at the chip level) score **high** — proving
the gap is the **method**, not the data or the system. Plus the direct Visual6502
transistor-level head-to-head (parallel track).

---

## 3. Phase B1 — attribution / saliency on agents

Subject: a DQN agent's action/Q. Each method emits an attribution map over inputs.

**(1) Measure** — vs the oracle's true causal map over pixels/objects:
- correlation (Spearman) with the true Δaction/ΔQ map;
- **deletion/insertion AUC measured on the true emulator** (remove top-attributed
  inputs, measure the *actual* effect — not a learned proxy);
- precision@k / pointing-game / object-hit vs the true causal top-k;
- **plausibility** (does it highlight human-sensible objects) for the X-vs-Y plot.

**(2) Ideal explanation** — the minimal set of pixels/objects whose change actually
flips the action, with correct sign/magnitude: the *true causal saliency*
(e.g., Pong: the ball and the paddle, not the score digits or background).

**(3) When right** — F: top-k attributed = true causal top-k. S: deleting them on
the real emulator changes the action as predicted (insertion/deletion faithfulness).
M: sparse, object-level. Explicitly *not* "looks like it's watching the ball"
(plausibility) unless the ball is the true cause *and* the map points there.

**(4) Best case** — a **faithfulness leaderboard with real ground truth**: popular
visual XAI (Grad-CAM, attention, vanilla saliency) sits near-chance on F while high
on plausibility — the measured, ground-truthed form of Atrey 2020 — whereas
IG / occlusion-on-the-true-emulator approach the ceiling. The deliverable claim:
"*plausible ≠ faithful*, and here is the gap, in numbers."

## 4. Phase B2 — mechanistic interpretability on agents (first-class)

Ground truth for the agent = the agent's **own true causal structure**, always
obtainable because we hold the weights *and* can intervene exactly on inputs and
activations. The known game variables are an extra semantic anchor.

**(1) Measure**
- **Activation patching / causal tracing:** patch activations clean↔corrupted;
  score = patched effect vs the **exact** intervention effect (the oracle on
  activations). Recovery of the truly-important components (P/R).
- **Attribution patching (+ edge AP):** gradient approximation; score =
  approximation error vs true patching, and important-edge recovery.
- **Sparse autoencoders:** train on activations; score = **feature ↔ known game
  variable** matching (probing accuracy / MI of a feature for ball-x, lives, …)
  *and* causal use (does patching the feature move the output as predicted?).
- **Circuits + causal scrubbing:** resample the hypothesised-irrelevant parts; if
  behavior is preserved, the hypothesis holds. Score = scrubbing-preserved
  performance + match to the known game logic.

**(2) Ideal explanation** — a circuit/feature-level account of *how the agent
computes its action*: e.g., "feature φ encodes ball-x, feature ψ encodes
paddle-x, circuit C computes ψ−φ and selects up/down" — verified to be the actual
computation.

**(3) When right** — F: recovered features/circuit = the agent's true causal
structure (patching-verified). S: the recovered circuit alone reproduces the
behavior (causal scrubbing passes; ablating the rest is harmless). M: minimal,
monosemantic features, matched to known variables.

**(4) Best case** — mechanistic methods score **high**, giving (a) the **first
validation of mechanistic interpretability against a complex system's ground
truth** (it really recovers the computation), and (b) a **calibration for SAEs**
(features map to known variables at a measured rate, with a measured monosemanticity
ceiling). A genuine contribution *to* mechanistic interpretability, not only a
critique. (Caveat to test: the agent may compute via features that don't cleanly
map to human game variables — itself a reportable finding.)

## 5. Phase D — behavioral / psychology probing

Subject: the agent (primary), the chip (stress test), as a "participant."

**(1) Measure** — psychophysics-style: vary one controlled factor (ball position,
distractor, reward cue, onset timing, masking), read action/Q, fit a behavioral
account (psychometric curve, inferred decision variable, bias, RT-analogue). Then
compare the **inferred** decision variable to the **true** causal driver (oracle).
Key metric: **"right-for-the-wrong-reasons" rate** — good behavioral fit, wrong
driver; plus held-out generalization of the behavioral law to novel stimuli.

**(2) Ideal explanation** — a behavioral law that names the agent's true decision
variables and their functional form (e.g., "P(up) is a logistic in
ball-y − paddle-y").

**(3) When right** — F: inferred variable = true causal driver (intervention-
verified). S: the law predicts behavior on **novel** stimuli (not just the probe
set) — directly answering Shiffrin–Mitchell's context-dependence / "right for the
wrong reasons" / training-contamination concerns. M: the simplest law that holds.

**(4) Best case** — the **first ground-truthed verdict on psychology-of-AI**:
a measured split of behavioral inferences into *trustworthy* (match truth +
generalize) vs *mirage* (fit but wrong driver / confounded), with the conditions
that predict each. E.g., "behaviorally-inferred decision variables match the true
driver X% of the time; failures are confounded by Z."

## 6. Phase C — synthesis (the cross-tradition result)

**(1) Measure** — put every method from A/B1/B2/D on the two shared axes
(faithfulness/sufficiency vs plausibility) with one common scoring protocol.

**(2/3) Ideal/right** — a single comparable F∧S∧M score per method.

**(4) Best case — the paper's headline:**
- **One figure, three traditions:** mechanistic-neuroscience, attribution+mech-interp,
  and behavioral, each scored against one ground truth, on a system complex enough
  to matter. The "plausible-but-wrong" quadrant is populated by the *popular*
  methods; the causal/mechanistic methods occupy the faithful region.
- **A benchmark artifact** (tasks + oracle + metrics) others can run.
- **A directions claim:** the science of understanding AI needs ground-truth
  validation, causal (not correlational) attribution, and method development
  *sieved on known systems first* — with our platform as that sieve.

---

## 7. Master table

| Phase | Measure (output → score) | Ideal explanation | Right when (F∧S∧M) | Best-case outcome |
|---|---|---|---|---|
| **A** chip / neuroscience | finding (graph, lesion map, tuning, components) → agreement with known mechanism | register-transfer account of the frame | recovered structure = true graph/role; predicts held-out lesions; module-level | quantified Kording: classical methods low-F, causal methods high-F — gap is the method |
| **B1** agent / attribution | attribution map → corr + deletion/insertion AUC on true emulator + precision@k | minimal true-causal pixels/objects | top-k = true causal top-k; deletion behaves as predicted; sparse | leaderboard: popular saliency plausible-but-unfaithful; IG/occlusion faithful |
| **B2** agent / mechanistic | patch/SAE/circuit → vs exact patch + feature↔known-var + scrubbing | circuit/feature account of the agent's computation | recovered = agent's true causal structure; circuit reproduces behavior; monosemantic | first ground-truth validation of mech-interp + SAE calibration |
| **D** agent / behavioral | psychometric fit → inferred driver vs true driver + generalization | behavioral law over true decision variables | inferred var = true driver; predicts novel stimuli; simplest | first ground-truthed verdict: trustworthy vs mirage behavioral inferences |
| **C** synthesis | all methods on faithfulness-vs-plausibility | one comparable F∧S∧M score | — | three-tradition headline figure + benchmark + directions |

> Outcomes in §2–§6 are *hypotheses* (our predictions), to be confirmed by the
> experiments — the pilots (`tools/xai_study/*`) test the two riskiest links first:
> the oracle, and one method per tradition.
