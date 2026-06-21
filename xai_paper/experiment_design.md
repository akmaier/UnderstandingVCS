# Experiment design — Paper 2 (the full experiment list)

> **Role of this file:** the *experiments* — what we run and how we score each one.
> The *storyline* and the Nature structure live in
> [`xai_paper_plan.md`](xai_paper_plan.md). See [`README.md`](README.md) for the
> document rules. **Subject in every phase: the Atari VCS itself** (chip + program +
> game logic) — *no learned agents*. Every citation below is **to be verified to
> Paper-1's no-hallucination standard before it enters the bib.**

Contents: §0 correctness criterion · §1 oracle · §2 obtaining T3 · §3 substrate audit ·
§4–§7 Phases A–D (each with an `Analysis | Finding | Measured score` table) · §8 method
matrix · §9 compute · §10 Phase E (deferred) · §11 master table.

---

## 0. When is an explanation "right"? — the correctness triad

An explanation `Ê` of how output `y` arises is **right** iff it is (operationalizing
Barbiero et al. 2025; Marr's levels; Lazebnik's "fix-the-radio"):

- **(F) Faithful** — claimed causes/structure are the *true* causes (intervention-
  verified): precision/recall, sign, magnitude vs the oracle.
- **(S) Sufficient** — predicts behavior under *unseen* interventions / could
  regenerate it: held-out interchange accuracy (causal-abstraction / scrubbing style).
- **(M) Minimal & at the right level** — parsimonious, no spurious parts, at the
  algorithmic level (registers/RAM/opcodes; the program's variables/circuits/decision
  logic), matching the known module decomposition.

`right = F ∧ S ∧ M`. **Two reporting axes (the headline plot):** *faithfulness*
(X, vs ground truth) vs *human-plausibility* (Y). Danger zone = high plausibility, low
faithfulness — where we predict the popular methods land.

---

## 1. The ground-truth oracle (shared measurement instrument)

For any VCS output `y` (a pixel, the score, a game event, a future state) and candidate
cause `u` (ROM byte, RAM cell, register, opcode, joystick input, or an internal state
variable):

- **Causal effect (exact):** intervene on `u` (occlude/clamp/replace/resample), re-run
  deterministically, record Δ`y`. Bit-exact; no world-model assumed. True causal map =
  `{Δy(u)}`.
- **Differential effect:** `∂y/∂u` through the differentiable substrate; raw gradient
  and Integrated Gradients (path integral to a baseline state).
- **Cross-check:** correlation between intervention and gradient; disagreement flags
  non-smooth/saturated points and is reported.
- **Known semantics, in tiers:** **T1** causal (exact, all 64) · **T2** hardware roles
  & data-flow (exact, all 64) · **T3** game-concept labels (ball-x, lives, score) —
  *partial*, see §2; needed only for the *semantic-level* metrics.

Running example: **Pong** — outputs: score, ball-pixel; known variables: ball/paddle
positions, score (RAM); behavioral subject: the **CPU opponent** (its paddle-tracking
routine, in the ROM).

---

## 2. How we obtain T3 (game-concept labels), and what the T3 papers do

Two public sources give *candidate* labels; our substrate makes them *causal* and
extends them.

- **AtariARI** (Anand et al. 2019, NeurIPS — *Atari Annotated RAM Interface*) — **22
  games**. Labels from commented disassemblies/source (Engelhardt & Jentzsch; CPUWIZ)
  locating RAM bytes for sprite positions, score, lives; a gym wrapper emits a label
  per frame. **Used as a linear-probing benchmark** (does a representation encode the
  variables? F1). Caveat: raw RAM (x,y) is *not aligned* to the rendered position
  (render-time offsets).
- **OCAtari** (Delfosse et al. 2023) — **40+ games**, two modes: **RAM mode** (maps RAM
  to objects *with offsets applied*, fixing the misalignment) and **Vision mode**
  (color-filter object detection on the rendered frame); the two cross-validate.
  HackAtari (Delfosse 2024) adds controlled variants.

**Our procedure (the substrate's unique lever):** (1) import OCAtari/AtariARI as
candidates; (2) **verify causally by intervention** — set the byte, re-render, confirm
the object moves to the predicted place on the bit-exact framebuffer (upgrades a
*correlational* label to a *verified causal* one; auto-corrects offsets); (3)
**discover** new labels by RAM↔framebuffer correlation + intervention sweeps; (4)
coverage ~22 → 40+ of our 64, extendable. **Thesis tie-in:** probing shows info is
*present*, not *used* (Hewitt & Liang 2019 control tasks — our A3/Phase-C trap); our
intervention test is strictly stronger.

---

## 3. Substrate-feasibility audit (which phases run on the current substrate)

Substrate = jutari/jaxtari as in Paper 1 (exact forward, exact interventions, full
observability, gradients of any output w.r.t. any input, GPU-batched SOFT-STE). The
subject *is* the substrate, so it is both instrument and subject throughout.

| Phase | Substrate capability required | In current substrate? | Other ingredient | Evaluable now? |
|---|---|---|---|---|
| **A** | full state observability + exact interventions | ✅ | T3 only for A3 game-concept scoring | **Yes** |
| **B** | exact intervention + gradient of an output w.r.t. inputs/state | ✅ (P8 already does this) | T3 only for object/concept attribution | **Yes** |
| **C** | clamp/resample state between runs; record state trajectory; gradients | ✅ | standard SAE training code; T3 for semantic checks | **Yes** |
| **D** | set state + re-render controlled stimuli; read program response | ✅ | T3 / vision for the situation variable | **Yes** |

Cross-cutting (neither is a new substrate): **T3 labels** — external knowledge, but
derivable/verifiable with our own interventions (§2); gates only semantic-level
metrics. **Long-horizon end-to-end gradients** (credit through many steps) — the one
item that could stress the substrate; the core single-step metrics don't need it.
**Bottom line: all phases are evaluable on the current substrate;** the cluster is for
*scale* (§9), not capability.

---

## 4. Phase A — neuroscience / mechanistic battery on the VCS (Kording, quantified)

| Analysis | Finding (what the method outputs) | Measured score |
|---|---|---|
| **A1** connectomics / data-flow graph | recovered dependency graph over state vars | precision/recall + graph-edit-distance vs the *true* read/write graph |
| **A2** single-unit lesions | per-unit importance map (boot/run still works?) | rank-correlation of importance with the unit's *true* role; #units flagged "specific" that are generic |
| **A3** tuning curves | per-unit tuning to luminance / a game variable | spurious-tuning rate (strongly-tuned units whose tuning ≠ true role) [T3 for game-var version] |
| **A4** spike-word / pairwise correlations | pairwise + global correlation structure | weak-pairwise/strong-global reproduced; vs true coupling |
| **A5** local field potentials | regional pooled-activity power spectra | %-variance that is the known clocks (frame/scanline) → epiphenomenal |
| **A6** Granger causality | inferred CPU/TIA/RIOT causal edges | false-edge / missed-edge rate vs true data-flow |
| **A7** dim-reduction (NMF/PCA) | latent components of the state tensor | matched-component fraction vs known signals (clock, R/W, vsync) |
| **A8** whole-state recording | full RAM+register state map over time | descriptive baseline |

**Ideal explanation:** the register-transfer account of the frame (known exactly).
**Right when:** recovered graph/role = true (F); predicts held-out lesions (S);
module-level (M). **Best case:** a *quantified* Kording — classical methods low-F
despite rich structure, while the causal operators (§6) score high → the gap is the
*method*. (Parallel track: Visual6502 transistor-level head-to-head.)

## 5. Phase B — attribution / XAI on the VCS's input→output computation

Explain a chosen output `y` (pixel/score/event) from its causes (ROM/RAM/registers/
inputs). Score every map against the §1 oracle.

| Analysis (named method) | Finding (output) | Measured score |
|---|---|---|
| Vanilla gradient (Simonyan et al. 2014) | saliency map over causes | corr + deletion/insertion AUC on the true VCS + precision@k vs true causal top-k |
| Grad×Input / DeepLIFT (Shrikumar et al. 2017) | attribution map | as above (+ completeness where defined) |
| Guided Backprop (Springenberg et al. 2015) | saliency map | as above + sanity-check pass/fail (Adebayo et al. 2018) |
| SmoothGrad (Smilkov et al. 2017) | noise-averaged saliency | as above |
| **Integrated Gradients** (Sundararajan et al. 2017; our P8) | path-integrated attribution | corr + del/ins AUC + completeness; baseline-sensitivity sweep |
| Expected Gradients (Erion et al. 2021) | baseline-averaged IG | as above |
| Occlusion (Zeiler & Fergus 2014) | Δ`y` per occluded region | del/ins AUC; ≈ a coarse intervention oracle |
| Meaningful / extremal perturbation (Fong & Vedaldi 2017; Fong et al. 2019) | learned minimal mask | mask IoU vs the true causal set; del/ins |
| RISE (Petsiuk et al. 2018) | randomized-mask saliency | corr + del/ins |
| LIME (Ribeiro et al. 2016) | local linear weights | corr vs true map; stability across seeds |
| KernelSHAP / Shapley sampling (Lundberg & Lee 2017; Štrumbelj & Kononenko 2014) | Shapley values | corr vs true; convergence vs compute |
| On-distribution counterfactual (state-set + re-render; cf. Olson 2021; Atrey 2020) | minimal valid counterfactual edit | validity (re-renders) + minimality vs true minimal set |
| **N/A** Grad-CAM/++ (Selvaraju 2017; Chattopadhyay 2018), attention rollout (Abnar & Zuidema 2020), VIPER (Bastani 2018) | needs NN layers / a policy | *does not apply* — recorded finding |

**Ideal:** the minimal true-causal causes of `y`, with sign/magnitude. **Right when:**
top-k = true causal top-k (F); deletion behaves as predicted (S); sparse (M).
**Best case:** a faithfulness leaderboard — plausible ≠ faithful, in numbers, on the
system itself.

## 6. Phase C — mechanistic interpretability on the VCS (known-circuit testbed)

The state trajectory = "activations"; the program's data-flow = the "circuit" — both
known. Ground truth = T1/T2 (always) + T3 (where labeled).

| Analysis (named method) | Finding (output) | Measured score |
|---|---|---|
| Activation patching / causal mediation (Vig et al. 2020; ROME causal tracing, Meng et al. 2022) | per-site causal effect | recovered effect vs the *exact* patch; important-site P/R vs true data-flow |
| Interchange interventions / DAS (Geiger et al. 2021, 2023, 2024) | aligned causal variables | interchange accuracy; alignment vs the true variable |
| Attribution patching / edge AP (Nanda 2023; Syed et al. 2023) | gradient-approx site/edge effects | approximation error vs true patching; edge P/R |
| Path patching / IOI circuit (Wang et al. 2022; Goldowsky-Dill et al. 2023) | recovered circuit (path set) | circuit P/R vs the true routine |
| ACDC — automatic circuit discovery (Conmy et al. 2023) | auto-discovered circuit graph | edge P/R + scrubbing-preserved performance vs true data-flow |
| Sparse autoencoders (Cunningham et al. 2023; Bricken et al. 2023; Templeton et al. 2024) | learned features over state | feature↔known-variable match (probe F1/MI) + causal use (patch effect) + monosemanticity |
| NMF/PCA dictionaries | latent components | matched-component fraction vs known variables |
| Causal scrubbing (Chan et al. 2022) | hypothesis pass/fail | scrubbing-preserved performance vs the true routine |
| Linear probing + control tasks (Alain & Bengio 2017; Hewitt & Liang 2019) | concept decodability | accuracy **and** selectivity (probe − control) → present-vs-used gap |
| Logit / tuned lens (nostalgebraist 2020; Belrose et al. 2023) | per-stage readout of state | readout fidelity vs the true intermediate value |

**Ideal:** a circuit/feature account of *how the program computes `y`*, verified
against the disassembly. **Right when:** recovered = true data-flow (F); the circuit
alone reproduces behavior (S); minimal/monosemantic (M). **Best case:** the first
validation of mech-interp against a *known* circuit + an SAE calibration (feature↔
variable rate, monosemanticity ceiling).

## 7. Phase D — behavioral / psychology probing of the game's own logic

Subject: a game's built-in decision logic (e.g., Pong's CPU opponent), as a
"participant." Present controlled situations by state-set + re-render; read the
program's response.

| Analysis (named method) | Finding (output) | Measured score |
|---|---|---|
| Psychometric-function fitting (Wichmann & Hill 2001) | response curve vs a stimulus factor (threshold, slope) | does the inferred factor = the true driver? threshold vs the coded boundary |
| Method of constant stimuli / adaptive staircases | decision-boundary estimate | vs the true boundary in the code |
| Signal-detection theory (Green & Swets 1966) | d′, criterion | vs the true (deterministic) decision rule |
| Reverse correlation / classification images (Ahumada 1971; Murray 2011) | recovered stimulus "template" | overlap with the true input region the code reads |
| Drift-diffusion / sequential sampling (Ratcliff 1978) | inferred decision variable + "frames-to-respond" | inferred variable vs true driver; latency vs the true code latency |
| Ideal-observer analysis | the optimal-strategy account | gap to the actual (often suboptimal) coded strategy |
| Cognitive-task battery, Binz & Schulz (2023) style; Shiffrin & Mitchell (2023) caveats | behavioral "trait" inferences | "right-for-the-wrong-reasons" rate vs the code; generalization to novel stimuli |

**Ideal:** a behavioral law over the program's true decision variables. **Right when:**
inferred variable = true driver (F); predicts novel stimuli (S); simplest law (M).
**Best case:** the first ground-truthed verdict on behavioral-probing methodology —
trustworthy vs mirage, with the conditions that predict each.

---

## 8. Method matrix — expected success vs failure (a prediction to test)

| Method (category) | Applies to VCS? | Expected vs ground truth | Why |
|---|---|---|---|
| Activation patching / causal tracing | ✓ | **Succeed** | causal by construction; comparable to the oracle |
| Causal scrubbing | ✓ | **Succeed** | we *have* ground-truth hypotheses to scrub |
| Integrated Gradients | ✓ | **Succeed/partial** | completeness + exact-forward gradients; baseline-sensitive |
| Occlusion on the true VCS | ✓ | **Succeed/partial** | faithful when the perturbation is a valid intervention |
| On-distribution counterfactuals | ✓ | **Partial→Succeed** | set-state-and-re-render removes Atrey's off-manifold objection |
| Attribution / edge patching | ✓ | **Partial** | cheap approximation; score the approx error vs true patching |
| Sparse autoencoders | ✓ | **Partial** | recovers features; faithfulness-to-computation debated |
| Circuit discovery (ACDC, path patching) | ✓ | **Partial→Succeed** | goal state = our ground truth; measure recovered vs true |
| Linear probing | ✓ | **Partial / misleading** | present ≠ used — the tuning-curve trap |
| SHAP / LIME | ✓ | **Partial/Fail** | baseline/sampling-dependent; approximations diverge |
| Vanilla gradient / saliency | ✓ | **Fail** | noisy, gradient-shattering, non-causal |
| Grad-CAM/++ , attention, VIPER | **✗ N/A** | — | need conv maps / attention / a learned policy |

Headline expectation: **causal/gradient/mechanistic methods pass; correlational and
architecture-specific methods fail or don't apply** — a measured sharpening of
Kording's lesson and of Atrey 2020.

## 9. Compute plan

Pilots local; full runs on the LME cluster. Enabler: SOFT-STE forward is bit-exact to
HARD and GPU-batchable (`vmap`), so the oracle and gradients run *batched on GPU*.

| Workload | Where |
|---|---|
| All four pilots (1 game each: A lesion/tuning/dim-red · B IG-vs-oracle · C patching+SAE · D one psychophysics probe) | **local** (M1 Max) |
| Full lesion + occlusion sweeps (RAM × ROM × opcodes × outputs × games) | **cluster (CPU array / GPU)** |
| Batched gradients / IG over outputs × games | **cluster (GPU)** |
| SAE training on recorded state trajectories | **cluster (GPU)** |

Reuse Paper-1 setup: LME Slurm, `/cluster/maier`, `tools/cluster/*.sbatch`, the jaxtari
GPU venv. No agents → no agent training/zoo.

## 10. Phase E — synthesis (deferred by design)

Designed **once results are in**. It will: place every method on the shared
faithfulness-vs-plausibility axes (the headline figure), package the scoring suite as a
reusable **benchmark**, demonstrate a faithful method, and articulate directions.
**Semantics (T3) will be central here** (the cross-tradition comparison leans on the
game-concept level). Not specified in advance.

## 11. Master table

| Phase | Measure (output → score) | Ideal | Right when (F∧S∧M) | Best case |
|---|---|---|---|---|
| **A** neuroscience | finding → agreement with known mechanism | register-transfer account | recovered = true; predicts held-out lesions; module-level | quantified Kording: classical low-F, causal high-F |
| **B** attribution | attribution map → corr + del/ins AUC + precision@k | minimal true-causal causes | top-k = true; deletion as predicted; sparse | leaderboard: plausible ≠ faithful |
| **C** mechanistic | patch/SAE/circuit → vs exact patch + feature↔var + scrubbing | circuit/feature account | recovered = true data-flow; reproduces behavior; monosemantic | first mech-interp validation on a known circuit |
| **D** behavioral | psychometric fit → inferred driver vs true code + generalization | behavioral law over true variables | inferred = true driver; predicts novel stimuli; simplest | first ground-truthed verdict: trustworthy vs mirage |
| **E** synthesis | (deferred) all methods on faithfulness-vs-plausibility | one comparable F∧S∧M score | — | headline figure + benchmark + directions; semantics central |

> Outcomes in §4–§7 and §10 are *hypotheses* (predictions), confirmed by the
> experiments. Pilots test the riskiest links first: the oracle, and one method per
> phase. Every phase runs on the current substrate (§3).
