# Experiment design — Paper 2 (the full experiment list)

> **Role of this file:** the *experiments* — what we run and how we score each one.
> The *storyline*, the Nature structure, and the **Discussion / direction** (incl. the
> general-T3-methods survey and the software-reverse-engineering framing) live in
> [`xai_paper_plan.md`](xai_paper_plan.md). See [`README.md`](README.md) for the
> document rules. **Subject in every phase: the Atari VCS itself** (chip + program +
> game logic) — *no learned agents*. Every citation below is **to be verified to
> Paper-1's no-hallucination standard before it enters the bib.**

Contents: §0 correctness criterion · §1 oracle · §2 obtaining T3 · §3 substrate audit ·
§4–§7 Phases A–D · §8 Phase E (semantic recovery) · §9 method matrix · §10 compute ·
§11 master table. (Each phase leads with an `Analysis | Finding | Measured score` table.)

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

> Note: the **general** question — "are there general methods for obtaining T3, and
> which direction is most promising?" — and the **software-reverse-engineering
> framing** of T3 are *direction/discussion*, so they live in
> [`xai_paper_plan.md`](xai_paper_plan.md) (Discussion), not here. This §2 is the
> concrete procedure *we run* to obtain our labels; the *recovery as an experiment*
> is Phase E (§8).

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
| **E** | trace logging; **membership + equivalence queries** (exact teacher); the ROM binary | ✅ (the bit-exact emulator is an exact MAT teacher) | RE tooling (decompiler, Daikon, L* lib); T3 = the *target* to score against | **Yes** |

Cross-cutting (neither is a new substrate): **T3 labels** — external knowledge, but
derivable/verifiable with our own interventions (§2); gates the semantic-level metrics
(and is the *target* of Phase E). **Long-horizon end-to-end gradients** (credit through
many steps) — the one item that could stress the substrate; the core single-step
metrics don't need it. **Bottom line: all phases are evaluable on the current
substrate;** the cluster is for *scale* (§10), not capability.

---

## 4. Phase A — neuroscience / mechanistic battery on the VCS (Kording, quantified)

| Analysis | Finding (what the method outputs) | Measured score | Needs T3? |
|---|---|---|---|
| **A1** connectomics / data-flow graph | recovered dependency graph over state vars | precision/recall + graph-edit-distance vs the *true* read/write graph | **No** (T1/T2) |
| **A2** single-unit lesions | per-unit importance map (boot/run still works?) | rank-correlation of importance with the unit's *true* role; #units flagged "specific" that are generic | **No** (T1 causal role); game-concept role naming optional |
| **A3** tuning curves | per-unit tuning to luminance / a game variable | spurious-tuning rate (strongly-tuned units whose tuning ≠ true role) | **Game-variable version only** (scoring); hardware-tuning version: No |
| **A4** spike-word / pairwise correlations | pairwise + global correlation structure | weak-pairwise/strong-global reproduced; vs true coupling | **No** (T1) |
| **A5** local field potentials | regional pooled-activity power spectra | %-variance that is the known clocks (frame/scanline) → epiphenomenal | **No** (T2 clocks) |
| **A6** Granger causality | inferred CPU/TIA/RIOT causal edges | false-edge / missed-edge rate vs true data-flow | **No** (T1) |
| **A7** dim-reduction (NMF/PCA) | latent components of the state tensor | matched-component fraction vs known signals (clock, R/W, vsync) | **No** (T2 signals); matching to game variables optional |
| **A8** whole-state recording | full RAM+register state map over time | descriptive baseline | **No** |

**Ideal explanation:** the register-transfer account of the frame (known exactly).
**Right when:** recovered graph/role = true (F); predicts held-out lesions (S);
module-level (M). **Best case:** a *quantified* Kording — classical methods low-F
despite rich structure, while the causal operators (§6) score high → the gap is the
*method*. (Parallel track: Visual6502 transistor-level head-to-head.)

## 5. Phase B — attribution / XAI on the VCS's input→output computation

Explain a chosen output `y` (pixel/score/event) from its causes (ROM/RAM/registers/
inputs). Score every map against the §1 oracle.

| Analysis (named method) | Finding (output) | Measured score | Needs T3? |
|---|---|---|---|
| Vanilla gradient (Simonyan et al. 2014) | saliency map over causes | corr + deletion/insertion AUC on the true VCS + precision@k vs true causal top-k | No* |
| Grad×Input / DeepLIFT (Shrikumar et al. 2017) | attribution map | as above (+ completeness where defined) | No* |
| Guided Backprop (Springenberg et al. 2015) | saliency map | as above + sanity-check pass/fail (Adebayo et al. 2018) | No* |
| SmoothGrad (Smilkov et al. 2017) | noise-averaged saliency | as above | No* |
| **Integrated Gradients** (Sundararajan et al. 2017; our P8) | path-integrated attribution | corr + del/ins AUC + completeness; baseline-sensitivity sweep | No* |
| Expected Gradients (Erion et al. 2021) | baseline-averaged IG | as above | No* |
| Occlusion (Zeiler & Fergus 2014) | Δ`y` per occluded region | del/ins AUC; ≈ a coarse intervention oracle | No* |
| Meaningful / extremal perturbation (Fong & Vedaldi 2017; Fong et al. 2019) | learned minimal mask | mask IoU vs the true causal set; del/ins | No* (object-set IoU needs T3) |
| RISE (Petsiuk et al. 2018) | randomized-mask saliency | corr + del/ins | No* |
| LIME (Ribeiro et al. 2016) | local linear weights | corr vs true map; stability across seeds | No* |
| KernelSHAP / Shapley sampling (Lundberg & Lee 2017; Štrumbelj & Kononenko 2014) | Shapley values | corr vs true; convergence vs compute | No* |
| On-distribution counterfactual (state-set + re-render; cf. Olson 2021; Atrey 2020) | minimal valid counterfactual edit | validity (re-renders) + minimality vs true minimal set | **Object-level: yes** (which bytes = the object); pixel-level: no |
| **N/A** Grad-CAM/++ (Selvaraju 2017; Chattopadhyay 2018), attention rollout (Abnar & Zuidema 2020), VIPER (Bastani 2018) | needs NN layers / a policy | *does not apply* — recorded finding | — |

> **\* T3 enters Phase B only at the *object level*.** The faithfulness metrics —
> correlation, deletion/insertion AUC, and precision@k over *cells/pixels* — score
> against the causal map (T1) and need **no T3**. T3 is required only for the
> *object-level read-outs* (object-hit / pointing-game, "the map points to the ball")
> and for *object-level* counterfactuals (removing a named object) — i.e., to name the
> causes, not to find them.

**Ideal:** the minimal true-causal causes of `y`, with sign/magnitude. **Right when:**
top-k = true causal top-k (F); deletion behaves as predicted (S); sparse (M).
**Best case:** a faithfulness leaderboard — plausible ≠ faithful, in numbers, on the
system itself.

## 6. Phase C — mechanistic interpretability on the VCS (known-circuit testbed)

The state trajectory = "activations"; the program's data-flow = the "circuit" — both
known. Ground truth = T1/T2 (always) + T3 (where labeled).

| Analysis (named method) | Finding (output) | Measured score | Needs T3? |
|---|---|---|---|
| Activation patching / causal mediation (Vig et al. 2020; ROME causal tracing, Meng et al. 2022) | per-site causal effect | recovered effect vs the *exact* patch; important-site P/R vs true data-flow | **No** (T1/T2); game-name optional |
| Interchange interventions / DAS (Geiger et al. 2021, 2023, 2024) | aligned causal variables | interchange accuracy; alignment vs the true variable | **If aligning to a game-concept variable: yes** (the target variable); hardware variable: No |
| Attribution patching / edge AP (Nanda 2023; Syed et al. 2023) | gradient-approx site/edge effects | approximation error vs true patching; edge P/R | **No** (T1) |
| Path patching / IOI circuit (Wang et al. 2022; Goldowsky-Dill et al. 2023) | recovered circuit (path set) | circuit P/R vs the true routine | **No** (T1); game-name optional |
| ACDC — automatic circuit discovery (Conmy et al. 2023) | auto-discovered circuit graph | edge P/R + scrubbing-preserved performance vs true data-flow | **No** (T1/T2) |
| Sparse autoencoders (Cunningham et al. 2023; Bricken et al. 2023; Templeton et al. 2024) | learned features over state | feature↔known-variable match (probe F1/MI) + causal use (patch effect) + monosemanticity | **Game-variable matching: yes**; hardware-signal match + causal-use: No |
| NMF/PCA dictionaries | latent components | matched-component fraction vs known variables | **Game-variable matching: yes**; hardware signals: No |
| Causal scrubbing (Chan et al. 2022) | hypothesis pass/fail | scrubbing-preserved performance vs the true routine | **No** (T1) |
| Linear probing + control tasks (Alain & Bengio 2017; Hewitt & Liang 2019) | concept decodability | accuracy **and** selectivity (probe − control) → present-vs-used gap | **Target** — the probe *needs* the concept labels (T3 for game concepts; T2 for hardware) |
| Logit / tuned lens (nostalgebraist 2020; Belrose et al. 2023) | per-stage readout of state | readout fidelity vs the true intermediate value | **No** (T1/T2) |

> **T3 in Phase C** is needed for the *semantic* layer only: naming a recovered
> feature/circuit/variable in game-concept terms (SAE/NMF game-variable matching;
> probing *for* a game concept; DAS aligned to a game variable). The *causal* core —
> patching/scrubbing/circuit recovery scored against the exact patch and the true
> data-flow — runs on T1/T2 alone. (Probing is the one method that *requires* labels
> as input, by definition.)

**Ideal:** a circuit/feature account of *how the program computes `y`*, verified
against the disassembly. **Right when:** recovered = true data-flow (F); the circuit
alone reproduces behavior (S); minimal/monosemantic (M). **Best case:** the first
validation of mech-interp against a *known* circuit + an SAE calibration (feature↔
variable rate, monosemanticity ceiling).

## 7. Phase D — behavioral / psychology probing of the game's own logic

Subject: a game's built-in decision logic (e.g., Pong's CPU opponent), as a
"participant." Present controlled situations by state-set + re-render; read the
program's response. **Phase D is the semantic-grounding bridge: it establishes the
symbol→concept link that feeds the documentation recovered in Phase E.**

| Analysis (named method) | Finding (output) | Measured score | Needs T3? |
|---|---|---|---|
| Psychometric-function fitting (Wichmann & Hill 2001) | response curve vs a stimulus factor (threshold, slope) | does the inferred factor = the true driver? threshold vs the coded boundary | **Yes — stimulus + scoring** |
| Method of constant stimuli / adaptive staircases | decision-boundary estimate | vs the true boundary in the code | **Yes — stimulus + scoring** |
| Signal-detection theory (Green & Swets 1966) | d′, criterion | vs the true (deterministic) decision rule | **Yes — stimulus + scoring** |
| Reverse correlation / classification images (Ahumada 1971; Murray 2011) | recovered stimulus "template" | overlap with the true input region the code reads | **No** (template vs the code's *read-set* = T1/T2); naming = T3 |
| Drift-diffusion / sequential sampling (Ratcliff 1978) | inferred decision variable + "frames-to-respond" | inferred variable vs true driver; latency vs the true code latency | **Yes — scoring** (name the driver); stimulus optional |
| Ideal-observer analysis | the optimal-strategy account | gap to the actual (often suboptimal) coded strategy | **Yes — scoring** (the coded strategy's variables) |
| Cognitive-task battery, Binz & Schulz (2023) style; Shiffrin & Mitchell (2023) caveats | behavioral "trait" inferences | "right-for-the-wrong-reasons" rate vs the code; generalization to novel stimuli | **Yes — stimulus + scoring** |

> **Phase D is the most T3-dependent**: to vary a *named* game factor we set the
> corresponding RAM byte (needs T3), and to score "inferred driver = true driver" we
> must name the true driver (T3). The exception is *reverse correlation*, whose
> recovered template can be scored against the code's read-set (T1/T2 data-flow) with
> no labels. Where T3 is missing, stimuli can still be varied via raw inputs (joystick)
> or vision-selected frames, at the cost of a coarser "factor."

**Ideal:** a behavioral law over the program's true decision variables. **Right when:**
inferred variable = true driver (F); predicts novel stimuli (S); simplest law (M).
**Best case:** the first ground-truthed verdict on behavioral-probing methodology —
trustworthy vs mirage, with the conditions that predict each.

## 8. Phase E — semantic recovery (design recovery): reconstructing the documentation

The constructive goal. The IEEE definition of *software* (programs + procedures +
**documentation** + data) sets the **bar**: to understand the VCS program is to
*recover its documentation/design*, not merely describe its behavior. The deployed ROM
is software with that documentation stripped, so recovery = **reverse engineering /
design recovery** (Chikofsky & Cross 1990), which by definition needs *external
information added to observations* — supplied here by Phase D's behavioral grounding and
the exact oracle. We combine A–D and the software-RE toolkit to reconstruct the design
and **score the recovered documentation against the truth we hold**. This is framed
honestly as *measuring the gap to the bar*, not as "solving recovery."

| Analysis (named method) | Finding (output) | Measured score | Needs T3? |
|---|---|---|---|
| **E1** Variable / data-dictionary recovery — binary type/var recovery (TIE, Lee et al. 2011; Howard, Slowinska et al. 2011; REWARDS, Lin et al. 2010) + neural name/type recovery (DEBIN, He et al. 2018; DIRE, Lacomis et al. 2019; DIRTY, Chen et al. 2022) + **Phase-D behavioral grounding** | candidate RAM/register → concept/type map (a "data dictionary") | precision/recall + type-accuracy vs the *true* data dictionary (T3 truth we hold); fraction grounded *observationally* | **This IS T3 recovery** — scored vs held-out T3 |
| **E2** Routine / concept recovery — concept assignment (Biggerstaff et al. 1993/94); neural function naming (Nero, David et al. 2020) + **Phase-C circuits** | "this routine = ball-bounce / opponent-AI" labels | match to the true disassembled routine; behavior-equivalence under scrubbing | naming = T3; structure = T1/T2 |
| **E3** Specification / behavior recovery — dynamic invariant detection (Daikon, Ernst et al. 2007); specification mining (Ammons et al. 2002); **active automata / model learning** (Angluin 1987 L*; Vaandrager 2017) | inferred invariants / state machine / behavioral spec | held-out behavioral prediction vs true rules; invariant P/R; learned-automaton equivalence (**exact**, via the bit-exact teacher) | **No** for the behavioral spec (vs true rules = T1/T2); naming states/vars = T3 |
| **E4** Redocumentation / decompilation & gap-to-bar — decompilation (Ghidra/Hex-Rays; LLM4Decompile 2024; Pearce et al. 2022) assembling E1–E3 | human-readable design doc | **fraction of the IEEE "software" recovered** (documentation %); correctness vs the true design | naming = T3 |

**Why this is uniquely possible here.** (i) *Active automata learning* (E3) needs a
"minimally adequate teacher" answering membership **and** equivalence queries — our
bit-exact emulator answers both *exactly* (equivalence against the reference), which is
normally the hard part. (ii) Daikon-style invariant detection (E3) is the behavioral/
observational link of point (b), made concrete. (iii) We hold the *true* design, so we
can score genuine **recovery** — unlike the field's workaround of **reimplementation**
(AtariARI hand-reads disassemblies; OCAtari hand-codes extractors; JAXAtari fully
reimplements), which *re-authors* the documentation rather than recovering it, and is
circular as a ground truth.

**Ideal:** the recovered documentation == the true design (the IEEE "software" made
whole). **Right when:** recovered dictionary/routines/spec match truth (F), predict
held-out behavior (S), at the design level (M). **Best case:** a measured **recovery
rate toward the bar** showing (i) behavioral grounding is *necessary* to cross from
operational to design semantics, and (ii) current methods fall short of full recovery —
with our substrate as the first *non-circular* yardstick. (The cross-tradition
comparison of A–E on the shared faithfulness axes — the former "synthesis" — is part of
the **Results reporting**, not a separate phase.)

---

## 9. Method matrix — expected success vs failure (a prediction to test)

| Method (category) | Applies to VCS? | Expected vs ground truth | Why |
|---|---|---|---|
| Activation patching / causal tracing (C) | ✓ | **Succeed** | causal by construction; comparable to the oracle |
| Causal scrubbing (C) | ✓ | **Succeed** | we *have* ground-truth hypotheses to scrub |
| Integrated Gradients (B) | ✓ | **Succeed/partial** | completeness + exact-forward gradients; baseline-sensitive |
| Occlusion on the true VCS (B) | ✓ | **Succeed/partial** | faithful when the perturbation is a valid intervention |
| On-distribution counterfactuals (B) | ✓ | **Partial→Succeed** | set-state-and-re-render removes Atrey's off-manifold objection |
| Attribution / edge patching (C) | ✓ | **Partial** | cheap approximation; score the approx error vs true patching |
| Sparse autoencoders (C) | ✓ | **Partial** | recovers features; faithfulness-to-computation debated |
| Circuit discovery (ACDC, path patching) (C) | ✓ | **Partial→Succeed** | goal state = our ground truth; measure recovered vs true |
| Linear probing (C) | ✓ | **Partial / misleading** | present ≠ used — the tuning-curve trap |
| SHAP / LIME (B) | ✓ | **Partial/Fail** | baseline/sampling-dependent; approximations diverge |
| Vanilla gradient / saliency (B) | ✓ | **Fail** | noisy, gradient-shattering, non-causal |
| Grad-CAM/++ , attention, VIPER (B) | **✗ N/A** | — | need conv maps / attention / a learned policy |
| **Active automata / model learning** (E) | ✓ | **Succeed** (finite-state parts) | exact MAT teacher (membership + equivalence) from the bit-exact emulator |
| **Dynamic invariant detection — Daikon** (E) | ✓ | **Partial→Succeed** | recovers true invariants from traces; coverage-limited to what traces exercise |
| **Binary type/variable recovery — TIE/DIRTY** (E) | ✓ | **Partial** | recovers types/structure; *names* are guesses (no intent in the binary) |
| **Decompilation (+LLM)** (E) | ✓ | **Partial** | recovers code structure, not intent; redocumentation hallucination risk |
| **Concept assignment / feature location** (E) | ✓ | **Partial / hard** | the AI-hard mapping; needs the behavioral anchor (D) |

Headline expectation: **causal/gradient/mechanistic methods pass; correlational and
architecture-specific methods fail or don't apply; recovery (E) succeeds for behavior/
structure but stalls on *intent/names* without behavioral grounding** — a measured
sharpening of Kording's lesson and of Atrey 2020, and a measured statement of how far
reverse engineering gets toward the documentation bar.

## 10. Compute plan

Pilots local; full runs on the LME cluster. Enabler: SOFT-STE forward is bit-exact to
HARD and GPU-batchable (`vmap`), so the oracle and gradients run *batched on GPU*.

| Workload | Where |
|---|---|
| All five pilots (1 game each: A lesion/tuning/dim-red · B IG-vs-oracle · C patching+SAE · D one psychophysics probe · E invariant-mining + L* on one routine) | **local** (M1 Max) |
| Full lesion + occlusion sweeps (RAM × ROM × opcodes × outputs × games) | **cluster (CPU array / GPU)** |
| Batched gradients / IG over outputs × games | **cluster (GPU)** |
| SAE training on recorded state trajectories | **cluster (GPU)** |
| Phase-E recovery (trace logging for Daikon; query-driven automata learning; decompilation) × games | **cluster (CPU array)** — many short deterministic re-runs / queries |

Reuse Paper-1 setup: LME Slurm, `/cluster/maier`, `tools/cluster/*.sbatch`, the jaxtari
GPU venv. No agents → no agent training/zoo.

## 11. Master table

| Phase | Measure (output → score) | Ideal | Right when (F∧S∧M) | Best case |
|---|---|---|---|---|
| **A** neuroscience | finding → agreement with known mechanism | register-transfer account | recovered = true; predicts held-out lesions; module-level | quantified Kording: classical low-F, causal high-F |
| **B** attribution | attribution map → corr + del/ins AUC + precision@k | minimal true-causal causes | top-k = true; deletion as predicted; sparse | leaderboard: plausible ≠ faithful |
| **C** mechanistic | patch/SAE/circuit → vs exact patch + feature↔var + scrubbing | circuit/feature account | recovered = true data-flow; reproduces behavior; monosemantic | first mech-interp validation on a known circuit |
| **D** behavioral | psychometric fit → inferred driver vs true code + generalization | behavioral law over true variables | inferred = true driver; predicts novel stimuli; simplest | first ground-truthed verdict: trustworthy vs mirage |
| **E** semantic recovery | recovered data-dictionary/routines/spec → vs the true design | the recovered documentation = the true design (IEEE "software" whole) | recovered = true (F); predicts held-out behavior (S); design-level (M) | a measured recovery rate to the bar; behavioral grounding shown necessary; first non-circular yardstick |

> Outcomes in §4–§8 are *hypotheses* (predictions), confirmed by the experiments.
> Pilots test the riskiest links first: the oracle, and one method per phase. Every
> phase runs on the current substrate (§3). The general-T3-methods survey and the
> software-RE direction are in `xai_paper_plan.md` (Discussion).
