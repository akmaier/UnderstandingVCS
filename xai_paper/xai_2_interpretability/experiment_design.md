# P2 — Experiment design (Phases A, B, C) + the SHARED oracle / T3 / triad

> **Role of this file:** the experiments and how each is scored. Storyline → `plan.md`;
> program map → [`../general_paper_plan.md`](../general_paper_plan.md).
>
> **⚠ Testbed redesign (READ BEFORE RE-RUNNING).** The committed Phase-A/B/C runners
> analyse **boot/attract NOOP states** at a fixed `target_frame=120, horizon=30` with an
> all-NOOP action stream, route gradient methods through the **naive** (vanishing) path,
> attribute over **RAM cells only**, and let different methods explain **different outputs**
> — five design flaws that break comparability and starve the oracle. The corrected
> protocol (shared **random-action gameplay** states with boot < 10%, an oracle
> **cause-density gate**, a single **shared screen-buffer output** per game, **soft-mode /
> bilinear-sampler** gradients for position, and direct **screen-buffer gradient maps**),
> the confirmation of each flaw against the code/paper, and the full "experiments to repeat"
> table live in [`experiment_redesign.md`](experiment_redesign.md). The **§0 triad, §1
> oracle, §2 T3, the 6 core games, seed=0 determinism, and the shared-testbed principle
> below are unchanged** — only the *states* and the *shared explained output* change.
> **§0–§3 are the SHARED foundation** (correctness triad, ground-truth oracle, T3
> procedure, substrate audit) — P3 (`../xai_3_psychology/`) and P4
> (`../xai_4_recovery/`) **reference these rather than repeating them.** Subject: the
> **Atari VCS itself** (no agents; that is P5). Every citation is **to be verified to
> Paper-1's no-hallucination standard before it enters the bib.**

Contents: §0 correctness triad · §1 oracle · §2 obtaining T3 · §3 substrate audit ·
§4 Phase A · §5 Phase B · §6 Phase C · §7 method matrix · §8 compute · §9 master table.

---

## 0. When is an explanation "right"? — the correctness triad  *(shared)*

An explanation `Ê` of how output `y` arises is **right** iff it is (operationalizing
ground-truth recovery; Marr's levels; Lazebnik's "fix-the-radio"):

- **(F) Faithful** — claimed causes/structure are the *true* causes (intervention-
  verified): precision/recall, sign, magnitude vs the oracle (§1).
- **(S) Sufficient** — predicts behavior under *unseen* interventions / could regenerate
  it. **Scored against the exact intervention oracle (§1), not against any
  interpretability method:** apply a held-out intervention, predict `y` from `Ê`,
  compare to the true `y` on the bit-exact re-run. *(This avoids circularity — do not
  define S via scrubbing/interchange, which are themselves Phase-C methods under test.)*
- **(M) Minimal & at the right level** — parsimonious, no spurious parts, at the
  algorithmic level (registers/RAM/opcodes; the program's variables/circuits), matching
  the known module decomposition.

`right = F ∧ S ∧ M`. **Two reporting axes (the headline plot):** *faithfulness*
(X, vs ground truth) vs *human-plausibility* (Y). Danger zone = high plausibility, low
faithfulness.

---

## 1. The ground-truth oracle (shared measurement instrument)  *(shared)*

For any VCS output `y` (pixel, score, game event, future state) and candidate cause `u`
(ROM byte, RAM cell, register, opcode, joystick input, or internal state variable):

- **Causal effect (exact) — the primary oracle.** Intervene on `u`
  (occlude/clamp/replace/resample), re-run deterministically, record Δ`y`. Bit-exact;
  no world-model assumed. True causal map = `{Δy(u)}`.
- **Differential effect (companion, content-path only).** `∂y/∂u` through the
  differentiable substrate. **Caveat (from Paper 1):** the STE gradient routes to the
  *content* path (register/colour/graphics-bit values); the *discrete index* (e.g.
  sprite position via strobe timing) has **zero naive gradient** and needs the bilinear
  sampler. So the gradient is the oracle's companion **only for content outputs**; for
  *position/index/event* outputs the **intervention oracle is the sole ground truth**,
  and gradient methods are evaluated *as methods under test*, not as oracle.
- **Cross-check:** correlation between intervention and gradient on the content path;
  disagreement (non-smooth/saturated/index points) is reported, not hidden.
- **Known semantics, in tiers:** **T1** causal (exact, all 64) · **T2** hardware roles &
  data-flow (exact, all 64) · **T3** game-concept labels — *partial*, see §2; needed
  only for *semantic-level* metrics.

Running example: **Pong** — outputs: score, ball-pixel; known variables: ball/paddle
positions, score (RAM).

---

## 2. How we obtain T3 (game-concept labels)  *(shared)*

Two public sources give *candidate* labels; our substrate makes them *causal* and
extends them.

- **AtariARI** (Anand et al. 2019, NeurIPS) — 22 games; labels from commented
  disassemblies (Engelhardt & Jentzsch; CPUWIZ); a gym wrapper emits a label/frame; used
  as a *linear-probing benchmark*. Caveat: raw RAM (x,y) is not aligned to the rendered
  position (render-time offsets).
- **OCAtari** (Delfosse et al. 2023) — 40+ games; **RAM mode** (offsets applied, fixing
  the misalignment) + **Vision mode** (color-filter object detection); cross-validated.

**Our procedure:** (1) import OCAtari/AtariARI as candidates; (2) **verify causally by
intervention** — set the byte, re-render, confirm the object moves to the predicted place
on the bit-exact framebuffer (upgrades correlational→verified-causal; auto-corrects
offsets); (3) **discover** new labels by RAM↔framebuffer correlation + intervention
sweeps; (4) coverage ~22→40+ of our 64, extendable. **Thesis tie-in:** probing shows
info is *present*, not *used* (Hewitt & Liang 2019 control tasks); our intervention test
is strictly stronger. *(The general "are there general methods for obtaining T3, and
which direction is most promising" survey is a direction/discussion — in `plan.md`
(Discussion) and developed in depth in P4.)*

---

## 3. Substrate-feasibility audit  *(shared)*

Substrate = jutari/jaxtari (exact forward, exact interventions, full observability,
content-path gradients, GPU-batched SOFT-STE). The subject *is* the substrate.

| Phase | Substrate capability required | In current substrate? | Other ingredient | Evaluable now? |
|---|---|---|---|---|
| **A** | full state observability + exact interventions | ✅ | T3 only for A3 game-concept scoring | **Yes** |
| **B** | exact intervention + content-path gradient of `y` w.r.t. inputs/state | ✅ (P8 already does this) | T3 only for object/concept attribution | **Yes** |
| **C** | clamp/resample state between runs; record state trajectory; gradients | ✅ | standard SAE training code; T3 for semantic checks | **Yes** |
| (D → P3, E → P4) | — | — | — | (audited in their folders; both reuse §1–§2) |

**Caveats:** (i) "evaluable now" = *no new substrate capability needed*; external tooling
(SAE training, RE tools) still has to be wired (`tools/xai_study` is specs-only). (ii)
**Conformance horizon:** Paper-1 bit-exactness holds within a bounded frame window
(≈30-frame RAM / 60-frame screen on a fixed action stream); scored claims must stay
inside the validated horizon or re-validate. (iii) **Long-horizon end-to-end gradients**
(credit through many steps) could stress the substrate — single-step metrics don't need
them; pilot before assuming. (iv) **Analysis state must be input-driven gameplay, not the
NOOP attract loop** — see [`experiment_redesign.md`](experiment_redesign.md) Problem 1; the
current runners violate this and must move to seed=0 random-action states with an oracle
cause-density gate before scoring. (v) **Position/index outputs must use the bilinear
sampler** (soft-mode), not the naive integer path which is provably zero (Paper 1
Fig. `fig:xai`(c); redesign Problem 2).

---

## 4. Phase A — neuroscience / mechanistic battery on the VCS (Kording, quantified)

| Analysis | Finding (what the method outputs) | Measured score | Needs T3? |
|---|---|---|---|
| **A1** connectomics / data-flow graph | recovered dependency graph over state vars | precision/recall + graph-edit-distance vs the *true* read/write graph | **No** (T1/T2) |
| **A2** single-unit lesions | per-unit importance map (boot/run still works?) | rank-correlation of importance with the unit's *true* role; #units flagged "specific" that are generic | **No** (T1 causal role); naming optional |
| **A3** tuning curves | per-unit tuning to luminance / a game variable | spurious-tuning rate (strongly-tuned units whose tuning ≠ true role) | **Game-variable version only**; hardware-tuning: No |
| **A4** spike-word / pairwise correlations | pairwise + global correlation structure | weak-pairwise/strong-global reproduced; vs true coupling | **No** (T1) |
| **A5** local field potentials | regional pooled-activity power spectra | %-variance that is the known clocks (frame/scanline) → epiphenomenal | **No** (T2 clocks) |
| **A6** Granger causality | inferred CPU/TIA/RIOT causal edges | false-edge / missed-edge rate vs true data-flow | **No** (T1) |
| **A7** dim-reduction (NMF/PCA) | latent components of the state tensor | matched-component fraction vs known signals (clock, R/W, vsync) | **No** (T2); game-var matching optional |
| **A8** whole-state recording | full RAM+register state map over time | descriptive baseline | **No** |

**Ideal:** the register-transfer account of the frame (known exactly). **Right when:**
recovered graph/role = true (F); predicts held-out lesions (S); module-level (M).
**Best case:** a *quantified* Kording — classical methods low-F despite rich structure,
while the causal operators (§6) score high → the gap is the *method*. **Novelty note:**
the delta over Jonas & Kording 2017 (who ran this battery on the same chip) is (i) the
*architectural*-level subject, (ii) the *quantitative* F/S/M score with the high-F causal
contrast they lacked, (iii) intervention-verified scoring. A is the calibration baseline,
not the headline. (Optional parallel Visual6502 transistor track — but that *is* J&K's
level; feature or cut after pilots.)

## 5. Phase B — attribution / XAI on the VCS's input→output computation

Explain a chosen output `y` (pixel/score/event) from its causes (ROM/RAM/registers/
inputs). Score every map against the §1 oracle.

| Analysis (named method) | Finding (output) | Measured score | Needs T3? |
|---|---|---|---|
| Vanilla gradient (Simonyan et al. 2014) | saliency over causes | corr + deletion/insertion AUC on the true VCS + precision@k vs true causal top-k | No* |
| Grad×Input / DeepLIFT (Shrikumar et al. 2017) | attribution map | as above (+ completeness where defined) | No* |
| Guided Backprop (Springenberg et al. 2015) | saliency map | as above + sanity-check (Adebayo et al. 2018) | No* |
| SmoothGrad (Smilkov et al. 2017) | noise-averaged saliency | as above | No* |
| **Integrated Gradients** (Sundararajan et al. 2017; our P8) | path-integrated attribution | corr + del/ins AUC + completeness; baseline sweep | No* |
| Expected Gradients (Erion et al. 2021, *NMI*) | baseline-averaged IG | as above | No* |
| Occlusion (Zeiler & Fergus 2014) | Δ`y` per occluded region | del/ins AUC; ≈ coarse intervention oracle | No* |
| Meaningful/extremal perturbation (Fong & Vedaldi 2017; Fong et al. 2019) | learned minimal mask | mask IoU vs true causal set; del/ins | No* (object-set IoU needs T3) |
| RISE (Petsiuk et al. 2018) | randomized-mask saliency | corr + del/ins | No* |
| LIME (Ribeiro et al. 2016) | local linear weights | corr vs true map; stability | No* |
| KernelSHAP / Shapley sampling (Lundberg & Lee 2017; Štrumbelj & Kononenko 2014) | Shapley values | corr vs true; convergence vs compute | No* |
| On-distribution counterfactual (state-set + re-render; cf. Olson 2021; Atrey 2020) | minimal valid counterfactual edit | validity + minimality vs true minimal set | **Object-level: yes**; pixel-level: no |
| **N/A** Grad-CAM/++ (Selvaraju 2017; Chattopadhyay 2018), attention rollout (Abnar & Zuidema 2020), VIPER (Bastani 2018) | needs NN layers / a policy | *does not apply* — recorded finding | — |

> **\* T3 enters B only at the *object level*** (object-hit/pointing-game; object-level
> counterfactuals). Cell/pixel faithfulness (corr, del/ins AUC, precision@k) scores
> against T1 and needs no T3. **Gradient methods (vanilla, IG, …) are evaluated as
> methods here** — for *position/index* outputs the gradient is the surrogate-sampler's,
> so the *intervention* oracle is the reference (see §1).

**Ideal:** the minimal true-causal causes of `y`. **Right when:** top-k = true causal
top-k (F); deletion behaves as predicted (S); sparse (M). **Best case:** a faithfulness
leaderboard — plausible ≠ faithful, in numbers, on the system itself.

> **Shared-output requirement (redesign Problem 4).** `y` must be **one shared,
> well-specified output per game**, fixed before any method runs, and preferably a
> **screen-buffer target** — not a per-method most-causally-active byte. In the committed
> records the methods drift: vanilla saliency explains `content_ram_index=54` while
> Grad×Input/DeepLIFT explains `score@ram[49]`, so the "true causal region" differs by
> method and the leaderboard is not apples-to-apples. All gradient methods must run through
> the **soft-mode content path** (content targets) and the **bilinear sampler** (position/
> index targets); add **screen-buffer gradient maps** (∂y/∂pixel, ∂pixel/∂cause) as
> first-class outputs. See [`experiment_redesign.md`](experiment_redesign.md).

## 6. Phase C — mechanistic interpretability on the VCS (known-circuit testbed)

State trajectory = "activations"; the program's data-flow = the "circuit" — both known.
Ground truth = T1/T2 (always) + T3 (where labeled).

| Analysis (named method) | Finding (output) | Measured score | Needs T3? |
|---|---|---|---|
| Activation patching / causal mediation (Vig et al. 2020; ROME causal tracing, Meng et al. 2022) | per-site causal effect | recovered effect vs the *exact* patch; site P/R vs true data-flow | **No** (T1/T2); naming optional |
| Interchange interventions / DAS (Geiger et al. 2021 causal abstraction; DAS, Geiger et al. 2023) | aligned causal variables | interchange accuracy; alignment vs the true variable | **If aligning to a game-concept variable: yes**; hardware variable: No |
| Attribution patching / edge AP (Nanda 2023; Syed et al. 2023) | gradient-approx site/edge effects | approx error vs true patching; edge P/R | **No** (T1) |
| Path patching / IOI circuit (Wang et al. 2022; Goldowsky-Dill et al. 2023) | recovered circuit (path set) | circuit P/R vs the true routine | **No** (T1); naming optional |
| ACDC — automatic circuit discovery (Conmy et al. 2023) | auto-discovered circuit graph | edge P/R + scrubbing-preserved performance vs true data-flow | **No** (T1/T2) |
| Sparse autoencoders (Cunningham et al. 2023; Bricken et al. 2023; Templeton et al. 2024) | learned features over state | feature↔known-variable match (probe F1/MI) + causal use (patch effect) + monosemanticity | **Game-variable matching: yes**; hardware + causal-use: No |
| NMF/PCA dictionaries | latent components | matched-component fraction vs known variables | **Game-variable matching: yes**; hardware: No |
| Causal scrubbing (Chan et al. 2022) | hypothesis pass/fail | scrubbing-preserved performance vs the true routine | **No** (T1) |
| Linear probing + control tasks (Alain & Bengio 2017; Hewitt & Liang 2019) | concept decodability | accuracy **and** selectivity (probe − control) → present-vs-used gap | **Target** — the probe *needs* the labels (T3 for game concepts; T2 for hardware) |
| Logit / tuned lens (nostalgebraist 2020; Belrose et al. 2023) | per-stage readout of state | readout fidelity vs the true intermediate value | **No** (T1/T2) |

> **T3 in C** = the *semantic* layer only (naming a feature/circuit/variable in game
> terms). The causal core (patching/scrubbing/circuit recovery vs exact patch + true
> data-flow) runs on T1/T2. Probing is the one method that *requires* labels by
> definition.

**Ideal:** a circuit/feature account of *how the program computes `y`*, verified against
the disassembly. **Right when:** recovered = true data-flow (F); the circuit alone
reproduces behavior, scored on held-out interventions (S); minimal/monosemantic (M).
**Best case:** the first validation of mech-interp against a *known* circuit on a real
artifact + an SAE calibration (feature↔variable rate, monosemanticity ceiling).

---

## 7. Method matrix — expected success vs failure (a prediction to test)

| Method (category) | Applies to VCS? | Expected vs ground truth | Why |
|---|---|---|---|
| Activation patching / causal tracing (C) | ✓ | **Succeed** | causal by construction; comparable to the oracle |
| Causal scrubbing (C) | ✓ | **Succeed** | we *have* ground-truth hypotheses to scrub |
| Integrated Gradients (B) | ✓ (content path) | **Succeed/partial** | completeness + content-path gradients; baseline-sensitive; vanishes on index outputs |
| Occlusion on the true VCS (B) | ✓ | **Succeed/partial** | faithful when the perturbation is a valid intervention |
| On-distribution counterfactuals (B) | ✓ | **Partial→Succeed** | set-state-and-re-render removes Atrey's off-manifold objection |
| Attribution / edge patching (C) | ✓ | **Partial** | cheap approximation; score the approx error vs true patching |
| Sparse autoencoders (C) | ✓ | **Partial** | recovers features; faithfulness-to-computation debated |
| Circuit discovery (ACDC, path patching) (C) | ✓ | **Partial→Succeed** | goal state = our ground truth |
| Linear probing (C) | ✓ | **Partial / misleading** | present ≠ used — the tuning-curve trap |
| SHAP / LIME (B) | ✓ | **Partial/Fail** | baseline/sampling-dependent; approximations diverge |
| Vanilla gradient / saliency (B) | ✓ | **Fail** | noisy, shattering, non-causal; zero on index outputs |
| Grad-CAM/++ , attention, VIPER (B) | **✗ N/A** | — | need conv maps / attention / a learned policy |

Headline expectation: **causal/gradient/mechanistic methods pass; correlational and
architecture-specific methods fail or don't apply** — a measured sharpening of Kording's
lesson and of Atrey 2020.

## 8. Compute plan

Pilots local; full runs on the LME cluster. Enabler: SOFT-STE forward is bit-exact to
HARD and GPU-batchable (`vmap`).

| Workload | Where |
|---|---|
| Pilots (1 game each: A lesion/tuning/dim-red · B IG-vs-oracle · C patching+SAE) | **local** (M1 Max) |
| Full lesion + occlusion sweeps (RAM × ROM × opcodes × outputs × games) | **cluster (CPU array / GPU)** |
| Batched gradients / IG over outputs × games | **cluster (GPU)** |
| SAE training on recorded state trajectories | **cluster (GPU)** |

Reuse Paper-1 setup: LME Slurm, `/cluster/maier`, `tools/cluster/*.sbatch`, jaxtari GPU
venv. No agents → no agent training/zoo.

## 9. Master table (P2 phases)

| Phase | Measure (output → score) | Ideal | Right when (F∧S∧M) | Best case |
|---|---|---|---|---|
| **A** neuroscience | finding → agreement with known mechanism | register-transfer account | recovered = true; predicts held-out lesions; module-level | quantified Kording: classical low-F, causal high-F |
| **B** attribution | attribution map → corr + del/ins AUC + precision@k | minimal true-causal causes | top-k = true; deletion as predicted; sparse | leaderboard: plausible ≠ faithful |
| **C** mechanistic | patch/SAE/circuit → vs exact patch + feature↔var + scrubbing | circuit/feature account | recovered = true data-flow; reproduces behavior; monosemantic | first mech-interp validation on a known *real* circuit |

> Outcomes in §4–§6 are *hypotheses*, confirmed by the experiments. Pilots test the
> riskiest links first: the oracle, and one method per phase. (D → `../xai_3_psychology/`,
> E → `../xai_4_recovery/`; both reuse §0–§3.)
