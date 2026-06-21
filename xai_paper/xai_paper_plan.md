# Paper 2 — Plan: a fully-known, differentiable model organism for the science of understanding AI

**Working title:** *Do our methods recover what is true? A fully-known,
differentiable model organism for evaluating interpretability.*

**One-line thesis.** There are **three traditions** for understanding an opaque
system: the **mechanistic/neuroscience** tradition (Jonas & Kording 2017 — probe
the substrate), the **attributional/XAI** tradition (saliency, Grad-CAM,
attribution/activation patching, SAEs, circuits), and the **behavioral/psychology**
tradition (Binz & Schulz 2023; Shiffrin & Mitchell 2023 — treat the system as a
participant). Each "finds structure," but none can be *validated* without ground
truth — exactly Kording's warning, and exactly Shiffrin & Mitchell's caution about
psychology-of-AI. Our Atari VCS (Paper 1) is **fully specified, bit-exact, and
end-to-end differentiable**, so we can compute the *true* causal attribution for
any output and, for the first time, **score all three traditions against ground
truth** on a system complex enough to matter. We adopt a formal definition of
interpretability (Barbiero et al. 2025) operationalized as *ground-truth
recovery*, replicate the Kording battery, audit today's XAI **including the
mechanistic-interpretability toolkit (activation/attribution patching, sparse
autoencoders, circuits)**, test behavioral/psychology-style probing, measure where
each diverges from truth, and from that argue what the science of understanding AI
actually needs.

> **Status: PLAN — experiment-prep adopted.** Template fetched into `paper/`
> (Springer Nature, `sn-nature`). Decisions taken (this revision):
> 1. **Conduct the experiments; decide scope later.** Both granularities stay in
>    scope — architectural state (our level) leads; a transistor-level head-to-head
>    via the Visual6502 netlist runs as a parallel track and we cut whichever the
>    pilots show is weaker (§4, §8).
> 2. **Compute:** pilots run locally; full Phase-A sweeps and all of Phase B run on
>    the LME GPU cluster — see the compute plan (§4.6). Short answer: **yes, the
>    cluster is needed for the full runs.**
> 3. **Method coverage reviewed** (§4.4–4.5): our Paper-1 literature is strong on
>    classic XRL but misses the 2023–25 mechanistic-interpretability wave
>    (activation/attribution patching, causal scrubbing, SAEs, probing, circuits).
>    These are added — they are the most ground-truth-relevant methods we test.
> Experiment scaffolding: `tools/xai_study/` (specs + harness stubs).

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
3. **A faithfulness audit of modern XAI — attribution *and* mechanistic.**
   Saliency, Grad-CAM/++, Integrated Gradients, perturbation/occlusion, attention,
   counterfactual **and, as first-class targets, the mechanistic-interpretability
   toolkit — activation patching, attribution patching, sparse autoencoders,
   circuit discovery** — applied to DQN agents and scored against the *true* causal
   attribution from the exact `emulator ∘ agent` pipeline.
4. **A ground-truth test of the *behavioral/psychology* tradition.** Treat the
   agent (and the chip) as a psychology participant — controlled-stimulus
   ("psychophysics") probes — and ask whether the inferred behavioral account
   matches the known mechanism. This puts Shiffrin & Mitchell's caution to a
   *measurable* test (a first).
5. **A formal, operationalized notion of interpretability.** Adopt Barbiero et
   al.'s (2025) definition and operationalize "an explanation is correct" as
   *recovering the ground-truth causal structure*; include interpretable-by-design
   models as a comparison arm.
6. **Evidence of the discrepancy + concrete new directions:** where each tradition
   fails, why, and what a faithful, ground-truth-validated science of understanding
   AI should look like (causal/known-operator attribution; ground-truth-scored
   benchmarking; which behavioral inferences are trustworthy).

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
3. **Methods under test — two families, both first-class.**
   - **B1 Attribution / saliency:** Greydanus perturbation, Grad-CAM (Selvaraju) +
     Grad-CAM++ (Chattopadhyay), Integrated Gradients, occlusion, attention
     (Nikulin/Mott), counterfactual (Atrey/Olson), SHAP/LIME; interpretable-by-design
     baselines (XDQN Kontogiannis; Barbiero-style). Survey scope from our XRL notes
     (Qing 2022, Vouros 2023, Cheng 2025, Saulières 2025).
   - **B2 Mechanistic (first-class):** **activation patching / causal tracing,
     attribution patching (+ edge AP), sparse autoencoders / dictionary learning,
     circuit discovery, linear probing.** Capture agent activations; patch by
     resample/clamp; train SAEs on activations; discover candidate circuits — then
     **score the recovered circuit / features / patched effects against the *true*
     causal structure** (for the chip in Phase A the true circuits are *known*; for
     the agent the oracle is the reference). This is the modern, causal core and a
     contribution back to mechanistic-interpretability, not only XRL.
4. **Faithfulness metrics vs ground truth.** Correlation with the true causal map;
   deletion/insertion AUC on the *true* emulator (not a proxy); pointing-game /
   object-hit rate; (mechanistic) recovered-effect vs exact-patch agreement,
   feature↔known-variable matching for SAEs.
5. **Result:** quantify the discrepancy — which methods (attribution *and*
   mechanistic) track the true causes vs produce plausible-but-wrong accounts;
   expected: causal/patching/IG pass, popular visual saliency fails (a measured
   sharpening of Atrey 2020).

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

### Phase D — Behavioral / psychology probing, scored against ground truth

The third tradition (Binz & Schulz 2023; Shiffrin & Mitchell 2023): treat the
system as a *participant* and infer its "cognition" from behavior. Their open
worry — does behavioral probing reveal the true mechanism, or just plausible
correspondences? — is *unanswerable for LLMs* because there is no ground truth.
**We can answer it.**

- **Subjects:** the DQN agent (primary) and, as a stress test, the chip.
- **Probes:** controlled-stimulus / psychophysics-style experiments — vary one
  factor of the input (object position, distractor presence, reward cue) and read
  the behavioral response (action, Q); fit "cognitive" accounts (decision
  variables, biases, tuning) the way a psychologist would.
- **The test:** does the inferred behavioral account match the *known* causal
  mechanism (Phase-A ground truth for the chip; the oracle + the agent's true
  decision variables for the agent)? Quantify "right for the wrong reasons."
- **Contribution:** the first ground-truthed verdict on psychology-of-AI methods —
  which behavioral inferences are trustworthy, which are anthropomorphic mirages.

## 4.4 Method matrix — where we expect success vs failure

The expected **punchline** (and the paper's narrative): methods that are *causal /
intervention-grounded* should track our ground truth; methods that are
*correlational, attention-based, off-manifold, or spatially coarse* should produce
plausible-but-wrong explanations — the exact analogue of Kording's lesson that
correlational neuroscience methods mislead even with unlimited data.

| Method (category) | In our lit? | Expected vs ground truth | Why |
|---|---|---|---|
| **Activation patching / causal tracing** (causal, mechanistic) | ✗ add | **Succeed** | causal by construction; directly comparable to our intervention oracle |
| **Causal scrubbing** (hypothesis test) | ✗ add | **Succeed** | we *have* ground-truth hypotheses to scrub against — ideal testbed |
| **Integrated Gradients** (axiomatic gradient) | ~partly (our P8) | **Succeed/partial** | completeness axiom + exact-forward gradients on our substrate; baseline-sensitive |
| **Occlusion/perturbation on the true emulator** | ✓ (Greydanus, Iyer/Anderson) | **Succeed/partial** | faithful when the perturbation is a valid intervention; Greydanus uses Gaussian blur (off-distribution) and smears small sprites |
| **Attribution patching / edge attribution patching** (gradient approx of patching) | ✗ add | **Partial** | cheap approximation of patching; we can score the approximation error against true patching |
| **Sparse autoencoders / dictionary learning** (feature disentangling) | ~1 note | **Partial** | recovers interpretable features; score them against the *known* game variables; faithfulness-to-computation debated |
| **Circuit / mechanistic analysis** | ~1 note | **Partial→Succeed** | the goal state is exactly our ground truth; measure recovered circuit vs true data-flow |
| **Linear probing** (concept readout) | ✗ add | **Partial / misleading** | detects info is *present*, not that it is *used* — the tuning-curve trap (Phase A A3) |
| **SHAP / LIME / Shapley** (model-agnostic attribution) | ✓ (surveys) | **Partial/Fail** | baseline- and sampling-dependent; approximations diverge; costly |
| **Grad-CAM / Grad-CAM++** (CAM) | ✓ | **Fail/partial** | last-conv coarse; built for classification CNNs; misses tiny Atari sprites |
| **Vanilla gradient / raw saliency** | ✓ | **Fail** | noisy, gradient-shattering, non-causal |
| **Attention maps** (Mott, Nikulin free-lunch) | ✓ | **Fail** | attention ≠ attribution (well documented); expect divergence from true causes |
| **Counterfactual states** (Olson; Atrey) | ✓ | **Partial/Fail** | generative counterfactuals go off-manifold; Atrey already argues "exploratory not explanatory" — we quantify it |
| **Surrogate trees / policy distillation** (VIPER, XDQN) | ✓ | **Partial** | global fidelity ≠ local faithfulness; can misattribute |
| **Reward decomposition** (RDX/MSX) | ✓ | **Succeed (own task)** | explains *which goal*, not input attribution; we know the true reward structure |

Headline expected result: **causal/gradient methods grounded in real interventions
pass the ground-truth test; the most *popular* visual XAI (Grad-CAM, attention,
vanilla saliency) fails it** — a measured, ground-truthed sharpening of Atrey 2020.

## 4.5 Newest methods we must add (gap from the Paper-1 literature)

Our Paper-1 notes (RL-centric XRL surveys) **miss the 2023–25 mechanistic-
interpretability wave**, which is the most relevant to a *causal ground-truth*
benchmark. Add to the literature/bib and the Phase-B matrix (verify each entry to
Paper-1's no-hallucination standard before it enters the bib):

- **Activation patching / causal tracing** — Meng et al. 2022 (ROME); Heimersheim & Nanda 2024, "How to use and interpret activation patching".
- **Attribution patching** — Nanda 2023; **Edge Attribution Patching**, Syed et al. 2023.
- **Causal scrubbing** — Chan et al. 2022 (Redwood).
- **Sparse autoencoders for interpretability** — Cunningham et al. 2023; Bricken et al. 2023 ("Towards Monosemanticity").
- **Unifying theory** — Geiger et al. 2023/25, "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability".
- **Field reviews** — Bereska & Gavves 2024 (mech-interp for AI safety).
- **Behavioral / psychology tradition (Phase D)** — Binz & Schulz 2023 ("Using
  cognitive psychology to understand GPT-3", PNAS); Shiffrin & Mitchell 2023
  ("Probing the psychology of AI models", PNAS) [in `papers/`].
- **Formal definition of interpretability** — Barbiero et al. 2025, "Foundations
  of Interpretable Models" (arXiv:2508.00545) — our definitional anchor +
  interpretable-by-design comparison arm.
- Also confirm we cite the RL-native pieces already in our notes: object saliency (Iyer/Anderson), attention agents (Mott 2019), StateMask (Cheng 2023), EDGE (Guo 2021), HIGHLIGHTS (Amir & Amir).

These newest methods are also the paper's *positive* story: they are exactly the
causal tools we expect to pass, so testing them on ground truth is a contribution
to the mechanistic-interpretability community, not only to XRL.

## 4.6 Compute plan — do we need the cluster?

**Yes for the full runs; pilots are local.** Key enabler from Paper 1: the
**SOFT-STE forward pass is bit-exact to HARD and GPU-batchable** (`vmap`,
millions of env-steps/s), so the exact intervention oracle and end-to-end
gradients can run *batched on GPU* rather than on the slow CPU-only HARD path.

| Workload | Where | Why |
|---|---|---|
| Phase-A pilot (1 game: lesion + tuning + dim-reduction, scored) | **local** (M1 Max) | small; reuses Paper-1 tooling |
| Phase-B pilot (1 agent, IG vs intervention oracle, 1 game) | **local** | proves the oracle pipeline |
| Phase-A full lesion sweeps (RAM bits × ROM bytes × opcodes × games) | **cluster (CPU array jobs)** | embarrassingly parallel; thousands of short deterministic re-runs |
| Phase-B intervention oracle (per-pixel/object occlusion × states × agents × games) | **cluster (GPU)** | huge forward count; batched SOFT-STE exact-forward on GPU |
| Phase-B end-to-end gradients through `emulator ∘ agent` | **cluster (GPU)** | the differentiable path is jit+vmap, GPU-bound |
| SAE training on agent activations; any agent (re)training | **cluster (GPU)** | standard DL training |

Reuse the Paper-1 cluster setup verbatim: LME Slurm, repo on `/cluster/maier`,
the `tools/cluster/*.sbatch` pattern + the jaxtari GPU venv (jax[cuda12]). Prefer
**existing zoo agents** (Such et al. 2019) over training to keep GPU cost down.
Add: agent-activation capture + patching infra + SAE training (GPU) for Phase B2;
behavioral-probe sweeps (Phase D) are many short rollouts (cluster, CPU/GPU).

## 5. Discussion — three traditions of understanding AI, scored against truth

- **Three traditions, one ground truth.** Mechanistic/neuroscience (Phase A),
  attributional/XAI incl. mechanistic-interp (Phase B), and behavioral/psychology
  (Phase D) are the three ways the community tries to understand opaque systems.
  Each "finds structure"; our platform is the first place all three can be
  *validated* against known truth on a complex system.
- **What "interpretable" means.** We adopt Barbiero et al.'s (2025) formal
  definition and operationalize it as *ground-truth recovery* — making "is this
  explanation correct?" a measurable question rather than a matter of taste.
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
- **And beyond to psychology-of-AI.** Shiffrin & Mitchell ask whether treating a
  model as a psychology participant reveals its true mechanism — unanswerable for
  LLMs (no ground truth), answerable here. We give the behavioral tradition its
  first ground-truthed verdict, completing the mechanistic↔attributional↔behavioral
  triangle — the natural arena for a cognitive scientist + ML co-authorship.

## 6. Paper structure (Nature Portfolio format)

`paper/main.tex` is wired to `\documentclass[sn-nature]{sn-jnl}` (template files
fetched). Nature-style layout:

- **Title; Abstract** (~150–200 words, unreferenced).
- **Main text** (Nature has no rigid IMRaD): Introduction → Results
  (Phase A mechanistic/Kording scoring; Phase B attribution **and**
  mechanistic-interp audit; Phase D behavioral probing; the cross-tradition
  discrepancy; a faithful-method demonstration) → Discussion (§5).
- **Main display items** (~6 figures): (1) the platform & ground-truth oracle;
  (2) the three traditions scored side by side (headline); (3) Kording battery
  scored; (4) attribution vs mechanistic-interp faithfulness on agents;
  (5) behavioral-probe verdict vs truth; (6) failure taxonomy / what to do instead.
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

- **Granularity vs Kording.** *Decided: run both, cut later.* Architectural level
  (our level, where "understanding" is defined) leads; the Visual6502 transistor
  netlist runs as a parallel A1/A2 track for a direct head-to-head with Kording.
  Pilots decide which we feature. (Visual6502 netlist is public; importing it is a
  bounded add, not a new simulator.)
- **Agent training cost.** Use existing zoo agents where possible (jaxtari GPU
  rollouts help); training from scratch is a time sink.
- **Defining "the true causal map."** Intervention-based vs gradient-based ground
  truth can disagree at non-smooth points; we report both and treat agreement as
  part of the result.
- **Scope control.** Phase A + B alone is a strong paper; C2 (new methods) can be
  a follow-up if it bloats.

## 9. Phasing / next actions

**Prepared (this revision):** scaffolding at `tools/xai_study/` — specs +
harness stubs for the ground-truth oracle, Phase A, and Phase B; method matrix and
compute plan locked above.

Immediate (the two pilots de-risk the whole paper; both run **locally**):
1. **Phase-A pilot** — `tools/xai_study/phaseA_kording/`: lesion + tuning-curve +
   dim-reduction on Space Invaders, with ground-truth scoring (reuses Paper-1
   tooling). Proves the scoring concept.
2. **Phase-B pilot** — `tools/xai_study/phaseB_agents/`: one DQN agent (zoo) +
   Integrated Gradients vs the exact intervention oracle on one game. Proves the
   ground-truth attribution pipeline.

Fast-follow pilots once the oracle works:
3. **Phase-B2 mechanistic pilot** — activation patching + one SAE on the agent;
   score the patched effect vs the exact patch and SAE features vs known variables.
4. **Phase-D behavioral pilot** — one psychophysics-style probe of the agent;
   compare the inferred decision variable to the true one.

Then: secure a zoo agent set; stand up the cluster runs (§4.6); expand to the full
method matrix (§4.4) across attribution + mechanistic + behavioral and multiple
games; build the benchmark (C1); draft `main.tex` (§6). Lock
journal/article-type/authors after the pilots (§7).

Author fit (provisional, TBD): A. Maier, S. Bayer, P. Krauss (+ collaborators);
Krauss's neuroscience/neuroprosthetics background anchors the §5 bridge.
