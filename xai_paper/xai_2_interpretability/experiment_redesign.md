# P2 — Experiment redesign (testbed fix: gameplay states, soft-mode gradients, screen-buffer targets)

> **Role of this file:** the *diagnosis + fix* for the Phase-A/B/C testbed. It records
> five design problems found in the committed runners/records, cross-references each to
> the paper section or code line that confirms (or corrects) it, gives the redesigned
> protocol, states what stays unchanged, and lists the experiments to repeat.
> Storyline → [`plan.md`](plan.md); experiments/scoring → [`experiment_design.md`](experiment_design.md)
> (§0–§3 shared foundation); Paper-1 grounding → `jutari_paper/paper/{paper,supplementary}.tex`.
>
> **This document changes the *testbed and targets*, not the thesis.** The F∧S∧M triad,
> the §1 oracle, the 6 core games, seed=0 determinism, and the shared-testbed principle
> all stand. The semantic-gap claim is *strengthened*, not weakened, by removing the
> boot/attract artifact and by turning the sampler on for the gradient family (this is the
> keystone "sampler-on" experiment the storyline already commits to — `plan.md` §"the
> move (half 2)", Display item 7).

---

## What Paper 1 actually provides (ground the fix in the substrate's real capabilities)

Read against `jutari_paper/paper/paper.tex` and `.../supplementary.tex`, the emulator
gives **three** distinct gradient channels, not one:

1. **Content-path gradient** — register/colour/graphics-bit/RAM values, via the one-hot
   `peek` read (paper.tex Eq. (1) `peek`, lines 357–366; supplementary Eq. (S-peek), lines
   59–64). The STE (paper.tex Eq. (5), lines 441–451; Corollary 1, lines 550–560) makes the
   forward bit-exact and routes the backward pass through the soft value. This is the
   channel the current Phase-B runners use.
2. **Position/index gradient via the bilinear sampler** — the *one* quantity a stored
   switch cannot differentiate is a discrete **index** (sprite column from strobe timing);
   a sub-pixel triangular (bilinear) kernel restores it (paper.tex lines 459–467, and the
   proof-of-concept §"Ground-Truth Gradients", lines 617–662; Fig. `fig:xai`). The naive
   index gradient is **identically zero** (paper.tex line 649; Fig. `fig:xai`(c)) — exactly
   the artifact Problem 2 flags. The sampler puts **100%** of ∂screen/∂RIGHT mass on the
   cannon edges and recovers the inverse control map (+35.7/−35.7 for RIGHT/LEFT, 0 up/down;
   paper.tex lines 656–662). Reference implementation: `tools/xai_si_gradient/si_joystick_gradient.jl`
   (the `occ`/`render`/`fwd_saliency`/`objective` sampler path, lines 48–110) and its figure
   `tools/xai_si_gradient/si_joystick_fig.py`.
3. **Screen-framebuffer gradient** — the full state `x` in Theorem 1 explicitly includes
   the frame buffer (paper.tex lines 472–474; supplementary lines 47–49), and every gradient
   figure in Paper 1 is an **image-domain** ∂screen/∂control map (paper.tex Fig. `fig:xai`(b),
   supplementary Fig. `fig:alphatemp` "screen-space directional-derivative saliency", lines
   366–367, 402–413). So ∂(screen pixel)/∂(cause) is a first-class, demonstrated output, not
   a proxy.

**Operating point:** SOFT-STE is forward bit-exact at any finite `T` (Thm 1); the fully
relaxed forward is bit-exact only in the corner α=6, T=0.14 (supplementary Table
`tab:relax-exact`, lines 468–508; §"Choosing α and T", lines 643–666). **Conformance
horizon:** bit-exactness is guaranteed only within ≈30-frame-RAM / 60-frame-screen of the
fixed action stream (supplementary Table `tab:s-params`, line 903; already flagged in
`experiment_design.md` §3 caveat ii). Any redesigned state must stay inside this window or
re-validate.

---

## Problem 1 — the testbed is boot/attract frames (CONFIRMED)

**Claim.** Phase-A/B/C fix `target_frame=120, horizon=30` under an **all-NOOP** stream after
a 60-NOOP+4-RESET boot, so ~100% of analysed states are boot/attract frames — causally
inert for games that need input to start.

**Confirmed in code.**
- `tools/xai_study/phaseB_attribution/ig_baseline_sweep.jl:823` — `target_frame = 120; horizon = 30`.
- `.../ig_baseline_sweep.jl:525` — `actions = fill(0, total)` (pure NOOP; `total = target_frame + horizon`).
- `.../ig_baseline_sweep.jl:530` — `checkpoint = boot_replay(actions, target_frame; game = game)` (NOOP boot-replay).
- Expected Gradients' reference pool is the **same NOOP trace**: `expected_gradients.jl:137–148`
  `reference_pool(...)` steps `acts = fill(0, nframes)` and records frames 1..target_frame;
  `expected_gradients.jl:435` again `actions = fill(0, total)`. Docstring lines 134–136:
  "all on-distribution and reachable under the same NOOP trace."

**Consequences confirmed in the committed records.**
- **seaquest content oracle is near-empty:** `out/ig_baseline_sweep_seaquest_content.json`
  has **2 of 48** non-zero oracle causes (`ram[60]:set`=241.0, `ram[118]:set`=17.0). A
  near-flat oracle column starves precision@k / correlation for every method scored there.
- **Expected Gradients gives 0 on constant bytes:** EG weights each cause by `(x − x0)`
  (`expected_gradients.jl:161`, Erion Eq. 5). If the causal byte is **constant over the NOOP
  reference pool**, `(x − x0)=0` and EG is identically zero regardless of the gradient — an
  artifact of the reference distribution, not of the method. (The magnitude "0 on 5/6 content
  games" in the original note is the right *shape* of the failure; treat the exact count as
  a to-recount number after we move states — the mechanism, `(x−x0)=0` on constant bytes over
  a NOOP pool, is what is load-bearing and is confirmed.)

**Correction to the comment.** The comment says "~100% of analysed states are boot/attract."
Precisely: **all** analysed states are on the single deterministic NOOP trajectory; whether a
given game is still in attract mode at f120 is game-dependent, but the *stream carries no
input*, so no game reaches genuine input-driven gameplay. The fix (random-action gameplay) is
correct; the framing should be "no input-driven gameplay is exercised," which is the stronger
and exactly-true statement.

**Fix.** Analyse **shared random-action gameplay states** (seed=0 deterministic PRNG action
stream; **no DQN/agent needed** — random inputs suffice to leave attract mode and populate the
causal state). Constraints: (i) boot/attract frames **< 10%** of the test set; (ii) an
**oracle cause-density gate** that rejects any candidate analysis state whose non-zero oracle
cause-set is below a threshold (e.g. ≥ K causes with |Δy| above a floor), so we never score
methods on a near-flat oracle column; (iii) keep the state **shared across all methods** for
comparability (the shared-testbed principle, unchanged). Stay inside the conformance horizon:
a random-action prefix of length `L_pre` followed by the `horizon` window must keep
`L_pre + horizon` within the validated screen window, or re-validate bit-exactness for that
stream (the runners already `assert_bit_exact`, `ig_baseline_sweep.jl:526`).

---

## Problem 2 — gradient methods use the naive path, not the sampler/soft-mode (CONFIRMED, and it is an artifact)

**Claim.** The gradient methods vanish to 0 on discrete position/index outputs; this is an
artifact of the naive integer-dispatch gradient, not a real limitation — Paper 1's soft-mode
+ bilinear sampler *do* provide position gradients.

**Confirmed.**
- The vanishing is real and by design on the **naive** path: paper.tex line 649 "a naive
  gradient is identically zero"; Fig. `fig:xai`(c). The runner records the same:
  `out/saliency_pong_content.json` `auc_note` — "NaN = ... the vanishing position output";
  `experiment_design.md` §1 caveat and §5 note ("gradient methods ... zero on index outputs").
- The **fix already exists** in Paper 1: the sampler (`paper.tex:459–467`, 649–662) and its
  runnable demo `tools/xai_si_gradient/si_joystick_gradient.jl` (sampler `occ(px)`,
  `render(px)`, `fwd_saliency()`, inverse `objective(j)`; the naive `mask_int`/`cx_naive`
  path is kept only to show it vanishes, lines 74–125). Cross-variant agreement (Thm 1) is
  asserted at lines 131–135.

**Fix.** Route **every** Phase-B gradient method — vanilla saliency, Grad×Input/DeepLIFT,
SmoothGrad, Integrated Gradients, Expected Gradients, Guided Backprop — through the
**soft-mode / bilinear-sampler path** for position/index targets, so the position gradient is
**real and reported**, not silent. Content targets already flow through the one-hot content
path (keep). Report both the naive (vanishing) and sampler (real) gradients side by side for
the position target, exactly as Paper 1's figure does — this *is* the storyline's "sampler-on"
keystone (`plan.md` Display item 7, §"the move (half 2)"): faithfulness rises off the provable
zero, and we then show semantic recovery still stays zero.

---

## Problem 3 — attribution is RAM-only; add differentiable screen-buffer gradients (CONFIRMED, correcting one detail)

**Claim.** Attribution is over the 128 RAM cells; the screen framebuffer is differentiable,
so add ∂output/∂screen-pixel and ∂screen-pixel/∂cause, removing the RAM-cell→screen footprint
proxy the audit website uses.

**Confirmed.** The runners score over RAM cells (`ig_baseline_sweep.jl` cause set is
`ram[..]` + one `tia[COLUBK]` + `joystick:RIGHT`, e.g. `out/saliency_pong_content.json`
`cause_names`). The audit website reconstructs the image-domain view from a **perturbation
footprint**, not a gradient: `docs/cell_footprints.jl` header — "footprint of a cell is the
set of screen pixels that change when the cell is perturbed and the emulator is re-run
bit-exact." That is a coarse intervention proxy, exactly as the comment says.

**Correction / precision.** The screen buffer is in `x` and is differentiable (Thm 1), so a
**direct** ∂(screen pixel)/∂(cause) map is available and is what Paper 1 already plots
(`fig:xai`(b), supplementary `fig:alphatemp`). Two distinct maps are wanted and should not be
conflated:
- **∂output/∂screen-pixel** — when the explained output is itself a screen-buffer target
  (Problem 4), this is the saliency over the picture.
- **∂screen-pixel/∂cause** — the image-domain attribution of a single cause (RAM cell,
  register, joystick) onto the picture, the Paper-1 cannon-figure object.

**Fix.** Add a **screen-buffer gradient map** as a first-class output for every applicable
Phase-B/C method, computed directly through the differentiable renderer (soft-mode, and the
sampler where the cause is a position/index). Replace the website's footprint *proxy* with
the true gradient where a gradient exists; keep the intervention footprint only as the
oracle/ground-truth overlay (it is the exact causal map, §1).

---

## Problem 4 — methods explain different outputs → not comparable (CONFIRMED, exactly as stated)

**Claim, verified byte-for-byte in the committed records.**
- `out/saliency_pong_content.json`: `extra.content_ram_index = 54`; `output_note` = "CONTENT
  output: RAM byte 54 ... the most causally-active candidate concept byte." → vanilla saliency
  explains **RAM[54]'s content**.
- `out/gradxinput_deeplift_pong_content.json`: `extra.score_cell = {concept: ball_x,
  ram_index: 49}`; `output_note` = "CONTENT (score) output (score@ram[49])." → Grad×Input/
  DeepLIFT explains the **SCORE at RAM[49]**.

Two different targets ⇒ two different "true causal regions" ⇒ the leaderboard is not
apples-to-apples. (The mechanism is `pick_content_idx`, `ig_baseline_sweep.jl:536`, choosing a
per-method most-causally-active byte, which drifts by method/state.)

**Fix.** Define **one shared, well-specified explained output per game**, fixed before any
method runs, that **all** methods explain. Prefer a **screen-buffer target** (a specified
pixel/region — e.g. the ball, the player sprite — via the differentiable renderer), which is
the natural common target for RAM-cause, register-cause, and joystick-cause methods alike, and
which is directly plottable (Problem 3). Where a RAM-cell target is genuinely needed (Phase C
probing/patching), fix it once per game and reuse it across methods. The oracle's "true causal
region" is then defined **once** per game/output and shared.

---

## Problem 5 — footprint/downstream artifact bleeds into the score display (CONFIRMED)

**Claim.** Because the analysis state is a 30-frame NOOP window, perturbing a game-state cell
(e.g. pong RAM[49]) propagates to the SCORE within the window, so the "causal region" bleeds
into the score display — a downstream game-outcome effect, not direct pixel control.

**Confirmed / refined.** The oracle scores Δy over a `horizon`-frame re-run
(`ig_baseline_sweep.jl:497`, `deletion_insertion_auc` over `actions[target_frame+1:...]`), so
any cause that changes the score within 30 frames lights the on-screen score digits. That is
a real *downstream* causal effect (the oracle is honest), but it conflates **direct pixel
control** (this cell paints these pixels *now*) with **game-outcome influence** (this cell
changes what the score becomes over the window). For a per-pixel saliency claim we want the
former.

**Fix.** Two complementary levers: (i) score a **direct screen-buffer gradient** ∂(screen
pixel)/∂(cause) at the analysis frame (no horizon roll-forward → no score-display bleed) as
the primary attribution, keeping the horizon-based intervention oracle as the ground-truth
comparison; (ii) when a horizon is used, **report the footprint decomposition** (direct-frame
vs horizon-accumulated) so the score-digit bleed is visible and separable rather than silently
folded into "the causal region." The gameplay-state move (Problem 1) also reduces the
degenerate case where the *only* thing a NOOP window can causally move is the score.

---

## The redesigned protocol (shared across A, B, C)

1. **Shared gameplay states.** Replace the all-NOOP stream with a **seed=0 deterministic
   random-action** stream. For each of the 6 core games, roll a random-action prefix to a
   fixed analysis frame `f*`, chosen so the game is in **input-driven gameplay** (attract/boot
   frames < 10% of the analysis set across games). The state is **shared** across every method
   (comparability). No agent/DQN. Assert bit-exactness of the exact stream (existing
   `assert_bit_exact`) and keep `prefix + horizon` inside the conformance horizon.
2. **Oracle cause-density gate.** Before scoring, run the §1 intervention oracle at the
   candidate state; **reject** states whose non-zero cause-set is sparse (below a fixed count
   with |Δy| above a floor). This kills the seaquest-style near-flat column at the source.
   Record the accepted state + its cause-density so the gate is auditable.
3. **One shared explained output per game.** Fix a single, well-specified output — preferably
   a **screen-buffer target** (a named pixel/region) — before any method runs. All methods
   explain it; the oracle's true-causal region is computed once and shared.
4. **Soft-mode / sampler gradients for position.** All gradient-family methods run through the
   soft-mode content path (content targets) **and** the bilinear sampler (position/index
   targets), so the position gradient is real. Report naive-vs-sampler side by side (the
   keystone).
5. **Screen-buffer gradient maps.** Add ∂output/∂screen-pixel and ∂screen-pixel/∂cause as
   first-class image-domain outputs (direct through the differentiable renderer), replacing
   the website's footprint *proxy* where a gradient exists; keep the intervention footprint as
   the exact-oracle overlay only.
6. **Direct-frame primary, horizon as comparison.** Score the direct screen-buffer gradient at
   `f*` as the primary map (no score-display bleed); use the horizon-based intervention oracle
   as ground truth and report the direct-vs-horizon footprint split.

## What stays unchanged (do not drift)

- The **§1 ground-truth oracle** (exact intervention Δy) — the measurement instrument is
  unchanged; only the *states* and *shared output* it is evaluated at change.
- The **F∧S∧M correctness triad** (`experiment_design.md` §0) — definitions and scoring axes
  stand; the semantic-gap claims (ACDC S=0.44; SAE matched=1.0/F=0.04; decodable-but-unused
  cell) are re-measured at gameplay states but the *thesis* is unchanged (and strengthened by
  the sampler-on keystone).
- The **6 core games** (`gradxinput.jl:122`): pong, breakout, space_invaders, seaquest,
  ms_pacman, qbert.
- **seed=0 determinism** and the **shared-testbed principle** (one testbed, all methods).
- Phase E remains **specified-only** (deferred to P4); D → P3; agents → P5.

---

## Experiments to repeat

Legend — **Re-run** = same runner/method, new shared testbed (gameplay state + shared output +
soft/sampler + screen-buffer map); **New** = additional artifact not previously produced.

| Phase | Method / item | Games | What changes | Re-run vs New | Why |
|---|---|---|---|---|---|
| **B** | Vanilla saliency (Simonyan 2014) | 6 core | gameplay state; shared screen-buffer output; **sampler on** for position | Re-run + New (position gradient) | Naive path gave 0 on index (P2); sampler makes it faithful; shared output makes it comparable (P4). |
| **B** | Grad×Input / DeepLIFT (Shrikumar 2017) | 6 core | gameplay state; **explain the shared output, not score@ram[49]**; sampler on | Re-run | It currently explains a *different* target than saliency (P4, verified: ram[49] vs ram[54]). |
| **B** | SmoothGrad (Smilkov 2017) | 6 core | gameplay state; shared output; sampler on; noise over the sampler path | Re-run | Same naive-vanishing + comparability fixes. |
| **B** | Integrated Gradients (Sundararajan 2017) | 6 core | gameplay state; shared output; sampler-on position map; screen-buffer IG | Re-run + New (screen-buffer IG) | Content IG fine; position IG must use sampler; add image-domain IG (P2, P3). |
| **B** | Expected Gradients (Erion 2021) | 6 core | **gameplay reference pool** (random-action states, not the NOOP tape) + gameplay analysis state; sampler on | Re-run | `(x−x0)=0` on constant bytes over the NOOP pool zeroed EG (P1); a varied reference pool restores it. |
| **B** | Guided Backprop (Springenberg 2015) | 6 core | gameplay state; shared output; sampler on; sanity-check (Adebayo 2018) | Re-run | Naive-vanishing + comparability. |
| **B** | Occlusion (Zeiler 2014) | 6 core | gameplay state; shared output; oracle cause-density-gated | Re-run | Was scored on near-flat oracle columns (seaquest 2/48); perturbation methods need a causally-live state (P1). |
| **B** | RISE (Petsiuk 2018) | 6 core | gameplay state; shared output | Re-run | Same: mask methods need a live causal state to score against. |
| **B** | LIME (Ribeiro 2016) | 6 core | gameplay state; shared output | Re-run | Same. |
| **B** | KernelSHAP / Shapley (Lundberg 2017) | 6 core | gameplay state; shared output | Re-run | Same. |
| **B** | Meaningful/extremal perturbation (Fong 2017/2019) | 6 core | gameplay state; shared output; mask-IoU vs shared true-causal set | Re-run | Needs a non-degenerate true-causal set to form the IoU (P1, P4). |
| **B** | On-distribution counterfactual (Olson 2021; cf. Atrey 2020) | 6 core | gameplay state; shared output; minimal-edit vs shared true minimal set | Re-run | Minimality is meaningless on a near-inert NOOP state (P1). |
| **B** | Screen-buffer gradient maps (∂out/∂pixel, ∂pixel/∂cause) | 6 core | **new**: direct image-domain gradient via the differentiable renderer | New | Removes the website footprint *proxy* (P3); replaces the RAM-only view; direct-frame avoids score bleed (P5). |
| **B** | N/A audit (Grad-CAM/++, attention rollout, VIPER) | — | none | Re-run (record) | "Does not apply" finding is state-independent; regenerate the record for consistency only. |
| **§1** | **Oracle recomputation** | 6 core | recompute Δy at the **gameplay states** and for the **shared screen-buffer output**; apply/emit the cause-density gate | Re-run (mandatory, first) | The oracle defines every method's ground truth; the whole testbed hinges on it (P1, P4). Must run before any method. |
| **§1** | Cause-density gate | 6 core | **new**: state-acceptance filter on oracle sparsity | New | Prevents scoring on near-flat columns (seaquest 2/48) (P1). |
| **A** | A1 connectomics / data-flow graph | 6 core (+64 where cheap) | gameplay states | Re-run | Read/write graph should be exercised by input-driven execution, not attract loops. |
| **A** | A2 single-unit lesions | 6 core | gameplay states; cause-density gate | Re-run | Lesion importance is degenerate if the unit is inert in attract mode (P1). |
| **A** | A3 tuning curves | 6 core | gameplay states (game-variable tuning needs varied inputs) | Re-run | Tuning to a game variable requires the variable to *vary* — NOOP freezes it. |
| **A** | A4 pairwise/global correlations | 6 core | gameplay states | Re-run | Coupling structure differs under input-driven execution. |
| **A** | A5 local field potentials | 6 core | gameplay states | Re-run | Clock/scanline power spectra should be measured on live play. |
| **A** | A6 Granger causality | 6 core | gameplay states | Re-run | Inferred edges need input-driven dynamics to be non-trivial. |
| **A** | A7 dim-reduction (NMF/PCA) | 6 core | gameplay states | Re-run | Latent components of a frozen NOOP state are degenerate. |
| **A** | A8 whole-state recording | 6 core | gameplay states | Re-run | Descriptive baseline must match the new testbed. |
| **C** | Activation patching / causal mediation (Vig 2020; ROME 2022) | 6 core | gameplay states; shared output | Re-run | Per-site effects are near-zero on an inert state; needs live causal flow (P1). |
| **C** | Interchange / DAS (Geiger 2021/2023) | 6 core | gameplay states; align to shared variable/output | Re-run | DAS=1.0 "by construction"; keep, but at a state where the variable is causally live. |
| **C** | Attribution patching / edge AP (Nanda 2023; Syed 2023) | 6 core | gameplay states; **sampler-aware** gradient approx | Re-run | Gradient approx inherits the naive-vanishing bug on position (P2). |
| **C** | Path patching / IOI circuit (Wang 2022) | 6 core | gameplay states | Re-run | Recovered path set needs a live routine. |
| **C** | ACDC (Conmy 2023) | 6 core | gameplay states; scrubbing vs true data-flow | Re-run | The S=0.44 separation must be re-measured at a gameplay state (thesis unchanged). |
| **C** | Sparse autoencoders (Cunningham/Bricken 2023) | 6 core | SAEs trained on **gameplay** state trajectories | Re-run (retrain) | The matched=1.0 / F=0.04 separation must come from live features, not attract-loop features. |
| **C** | NMF/PCA dictionaries | 6 core | gameplay trajectories | Re-run | Same as A7 at the circuit level. |
| **C** | Causal scrubbing (Chan 2022) | 6 core | gameplay states | Re-run | Hypothesis pass/fail needs a live routine. |
| **C** | Linear probing + control tasks (Alain 2017; Hewitt 2019) | 6 core | gameplay states; decodable-but-unused cell re-identified live | Re-run | The "present ≠ used" separation (cell 84) must be re-counted at a gameplay state. |
| **C** | Logit / tuned lens | 6 core | gameplay states | Re-run | Readout fidelity vs live intermediate values. |
| **Compare** | Faithfulness leaderboard / danger-zone plot | 6 core, all A–C | rebuilt from all re-run records on the shared testbed | Re-run (aggregate) | Aggregation is only valid once every method shares one testbed + output (P4). |
| **Compare** | **Sampler-on keystone** (fig7) | 6 core (gradient family) | naive-vs-sampler faithfulness + zero semantic recovery | Re-run + New | The storyline's half-2 experiment; now runs on real gameplay states (P2). |
| **Site** | `docs/cell_footprints.jl` | pong (+ core) | gameplay analysis frame; emit **screen-buffer gradient** alongside the footprint | Re-run + New | Replace the RAM→screen *proxy* with the true gradient (P3, P5). |
| **Site** | `docs/gen_method_figures.py` | all methods | regenerate image-domain figures from the new records + screen-buffer maps | Re-run | Figures must reflect the new shared output + sampler gradients. |
| **Site** | `docs/build_pages.py` | all pages | rebuild HTML from the regenerated manifest/records | Re-run | Publish the corrected audit. |

**Run order.** §1 oracle recomputation + cause-density gate and the shared-output definition
**first** (they define ground truth for everyone); then Phase B/A/C methods on the shared
states; then the leaderboard aggregation; then the audit-website regeneration
(`cell_footprints.jl` → `gen_method_figures.py` → `build_pages.py`).
