# P3 — Experiment design (Phase D: behavioral / psychology probing)

> **Role of this file:** the experiments and how each is scored. Storyline → `plan.md`.
> **The correctness triad (§0), the ground-truth oracle (§1), the T3 procedure (§2),
> and the substrate audit (§3) are SHARED and defined in
> [`../xai_2_interpretability/experiment_design.md`](../xai_2_interpretability/experiment_design.md)
> §0–§3 — reused here, not repeated.** Subject: a game's own decision logic (no learned
> agents). Citations: verify before bib.

## Substrate feasibility (delta vs the shared §3 audit)
Phase D needs: **set state + re-render controlled stimuli**, and **read the program's
response** — both available now. **T3 is the most load-bearing tier here** (to vary a
*named* game factor we set the corresponding RAM byte; to score "inferred driver = true
driver" we must name the true driver). Restrict scored claims to games with verified T3
(shared §2). Where T3 is missing, vary raw inputs (joystick) or vision-selected frames
at the cost of a coarser factor. Stay within the conformance horizon (shared §3).

## Phase D — methods

| Analysis (named method) | Finding (output) | Measured score | Needs T3? |
|---|---|---|---|
| Psychometric-function fitting (Wichmann & Hill 2001) | response curve vs a stimulus factor (threshold, slope) | inferred factor = true driver? threshold vs the coded boundary | **Yes — stimulus + scoring** |
| Method of constant stimuli / adaptive staircases | decision-boundary estimate | vs the true boundary in the code | **Yes — stimulus + scoring** |
| Signal-detection theory (Green & Swets 1966) | d′, criterion | vs the true (deterministic) decision rule | **Yes — stimulus + scoring** |
| Reverse correlation / classification images (Ahumada 1971; Murray 2011) | recovered stimulus "template" | overlap with the true input region the code reads | **No** (template vs the code's *read-set* = T1/T2); naming = T3 |
| Drift-diffusion / sequential sampling (Ratcliff 1978) | inferred decision variable + "frames-to-respond" | inferred variable vs true driver; latency vs the true code latency | **Yes — scoring**; stimulus optional |
| Ideal-observer analysis | the optimal-strategy account | gap to the actual (often suboptimal) coded strategy | **Yes — scoring** |
| Cognitive-task battery, Binz & Schulz (2023) style; Shiffrin & Mitchell (2023) caveats | behavioral "trait" inferences | "right-for-the-wrong-reasons" rate vs the code; generalization to novel stimuli | **Yes — stimulus + scoring** |

> **Phase D is the most T3-dependent** phase (named-factor stimuli + named true driver).
> The exception is *reverse correlation*, whose recovered template scores against the
> code's read-set (T1/T2) with no labels.

**Ideal:** a behavioral law over the program's true decision variables (e.g., "the
opponent moves toward `sign(ball_y − paddle_y)` with a 1-frame lag and a dead-zone").
**Right when:** inferred variable = true driver (F, verified against the code); predicts
novel stimuli (S, on the exact re-run); simplest law (M). **Best case:** the first
ground-truthed verdict — a measured *trustworthy vs mirage* split with the conditions
that predict each, and the boundary (externally-manifest meanings only).

## Pilot (local)
`pilot_psychophysics`: one controlled-factor probe of one game's opponent logic
(Pong CPU paddle); report inferred decision variable vs the true code; reuse the shared
oracle (P2 §1) and `tools/xai_study/phaseD_behavioral/`.

## Scale-out (cluster)
probe batteries × factors × games — many short deterministic re-runs (CPU/GPU).
