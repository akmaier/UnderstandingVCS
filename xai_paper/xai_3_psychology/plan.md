# P3 — Paper plan (storyline & venue structure)

> **Role of this file:** the storyline + venue structure. Experiments → `experiment_
> design.md`. Program map + shared invariants → [`../general_paper_plan.md`](../general_paper_plan.md).
> **Shared oracle / T3 / correctness triad are defined in
> [`../xai_2_interpretability/experiment_design.md`](../xai_2_interpretability/experiment_design.md)
> §0–§3 — reference, don't repeat.** Status: carved plan; awaits P2 results.

## P3 in one line
*The psychology of a known machine.* Treat a game's **own hand-coded decision logic**
(e.g., Pong's CPU opponent) as a "participant," probe it with behavioral / psychology-of-
AI methods, and give the **first ground-truthed verdict** on whether those inferences
recover the true mechanism. This is **Phase D** of the program.

## Metadata
- **Working title:** *The psychology of a known machine: a ground-truth test of
  behavioral inference.*
- **Authors (provisional):** A. Maier, P. Krauss, S. Bayer (+ a cognitive-science
  collaborator — recommended for this venue).
- **Target:** **PNAS** (primary). Binz & Schulz 2023 and Shiffrin & Mitchell 2023 — the
  exact debate this answers — are PNAS. Alternatives: NMI; *Cognition* / CogSci.

## The gap it fills
Binz & Schulz (2023) propose treating an AI as a participant in cognitive-psychology
experiments; Shiffrin & Mitchell (2023) warn this may reveal *plausible correspondences,
not the true mechanism* — and for an LLM there is **no ground truth to check against**.
A game's decision logic is a *known* decision-maker (the mechanism is in the ROM), so we
**can** check. P3 is the missing experiment.

## Subject
A game's built-in decision logic — Pong's CPU opponent (paddle tracking), Space
Invaders' alien-movement / fire logic, pursuit/evasion AIs. Controlled stimuli are
created by **state-set + re-render** on the bit-exact substrate; the response is read
from the framebuffer / RAM. No learned agents.

## Why behavioral probing matters beyond "another tradition"
Operational semantics tells you a byte's *mechanism*; it never tells you it *means*
"ball-x." That symbol→concept link can only be established **observationally** — vary it,
watch the response, name the variable. So behavioral probing is the **semantic-grounding
bridge** from mechanism to meaning, and it is the engine that *feeds documentation
recovery* (P4). P3 establishes how trustworthy that bridge is.

## Contributions
1. The **first ground-truthed verdict** on psychology-of-AI / behavioral-probing
   methodology — a measured split into *trustworthy* (matches truth + generalizes) vs
   *mirage* (fits but wrong driver), with the conditions that predict each.
2. A reusable **behavioral-probing harness** on the VCS (controlled-stimulus generation
   + the true-driver oracle).
3. Evidence on the Shiffrin–Mitchell vs Binz–Schulz debate, with ground truth.
4. The **semantic-grounding** result connecting behavior to documentation (sets up P4).

## Limits (state plainly)
Behavioral grounding recovers only **externally manifest** meanings (something on
screen / in the score changes). Pure internal bookkeeping with no observable correlate
needs finer interventions and is out of behavioral reach — a reportable boundary.

---

# Paper structure (PNAS)
- **Significance statement** (PNAS-required, ≤120 words): we can finally check whether
  "doing psychology on a machine" recovers its true mechanism — and we measure it.
- **Abstract.**
- **Introduction:** the behavioral tradition; the no-ground-truth problem; the known-
  machine opportunity.
- **Results:** the Phase-D experiments (see `experiment_design.md`) — psychometric /
  SDT / reverse-correlation / drift-diffusion / cognitive-battery probes, each scored vs
  the true code; the trustworthy-vs-mirage map.
- **Discussion:** what behavioral inference can and cannot recover; implications for
  psychology-of-AI (LLMs); the bridge to documentation recovery (P4).
- **Materials and Methods:** the VCS substrate, the shared oracle (cite P2 §1), the
  controlled-stimulus protocol, the true-driver ground truth.
- **Data/Code availability; Author contributions; Competing interests.**

## Open questions
- Full standalone (PNAS) vs ride-along in P2 — decided **standalone** (different
  reviewer community); revisit only if P2 wants a bigger flagship.
- Which games' AIs give the cleanest decision variables for the headline (Pong opponent
  is the safe lead).
