# Phase D — behavioral / psychology probing of the game's own logic (spec)

The third tradition (Binz & Schulz 2023; Shiffrin & Mitchell 2023 — in `papers/`):
infer a system's "cognition" from behavior. Their open worry — does behavioral probing
reveal the true mechanism or just plausible correspondences? — is unanswerable for
LLMs (no ground truth) but **answerable here on a known decision-maker: the game's own
logic** (no learned agents).

## Subject
A game's built-in decision logic — e.g., Pong's CPU opponent (paddle-tracking), Space
Invaders' alien-movement / fire logic, pursuit/evasion AIs. The true mechanism is in
the ROM (recoverable via Phase A/C).

## Probes (psychophysics-style)
Set state + re-render to present a controlled situation, then read the program's
response:
- vary object position/distance, timing, distractors, masking;
- read the resulting game behavior (the opponent's move, a fire event);
- fit "cognitive" accounts (decision variables, biases, reaction-time analogues).

## The test (vs ground truth)
Does the inferred behavioral account match the **true code**?
- compare the inferred decision variable to the true driver (the disassembled
  routine / the oracle);
- quantify "right for the wrong reasons" (good fit, wrong driver);
- test generalization of the inferred law to novel stimuli.

## Pilot (local)
`pilot_psychophysics.{py,jl}`: one controlled-factor probe of one game's opponent
logic; report inferred decision variable vs the true code.

## Scale-out (cluster)
probe batteries × factors × games — many short deterministic re-runs (CPU/GPU).

Outputs: `out/behavioral_<probe>_<game>.*` + a "trustworthy vs mirage" table feeding
the cross-tradition comparison and Phase E (semantic recovery).
