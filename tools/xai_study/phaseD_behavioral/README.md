# Phase D — behavioral / psychology probing, scored vs ground truth (spec)

The third tradition (Binz & Schulz 2023; Shiffrin & Mitchell 2023 — in `papers/`):
treat the system as a *participant* and infer its "cognition" from behavior. Their
open worry — does behavioral probing reveal the true mechanism or just plausible
correspondences? — is unanswerable for LLMs (no ground truth) but **answerable
here**.

## Subjects
- DQN agent (primary); the chip as a stress test.

## Probes (psychophysics-style)
Vary one input factor under control and read the behavioral response:
- object position / distance, distractor presence, reward-cue manipulation,
  stimulus onset timing, masked vs visible objects;
- read action choice and Q-values; fit "cognitive" accounts (decision variables,
  biases, tuning, reaction-time analogues).

## The test (vs ground truth)
Does the inferred behavioral account match the **known** mechanism?
- agent: compare inferred decision variable to the true one (oracle + known game
  variables);
- chip: compare to Phase-A ground truth.
- Quantify "right for the wrong reasons": behavioral fit that does *not* match the
  true causal driver.

## Pilot (local)
`pilot_psychophysics.py`: one controlled-factor probe of one agent on one game;
report inferred decision variable vs the true causal driver.

## Scale-out (cluster)
Probe batteries × factors × agents × games — many short rollouts (CPU/GPU).

Outputs: `out/behavioral_<probe>_<agent>_<game>.*` + a "trustworthy vs mirage"
table feeding the cross-tradition comparison and benchmark C1.
