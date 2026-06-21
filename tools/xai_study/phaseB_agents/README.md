# Phase B — modern XAI on DQN agents, scored vs the oracle (spec)

Audit today's interpretability methods on DQN agents by scoring their attributions
against the `ground_truth/` oracle (exact intervention + end-to-end gradient).

## Agents
Prefer existing zoo agents (Such et al. 2019, Atari Model Zoo); Mnih-2015 DQN.
Run them on jutari/jaxtari. Avoid training from scratch (GPU cost) unless needed.

## Methods under test (see plan §4.4 for success/failure predictions)
- **Classic / visual:** vanilla gradient, Integrated Gradients, SmoothGrad,
  Grad-CAM, Grad-CAM++, occlusion/perturbation (Greydanus), object saliency
  (Iyer/Anderson), attention (Mott, Nikulin), SHAP/LIME, counterfactual (Atrey/Olson).
- **Newest / causal (add — see §4.5):** activation patching / causal tracing,
  attribution patching (+ edge AP), causal scrubbing, sparse autoencoders,
  circuit analysis, linear probing.
- **RL-native:** reward decomposition (RDX/MSX), surrogate trees (VIPER/XDQN).

## Faithfulness metrics (vs ground truth)
- correlation with the true causal map;
- deletion / insertion AUC measured on the **true emulator** (not a proxy);
- pointing-game / object-hit rate against true causal objects;
- (causal methods) agreement of patched effect with the exact intervention effect.

## Pilot (`pilot_ig_vs_oracle.{py}`, local)
One agent + Integrated Gradients vs the intervention oracle on one game (Pong/SI):
compute IG saliency, compute the oracle map, report the faithfulness metrics.
Proves the end-to-end `emulator ∘ agent` attribution pipeline.

## Scale-out (cluster, GPU)
Oracle over per-pixel/object occlusions × states × agents × games (batched
SOFT-STE exact-forward); end-to-end gradients; SAE training. See plan §4.6.

Outputs: `out/faithfulness_<method>_<agent>_<game>.*` + a leaderboard table
(method → faithfulness), feeding benchmark C1.
