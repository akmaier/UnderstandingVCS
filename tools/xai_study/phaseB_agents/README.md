# Phase B — XAI on DQN agents, scored vs the oracle (spec)

Audit interpretability methods on DQN agents by scoring their explanations against
the `ground_truth/` oracle (exact intervention + end-to-end gradient). **Two
first-class families:** B1 attribution/saliency and B2 mechanistic-interpretability.

## Agents
Prefer existing zoo agents (Such et al. 2019, Atari Model Zoo); Mnih-2015 DQN.
Run on jutari/jaxtari. Avoid training from scratch (GPU cost) unless needed.

## B1 — Attribution / saliency
vanilla gradient, Integrated Gradients, SmoothGrad, Grad-CAM, Grad-CAM++,
occlusion/perturbation (Greydanus), object saliency (Iyer/Anderson), attention
(Mott, Nikulin), SHAP/LIME, counterfactual (Atrey/Olson); interpretable-by-design
baselines (XDQN; Barbiero-style).

## B2 — Mechanistic interpretability (first-class)
**activation patching / causal tracing, attribution patching (+ edge AP), sparse
autoencoders / dictionary learning, circuit discovery, linear probing.**
Infra to build: agent-activation capture, patch (resample/clamp) hooks, SAE
training, circuit search. Score recovered structure against truth:
- patched effect vs the **exact** intervention effect (the oracle);
- SAE features ↔ **known game variables** (we know them);
- recovered circuit ↔ true data-flow (for the chip in Phase A this is exact).

## Faithfulness metrics (vs ground truth)
correlation with the true causal map; deletion/insertion AUC on the *true*
emulator; pointing-game/object-hit; (B2) recovered-effect vs exact-patch
agreement; feature↔variable matching.

## Pilots (local)
- `pilot_ig_vs_oracle.py` (B1): one agent + Integrated Gradients vs the
  intervention oracle on one game; report faithfulness. Proves the pipeline.
- `pilot_patch_sae.py` (B2): activation patching + one SAE on the same agent;
  patched-effect vs exact-patch; SAE-feature vs known-variable matching.

## Scale-out (cluster, GPU)
Oracle over per-pixel/object occlusions × states × agents × games (batched
SOFT-STE exact-forward); end-to-end gradients; SAE training. See plan §4.6.

Outputs: `out/faithfulness_<method>_<agent>_<game>.*` → leaderboard (method →
faithfulness), feeding benchmark C1.
