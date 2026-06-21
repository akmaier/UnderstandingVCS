# Phase B — attribution / XAI on the VCS, scored vs the oracle (spec)

**Subject: the VCS as a function** (no agents). Explain a chosen VCS output `y`
(a pixel, the score, a game event) from its causes (ROM bytes, RAM cells, registers,
joystick inputs), and score each attribution map against the `ground_truth/` oracle.

## Methods that apply to a non-NN computation
vanilla gradient / saliency, **Integrated Gradients** (extends Paper-1 P8),
SmoothGrad, occlusion/perturbation on the true VCS (Greydanus, Iyer/Anderson),
SHAP/LIME (model-agnostic), counterfactual states (made *on-distribution* here: set
state, re-render). Survey scope from XRL notes (Qing 2022, Vouros 2023, Cheng 2025,
Saulières 2025).

## Explicit N/A finding
Grad-CAM/Grad-CAM++ (need conv feature maps), attention maps, and policy-distillation
surrogates are NN-architecture-specific and do **not** apply to the VCS — recorded as
a measured statement about the narrowness of popular XAI.

## Faithfulness metrics (vs ground truth)
correlation with the true causal map; deletion/insertion AUC on the *true* VCS;
precision@k / pointing-game / object-hit vs the true causal top-k; plausibility (for
the faithfulness-vs-plausibility plot).

## Pilot (local)
`pilot_ig_vs_oracle.py`: Integrated Gradients vs the exact intervention oracle for one
output (e.g., the Pong score) on one game; report faithfulness. Proves the pipeline
(builds on `ground_truth/oracle_pong`).

## Scale-out (cluster, GPU)
Attribution + oracle over outputs × causes × games (batched SOFT-STE exact-forward).
See plan §4.6.

Outputs: `out/faithfulness_<method>_<game>_<output>.*` → leaderboard (method →
faithfulness), feeding the cross-tradition comparison (Results reporting) and Phase E
(semantic recovery).
