"""The XAI primer page (docs/primer.html): the shared vocabulary every method guide
(docs/guides/<key>.py) builds on. Rendered by build_pages.build_primer().

Written for a deep-learning beginner. It introduces explainable AI, the Atari 2600 VCS
as a "network whose wiring we know", the intervention oracle that supplies ground
truth, the scoring, and a reading order through the 30 methods. Code links use the
same (path, find) scheme as the guides, resolved to line numbers at build time.
"""

W = "https://en.wikipedia.org/wiki/"

# (path, find) code anchors used below — resolved by build_pages._find_line
CODE = {
    "env_step": ("jutari/src/env/StellaEnvironment.jl", "function env_step!"),
    "env_reset": ("jutari/src/env/StellaEnvironment.jl", "function env_reset!"),
    "run_until_frame": ("jutari/src/Console.jl", "function run_until_frame!"),
    "cpu_step": ("jutari/src/Console.jl", "function console_step!"),
    "soft_ram_peek": ("jutari/src/diff/SoftStep.jl", "function soft_ram_peek"),
    "soft_system_step": ("jutari/src/diff/SoftSystem.jl", "function soft_system_step"),
    "intervene_ram": ("tools/xai_study/common/jutari_oracle.jl", "function intervene_ram!"),
    "run_intervention": ("tools/xai_study/ground_truth/oracle_intervene.jl", "function run_intervention"),
    "build_causes": ("tools/xai_study/ground_truth/oracle_intervene.jl", "function build_pong_causes"),
    "testbed": ("tools/xai_study/common/shared_testbed_impl.jl", "function build_shared_testbed"),
    "actions": ("tools/xai_study/common/shared_testbed_impl.jl", "function shared_gameplay_actions"),
    "sampler": ("tools/xai_study/common/shared_testbed_impl.jl", "function _st_sampler_position_read"),
    "oracle_abs": ("tools/xai_study/phaseB_attribution/saliency.jl", "function oracle_abs_delta"),
    "delins": ("tools/xai_study/phaseB_attribution/saliency.jl", "function deletion_insertion_auc"),
    "pearson": ("tools/xai_study/phaseB_attribution/pilot_ig_vs_oracle.jl", "function pearson("),
    "patk": ("tools/xai_study/phaseB_attribution/pilot_ig_vs_oracle.jl", "function precision_at_k("),
    "e2e": ("tools/xai_study/phaseB_attribution/e2e_gradient.jl", "function e2e_vjp"),
}

# Families: (anchor, title, blurb, [method keys]) — the reading order of the catalogue
FAMILIES = [
    ("fam-grad", "1 · Gradient methods &mdash; &ldquo;how sensitive is the output?&rdquo;",
     "Differentiate the output with respect to the input. One backward pass, no re-runs. They need a "
     "differentiable model, which is why the VCS has a differentiable twin. Start with saliency; every "
     "other member fixes one of its problems (scale, noise, saturation, the choice of baseline).",
     ["saliency", "gradxinput", "smoothgrad", "guided_backprop", "ig_baseline_sweep",
      "expected_gradients"]),
    ("fam-pert", "2 · Perturbation methods &mdash; &ldquo;what happens if I change it?&rdquo;",
     "Treat the model as a black box: change inputs, re-run, watch the output. No gradients needed. "
     "On the VCS, re-running the model means re-running the real emulator, so these are the closest "
     "relatives of the oracle. They differ in <i>what</i> they change and how they summarise it.",
     ["occlusion", "perturbation", "rise", "lime", "kernelshap", "counterfactual"]),
    ("fam-mech", "3 · Mechanistic interpretability &mdash; &ldquo;which internal parts compute it?&rdquo;",
     "Open the box. Patch internal activations between two runs, find circuits, learn feature "
     "dictionaries, probe for information. Developed for transformers and LLMs; here the "
     "&ldquo;activations&rdquo; are RAM bytes and registers of a machine whose circuit we can read off "
     "the ROM.",
     ["activation_patching", "attribution_patching", "path_patching", "acdc", "das",
      "causal_scrubbing", "sae", "dictionaries", "linear_probing", "logit_lens"]),
    ("fam-neuro", "4 · The neuroscience battery &mdash; &ldquo;could a neuroscientist understand a "
     "microprocessor?&rdquo;",
     "The classic toolbox of systems neuroscience &mdash; lesions, tuning curves, correlations, field "
     "potentials, Granger causality, dimensionality reduction &mdash; applied to a chip, after Jonas "
     "&amp; Kording (2017). Many of these have direct deep-learning analogues (ablations, neuron "
     "selectivity, PCA of activations).",
     ["A1_connectomics", "A2_lesions", "A3_tuning", "A4_correlations", "A5_lfp", "A6_granger",
      "A7_dimred", "A8_wholestate"]),
    ("fam-na", "5 · When a method does not apply",
     "Some methods assume things the VCS does not have (e.g. attention heads, a token vocabulary). "
     "Marking them not-applicable, with the reason, is part of an honest benchmark.",
     ["na_audit"]),
]

GLOSSARY = [
    ("Attribution", "A score per input (or per internal unit) saying how much it contributed to an output."),
    ("Faithfulness (F)", "Does the explanation agree with what the model actually does? Here: correlation "
     "of the method's scores with the oracle's true effects."),
    ("Sufficiency (S)", "Can the explanation <i>predict</i> what happens under new, held-out interventions?"),
    ("Minimality (M)", "Does the explanation name only what is needed? True minimal cause set / named set."),
    ("Intervention / do(&middot;)", "Setting a variable to a value by force and re-running, as opposed "
     "to merely observing it. Written \\(do(u := v)\\)."),
    ("Oracle", "Here: the exact answer to &ldquo;what does intervention \\(u\\) do to output \\(y\\)?&rdquo;, "
     "obtained by actually re-running the bit-exact emulator."),
    ("Cause \\(u\\)", "One candidate intervention: a RAM byte set to a new value or zeroed, a TIA colour "
     "register, or a different joystick action."),
    ("Content output", "A value the machine stores (one RAM byte at the end of the horizon), e.g. the score."),
    ("Position output", "How many screen pixels differ from the un-intervened frame: depends on <i>where</i> a sprite is drawn."),
    ("Vanishing gradient", "A gradient that is exactly zero although the input matters, because the "
     "function is flat or piecewise constant around the current point."),
    ("Straight-through estimator (STE)", "Use the exact (hard, discrete) value in the forward pass and "
     "a smooth surrogate's derivative in the backward pass."),
    ("Bit-exact", "Every RAM byte, register and pixel identical to the reference emulator (xitari / the "
     "Arcade Learning Environment) at every frame."),
    ("Frame", "One video picture: 262 scanlines, about 1/60 s, a few thousand CPU instructions."),
    ("Horizon", "The number of frames played after the intervention before the output is read (15 here)."),
    ("Baseline / reference", "The &ldquo;neutral&rdquo; input some methods compare against (Integrated "
     "Gradients, SHAP, occlusion). Choosing it is a modelling decision, not a detail."),
    ("Superposition", "A network storing more features than it has neurons, by overlapping them; the "
     "reason sparse autoencoders are used."),
]
