# Beginner's guide — Guided Backpropagation (+ the Adebayo sanity check).
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="guided_backprop",
    level="Gradient family · a modified backward pass",

    wiki=[
        ("Rectified linear unit (ReLU)", "https://en.wikipedia.org/wiki/Rectified_linear_unit"),
        ("Backpropagation", "https://en.wikipedia.org/wiki/Backpropagation"),
        ("Edge detection", "https://en.wikipedia.org/wiki/Edge_detection"),
        ("Saliency map", "https://en.wikipedia.org/wiki/Saliency_map"),
    ],
    papers=[
        ("Springenberg, Dosovitskiy, Brox &amp; Riedmiller (2015). <i>Striving for Simplicity: "
         "The All Convolutional Net.</i> ICLR workshop (introduces guided backpropagation).",
         "https://arxiv.org/abs/1412.6806"),
        ("Adebayo et al. (2018). <i>Sanity Checks for Saliency Maps.</i> NeurIPS.",
         "https://arxiv.org/abs/1810.03292"),
        ("Nie, Zhang &amp; Patel (2018). <i>A Theoretical Explanation for Perplexing Behaviors of "
         "Backpropagation-based Visualizations.</i> ICML.",
         "https://arxiv.org/abs/1805.07039"),
    ],

    idea=r"""
<p>Vanilla saliency maps of image classifiers look grainy. Positive and negative gradient signals
mix as they flow back through the network. <b>Guided backpropagation</b> changes one rule in the
backward pass to clean this up. At every <b>ReLU</b> (rectified linear unit, \(\max(0, z)\), the
standard on/off nonlinearity of deep nets) it lets a gradient through only if the unit was
<i>active</i> in the forward pass <b>and</b> the gradient arriving from above is <i>positive</i>.
Negative signals (&ldquo;this feature argues <i>against</i> the class&rdquo;) are thrown away.</p>
<p>The result looks spectacular: crisp, edge-like pictures of the object. That is why the method
became popular. But it is no longer the gradient of anything. It is a hand-modified signal.
Adebayo et al. (2018) asked a simple question: <i>does the map change if we destroy the model?</i>
For guided backprop it barely does. The map behaves more like an edge detector applied to the
input than an explanation of the network. This test, the <b>model-randomisation sanity check</b>,
is part of our experiment, run on the machine itself.</p>""",

    nn=r"""
<p>Take one ReLU unit \(h = \max(0, z)\) and let \(g = \partial y/\partial h\) be the gradient arriving
from above. The three backward rules differ only in what they send down to \(z\):</p>
\[ \text{backprop: } \mathbb{1}[z>0]\,g, \qquad
   \text{deconvnet: } \mathbb{1}[g>0]\,g, \qquad
   \text{guided: } \mathbb{1}[z>0]\,\mathbb{1}[g>0]\,g . \]
<p>Guided backprop combines the two masks. Applied at every ReLU of the network it yields a map
\(G(x)\) that, unlike saliency, is not the derivative \(\nabla f(x)\).</p>
<p><b>Toy example.</b> Two inputs, two hidden ReLUs, one output:
\(h_1 = \max(0, 2x_1 + x_2)\), \(h_2 = \max(0, x_1 - x_2 + 1)\), \(y = h_1 - h_2\).
At \(x = (1, 1)\): \(h_1 = 3\), \(h_2 = 1\), \(y = 2\). Both units are active. From above,
\(\partial y/\partial h = (1, -1)\).</p>
<ul>
<li><b>Vanilla gradient:</b> \(1\cdot(2, 1) + (-1)\cdot(1, -1) = (1, 2)\). It says \(x_2\) matters
most.</li>
<li><b>Guided:</b> the \(-1\) arriving at \(h_2\) is negative, so it is zeroed. Only
\(1\cdot(2,1) = (2, 1)\) survives. It says \(x_1\) matters most.</li>
</ul>
<p>The ranking flipped. \(h_2\) <i>inhibits</i> the output and \(x_2\) acts partly by reducing that
inhibition. Guided backprop discards inhibition, so it cannot see this. Check the truth directly:
raising \(x_2\) by 0.1 raises \(y\) by 0.2, raising \(x_1\) by 0.1 raises it by only 0.1. Here the
vanilla gradient was right and the &ldquo;cleaner&rdquo; map was wrong.</p>""",

    recipe=[
        r"Run the forward pass and record which ReLUs are active (\(z > 0\)).",
        r"Start the backward pass from the scalar output \(y\), as in vanilla saliency.",
        r"At every ReLU, pass the gradient down only where \(z > 0\) <b>and</b> the incoming gradient is \(> 0\); set everything else to 0.",
        r"Continue to the input. Display the (usually absolute) result as a heat-map.",
        r"Sanity check (Adebayo): randomise the model's weights, recompute the map, and verify it <i>changes</i>. A map that stays the same cannot be explaining the model.",
    ],

    atari_map=[
        (r"The network \(f\)", r"The Atari 2600 VCS running the ROM, emulated bit-exactly by <code>jutari</code>; "
         r"its differentiable twin is <code>JuTari.Diff</code>."),
        (r"The ReLUs", r"There are none. The soft VCS's nonlinearities are straight-through round/clamp "
         r"(identity backward), the stop-gradient branch estimator, and the distance-softmax read used only in the relaxation study. "
         r"So the guided rule is applied once, to the assembled gradient over RAM."),
        (r"The input \(x\)", r"The 128 RAM bytes at the intervention frame \(f^*\)."),
        (r"The output \(y\)", r"A content output (one RAM byte read one-hot) and a position output "
         r"(screen pixels changed versus the un-intervened frame, made differentiable by the bilinear sampler)."),
        (r"Randomising the weights", r"Randomising the <b>program</b>: every ROM byte is replaced by a random "
         r"byte, except the CPU's reset/interrupt vectors so the console still boots."),
        (r"&ldquo;Is the map right?&rdquo;", r"The intervention oracle: poke each cause, re-run the real ROM, measure \(|\Delta y|\)."),
    ],

    atari=r"""
<p>Our substrate has no deep stack of ReLUs. The runner says so plainly, and implements guided
backprop as the rectification rule \(g \mapsto g\odot\mathbb{1}[g>0]\) applied to the gradient
\(\partial y/\partial u\) over the RAM vector \(u\), followed by an absolute value. It returns
the guided map <i>and</i> the vanilla \(|\partial y/\partial u|\), so the effect of the rule can
be measured directly.</p>
<ul>
<li><b>Content output.</b> \(y = \) <code>soft_ram_peek(ram, idx)</code>, whose gradient is the one-hot
vector \(e_{\text{idx}}\). It has no negative entries, so there is nothing to suppress: the guided
map equals vanilla saliency exactly. The runner records this as <code>guided_equals_vanilla</code>.
The byte <code>idx</code> is the candidate the oracle shows to be most causally active, the same
choice as the SmoothGrad runner, so only the backward rule differs.</li>
<li><b>Is the rule real?</b> To show it is not a no-op stub, <code>guided_demo_rectifies</code> applies it
to a small linear surrogate with mixed-sign weights and asserts that every negative entry is
zeroed.</li>
<li><b>Position output.</b> The naive gradient is zero (sprite position is set by strobe timing). The
bilinear sampler restores one, but its sign depends on which side of the screen cell the sprite
edge sits: moving the sprite can <i>reduce</i> the cell's occupancy. A negative slope is exactly
what the guided rule deletes. So guided backprop can throw away the very gradient the sampler
restored. The runner reports the raw sampler gradient next to the rectified one.</li>
<li><b>The Adebayo check.</b> <code>randomize_rom</code> fills the cartridge with random bytes but keeps the
last 6 bytes of each 4&nbsp;KB bank (the 6502 reset/IRQ/NMI vectors). Three randomised programs are
booted and replayed to \(f^*\). The runner counts how many RAM bytes differ from the real game,
recomputes the guided map, and measures its cosine similarity to the real map. The scrambled programs
reach very different RAM states. Yet the content map is identical, because the one-hot Jacobian
does not depend on what the program computed. The runner classifies this as a genuine failure
(&ldquo;model-invariant saliency, constant Jacobian&rdquo;), not as a harness error.</li>
<li><b>Scoring</b> is shared with the other gradient methods: correlation and precision@3 against the
oracle, and deletion/insertion curves re-run on the real ROM, plus an oracle-vs-itself control.</li>
</ul>""",

    code=[
        dict(path="tools/xai_study/phaseB_attribution/guided_backprop.jl", find="guided_rule(g::AbstractVector{<:Real}) = map",
             role="xai", desc="The guided rule: keep positive gradient entries, zero the negative ones."),
        dict(path="tools/xai_study/phaseB_attribution/guided_backprop.jl", find="function guided_backprop_over_ram(",
             role="xai", desc="One Zygote gradient over RAM, then the rule; returns guided and vanilla maps side by side."),
        dict(path="tools/xai_study/phaseB_attribution/guided_backprop.jl", find="function guided_demo_rectifies(",
             role="xai", desc="Proof the rule is wired: a mixed-sign surrogate whose negative gradients must be suppressed."),
        dict(path="tools/xai_study/phaseB_attribution/guided_backprop.jl", find="function randomize_rom(",
             role="xai", desc="Model randomisation for the VCS: random ROM bytes, CPU vectors preserved."),
        dict(path="tools/xai_study/phaseB_attribution/guided_backprop.jl", find="function adebayo_sanity(",
             role="xai", desc="The Adebayo sanity check: cosine/Spearman of the map under program randomisation, with a failure-mode diagnosis."),
        dict(path="tools/xai_study/phaseB_attribution/guided_backprop.jl", find="function compute_one(",
             role="xai", desc="One output: oracle &rarr; guided map &rarr; scores &rarr; deletion/insertion &rarr; control &rarr; sanity check."),
        dict(path="jutari/src/diff/SoftStep.jl", find="function soft_ram_peek",
             role="substrate", desc="The one-hot RAM read: a linear unit with a non-negative gradient, so the guided rule has nothing to remove."),
        dict(path="jutari/src/diff/StraightThrough.jl", find="@inline straight_through_round(x::Real)",
             role="substrate", desc="The straight-through round: one of the soft VCS's actual nonlinearities (exact forward, identity backward). Not a ReLU."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function _st_sampler_position_read",
             role="substrate", desc="The bilinear sampler; its triangular kernel has slopes of both signs."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function run_intervention",
             role="substrate", desc="The ground truth: poke a cause, re-run the real ROM, read the output."),
    ],

    reading=r"""
<p>On the <b>content</b> output the guided bars are the vanilla saliency bars, exactly. Any
agreement with the oracle is saliency's agreement. On the <b>position</b> output the map is often
empty, either because the naive gradient is zero or because the rule removed a negative sampler
slope. Then look at the sanity-check box. A cosine similarity near 1 between the real and the
scrambled-program maps means the map does not depend on the program. A faithful map must change
when the model is destroyed; this one stays the same. The oracle, by
contrast, re-runs the program and would see the difference at once.</p>""",

    pitfalls=[
        r"<b>Not a gradient.</b> Guided backprop is a modified signal. It has no completeness property and no clear causal meaning.",
        r"<b>Discards inhibition.</b> Inputs that matter by suppressing a unit are invisible to it, as the toy example shows.",
        r"<b>Model-insensitive.</b> Adebayo et al. and Nie et al. showed that guided maps largely reconstruct the input image and change little when weights are randomised. On the VCS content read the map does not change at all.",
        r"<b>Substrate mismatch.</b> The VCS has no ReLU layers, so here the method collapses to rectified vanilla saliency. That is an honest limit of transferring a network-specific rule to a different system.",
        r"<b>Looks are misleading.</b> Sharper is not truer. Only the intervention oracle decides.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseB_attribution/guided_backprop.jl --games core --sanity",
    try_note="Writes <code>guided_backprop_&lt;game&gt;_{content,ball_pixel}.json/.npz</code> and "
             "<code>guided_backprop_core_summary.json</code> into "
             "<code>tools/xai_study/phaseB_attribution/out/</code>. The sanity check boots three randomised "
             "ROMs per output; pass <code>--no-sanity</code> for a faster smoke run.",

    next=["saliency", "smoothgrad", "gradxinput", "occlusion"],
)
