# Beginner's guide — attribution patching / edge attribution patching.
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="attribution_patching",
    level="Mechanistic family · patching at gradient cost",

    wiki=[
        ("Taylor series", "https://en.wikipedia.org/wiki/Taylor_series"),
        ("Linear approximation", "https://en.wikipedia.org/wiki/Linear_approximation"),
        ("Finite difference", "https://en.wikipedia.org/wiki/Finite_difference"),
        ("Automatic differentiation", "https://en.wikipedia.org/wiki/Automatic_differentiation"),
    ],
    papers=[
        ("Nanda (2023). <i>Attribution Patching: Activation Patching At Industrial Scale.</i> "
         "Blog post.",
         "https://www.neelnanda.io/mechanistic-interpretability/attribution-patching"),
        ("Syed, Rager &amp; Conmy (2023). <i>Attribution Patching Outperforms Automated Circuit "
         "Discovery</i> (edge attribution patching).",
         "https://arxiv.org/abs/2310.10348"),
        ("Kram&aacute;r, Lieberum, Shah &amp; Nanda (2024). <i>AtP*: An efficient and scalable method for "
         "localizing LLM behaviour to components.</i> (Documents where the linear estimate fails, "
         "e.g. attention saturation, and proposes fixes.)",
         "https://arxiv.org/abs/2403.00745"),
    ],

    idea=r"""
<p>Activation patching is the gold standard of mechanistic interpretability, but it is slow. You need
one extra forward pass for <i>every</i> component you test. A large language model has millions of
neurons and thousands of attention heads at every token position, so exact patching of all of them is
out of reach. <b>Attribution patching</b> is the shortcut. It <b>predicts</b> the effect of every patch
at once from a single forward and backward pass.</p>
<p>The trick is the oldest one in calculus: near a point, a smooth function looks like a straight line.
If you know the <b>slope</b> of the output with respect to an activation (the gradient), and you know
how far the patch would move that activation, then slope &times; distance is a guess for how far the
output moves. That guess is a <b>first-order Taylor approximation</b>. It is exact when the output is
linear in the activation, and wrong when it is not. On the VCS we can measure the exact patch effect
for every site, so we can see precisely how much accuracy the shortcut trades for its speed.</p>""",

    nn=r"""
<p>Use the notation of activation patching: clean input \(x\), corrupted input \(\tilde x\),
component \(\ell\) with clean activation \(a = a_\ell(x)\) and corrupted activation
\(a' = a_\ell(\tilde x)\). The exact patch effect is
\(\Delta_\ell = y(x \mid a_\ell := a') - y(x)\). Attribution patching replaces it with</p>
\[ \widehat{\Delta}_\ell \;=\; \left.\frac{\partial y}{\partial a_\ell}\right|_{x} \cdot \,(a' - a). \]
<p>The gradient for <i>all</i> components comes from one backward pass on the clean run; the
\(a' - a\) terms come from one cached corrupted run. That is three passes in total, instead of one per
component. <b>Edge</b> attribution patching applies the same estimate to a connection between two
components (the sender&rsquo;s change times the receiver&rsquo;s gradient), so it can score every edge
of the computational graph cheaply.</p>
<p><b>Toy example.</b> Let \(y = a^2\) with clean \(a = 1\). The slope is \(2a = 2\).</p>
<ul>
<li>Small patch \(a' = 1.1\): estimate \(2 \times 0.1 = 0.2\); truth \(1.21 - 1 = 0.21\). Close.</li>
<li>Large patch \(a' = 4\): estimate \(2 \times 3 = 6\); truth \(16 - 1 = 15\). Badly off in size,
but the sign is right.</li>
<li>Patch at a flat point, clean \(a = 0\), \(a' = 3\): slope \(0\), estimate \(0\); truth \(9\). The
estimate misses the effect completely.</li>
</ul>
<p>The first two cases are why attribution patching is good at <i>which</i> components matter but
poor at <i>how much</i>. The third is the dangerous case: a real effect hidden behind a zero gradient,
just like saturated attention or a dead ReLU.</p>""",

    recipe=[
        "Run the corrupted input once and cache every activation \\(a'_\\ell\\).",
        "Run the clean input forward, caching \\(a_\\ell\\), and backward from the metric \\(y\\) to get every \\(\\partial y/\\partial a_\\ell\\).",
        "For each component, compute \\(\\widehat\\Delta_\\ell = \\partial y/\\partial a_\\ell \\cdot (a'_\\ell - a_\\ell)\\) (a dot product for vector activations).",
        "Rank components by \\(|\\widehat\\Delta_\\ell|\\) and keep the top ones.",
        "Verify the shortlist with exact activation patching; the estimate is a screen, not a verdict.",
    ],

    atari_map=[
        ("Activation \\(a_\\ell\\)", "One state site at frame \\(f^*\\): a candidate RAM byte or the "
         "<code>COLUBK</code> TIA register."),
        ("Corrupted activation \\(a'\\)", "The value from a donor run (LEFT or RIGHT pressed before "
         "\\(f^*\\)), or the directed value <code>base+17</code>."),
        ("Gradient \\(\\partial y / \\partial a\\)", "The local slope of the true 15-frame output, measured "
         "by poking the byte to \\(a \\pm 1\\) and re-running the real ROM (a central finite "
         "difference)."),
        ("Metric \\(y\\)", "The two most causally active candidate bytes and the count of changed "
         "screen pixels, read 15 frames later."),
        ("Exact patching (the reference)", "The full re-run with the patched value, which on the VCS "
         "equals the oracle."),
        ("An edge", "A (patched site, output) pair: one cell of the effect matrix."),
    ],

    atari=r"""
<p>The runner is built on top of the activation-patching harness. It uses the same shared gameplay
checkpoint, the same outputs and the same patch list: donor patches from LEFT/RIGHT runs, directed
<code>base+17</code> patches on every candidate byte, and one background-colour patch.</p>
<ol>
<li><b>The slope.</b> For each site the runner writes \(a+1\) and \(a-1\), re-runs 15 frames each,
and divides the difference by 2. At 0 or 255 it uses a one-sided difference. This is the
program&rsquo;s own local derivative, measured on the real machine. It is <i>not</i> a Zygote
backward pass. For a single-frame content read the two coincide, but the finite difference also
works across 15 frames of discrete code, where autodiff gives no useful answer.</li>
<li><b>The estimate.</b> \(\widehat\Delta = \text{slope} \times (a' - a)\) for every (site, output)
pair.</li>
<li><b>The truth.</b> The exact patch: write \(a'\), re-run, read the outputs.</li>
<li><b>Scores.</b> Pearson correlation of estimate vs truth over all cells, mean and relative error,
edge precision/recall (does the estimate fire where the truth fires?), and the fraction of active
edges where estimate and truth agree exactly (the &ldquo;linear regime&rdquo;).</li>
<li><b>Positive control.</b> A byte read back at the patch frame itself is perfectly linear,
\(y(a) = a\). For it the estimate must equal the truth for steps of 1 to 32. This proves the wiring
is right, so any error elsewhere is the program&rsquo;s nonlinearity.</li>
<li><b>Position gradient.</b> Because this is a gradient method, the runner also reports the naive
Zygote gradient of the screen-position output (identically zero) next to the bilinear-sampler
gradient, which restores a non-zero value.</li>
</ol>
<p>Note one honest consequence. On a network the gradient is nearly free. Here each slope costs two
re-runs, so the VCS version is not cheaper than exact patching. It is run to measure the
<i>approximation error</i>, not to save compute.</p>""",

    code=[
        dict(path="tools/xai_study/phaseC_mechanistic/attribution_patching.jl", find="function site_gradient(",
             role="xai", desc="The slope \\(\\partial y/\\partial a\\): central difference of two real re-runs at \\(a\\pm1\\)."),
        dict(path="tools/xai_study/phaseC_mechanistic/attribution_patching.jl", find="approx[i, :] = g .* Δa",
             role="xai", desc="The attribution-patching estimate: slope times the patch distance."),
        dict(path="tools/xai_study/phaseC_mechanistic/attribution_patching.jl", find="approx_fire = abs.(approx)",
             role="xai", desc="Edge precision/recall: the estimate&rsquo;s firing pattern vs the exact data-flow."),
        dict(path="tools/xai_study/phaseC_mechanistic/attribution_patching.jl", find="function linear_probe(",
             role="xai", desc="Positive control: a strictly linear site where the estimate must be exact."),
        dict(path="tools/xai_study/phaseC_mechanistic/attribution_patching.jl", find="sampler_pos_grad_max = Float64(",
             role="xai", desc="Naive (zero) vs bilinear-sampler Zygote gradient for the screen-position output."),
        dict(path="tools/xai_study/phaseC_mechanistic/activation_patching.jl", find="function run_patch(",
             role="xai", desc="The exact patch both the slope probes and the reference are built from."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function _st_sampler_position_read",
             role="substrate", desc="The bilinear sampler that gives the position output a usable gradient."),
        dict(path="jutari/src/diff/SoftStep.jl", find="function soft_ram_peek",
             role="substrate", desc="The differentiable one-hot RAM read the sampler gradient flows through."),
        dict(path="tools/xai_study/common/jutari_oracle.jl", find="function intervene_ram!(",
             role="substrate", desc="The byte write used for \\(a\\pm1\\) probes and for the exact patch."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(",
             role="substrate", desc="One bit-exact frame of the real machine."),
    ],

    reading=r"""
<p>Expect the estimate to agree with the truth on <i>which</i> edges exist far better than on their
<i>size</i>. Small donor patches and single-frame reads sit in the linear regime and match closely.
Large directed patches, bytes that wrap at 255, and bytes the game recomputes each frame break
linearity, and the scatter of estimate vs truth spreads out. The correlation is therefore good but
clearly below the exact method, and the positive control shows the gap is the program&rsquo;s fault,
not the harness&rsquo;s.</p>""",

    pitfalls=[
        "<b>Large patches are not small.</b> Taylor expansion holds near the clean point. A patch that moves an activation far (here, <code>base+17</code>) can land in a region the slope knows nothing about.",
        "<b>Zero gradient, real effect.</b> Saturated softmax, dead ReLUs or a threshold in the program give slope 0 and hide a large true effect. AtP* documents this for attention in LLMs.",
        "<b>Wrap-around and thresholds.</b> VCS bytes are 8-bit: 250 + 17 wraps to 11. Comparisons and branches make the output piecewise constant. Both break the linear model.",
        "<b>Cancellation.</b> Summing estimates over many components or positions can cancel positive and negative terms that are each large.",
        "<b>Use it as a filter.</b> The method is designed to shortlist candidates for exact patching, not to replace it.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseC_mechanistic/attribution_patching.jl --games core",
    try_note="Writes <code>attribution_patching_&lt;game&gt;.json/.npz</code> and a core summary into "
             "<code>tools/xai_study/phaseC_mechanistic/out/</code>. Three re-runs per patch (two slope "
             "probes plus the exact patch); minutes per game on one CPU core.",

    next=["activation_patching", "saliency", "acdc", "path_patching"],
)
