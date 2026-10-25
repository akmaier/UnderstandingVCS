# Beginner's guide — SmoothGrad.
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="smoothgrad",
    level="Gradient family · saliency averaged over noise",

    wiki=[
        ("Normal distribution", "https://en.wikipedia.org/wiki/Normal_distribution"),
        ("Gaussian noise", "https://en.wikipedia.org/wiki/Gaussian_noise"),
        ("Monte Carlo method", "https://en.wikipedia.org/wiki/Monte_Carlo_method"),
        ("Smoothing", "https://en.wikipedia.org/wiki/Smoothing"),
    ],
    papers=[
        ("Smilkov, Thorat, Kim, Vi&eacute;gas &amp; Wattenberg (2017). <i>SmoothGrad: removing noise "
         "by adding noise.</i> ICML workshop.",
         "https://arxiv.org/abs/1706.03825"),
        ("Hooker, Erhan, Kindermans &amp; Kim (2019). <i>A Benchmark for Interpretability Methods in "
         "Deep Neural Networks</i> (ROAR). NeurIPS.",
         "https://arxiv.org/abs/1806.10758"),
    ],

    idea=r"""
<p>Vanilla saliency maps of image networks look speckled. Neighbouring pixels get wildly different
values. Smilkov et al. argued that this is not what the network &ldquo;thinks&rdquo;. It is because
the gradient of a deep ReLU network jumps around from point to point: move the input a hair and
the gradient changes a lot. The map is accurate for that exact point, but noisy as a summary.</p>
<p><b>SmoothGrad</b> fixes the look with a simple trick: make many copies of the input, add a little
random noise to each, compute the saliency map of every copy, and <b>average</b> them. The random
jumps cancel out; what is consistently there survives. In the authors' words, it removes noise by
adding noise. It works as a wrapper around any gradient method. The catch is in the word
<i>look</i>. Smoothing makes maps cleaner to the eye, and whether it makes them more <i>faithful</i>
is a separate question. On the VCS we can answer that question exactly.</p>""",

    nn=r"""
<p>With \(y = f(x)\) as in vanilla saliency, SmoothGrad averages the gradient over Gaussian
(normal-distribution) noise of standard deviation \(\sigma\):</p>
\[ \mathrm{SG}_i(x) \;=\; \frac{1}{N}\sum_{k=1}^{N}
   \left|\frac{\partial f}{\partial x_i}\big(x + \varepsilon_k\big)\right|,
   \qquad \varepsilon_k \sim \mathcal{N}(0, \sigma^2 I). \]
<p>This is a <b>Monte Carlo</b> estimate (an average over random samples) of the expected gradient
in a fuzzy neighbourhood of \(x\). Smilkov et al. recommend \(\sigma\) at 10&ndash;20&nbsp;% of the
input range and \(N\) around 50. Some variants average the signed gradient; ours averages
absolute values.</p>
<p><b>Toy example: smoothing can wake up a dead gradient&hellip;</b> Take a single ReLU,
\(f(x) = \max(0, x)\), at \(x = -0.1\). The vanilla gradient is 0: saliency says the input
does not matter. Now draw \(N = 4\) noise values \(\varepsilon = (-0.8, 0.3, 1.2, -0.2)\). The noisy inputs are
\((-0.9, 0.2, 1.1, -0.3)\), with gradients \((0, 1, 1, 0)\). SmoothGrad gives \(0.5\). With many samples and
\(\sigma = 1\) it converges to \(P(x + \varepsilon > 0) \approx 0.46\). Because the kink is within
reach of the noise, some samples land where the slope is non-zero.</p>
<p><b>&hellip;but not a flat one.</b> Now take a step, \(f(x) = \mathbb{1}[x > 0]\). Its derivative
is 0 on both sides of the jump. Every noisy sample also has gradient 0, and the average of zeros is
zero. And for a straight line, \(f(x) = 3x\), every sample has gradient 3: smoothing changes
nothing. Remember both cases: the VCS gives us exactly these two.</p>""",

    recipe=[
        r"Choose the noise level \(\sigma\) (relative to the input range) and the sample count \(N\).",
        r"Draw \(N\) noise vectors \(\varepsilon_k \sim \mathcal{N}(0, \sigma^2 I)\), one value per input.",
        r"For each noisy copy \(x + \varepsilon_k\), run one forward and one backward pass and take \(|\partial y/\partial x_i|\).",
        r"Average the \(N\) maps. Optionally also keep the noise-free map for comparison.",
        r"Display the averaged map. Cost: \(N\) backward passes instead of one.",
    ],

    atari_map=[
        (r"The network \(f\)", r"The Atari 2600 VCS running the ROM, emulated bit-exactly by <code>jutari</code>; "
         r"its differentiable twin is <code>JuTari.Diff</code>."),
        (r"The input \(x\)", r"The 128 RAM bytes at the intervention frame \(f^*\) (after 90 frames of seeded random play)."),
        (r"Input noise", r"Gaussian noise with \(\sigma = 8\) <b>raw byte units</b> (about 3&nbsp;% of the 0&ndash;255 "
         r"range) added to every RAM byte, \(N = 50\) copies, fixed seed."),
        (r"The output \(y\)", r"A content output (one RAM byte read one-hot) and a position output (screen pixels changed "
         r"versus the un-intervened frame, made differentiable by the bilinear sampler)."),
        (r"&ldquo;Is the smoothed map right?&rdquo;", r"The intervention oracle: poke each cause, re-run the real ROM, measure \(|\Delta y|\)."),
    ],

    atari=r"""
<p>The runner computes \(\mathrm{SG}(u) = \frac1N\sum_k |\partial y/\partial u|_{u+\varepsilon_k}\) over
the RAM vector \(u\) with Zygote, and returns the noise-free vanilla \(|\partial y/\partial u|\) next to it.
Both are scored against the oracle, so the effect of smoothing on <i>faithfulness</i> is measured
directly, not just its effect on looks.</p>
<ul>
<li><b>Noisy RAM is never executed.</b> A noisy copy holds values like 117.4 or &minus;3.1, which no real
RAM byte can take. That is fine here because the gradient is evaluated only on the differentiable
<i>read</i> of the output, not on the machine. The real VCS only ever sees integer pokes made by the
oracle and the deletion/insertion curves.</li>
<li><b>Content output = the straight line.</b> \(y\) is <code>soft_ram_peek(ram, idx)</code>, a linear
one-hot read. Its gradient is \(e_{\text{idx}}\) at every input, noisy or not. So SmoothGrad returns
exactly the vanilla map. The runner's <code>--sigma-sweep</code> option confirms this over five
\((\sigma, N)\) settings, from \((0, 1)\) to \((32, 100)\): the content-path correlation does not
move. The byte <code>idx</code> is the candidate the oracle shows to be most causally active.</li>
<li><b>Naive position output = the step.</b> The honest position gradient is zero everywhere (sprites are
placed by strobe timing). Averaging 50 zeros gives zero. Noise cannot manufacture a gradient the
machine does not have.</li>
<li><b>Sampler position output = the ReLU case.</b> The bilinear sampler draws the sprite at a continuous
column driven by one RAM byte, using a triangular kernel one pixel wide. Noise of 8 byte units
moves that column by several pixels, so many samples land where the kernel is flat or the
occupancy is clamped. The averaged gradient can therefore be weaker than the un-smoothed one, and
sometimes nearly zero. The runner records the raw sampler gradient and the smoothed one side by
side, and treats a smoothed-away gradient as an honest smoothing effect, not as a vanishing.</li>
<li><b>Scoring</b> follows the shared contract: correlation and precision@3 against the oracle,
deletion/insertion curves on the real ROM, and an oracle-vs-itself positive control.</li>
</ul>""",

    code=[
        dict(path="tools/xai_study/phaseB_attribution/smoothgrad.jl", find="function smoothgrad_over_ram(",
             role="xai", desc="The method: average \\(|\\nabla|\\) over \\(N\\) noisy RAM copies; also returns the noise-free map."),
        dict(path="tools/xai_study/phaseB_attribution/smoothgrad.jl", find="xn = x .+ Float32(sigma) .* Float32.(randn(rng, length(x)))",
             role="xai", desc="The noise: \\(\\sigma\\) in raw byte units, one Gaussian draw per RAM byte, seeded."),
        dict(path="tools/xai_study/phaseB_attribution/smoothgrad.jl", find="function sigma_sweep(",
             role="xai", desc="Robustness grid over \\(\\sigma\\) and \\(N\\): does smoothing change faithfulness, or only looks?"),
        dict(path="tools/xai_study/phaseB_attribution/smoothgrad.jl", find="function pick_content_idx(",
             role="xai", desc="Picks the content byte the oracle shows to be most causally active."),
        dict(path="tools/xai_study/phaseB_attribution/smoothgrad.jl", find="function compute_game(",
             role="xai", desc="One game: shared testbed &rarr; content and sampler-position outputs &rarr; naive vs sampler side by side."),
        dict(path="jutari/src/diff/SoftStep.jl", find="function soft_ram_peek",
             role="substrate", desc="The linear one-hot RAM read: its gradient is the same at every noisy input."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function _st_sampler_position_read",
             role="substrate", desc="The bilinear sampler for the position output, clamped to \\([0,1]\\)."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="_st_tri(t) = max(0f0",
             role="substrate", desc="The one-pixel triangular kernel: flat outside \\(|t|&lt;1\\), which is why large noise smooths its slope away."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function run_intervention",
             role="substrate", desc="The ground truth: poke a cause, re-run the real ROM, read the output."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!",
             role="substrate", desc="One frame of the bit-exact machine, used by the oracle and the deletion/insertion curves."),
    ],

    reading=r"""
<p>Put this page next to vanilla saliency. On the <b>content</b> output the two maps are the same, and
so is their agreement with the oracle: a single spike at the read byte, against an oracle that
usually lights up several causes. On the <b>position</b> output, the smoothed sampler map is often
fainter than the raw sampler map, and its agreement with the oracle stays weak. The lesson is
general. SmoothGrad changes how a map looks, but here it does not change how faithful the map is.
It cannot add causal information the gradient never carried.</p>""",

    pitfalls=[
        r"<b>Denoising is not faithfulness.</b> A cleaner map can be exactly as wrong as the noisy one. Hooker et al. found that SmoothGrad did not beat a random ranking under retraining-based evaluation (ROAR).",
        r"<b>Averaging zeros gives zero.</b> Where the function is flat over the whole noise neighbourhood (a step, a table look-up, sprite timing), SmoothGrad stays blind.",
        r"<b>\(\sigma\) is a free knob.</b> Too small and nothing is smoothed; too large and the map describes a different region of input space. On the VCS, 8 byte units already moves a sprite by several pixels.",
        r"<b>Off-distribution inputs.</b> Noisy copies are not valid inputs (non-integer, possibly negative bytes; for images, grainy photos). The method explains the model on inputs it would never see.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseB_attribution/smoothgrad.jl --games core",
    try_note="Writes <code>smoothgrad_&lt;game&gt;_{content,ball_pixel}.json/.npz</code> and "
             "<code>smoothgrad_core_summary.json</code> into "
             "<code>tools/xai_study/phaseB_attribution/out/</code>. Defaults: <code>--sigma 8 --n-samples 50</code>. "
             "Add <code>--sigma-sweep</code> to reproduce the \\(\\sigma/N\\) robustness grid (it re-runs every game per grid point).",

    next=["saliency", "ig_baseline_sweep", "guided_backprop", "rise"],
)
