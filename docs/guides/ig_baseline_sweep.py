# Beginner's guide — Integrated Gradients, with a baseline sweep.
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="ig_baseline_sweep",
    level="Gradient family · path integral from a baseline",

    wiki=[
        ("Line integral", "https://en.wikipedia.org/wiki/Line_integral"),
        ("Gradient theorem", "https://en.wikipedia.org/wiki/Gradient_theorem"),
        ("Riemann sum", "https://en.wikipedia.org/wiki/Riemann_sum"),
        ("Shapley value", "https://en.wikipedia.org/wiki/Shapley_value"),
    ],
    papers=[
        ("Sundararajan, Taly &amp; Yan (2017). <i>Axiomatic Attribution for Deep Networks.</i> ICML.",
         "https://arxiv.org/abs/1703.01365"),
        ("Sturmfels, Lundberg &amp; Lee (2020). <i>Visualizing the Impact of Feature Attribution "
         "Baselines.</i> Distill.",
         "https://doi.org/10.23915/distill.00022"),
    ],

    idea=r"""
<p>Vanilla saliency has a blind spot called <b>saturation</b>. Picture a network that is already
99.9&nbsp;% sure the photo shows a dog. Making the dog's ear a bit brighter cannot raise the score
much more, so the gradient there is almost zero. Yet the ear is a big reason the score is high.
The gradient only sees the flat top of the curve, not the climb that got there.</p>
<p><b>Integrated Gradients</b> (IG) looks at the whole climb. It chooses a <b>baseline</b>: an input
that means &ldquo;nothing&rdquo;, such as a black image. It walks in a straight line from the baseline
to the real input, adds up the gradients along the way, and multiplies by how far each input moved.
The payoff is <b>completeness</b>: the attributions add up exactly to the output at the input minus
the output at the baseline. The price is that the answer depends on which baseline you pick. So our
runner tries four baselines and reports how much the result moves.</p>""",

    nn=r"""
<p>With \(y = f(x)\) and a baseline \(x^0\), IG assigns input \(i\)</p>
\[ \mathrm{IG}_i(x; x^0) \;=\; (x_i - x^0_i)\int_0^1
   \frac{\partial f}{\partial x_i}\big(x^0 + \alpha\,(x - x^0)\big)\,d\alpha . \]
<p>The integral is a <b>line integral</b> of the gradient along the straight path. By the
<b>gradient theorem</b> (the fundamental theorem of calculus for paths) the pieces sum to the
total change: \(\sum_i \mathrm{IG}_i = f(x) - f(x^0)\). In practice the integral is a
<b>Riemann sum</b> over \(K\) points, e.g. midpoints \(\alpha_k = (k - \tfrac12)/K\).</p>
<p><b>Toy example (2 inputs).</b> \(f(x_1, x_2) = x_1 x_2\) at \(x = (2, 3)\), baseline \(x^0 = (0, 0)\).
On the path \(x(\alpha) = (2\alpha, 3\alpha)\) the gradient is \((3\alpha, 2\alpha)\). Then</p>
\[ \mathrm{IG}_1 = 2\int_0^1 3\alpha\,d\alpha = 3, \qquad
   \mathrm{IG}_2 = 3\int_0^1 2\alpha\,d\alpha = 3, \qquad
   3 + 3 = 6 = f(x) - f(x^0). \]
<p>Completeness holds. Compare: saliency gives \((3, 2)\), and Grad&times;Input gives \((6, 6)\), which sums to
12, twice the real change.</p>
<p><b>Saturation.</b> Let \(f(x) = \min(1, x)\) at \(x = 2\). The gradient is 0, so saliency says
&ldquo;irrelevant&rdquo;. IG from \(x^0 = 0\) integrates slope 1 over the first half of the path:
\(\mathrm{IG} = 2 \cdot \tfrac12 = 1 = f(2) - f(0)\). IG sees the climb.</p>
<p><b>Baseline dependence.</b> Back to \(x_1 x_2\), now with baseline \(x^0 = (2, 0)\). \(x_1\) does
not move, so \(\mathrm{IG}_1 = 0\), and \(\mathrm{IG}_2 = 3 \cdot 2 = 6\). Same model, same input:
all the credit moved to \(x_2\). The baseline defines the question being answered.</p>""",

    recipe=[
        r"Choose a baseline \(x^0\) (zeros, a mean input, noise, or a real reference input).",
        r"Pick \(K\) points on the straight path, \(x^0 + \alpha_k (x - x^0)\) with \(\alpha_k = (k - \tfrac12)/K\).",
        r"At each point run one backward pass to get \(\partial y/\partial x_i\).",
        r"Average the \(K\) gradients and multiply element-wise by \((x - x^0)\).",
        r"Check completeness: \(|\sum_i \mathrm{IG}_i - (y(x) - y(x^0))|\) should be small; if not, increase \(K\).",
        r"Repeat for several baselines and compare: the spread is part of the answer.",
    ],

    atari_map=[
        (r"The network \(f\)", r"The Atari 2600 VCS running the ROM, emulated bit-exactly by <code>jutari</code>; "
         r"its differentiable twin is <code>JuTari.Diff</code>."),
        (r"The input \(x\)", r"The 128 RAM bytes at the intervention frame \(f^*\) (after 90 frames of seeded random play)."),
        (r"The baseline \(x^0\)", r"Four choices: all-zeros RAM (headline), every byte set to the mean of \(x\), one "
         r"uniform-random RAM vector, and the <b>real RAM</b> at an earlier frame (\(f^*/2\)) of the same game."),
        (r"Points on the path", r"\(K = 64\) midpoint RAM vectors. They hold fractional byte values the real "
         r"machine cannot execute; only the differentiable read is evaluated there."),
        (r"The output \(y\)", r"A content output (one RAM byte read one-hot) and a position output (screen pixels changed "
         r"versus the un-intervened frame, made differentiable by the bilinear sampler)."),
        (r"&ldquo;Is the attribution right?&rdquo;", r"The intervention oracle: poke each cause, re-run the real ROM, measure \(|\Delta y|\)."),
    ],

    atari=r"""
<p>The runner integrates the gradient over the RAM vector with \(K = 64\) Riemann midpoints, for each
of four baselines, and scores each IG map against the oracle. It also records each baseline's
completeness error. The headline number uses the all-zeros baseline; the sweep sits beside it.</p>
<ul>
<li><b>What about non-integer RAM?</b> A point halfway along the path has bytes like 63.5. A real
6507 cannot run on that. The code never asks it to. At each path point it evaluates only the
differentiable read of the output (<code>soft_ram_peek</code> or the bilinear sampler), which is
defined for any real number. The machine itself only runs during the oracle and the
deletion/insertion curves, always on integer pokes. As in vanilla saliency, the 15 frames between
\(f^*\) and the readout are not differentiated.</li>
<li><b>Content output.</b> The read is linear, so the gradient is the same one-hot vector \(e_s\) at
every path point. IG reduces to \(\mathrm{IG}_s = x_s - x^0_s\) and 0 elsewhere. Completeness is exact
for every baseline. Changing the baseline only changes the height of one spike, so the ranking of
causes, and hence the correlation with the oracle, stays put. One exception: if the baseline happens
to equal the byte's value, the spike vanishes. The sweep makes this baseline degeneracy visible.</li>
<li><b>Position output.</b> The naive gradient is zero all along the path, and the integral of zero is
zero, for every baseline. With the sampler on, completeness gives
\(\mathrm{IG} = \text{sampler}(x) - \text{sampler}(x^0)\): the difference in the cell's occupancy
between the sprite drawn at the baseline's column and at the real one. If the cell is empty at
both ends, IG is zero even when the sprite passed through the cell on the way.</li>
<li><b>The four baselines.</b> <code>zeros</code> is the classic &ldquo;absent&rdquo; input.
<code>mean_ram</code> fills all 128 bytes with one number, the mean of the current RAM.
<code>random</code> is one seeded uniform draw in 0&ndash;255. <code>real_alt</code> is the genuine RAM at
frame \(f^*/2\) of the same seeded run. It is the only baseline that is itself a reachable game state.</li>
<li><b>Scoring</b> is the shared contract: correlation and precision@3 against the oracle,
deletion/insertion curves on the real ROM, and an oracle-vs-itself control.</li>
</ul>""",

    code=[
        dict(path="tools/xai_study/phaseB_attribution/ig_baseline_sweep.jl", find="function ig_over_ram(",
             role="xai", desc="The method: \\(K\\) midpoint gradients along the path, times \\((x - x^0)\\), plus the completeness error."),
        dict(path="tools/xai_study/phaseB_attribution/ig_baseline_sweep.jl", find="xα = x0 .+ α .* (x .- x0)",
             role="xai", desc="The interpolated RAM vector: fractional bytes, fed only to the differentiable read."),
        dict(path="tools/xai_study/phaseB_attribution/ig_baseline_sweep.jl", find="function make_baseline(",
             role="xai", desc="The four baselines of the sweep: zeros, mean_ram, random, real_alt."),
        dict(path="tools/xai_study/phaseB_attribution/ig_baseline_sweep.jl", find="alt_frame = max(1, tf ÷ 2)",
             role="xai", desc="The real-state baseline: the genuine RAM at half the intervention frame."),
        dict(path="tools/xai_study/phaseB_attribution/ig_baseline_sweep.jl", find="function score_ig_for_baseline(",
             role="xai", desc="Scores one baseline's IG against the oracle (correlation, precision@k, deletion/insertion, completeness)."),
        dict(path="tools/xai_study/phaseB_attribution/ig_baseline_sweep.jl", find="function deletion_insertion_auc(",
             role="xai", desc="Deletion/insertion curves: occlude or restore the top-ranked causes and re-run the real ROM."),
        dict(path="jutari/src/diff/SoftStep.jl", find="function soft_ram_peek",
             role="substrate", desc="The linear one-hot read: the same gradient at every point of the path."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function _st_sampler_position_read",
             role="substrate", desc="The bilinear sampler: defined for any real RAM value, so IG can integrate along the path."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function run_intervention",
             role="substrate", desc="The ground truth: poke a cause (integer value), re-run the real ROM, read the output."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!",
             role="substrate", desc="One frame of the bit-exact machine."),
    ],

    reading=r"""
<p>The headline bars use the zeros baseline. On the <b>content</b> output you see one spike at the
read byte, as in vanilla saliency. Its height is the byte's distance from the baseline. Across the
baseline sweep the spike grows or shrinks, but the correlation with the oracle barely moves. It drops to
nothing only when a baseline equals the byte. The completeness error is essentially zero
throughout. On the <b>position</b> output, agreement with the oracle stays low for every baseline.
Completeness is a bookkeeping guarantee about the <i>total</i>; it does not make IG notice the
several causes the oracle finds over 15 frames.</p>""",

    pitfalls=[
        r"<b>The baseline is a hidden choice.</b> Different baselines answer different questions (&ldquo;compared with what?&rdquo;). A black baseline gives zero credit to black pixels; Sturmfels et al. show how much maps change with it.",
        r"<b>Off-distribution path.</b> Points along the straight line are blends that no real input looks like. On the VCS they are fractional RAM states the machine cannot run, so IG explains the differentiable read, not the machine's behaviour on them.",
        r"<b>Completeness is not correctness.</b> Attributions that sum to the right total can still credit the wrong inputs. On discrete outputs the total itself can be zero.",
        r"<b>Riemann error.</b> Too few steps break completeness on curved functions. Check the completeness error, as the runner does.",
        r"<b>Integral of zero is zero.</b> Where the gradient vanishes along the whole path (sprite timing, table look-ups), no baseline helps.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseB_attribution/ig_baseline_sweep.jl --games core --baseline-sweep",
    try_note="Writes <code>ig_baseline_sweep_&lt;game&gt;_{content,ball_pixel}.json/.npz</code> (the sweep is under "
             "<code>extra.baseline_sweep</code>) and <code>ig_baseline_sweep_core_summary.json</code> into "
             "<code>tools/xai_study/phaseB_attribution/out/</code>. Use <code>--ig-steps N</code> to change \\(K\\) "
             "and <code>--baseline zeros|mean_ram|random|real_alt</code> to run a single baseline.",

    next=["expected_gradients", "gradxinput", "saliency", "kernelshap"],
)
