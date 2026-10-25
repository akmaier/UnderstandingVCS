# Beginner's guide — Expected Gradients.
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="expected_gradients",
    level="Gradient family · Integrated Gradients averaged over baselines",

    wiki=[
        ("Expected value", "https://en.wikipedia.org/wiki/Expected_value"),
        ("Monte Carlo integration", "https://en.wikipedia.org/wiki/Monte_Carlo_integration"),
        ("Shapley value", "https://en.wikipedia.org/wiki/Shapley_value"),
        ("Gradient theorem", "https://en.wikipedia.org/wiki/Gradient_theorem"),
    ],
    papers=[
        ("Erion, Janizek, Sturmfels, Lundberg &amp; Lee (2021). <i>Improving performance of deep "
         "learning models with axiomatic attribution priors and expected gradients.</i> Nature "
         "Machine Intelligence 3, 620&ndash;631.",
         "https://doi.org/10.1038/s42256-021-00343-w"),
        ("Lundberg &amp; Lee (2017). <i>A Unified Approach to Interpreting Model Predictions</i> "
         "(SHAP; Expected Gradients is its gradient-based relative). NeurIPS.",
         "https://arxiv.org/abs/1705.07874"),
        ("Sturmfels, Lundberg &amp; Lee (2020). <i>Visualizing the Impact of Feature Attribution "
         "Baselines.</i> Distill.",
         "https://doi.org/10.23915/distill.00022"),
    ],

    idea=r"""
<p>Integrated Gradients (IG) needs a <b>baseline</b>, an input that means &ldquo;nothing&rdquo;. As
the IG guide showed, the answer depends heavily on that choice: a black image, a blurred image and
random noise each give a different map. There is rarely a single right &ldquo;nothing&rdquo;.</p>
<p><b>Expected Gradients</b> (EG) sidesteps the choice. Instead of one hand-picked baseline it uses
a whole <b>reference distribution</b>: a pool of real inputs, for example images from the training
set. It runs IG from many randomly drawn references and averages the results. The attribution then
answers &ldquo;how does this input differ from a <i>typical</i> input, and which features carry the
difference?&rdquo; This links EG to <b>Shapley values</b> from game theory, which also average a
feature's contribution over many reference situations. EG is stable and keeps completeness <i>in
expectation</i>. But stable does not mean faithful, and averaging differences can cancel them.</p>""",

    nn=r"""
<p>With \(y = f(x)\) and a reference distribution \(D\), EG assigns input \(i\) the
<b>expected value</b> (average) of the IG integrand over references \(x'\) and path positions
\(\alpha\):</p>
\[ \mathrm{EG}_i(x) \;=\; \mathbb{E}_{x' \sim D,\;\alpha \sim U(0,1)}
   \Big[(x_i - x'_i)\,\frac{\partial f}{\partial x_i}\big(x' + \alpha(x - x')\big)\Big]. \]
<p>Erion et al. estimate it by <b>Monte Carlo integration</b>: draw \(M\) pairs
\((x'^{(m)}, \alpha_m)\), one gradient each, and average. There is no inner Riemann sum; the random
\(\alpha\) does the path integral. Summing over \(i\) gives completeness in expectation:
\(\sum_i \mathrm{EG}_i = f(x) - \mathbb{E}_{x'}[f(x')]\).</p>
<p><b>Toy example: averaging over baselines.</b> \(f(x_1, x_2) = x_1 x_2\) at \(x = (2, 3)\), with
\(D = \{(0,0),\,(2,0)\}\), each drawn half the time. From the IG guide: IG from \((0,0)\) is
\((3, 3)\); IG from \((2,0)\) is \((0, 6)\). EG is the average, \((1.5,\ 4.5)\). Check:
\(1.5 + 4.5 = 6 = f(x) - \tfrac12\big(f(0,0) + f(2,0)\big) = 6 - 0\). The baseline choice is no
longer a single arbitrary pick, but the result now depends on the pool \(D\).</p>
<p><b>Toy example: cancellation.</b> Let \(f(x) = x\) at \(x = 3\), with references
\(\{1, 5\}\). The two IG terms are \(3 - 1 = 2\) and \(3 - 5 = -2\). EG \(= 0\). Yet \(f\) depends on \(x\)
one-for-one. EG measures how far \(x\) is from the <i>average</i> reference. An input sitting at
the typical value gets no credit, however much the output depends on it.</p>""",

    recipe=[
        r"Build a reference pool \(D\) of real inputs (on a network: training examples).",
        r"Draw \(M\) references \(x'^{(m)}\) from \(D\) (with replacement) and one \(\alpha_m \sim U(0,1)\) for each.",
        r"For each pair, form \(x' + \alpha (x - x')\), run one backward pass, and multiply the gradient by \((x - x')\).",
        r"Average the \(M\) products: that is the EG attribution.",
        r"Optionally check completeness against \(f(x) - \overline{f(x')}\), and repeat with other seeds to measure stability.",
    ],

    atari_map=[
        (r"The network \(f\)", r"The Atari 2600 VCS running the ROM, emulated bit-exactly by <code>jutari</code>; "
         r"its differentiable twin is <code>JuTari.Diff</code>."),
        (r"The input \(x\)", r"The 128 RAM bytes at the intervention frame \(f^*\) (frame 90 of seeded random play)."),
        (r"The reference distribution \(D\)", r"The 90 genuine RAM states at frames 1&ndash;90 of the <b>same</b> seeded "
         r"gameplay run: every row is a reachable machine state."),
        (r"Monte Carlo samples", r"\(M = 64\) (reference, \(\alpha\)) draws per run, five independent seeds for the stability estimate."),
        (r"The output \(y\)", r"A content output (one RAM byte read one-hot) and a position output (screen pixels changed "
         r"versus the un-intervened frame, made differentiable by the bilinear sampler)."),
        (r"&ldquo;Is the attribution right?&rdquo;", r"The intervention oracle: poke each cause, re-run the real ROM, measure \(|\Delta y|\)."),
    ],

    atari=r"""
<p>The runner implements Erion's single-\(\alpha\)-per-reference estimator over the RAM vector: 64 draws
per run, each a uniformly sampled row of \(D\) and a uniform \(\alpha\). It repeats this with five seeds,
reports the mean pairwise cosine similarity between the seeds' maps (stability), and also scores
single-baseline IG with the zeros baseline on the same output for a head-to-head comparison.</p>
<ul>
<li><b>The reference pool.</b> \(D\) is built by replaying the shared seeded gameplay stream and
recording the RAM after each of the first 90 frames. An earlier version used a pool from a NOOP
(no-input) replay. There, many candidate bytes never changed, so \(x - x' = 0\) for every
reference and EG was exactly zero. The gameplay pool was introduced so the causal bytes actually
vary across references.</li>
<li><b>Interpolated RAM.</b> As in IG, the blended states \(x' + \alpha(x - x')\) have fractional bytes. They are
only fed to the differentiable read, never executed on the machine.</li>
<li><b>Content output.</b> The one-hot read has gradient \(e_s\) everywhere, so
\(\mathrm{EG}_s = \frac1M\sum_m (x_s - x'^{(m)}_s)\): the current byte minus the average of the
sampled references, and 0 for every other byte. This is the cancellation toy for real. A score byte
that only grows keeps a clear signal. A byte that sits near its own trajectory average gets almost
none, though the oracle may show it matters. The runner records the byte's value, its range over
\(D\), and whether EG vanished across all seeds.</li>
<li><b>Position output.</b> The naive gradient is zero for every reference. With the sampler, each draw
contributes \((x - x')\) times the kernel's slope at one random point of its path, a noisy one-sample
estimate of the occupancy difference between the sprite at the reference's column and at the real one.</li>
<li><b>Scoring</b> is the shared contract: correlation and precision@3 against the oracle,
deletion/insertion curves on the real ROM, and an oracle-vs-itself control.</li>
</ul>""",

    code=[
        dict(path="tools/xai_study/phaseB_attribution/expected_gradients.jl", find="function expected_gradients(",
             role="xai", desc="The method: \\(M\\) Monte Carlo (reference, \\(\\alpha\\)) draws, plus completeness in expectation."),
        dict(path="tools/xai_study/phaseB_attribution/expected_gradients.jl", find="acc .+= (x .- x0) .* g",
             role="xai", desc="One EG term: \\((x - x')\\) times the gradient at the interpolated state."),
        dict(path="tools/xai_study/phaseB_attribution/expected_gradients.jl", find="D = st.ref_pool",
             role="xai", desc="The reference pool used: real RAM states from the shared gameplay run."),
        dict(path="tools/xai_study/phaseB_attribution/expected_gradients.jl", find="function reference_pool(",
             role="xai", desc="The older NOOP-replay pool, kept for the legacy path; its constant bytes zeroed EG."),
        dict(path="tools/xai_study/phaseB_attribution/expected_gradients.jl", find="function mean_pairwise_cosine(",
             role="xai", desc="Stability across seeds: mean pairwise cosine of the per-cause maps."),
        dict(path="tools/xai_study/phaseB_attribution/expected_gradients.jl", find="function score_ig_zeros(",
             role="xai", desc="The head-to-head: single-baseline IG (zeros, 64 steps) on the same output."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function _st_reference_pool(",
             role="substrate", desc="Builds \\(D\\): replays the seeded gameplay stream and records RAM at frames 1&ndash;90."),
        dict(path="jutari/src/diff/SoftStep.jl", find="function soft_ram_peek",
             role="substrate", desc="The linear one-hot read: its gradient is \\(e_s\\) at every interpolated state."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function _st_sampler_position_read",
             role="substrate", desc="The bilinear sampler for the position output."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function run_intervention",
             role="substrate", desc="The ground truth: poke a cause, re-run the real ROM, read the output."),
    ],

    reading=r"""
<p>The stability score is high: different seeds give nearly the same map. That is EG's promise, and
it keeps it. But look at agreement with the oracle. On the <b>content</b> output the map is still a
single spike at the read byte, now sized by its distance from the average reference, and that
spike can shrink to almost nothing. On the <b>position</b> output agreement stays low. Overall EG is
among the least faithful attribution methods in the study while producing tidy, plausible-looking
maps. That combination, high plausibility and low faithfulness, is exactly what the intervention
oracle is designed to expose.</p>""",

    pitfalls=[
        r"<b>Stable is not faithful.</b> Averaging removes randomness from the baseline choice, not error. A map can be perfectly reproducible and wrong.",
        r"<b>Cancellation.</b> Signed differences from references above and below the input cancel. Inputs at their typical value get little credit even when the output depends on them.",
        r"<b>The pool is the new baseline.</b> Results depend on \(D\). A pool where an input never varies gives it zero credit, as the NOOP pool did on the VCS.",
        r"<b>Monte Carlo noise.</b> One gradient per draw is cheap but noisy; check stability across seeds, as the runner does.",
        r"<b>Still a gradient method.</b> Where the gradient vanishes (sprite timing), no reference distribution can bring it back.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseB_attribution/expected_gradients.jl --games core",
    try_note="Writes <code>expected_gradients_&lt;game&gt;_{content,ball_pixel}.json/.npz</code> and "
             "<code>expected_gradients_core_summary.json</code> into "
             "<code>tools/xai_study/phaseB_attribution/out/</code>. Defaults: <code>--eg-samples 64 "
             "--eg-stability-seeds 5</code>; the IG head-to-head uses <code>--ig-steps 64</code>.",

    next=["ig_baseline_sweep", "kernelshap", "gradxinput", "occlusion"],
)
