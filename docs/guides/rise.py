# Beginner's guide — RISE (randomized input sampling for explanation).
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="rise",
    level="Perturbation family · random joint masks",

    wiki=[
        ("Monte Carlo method", "https://en.wikipedia.org/wiki/Monte_Carlo_method"),
        ("Bernoulli distribution", "https://en.wikipedia.org/wiki/Bernoulli_distribution"),
        ("Conditional expectation", "https://en.wikipedia.org/wiki/Conditional_expectation"),
        ("Importance sampling", "https://en.wikipedia.org/wiki/Importance_sampling"),
    ],
    papers=[
        ("Petsiuk, Das &amp; Saenko (2018). <i>RISE: Randomized Input Sampling for Explanation of "
         "Black-box Models.</i> BMVC.",
         "https://arxiv.org/abs/1806.07421"),
        ("Hooker, Erhan, Kindermans &amp; Kim (2019). <i>A Benchmark for Interpretability Methods in Deep "
         "Neural Networks</i> (why deletion-style scores on masked inputs can mislead). NeurIPS.",
         "https://arxiv.org/abs/1806.10758"),
    ],

    idea=r"""
<p>Occlusion hides <i>one</i> part at a time. RISE hides <b>many parts at once, at random</b>. Throw
a random pattern of holes over the photo (about half the pixels hidden), run the network, write
down the dog score. Repeat a few hundred times with fresh random patterns. Now ask, for each pixel:
<i>when this pixel happened to be visible, how high was the score on average?</i> Pixels the network
needs will be visible mostly in the high-scoring runs. Pixels it ignores will be visible in high and
low runs alike.</p>
<p>This is a <b>Monte-Carlo</b> estimate: instead of trying every possible pattern (there are
\(2^n\)), you average over random samples, and the estimate gets better as you draw more. RISE is a
black-box method like occlusion. Because many parts are hidden <i>together</i>, it can notice parts
that only matter in combination, which occlusion cannot. The price is noise: with few masks, the
map is grainy.</p>""",

    nn=r"""
<p>Draw \(N\) random binary masks \(m_1,\dots,m_N \in \{0,1\}^n\). Each entry is an independent
coin flip: kept (\(1\)) with probability \(p\), hidden (\(0\)) otherwise (a <b>Bernoulli</b>
variable). Run the model on each masked input, \(y_k = f(x \odot m_k)\). The importance of input
\(i\) is</p>
\[ S_i \;=\; \frac{1}{N\,p}\sum_{k=1}^{N} y_k \, m_{k,i} \;\;\approx\;\; \mathbb{E}\big[\,y \mid m_i = 1\,\big] . \]
<p>That is the average output over the masks that <i>kept</i> input \(i\): a <b>conditional
expectation</b>. The \(1/(Np)\) factor turns the sum into that average, because about \(Np\) masks
keep \(i\). On images, RISE draws small grids, upsamples them smoothly and shifts them randomly, so
the holes are blobs rather than single pixels.</p>
<p><b>Toy example.</b> \(f(x) = 4x_1x_2 + x_3\) at \(x=(1,1,1,1)\) (\(x_4\) unused), \(p = 0.5\).
Enumerate all 16 masks instead of sampling. When \(x_1\) is kept, \(x_2\) is kept half the time and
\(x_3\) half the time, so \(\mathbb{E}[y\mid m_1{=}1] = 4\cdot0.5 + 0.5 = 2.5\). Likewise
\(S_2 = 2.5\), \(S_3 = 4\cdot 0.25 + 1 = 2\), \(S_4 = 1.5\). The unused \(x_4\) gets exactly the
<b>overall mean</b> \(\mathbb{E}[y] = 1.5\). Subtract it and you get \((1, 1, 0.5, 0)\): the
interaction pair ranks first, \(x_3\) second, \(x_4\) is zero. With \(p = 0.5\) this centred score is
half the average change caused by switching \(x_i\) on (the <i>Banzhaf value</i>, a close relative of
the Shapley value). Note: occlusion on the same toy gives \((4,4,1,0)\), and both agree on the
ranking here.</p>""",

    recipe=[
        r"Choose the output \(y\), the number of masks \(N\), the keep-probability \(p\) and what &ldquo;hidden&rdquo; means (here: set to 0).",
        r"Draw a random mask: keep each input with probability \(p\).",
        r"Hide the inputs whose mask bit is 0, run the model, record \(y_k\). Repeat \(N\) times.",
        r"For each input, average \(y_k\) over the masks that kept it: \(S_i = \frac{1}{Np}\sum_k y_k m_{k,i}\).",
        r"Normalise the map (subtract the mean) and display it.",
    ],

    atari_map=[
        (r"The network \(f\)", "The cartridge program on the bit-exact emulated <b>Atari 2600 VCS</b> (<code>jutari</code>)."),
        (r"The pixels \(x_i\)", "The <b>candidate RAM cells</b> (the bytes the oracle scores), de-duplicated: 8 to 23 per core game. TIA and joystick causes are not in RISE's universe."),
        (r"Masked input \(x\odot m\)", "<code>deepcopy</code> the checkpoint, <code>intervene_ram!(env, i, 0)</code> for every dropped cell, run 15 frames of the real ROM."),
        (r"\(N\) masks, keep-prob \(p\)", "\\(N = 500\\) masks, \\(p = 0.5\\), from a seeded random generator, so the run is reproducible."),
        (r"The output \(y\)", "A content RAM byte at \\(f^*{+}15\\), or the number of screen pixels that differ from the un-intervened frame."),
        ("&ldquo;Ground-truth importance&rdquo;", "The oracle's per-cause |&Delta;y|, from single-cause re-runs (+17 and &rarr;0 per byte, plus a TIA and a joystick do)."),
    ],

    atari=r"""
<p>The VCS version follows the paper's formula one-to-one; only the &ldquo;image&rdquo; changes.
The universe is the list of candidate RAM cells. Each of the \(N = 500\) masks keeps every cell with
probability \(p = 0.5\). For each mask, <code>masked_read_y</code> copies the checkpoint, zeroes the
dropped cells with <code>intervene_ram!</code>, plays the recorded 15 actions with
<code>env_step!</code>, and reads \(y\). Those 500 real-emulator runs are the whole cost of the
method. <code>rise_importance</code> then computes \(\frac{1}{Np}\sum_k y_k m_{k,i}\).</p>
<p><b>From cells to causes.</b> The oracle scores <i>causes</i>, not cells.
<code>rise_attr_per_cause</code> centres the cell scores by subtracting their mean <i>across
cells</i>, takes the absolute value, and gives each RAM cause the score of its cell (so a byte's
<code>:set</code> and <code>:occlude</code> causes get the same value). TIA and joystick causes get
0, because RISE never touches them.</p>
<p><b>Convergence.</b> RISE is a Monte-Carlo estimate, so the runner also rescores the map using only
the first \(N/8\), \(N/4\), \(N/2\) masks of the same pool. That shows how the agreement with the oracle
grows with compute, without drawing new masks.</p>
<p><b>How is this different from the oracle?</b> Three ways. (1) <b>Joint masks</b>: in every run
about half the cells are zeroed at once, so each \(y_k\) mixes many effects; the oracle changes one
cause at a time. (2) <b>Averaging</b>: RISE reports an average over random contexts, the oracle a
single exact difference at the real state. (3) <b>One direction</b>: RISE only ever sets cells to
0. On a <i>content</i> byte that responds smoothly to many cells, the average is informative. On the
<i>position</i> output, the effect of one cell is often an all-or-nothing jump that depends on which
other cells are zeroed with it, so the average is noisier.</p>""",

    code=[
        dict(path="tools/xai_study/phaseB_attribution/rise.jl", find="function rise_sample(",
             role="xai", desc="Draws \\(N\\) Bernoulli(\\(p\\)) masks and re-runs the real ROM once per mask."),
        dict(path="tools/xai_study/phaseB_attribution/rise.jl", find="function masked_read_y(",
             role="xai", desc="One masked run: <code>deepcopy</code>, zero every dropped cell, 15 &times; <code>env_step!</code>, read y."),
        dict(path="tools/xai_study/phaseB_attribution/rise.jl", find="function rise_importance(",
             role="xai", desc="Petsiuk's estimator \\(\\frac{1}{Np}\\sum_k y_k m_{k,i}\\)."),
        dict(path="tools/xai_study/phaseB_attribution/rise.jl", find="function rise_attr_per_cause(",
             role="xai", desc="Centre across cells, take |&middot;|, map cells to the oracle's causes; TIA/joystick get 0."),
        dict(path="tools/xai_study/phaseB_attribution/rise.jl", find="for frac in (8, 4, 2, 1)",
             role="xai", desc="The convergence sweep: rescoring from prefixes of the same mask pool."),
        dict(path="tools/xai_study/phaseB_attribution/rise.jl", find="n_masks = 500; keep_prob = 0.5",
             role="xai", desc="The defaults: 500 masks, keep-probability 0.5."),
        dict(path="tools/xai_study/common/jutari_oracle.jl", find="function intervene_ram!(",
             role="substrate", desc="The poke that zeroes a dropped cell in the emulated RAM."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(",
             role="substrate", desc="One frame of the real machine; each mask costs 15 of them."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function run_intervention(",
             role="substrate", desc="The oracle's single-cause re-run, which RISE's joint masks are compared with."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function build_shared_testbed(",
             role="substrate", desc="The shared gameplay checkpoint and the position output."),
    ],

    reading=r"""
<p>Bars are RISE scores per cause, next to the oracle's true effect. Two features are built in:
TIA and joystick causes are always zero for RISE, and the two causes of one byte always share a
height. Beyond that, compare the tallest bars. On content outputs expect reasonable agreement; on
position outputs expect a noisier map with a weaker match. The convergence numbers in the record show
whether more masks would still help: if agreement is still rising from \(N/2\) to \(N\), the estimate
has not settled.</p>""",

    pitfalls=[
        "<b>Monte-Carlo noise.</b> The variance falls only like \\(1/N\\). Few masks give a grainy, seed-dependent map.",
        "<b>Average over strange contexts.</b> With \\(p=0.5\\), every run zeroes about half the cells. The score answers &ldquo;how much does this cell help in a heavily damaged machine?&rdquo;, which can differ from its effect in the real, intact state.",
        "<b>Centring across cells.</b> Our map subtracts the mean of the cell scores, not the global mean output. When most cells are inert, the two are about the same (inert cells score the global mean). When most cells matter, the across-cell mean is pulled up and the absolute value can mis-rank cells.",
        "<b>Masks are the input type.</b> On images, the blob size of the upsampled masks is a hidden hyper-parameter. Here the masks are over a fixed candidate list; RAM bytes outside it cannot be credited.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseB_attribution/rise.jl --games core",
    try_note="Writes <code>rise_&lt;game&gt;_{content,ball_pixel}.json/.npz</code> into "
             "<code>tools/xai_study/phaseB_attribution/out/</code>, including the convergence sweep. "
             "Change the budget with <code>--n-masks</code> and <code>--keep-prob</code>.",

    next=["kernelshap", "lime", "occlusion", "perturbation"],
)
