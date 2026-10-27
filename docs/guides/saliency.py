# Beginner's guide — vanilla gradient saliency.
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="saliency",
    level="Gradient family · the starting point",

    wiki=[
        ("Saliency map", "https://en.wikipedia.org/wiki/Saliency_map"),
        ("Gradient", "https://en.wikipedia.org/wiki/Gradient"),
        ("Automatic differentiation", "https://en.wikipedia.org/wiki/Automatic_differentiation"),
        ("Backpropagation", "https://en.wikipedia.org/wiki/Backpropagation"),
    ],
    papers=[
        ("Simonyan, Vedaldi &amp; Zisserman (2014). <i>Deep Inside Convolutional Networks: "
         "Visualising Image Classification Models and Saliency Maps.</i> ICLR workshop.",
         "https://arxiv.org/abs/1312.6034"),
        ("Adebayo et al. (2018). <i>Sanity Checks for Saliency Maps.</i> NeurIPS.",
         "https://arxiv.org/abs/1810.03292"),
    ],

    idea=r"""
<p>Imagine a network that looks at a photo and says &ldquo;dog: 92&nbsp;%&rdquo;. You want to know
<b>which pixels it looked at</b>. The simplest question you can ask is: <i>if I change this one
pixel a tiny bit, how much does the dog score move?</i> A pixel whose small change moves the score
a lot is a pixel the network is <b>sensitive</b> to. Ask that question for every pixel at once and
paint the answers as a heat-map: that picture is the <b>saliency map</b>.</p>
<p>The &ldquo;how much does the output move per tiny change of the input&rdquo; number has a name
from calculus: the <b>partial derivative</b>. Collected over all inputs it is the
<b>gradient</b>. Deep-learning frameworks already compute gradients for training (that is what
backpropagation does), so a saliency map costs just <b>one backward pass</b>. That is why it is
the first attribution method everyone learns, and the baseline every other gradient method
(Grad&times;Input, SmoothGrad, Integrated Gradients, &hellip;) tries to improve.</p>""",

    nn=r"""
<p>Write the network as a function \(y = f(x)\): the input \(x = (x_1,\dots,x_n)\) is e.g. the
pixel vector, the output \(y\) is one scalar (the logit of the class you care about). Saliency
assigns input \(i\) the score</p>
\[ S_i(x) \;=\; \left|\,\frac{\partial f}{\partial x_i}(x)\,\right| . \]
<p>The derivative is evaluated <b>at the actual input</b> \(x\). It is a <i>local, first-order</i>
statement: it says how \(f\) behaves for infinitesimally small changes around \(x\), via the Taylor
approximation \(f(x+\delta) \approx f(x) + \nabla f(x)\cdot\delta\). Nothing more.</p>
<p><b>Toy example.</b> Let \(f(x_1,x_2) = 3x_1 + x_2^2\) at \(x = (1, 0)\). Then
\(\partial f/\partial x_1 = 3\) and \(\partial f/\partial x_2 = 2x_2 = 0\). Saliency says
&ldquo;only \(x_1\) matters&rdquo;. Yet setting \(x_2 = 5\) changes the output by 25, far more than
any reasonable change of \(x_1\). The gradient is blind to \(x_2\) because, <i>at this point</i>, the
curve is flat in that direction. Keep this example in mind: it is exactly what happens on the
Atari.</p>""",

    recipe=[
        "Pick the input \\(x\\) you want explained and the scalar output \\(y\\) (one logit, one pixel, one RAM byte).",
        "Run the forward pass and record it on an autodiff tape (PyTorch <code>requires_grad</code>, JAX <code>grad</code>, Julia <code>Zygote.gradient</code>).",
        "Run one backward pass from \\(y\\): this yields \\(\\partial y/\\partial x_i\\) for every input \\(i\\) at once.",
        "Take the absolute value (sign is usually dropped) and display it as a heat-map over the inputs.",
    ],

    atari_map=[
        ("The trained network \\(f\\)", "The game cartridge's program (the ROM), run by the 6507 CPU, "
         "together with the TIA video chip and RIOT timer: the whole <b>Atari 2600 VCS</b>, emulated "
         "bit-exactly by <code>jutari</code>."),
        ("The input \\(x\\)", "The machine's internal state at the intervention frame \\(f^*\\): the "
         "128 bytes of RAM (plus registers). This is the &ldquo;activation vector&rdquo; we explain."),
        ("The output \\(y\\)", "Two outputs per game: a <b>content</b> output (the value of one candidate RAM "
         "byte, the one the oracle finds most causally active) and a <b>position</b> output (how many "
         "screen pixels differ from the un-intervened frame 15 frames later)."),
        ("Backpropagation", "The <b>differentiable SOFT path</b> of jutari (<code>JuTari.Diff</code>): "
         "the same computation, written so Julia's Zygote can differentiate it. Its forward pass is "
         "bit-identical to the real machine (Theorem&nbsp;1 of Paper&nbsp;1)."),
        ("&ldquo;Ground-truth importance&rdquo;", "Something you never have for a real network. Here we "
         "do: the <b>intervention oracle</b> sets each candidate cause to a new value, re-runs the "
         "real ROM, and measures how much \\(y\\) actually changed: \\(|\\Delta y(u)|\\)."),
    ],

    atari=r"""
<p>On a neural net we never know whether a saliency map is <i>right</i>. On the VCS we do, because
we can simply re-run the machine. The experiment therefore has two halves:</p>
<ol>
<li><b>The method.</b> We stop the game at frame \(f^*\) (after 90 frames of seeded random play), take
the RAM vector there and compute \(|\partial y/\partial \mathrm{ram}_u|\) with Zygote through the
differentiable read of the output. Each candidate <i>cause</i> \(u\) (a RAM byte the game is known to
use for a score, a position, &hellip;) gets that number as its attribution; the oracle's one TIA colour
register and joystick cause, which the RAM gradient cannot see, get 0.</li>
<li><b>The truth.</b> For every cause \(u\) we poke a new value into the real emulator, play the same
15 frames, and measure the true effect \(|\Delta y(u)|\). That is 1 re-run per cause, no
approximations.</li>
</ol>
<p>The score is how well the two agree: Pearson / Spearman correlation, precision@3 (are the top-3
causes the same?) and deletion/insertion curves run on the real machine.</p>
<p><b>Content output.</b> \(y\) is a RAM byte read through <code>soft_ram_peek</code>, a
one-hot dot product \(y = e_{\text{idx}}\cdot \mathrm{ram}\). Its gradient is exactly the one-hot
vector \(e_{\text{idx}}\): saliency points at the byte itself and nothing else. That is
correct for &ldquo;which byte holds the score&rdquo;, but it says nothing about which <i>other</i>
bytes will change the score over the next 15 frames.</p>
<p><b>Position output.</b> This is the toy example's \(x_2\) for real. The VCS places a sprite
by <i>when</i> the program writes a strobe register during the scanline: a discrete event. Moving
the position byte by 0.001 moves the sprite by zero pixels, so the honest derivative is
<b>exactly zero</b> everywhere: the <i>vanishing gradient</i>. We report that naive zero and, side by
side, a gradient through a <b>bilinear sampler</b> that draws the sprite at a continuous column.
This is the same trick spatial-transformer networks use, and it restores a non-zero gradient.</p>
<p><b>The end-to-end arm.</b> The two reads above differentiate one line of code at \(f^*\). The
<code>e2e_gradient.jl</code> runner instead differentiates the content byte at \(f^*{+}15\)
<b>back through every CPU instruction of the 15 frames</b> (about 10<sup>5</sup> instructions,
checkpointed every 200). That is the analogue of backpropagating through a very deep, very
discrete network. The renderer (TIA) is still not differentiated.</p>
<p><b>What it found</b> (all 44 scored games, forward pass bit-exact in every one, 8&ndash;23 hours
of backward pass per game on one core). On most games the answer is the same map as the one-line read.
On the content byte the per-cause map is unchanged in 34 of 43 games finished so far, and in the position
output in 34 as well. The six core games show the patterns one by one:</p>
<ul>
<li><b>Content:</b> in five games the gradient through 65&thinsp;000&ndash;81&thinsp;000 instructions is
<i>exactly</i> the one-hot vector \(e_{\text{idx}}\) again (\(\lVert g\rVert_1 = 1\), one non-zero
entry), so its scores equal the one-line read's to the last digit (Breakout 0.750, Space Invaders 1.000,
Seaquest 0.017, Ms.&nbsp;Pac-Man 0.811, Q*bert 0.999). The byte simply persists or is incremented.
Every other way the oracle's causes reach it goes through a branch (&ldquo;if the ball hit the
paddle&hellip;&rdquo;), an indexed load or a hardware read, and none of those carries a gradient.</li>
<li><b>Pong, the instructive exception:</b> the gradient also lands on RAM[56] with
\(\partial y/\partial \mathrm{ram}_{56} = 15\), the horizon length. That is the signature of a
velocity being added to a position once per frame: \(x_{t+15} = x_t + 15\,v\), so
\(\partial x_{t+15}/\partial v = 15\). Backpropagation through time has discovered the game's
physics. RAM[56] is not among the candidate causes, though, so the scores again equal the one-line
read's (0.464), and the oracle's strongest cause still gets zero. Atlantis shows the same signature.</li>
<li><b>Position:</b> in Breakout and Space Invaders the gradient arriving at \(f^*\) is
<b>exactly zero</b>, although the sampler's gradient at \(f^*{+}15\) is non-zero (checked). Breakout
recomputes its paddle position every frame from the paddle controller, a hardware input the gradient
treats as a constant. The vanishing gradient, measured.</li>
<li><b>Across all games:</b> the gradient vanishes completely on 8 games' content byte and on 8 games'
position output. Three games lose a good content score that way (Enduro 1.000 to 0, Phoenix 0.909 to 0,
Air Raid 0.360 to 0). One game gains: in Centipede the content byte at \(f^*{+}15\) is carried over
from RAM[111] at \(f^*\), the oracle's strongest cause, and the score jumps from &minus;0.10 to 0.85.
On average the deep gradient is slightly <i>less</i> faithful than the one-line read (content 0.627 to
0.594, position 0.090 to 0.084).</li>
</ul>
<p>The lesson for neural networks: a deeper, more &ldquo;honest&rdquo; gradient is not automatically a
more faithful explanation. Through a computation made of decisions rather than smooth arithmetic,
backpropagation sees only the smooth sliver.</p>""",

    code=[
        dict(path="tools/xai_study/phaseB_attribution/saliency.jl", find="function saliency_over_ram",
             role="xai", desc="The method itself, in four lines: <code>abs.(Zygote.gradient(readf, ram))</code>."),
        dict(path="tools/xai_study/phaseB_attribution/saliency.jl", find="content_read(ram::AbstractVector",
             role="xai", desc="The differentiable content output: a one-hot read of RAM byte <code>idx</code>."),
        dict(path="tools/xai_study/phaseB_attribution/saliency.jl", find="position_read_zero(ram::AbstractVector",
             role="xai", desc="The naive position output: no differentiable path exists, so the gradient is identically zero."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function _st_sampler_position_read",
             role="xai", desc="The bilinear sampler that gives the position output a non-zero gradient."),
        dict(path="tools/xai_study/phaseB_attribution/saliency.jl", find="function compute_one",
             role="xai", desc="One experiment: oracle &rarr; saliency &rarr; correlation / precision@k &rarr; deletion/insertion on the real VCS &rarr; positive control."),
        dict(path="jutari/src/diff/SoftStep.jl", find="function soft_ram_peek",
             role="substrate", desc="The differentiable RAM read the gradient flows through (forward value exact, gradient one-hot)."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!",
             role="substrate", desc="One video frame of the real, bit-exact machine (used by the oracle and the deletion curves)."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function run_intervention",
             role="substrate", desc="The ground truth: poke a cause, re-run the ROM, read the output."),
        dict(path="tools/xai_study/phaseB_attribution/e2e_gradient.jl", find="function e2e_vjp",
             role="xai", desc="The true end-to-end gradient: checkpointed reverse mode through every instruction of the horizon."),
        dict(path="jutari/src/diff/SoftSystem.jl", find="function soft_system_step",
             role="substrate", desc="One differentiable CPU instruction on the full machine (TIA/RIOT reads replayed as constants)."),
    ],

    reading=r"""
<p>In the figure, the bars are saliency per cause and the dots/bars beside them are the oracle's true
effect. A faithful method puts its tall bars where the oracle's tall bars are. For the
<b>content</b> output you will see one spike, at the byte that <i>is</i> the output. For the
<b>naive position</b> output you will see nothing at all, because the gradient is zero. With the
sampler you see a single spike at the position byte. The oracle, meanwhile, usually lights up several
causes: the gradient is <i>local</i>, the oracle is <i>global</i>.</p>""",

    pitfalls=[
        "<b>Local is not global.</b> A gradient describes an infinitesimal neighbourhood. Real interventions are finite (here: set a byte to a new value), and on a discrete machine the two can disagree completely.",
        "<b>Vanishing / zero gradients</b> wherever the computation is piecewise constant (thresholds, comparisons, table look-ups, sprite positioning by timing). ReLU networks have the same problem in their flat regions; the VCS has it everywhere.",
        "<b>Saturation.</b> If a byte is already at a value where the output no longer responds (think of a saturated sigmoid), its gradient is ~0 even though it matters a lot.",
        "<b>Visual plausibility is not faithfulness.</b> Adebayo et al. showed saliency maps can look sensible even for a network with randomised weights. Here we never judge a map by looks: every score is against the oracle.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseB_attribution/saliency.jl --games core",
    try_note="Writes <code>saliency_&lt;game&gt;_{content,position}.json/.npz</code> into "
             "<code>tools/xai_study/phaseB_attribution/out/</code>. For the end-to-end variant run "
             "<code>e2e_gradient.jl --game pong</code> (8&ndash;23 hours per game on one CPU core; records "
             "<code>e2e_{saliency,gradxinput}_&lt;game&gt;_*</code> and <code>e2e_core_summary.json</code>).",

    next=["gradxinput", "smoothgrad", "ig_baseline_sweep", "occlusion"],
)
