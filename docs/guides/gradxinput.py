# Beginner's guide — Gradient x Input and DeepLIFT (rescale rule).
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="gradxinput",
    level="Gradient family · saliency weighted by the input",

    wiki=[
        ("Gradient", "https://en.wikipedia.org/wiki/Gradient"),
        ("Hadamard product (element-wise product)", "https://en.wikipedia.org/wiki/Hadamard_product_(matrices)"),
        ("Linear approximation", "https://en.wikipedia.org/wiki/Linear_approximation"),
        ("Taylor series", "https://en.wikipedia.org/wiki/Taylor_series"),
    ],
    papers=[
        ("Shrikumar, Greenside &amp; Kundaje (2017). <i>Learning Important Features Through "
         "Propagating Activation Differences</i> (DeepLIFT). ICML.",
         "https://arxiv.org/abs/1704.02685"),
        ("Ancona, Ceolini, &Ouml;ztireli &amp; Gross (2018). <i>Towards better understanding of "
         "gradient-based attribution methods for Deep Neural Networks.</i> ICLR.",
         "https://arxiv.org/abs/1711.06104"),
        ("Kindermans et al. (2017/2019). <i>The (Un)reliability of saliency methods.</i> "
         "(Input-shift critique of Grad&times;Input and DeepLIFT.)",
         "https://arxiv.org/abs/1711.00867"),
    ],

    idea=r"""
<p>Vanilla saliency answers one question: <i>how sensitive is the output to each input?</i> That
is useful, but it ignores how big the input actually is. A pixel can be very sensitive and still
be black (value 0). Then it contributes nothing to the score right now. <b>Gradient&times;Input</b>
fixes this by multiplying the two: the attribution of input \(i\) is its gradient
<i>times its own value</i>. The result reads as &ldquo;how much of the output this input is
responsible for, at the value it has&rdquo;.</p>
<p><b>DeepLIFT</b> (Deep Learning Important FeaTures) generalises the idea. It picks a
<b>reference</b> (or <b>baseline</b>) input that means &ldquo;nothing there&rdquo;, for example an
all-black image. It then splits the difference between the output at the real input and the output
at the reference among the inputs. The pieces add up exactly to that difference. This property is
called <b>completeness</b> (DeepLIFT calls it &ldquo;summation-to-delta&rdquo;). With an all-zero
reference and a linear model, DeepLIFT and Grad&times;Input give the same numbers. That is the
setting we use on the Atari.</p>""",

    nn=r"""
<p>As in vanilla saliency, write the network as \(y = f(x)\). Grad&times;Input and DeepLIFT
(rescale rule) assign input \(i\)</p>
\[ A^{\text{G}\times\text{I}}_i = x_i\,\frac{\partial f}{\partial x_i}(x),
\qquad
A^{\text{DL}}_i = (x_i - x^0_i)\; m_i ,\]
<p>where \(x^0\) is the reference and \(m_i\) is a <b>multiplier</b>. For a linear unit
\(m_i = \partial f/\partial x_i\). Through a nonlinearity such as a ReLU, the rescale rule
uses the <i>finite</i> slope \(m = \Delta \text{out}/\Delta \text{in}\) between reference and
input instead of the infinitesimal one. The product of input and gradient is the first-order Taylor
term \(\nabla f(x)\cdot x\): for a linear function with zero bias it is exactly \(f(x) - f(0)\).</p>
<p><b>Toy example 1 (linear).</b> Let \(f(x) = 2x_1 - x_2 + 0.5x_3\) at \(x = (4, 1, 0)\). The gradient
is \((2, -1, 0.5)\). Saliency says \(x_1\) matters most, then \(x_2\), then \(x_3\). Grad&times;Input
gives \((8, -1, 0)\): \(x_3\) gets <b>zero</b>, because it is switched off. The sum is
\(8 - 1 + 0 = 7 = f(x) - f(0)\). Completeness holds.</p>
<p><b>Toy example 2 (nonlinear).</b> Let \(f(x) = x^2\) at \(x = 3\), reference \(x^0 = 0\). The
gradient is 6, so Grad&times;Input says 18. But the output only changed by \(f(3) - f(0) = 9\).
DeepLIFT's rescale multiplier is \((9 - 0)/(3 - 0) = 3\), giving \(3 \cdot 3 = 9\): complete. So
Grad&times;Input is complete only for linear pieces; DeepLIFT repairs this by using finite
differences.</p>""",

    recipe=[
        r"Pick the input \(x\), the scalar output \(y\), and (for DeepLIFT) a reference \(x^0\). The usual default is all zeros.",
        r"Compute the gradient \(\partial y/\partial x_i\) with one backward pass, exactly as in vanilla saliency.",
        r"Multiply element-wise: \(x_i \cdot \partial y/\partial x_i\) (Grad&times;Input) or \((x_i - x^0_i)\cdot m_i\) (DeepLIFT).",
        r"Optionally check completeness: \(\sum_i A_i\) should equal \(y(x) - y(x^0)\).",
        r"Keep the sign (positive = pushes the output up) or take \(|A_i|\) for a heat-map.",
    ],

    atari_map=[
        (r"The network \(f\)", r"The Atari 2600 VCS running the game ROM, emulated bit-exactly by "
         r"<code>jutari</code>; its differentiable twin is the SOFT path <code>JuTari.Diff</code>."),
        (r"The input \(x\)", r"The 128 RAM bytes at the intervention frame \(f^*\) (after 90 frames of "
         r"seeded random play), as floats 0&ndash;255."),
        (r"The reference \(x^0\)", r"The all-zeros RAM vector, the &ldquo;absent&rdquo; state. With this "
         r"reference DeepLIFT and Grad&times;Input coincide byte for byte."),
        (r"The output \(y\)", r"A <b>content</b> output (one RAM byte read one-hot) and a <b>position</b> "
         r"output (screen pixels changed versus the un-intervened frame 15 frames later, made differentiable by the "
         r"bilinear sampler)."),
        (r"&ldquo;Is the attribution right?&rdquo;", r"The intervention oracle: poke each cause, re-run the "
         r"real ROM, measure \(|\Delta y|\)."),
    ],

    atari=r"""
<p>The runner computes three maps from <b>one</b> gradient: Grad&times;Input
\(u \odot \partial y/\partial u\), DeepLIFT \((u - 0)\odot \partial y/\partial u\), and vanilla saliency
\(|\partial y/\partial u|\) as the contrast. Here \(u\) is the RAM vector. Because the reference is
all zeros, the first two are identical. The record's headline number is the DeepLIFT
correlation with the oracle.</p>
<ul>
<li><b>Content output.</b> \(y\) is <code>soft_ram_peek(ram, s)</code>, a one-hot dot product. Its
gradient is the one-hot vector \(e_s\), so Grad&times;Input is a single spike of height
\(u_s\) at byte \(s\). Completeness is exact: \(\sum_i A_i = u_s = y(x) - y(0)\). The runner
checks this numerically. Note what the input factor can and cannot do here: it only <i>rescales</i>
the one spike saliency already had. The ranking of causes is unchanged, unless \(u_s = 0\).
Then the attribution vanishes entirely.</li>
<li><b>Choosing the byte.</b> That last case is why <code>select_content_cell</code> refuses a zero
byte. It also tests each candidate on the real VCS: poke it to value+17, run 15 frames, and keep
only bytes whose own change survives to the readout. Otherwise the game could overwrite the poke
within the horizon and leave an all-zero oracle map. A score byte is preferred when it qualifies.</li>
<li><b>Position output.</b> The naive gradient is exactly zero, because sprite positions are set by
strobe timing (see vanilla saliency). The runner therefore takes the gradient through the bilinear
sampler, which slides a sprite footprint along a continuous column driven by one position byte.
Grad&times;Input multiplies that gradient by the position byte's value. If the byte happens to be
0, the restored gradient is wiped out again. This is a real failure mode the runner records,
not a bug.</li>
<li><b>Scoring.</b> Each map is turned into one number per oracle cause (\(|A|\) at that RAM byte; 0 for
TIA-register and joystick causes the RAM map cannot see). It is compared with the oracle by
Pearson/Spearman correlation and precision@3, and by deletion/insertion curves that re-run the real
ROM. A positive control scores the oracle against itself.</li>
</ul>""",

    code=[
        dict(path="tools/xai_study/phaseB_attribution/gradxinput.jl", find="function attributions_over_ram(",
             role="xai", desc="The method: Grad&times;Input <code>x .* g</code>, DeepLIFT <code>(x .- x0) .* g</code>, saliency <code>abs.(g)</code>."),
        dict(path="tools/xai_study/phaseB_attribution/gradxinput.jl", find="function content_grad_over_ram(",
             role="xai", desc="One Zygote backward pass through the one-hot content read gives \\(\\partial y/\\partial u\\)."),
        dict(path="tools/xai_study/phaseB_attribution/gradxinput.jl", find="function deeplift_completeness_err(",
             role="xai", desc="Completeness check: \\(|\\sum A - (y(x) - y(x^0))|\\), asserted to be ~0."),
        dict(path="tools/xai_study/phaseB_attribution/gradxinput.jl", find="function select_content_cell(",
             role="xai", desc="Picks a non-zero, causally persistent content byte (tested on the real VCS) so the input factor cannot trivially zero it."),
        dict(path="tools/xai_study/phaseB_attribution/gradxinput.jl", find="g = Zygote.gradient(st.sampler_read, ram_now)[1]",
             role="xai", desc="Position output: the gradient is taken through the bilinear sampler instead of the (zero) naive read."),
        dict(path="tools/xai_study/phaseB_attribution/gradxinput.jl", find="function compute_game(",
             role="xai", desc="One game: shared testbed &rarr; oracle map &rarr; three attribution maps &rarr; scores &rarr; positive control."),
        dict(path="jutari/src/diff/SoftStep.jl", find="function soft_ram_peek",
             role="substrate", desc="The forward-exact one-hot RAM read; its gradient is the one-hot vector \\(e_s\\)."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function _st_sampler_position_read",
             role="substrate", desc="The bilinear sampler: a sprite whose column is a continuous function of one RAM byte."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function run_intervention",
             role="substrate", desc="The ground truth: poke a cause, re-run the ROM 15 frames, read the output."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!",
             role="substrate", desc="One frame of the bit-exact machine, used by the oracle and the deletion/insertion curves."),
    ],

    reading=r"""
<p>Compare the Grad&times;Input bars with the vanilla saliency page. On the <b>content</b> output
both show one spike at the same byte; Grad&times;Input's spike is just taller or shorter, scaled by
the byte's value. So its agreement with the oracle tracks saliency closely. A one-byte map can only partly
match an oracle that lights up several causes over 15 frames. On the <b>position</b> output, expect much weaker agreement: the sampler restores a
gradient, but it points at one byte, and that byte is not always the one the oracle ranks first.
Completeness holds exactly on content, yet it does not make the map more faithful. Summing to the
right total says nothing about whether the total is split among the right causes.</p>""",

    pitfalls=[
        r"<b>Zero inputs get zero credit.</b> A byte (or pixel) whose value is 0 receives no attribution, however important it is. On the VCS many flags and counters are legitimately 0.",
        r"<b>Reference dependence.</b> The all-zeros reference is a choice, not a law. Kindermans et al. showed that shifting every input by a constant changes Grad&times;Input/DeepLIFT maps although the model's decisions do not change.",
        r"<b>Completeness is not faithfulness.</b> Attributions can sum to the exact output change and still be assigned to the wrong inputs. On the content read here it is complete by construction because the read is linear.",
        r"<b>Inherits every gradient problem.</b> Multiplying by the input cannot create a gradient where there is none. The naive position gradient stays zero; only the sampler gives the method something to multiply.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseB_attribution/gradxinput.jl --games core",
    try_note="Writes <code>gradxinput_deeplift_&lt;game&gt;_{content,position}.json/.npz</code> and a "
             "per-output summary into <code>tools/xai_study/phaseB_attribution/out/</code>. Each game "
             "re-runs the real ROM once per cause for the oracle and twice per cause for the "
             "deletion/insertion curves, so expect minutes per game on one CPU core.",

    next=["saliency", "ig_baseline_sweep", "expected_gradients", "smoothgrad"],
)
