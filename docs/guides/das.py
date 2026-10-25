# Beginner's guide — interchange interventions / Distributed Alignment Search (DAS).
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="das",
    level="Mechanistic family · testing a hypothesis about a variable",

    wiki=[
        ("Causal model", "https://en.wikipedia.org/wiki/Causal_model"),
        ("Orthogonal matrix", "https://en.wikipedia.org/wiki/Orthogonal_matrix"),
        ("Rotation matrix", "https://en.wikipedia.org/wiki/Rotation_matrix"),
        ("Linear subspace", "https://en.wikipedia.org/wiki/Linear_subspace"),
        ("Mechanistic interpretability", "https://en.wikipedia.org/wiki/Mechanistic_interpretability"),
    ],
    papers=[
        ("Geiger, Lu, Icard &amp; Potts (2021). <i>Causal Abstractions of Neural Networks.</i> NeurIPS.",
         "https://arxiv.org/abs/2106.02997"),
        ("Geiger, Wu, Potts, Icard &amp; Goodman (2023). <i>Finding Alignments Between Interpretable "
         "Causal Variables and Distributed Neural Representations</i> (DAS). CLeaR 2024.",
         "https://arxiv.org/abs/2303.02536"),
        ("Makelov, Lange &amp; Nanda (2023). <i>Is This the Subspace You Are Looking For? An "
         "Interpretability Illusion for Subspace Activation Patching.</i> ICLR 2024.",
         "https://arxiv.org/abs/2311.17030"),
    ],

    idea=r"""
<p>Suppose you believe a network solves a task with a simple algorithm. For example, to compare two
numbers it first computes &ldquo;which one is bigger&rdquo; and then reports it. That simple algorithm
is a <b>high-level causal model</b>: a few named variables and the arrows between them. The question
is whether the network <i>really</i> contains a variable that plays the role of &ldquo;which one is
bigger&rdquo;. Causal abstraction answers it with an <b>interchange intervention</b>: take the hidden
state that should hold the variable from a <b>source</b> input, put it into the run on a <b>base</b>
input, and check whether the output changes exactly as the high-level model predicts when you swap
that variable.</p>
<p>The catch is that a network rarely stores a variable in one neuron. It may be spread across many
neurons, along some direction of the activation space. That is a <b>distributed representation</b>.
<b>Distributed Alignment Search (DAS)</b> therefore learns a <b>rotation</b> of the activation space.
After rotating, the first few coordinates should hold the variable, and swapping only those
coordinates should behave like swapping the variable. On the VCS we already know where every variable
lives, so the method turns into a clean test of the swap logic itself, and a lesson about what a
perfect score can and cannot prove.</p>""",

    nn=r"""
<p>Let \(h(x) \in \mathbb{R}^d\) be the hidden vector at some layer for input \(x\). DAS learns an
<b>orthogonal matrix</b> \(R\) (a rotation: \(R^\top R = I\), so no information is lost) and fixes a
number \(k\) of coordinates. The interchange intervention on base \(b\) with source \(s\) is</p>
\[ \hat h \;=\; R^\top\!\Big( (I - P_k)\,R\,h(b) \;+\; P_k\,R\,h(s) \Big), \]
<p>where \(P_k\) keeps only the first \(k\) rotated coordinates. The rest of the network then runs on
\(\hat h\). The high-level model predicts an output \(y^\ast(b, s)\): the answer you would get on
\(b\) if its variable took \(s\)&rsquo;s value. \(R\) is trained by gradient descent so the network&rsquo;s
output matches \(y^\ast\) on many \((b, s)\) pairs. The score is <b>interchange intervention accuracy
(IIA)</b>: the fraction of held-out pairs where the swapped run gives \(y^\ast\).</p>
<p><b>Toy example.</b> Let \(h = (h_1, h_2)\) with \(h_1 = u + v\) and \(h_2 = u - v\), where the
variable we care about is \(u\). No single neuron is \(u\). But rotate by 45&deg;:
\(R h = \tfrac{1}{\sqrt 2}(h_1 + h_2,\; h_1 - h_2) = \sqrt 2\,(u, v)\). Now the first rotated
coordinate is exactly \(u\) (times a constant). Base \(u{=}1, v{=}3\); source \(u{=}5, v{=}0\). Swap
coordinate 1: the patched state encodes \(u{=}5, v{=}3\), so \(h = (8, 2)\). That is precisely the
prediction &ldquo;base with source&rsquo;s \(u\)&rdquo;. Swapping neuron \(h_1\) alone would give
\(h = (5, -2)\), which decodes to \(u{=}1.5, v{=}3.5\): the wrong \(u\) and a corrupted \(v\).
This is why DAS looks for a direction, not a neuron.</p>""",

    recipe=[
        "Write down the high-level causal model and choose the variable \\(V\\) you want to locate.",
        "Choose a layer and a subspace size \\(k\\); initialise a rotation \\(R\\).",
        "Sample base/source pairs; compute the high-level prediction \\(y^\\ast(b,s)\\) for swapping \\(V\\).",
        "Run the interchange intervention in the rotated basis and train \\(R\\) so the network output matches \\(y^\\ast\\).",
        "Report IIA on held-out pairs, and compare it with a control (a random or wrong subspace).",
    ],

    atari_map=[
        ("Hidden vector \\(h\\) at a layer", "The VCS state at frame \\(f^*\\): 128 bytes of RAM plus TIA "
         "registers."),
        ("High-level variable \\(V\\)", "A named game variable from the candidate list (a position, a "
         "score digit, a timer), plus the background colour."),
        ("Learned rotation \\(R\\) and subspace \\(P_k\\)", "Not learned here. The aligned &ldquo;subspace&rdquo; "
         "is one <b>axis-aligned</b> byte: the program stores each variable in its own RAM cell."),
        ("Base / source inputs", "Base: the clean gameplay stream. Source: either a <i>directed</i> value "
         "(<code>base+17</code>) or a <i>natural</i> one (the same stream with RIGHT pressed before "
         "\\(f^*\\))."),
        ("IIA", "Fraction of variables whose cell, after the swap, holds the source value the high-level "
         "model predicts."),
        ("Wrong-subspace control", "Swap the <i>next</i> candidate cell instead and check variable \\(V\\): "
         "accuracy should collapse."),
    ],

    atari=r"""
<p>The runner reads its variables from the game&rsquo;s candidate file: each candidate RAM byte becomes a
variable whose value is the byte itself, plus a background-colour variable read from a screen pixel.
The claim being tested is that &ldquo;variable \(V\) lives in byte \(i\)&rdquo;. Everything runs on the
shared gameplay checkpoint at \(f^*\), with a 15-frame tail.</p>
<ol>
<li><b>Interchange-effect matrix.</b> For every cell \(i\), write the directed source value
<code>base+17</code> into a copy of the base checkpoint, run 15 frames, and read every variable \(j\).
The same swap is replayed from a fresh boot; the two matrices must be identical.</li>
<li><b>IIA of the true alignment.</b> For each RAM variable, swap its own cell and read the cell back
<i>at the swap frame</i>. It must hold the source value. This is done for the directed source and,
where the natural RIGHT-press source really differs from base, for that too.</li>
<li><b>Misaligned control.</b> Swap the next candidate cell in the list and read variable \(V\).
It matches the prediction only if the two cells happen to alias, so accuracy should drop to near
zero. This is what makes a perfect aligned score meaningful.</li>
<li><b>Alignment search.</b> For each variable, the runner picks the cell whose swap best reproduces
the variable&rsquo;s own 15-frame effect, and checks whether that is the true cell. Cells the program
overwrites within the horizon (<b>transient</b> cells) give no signal and are listed separately.</li>
</ol>
<p>Be clear about the difference from real DAS. There is <b>no learned rotation</b> and no gradient
training. The search space is the set of single bytes, because on the VCS the program&rsquo;s variables
are stored one per byte. The concept labels come from the candidate file, not from the method. So the
VCS run checks that interchange logic and the IIA metric reward the right alignment and punish a wrong
one; it does not show that DAS would <i>discover</i> the variable unaided.</p>""",

    code=[
        dict(path="tools/xai_study/phaseC_mechanistic/das.jl", find="function build_variables(",
             role="xai", desc="The high-level variables: one per labelled candidate byte, plus background colour."),
        dict(path="tools/xai_study/phaseC_mechanistic/das.jl", find="function interchange(",
             role="xai", desc="The interchange intervention: write source values into a copy of the base checkpoint, run the tail."),
        dict(path="tools/xai_study/phaseC_mechanistic/das.jl", find="src_at_t = continue_from(boot_replay(vcat(pre, ACT_RIGHT)",
             role="xai", desc="The natural source run: same gameplay prefix, RIGHT pressed before \\(f^*\\)."),
        dict(path="tools/xai_study/phaseC_mechanistic/das.jl", find="src_value = [align_kind[j]",
             role="xai", desc="The directed source values (<code>base+17</code>), so every cell gets a real swap."),
        dict(path="tools/xai_study/phaseC_mechanistic/das.jl", find="function self_after_interchange(",
             role="xai", desc="IIA readout: does the variable hold the source value right after the swap?"),
        dict(path="tools/xai_study/phaseC_mechanistic/das.jl", find="wrong = ram_js[mod1(k + 1",
             role="xai", desc="The misaligned negative control: swap the wrong cell, read the variable."),
        dict(path="tools/xai_study/common/jutari_oracle.jl", find="function intervene_ram!(",
             role="substrate", desc="The byte write that realises every interchange."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function build_shared_testbed(",
             role="substrate", desc="The shared base state every Phase C method sits on."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(",
             role="substrate", desc="One bit-exact frame; the 15-frame tail is a loop over it."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function run_intervention(",
             role="substrate", desc="The oracle. A single-cell interchange is the same operation as its <code>do(cell := value)</code>."),
    ],

    reading=r"""
<p>Expect the aligned IIA at the ceiling and the misaligned control near the floor. The gap between
the two is the real content: it shows the metric discriminates. Then look at the alignment search and
the transient list. Variables whose cell is rewritten every frame cannot be aligned by their downstream
effect, because there is none within the horizon. A perfect score here verifies a <i>given</i> label;
it is not evidence that the method found the label itself.</p>""",

    pitfalls=[
        "<b>A supplied variable is not a discovered one.</b> DAS tests an alignment you propose. If you choose the hypothesis from the same labels you score against, a perfect IIA is guaranteed in advance.",
        "<b>Interpretability illusions.</b> Makelov et al. show that a learned subspace can reach high IIA by activating a dormant pathway the model never uses on real inputs. Always compare against a control and check the subspace on natural data.",
        "<b>Too much freedom.</b> A large subspace or a very expressive map can make almost any hypothesis &ldquo;fit&rdquo;. Keep \\(k\\) small and report held-out accuracy.",
        "<b>Readout timing.</b> On the VCS, reading the variable right at the swap is trivially exact. Reading after 15 frames shows that many cells are overwritten by the program, so present is not the same as used.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseC_mechanistic/das.jl --games core",
    try_note="Writes <code>das_&lt;game&gt;.json/.npz</code> and <code>das_core_summary.json</code> into "
             "<code>tools/xai_study/phaseC_mechanistic/out/</code>. One re-run (plus one fresh-boot replay) "
             "per candidate cell; minutes per game on one CPU core.",

    next=["activation_patching", "causal_scrubbing", "linear_probing", "sae"],
)
