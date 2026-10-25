# Beginner's guide — linear probing with control tasks.
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="linear_probing",
    level="Mechanistic family · decoding what is represented",

    wiki=[
        ("Linear classifier", "https://en.wikipedia.org/wiki/Linear_classifier"),
        ("Ridge regression", "https://en.wikipedia.org/wiki/Ridge_regression"),
        ("Overfitting", "https://en.wikipedia.org/wiki/Overfitting"),
        ("Multiclass classification", "https://en.wikipedia.org/wiki/Multiclass_classification"),
    ],
    papers=[
        ("Alain &amp; Bengio (2017). <i>Understanding intermediate layers using linear classifier "
         "probes.</i> ICLR workshop.",
         "https://arxiv.org/abs/1610.01644"),
        ("Hewitt &amp; Liang (2019). <i>Designing and Interpreting Probes with Control Tasks.</i> EMNLP.",
         "https://arxiv.org/abs/1909.03368"),
        ("Belinkov (2022). <i>Probing Classifiers: Promises, Shortcomings, and Advances.</i> "
         "Computational Linguistics. (critique / survey)",
         "https://arxiv.org/abs/2102.12452"),
        ("Elazar, Ravfogel, Jacovi &amp; Goldberg (2021). <i>Amnesic Probing: Behavioral Explanation with "
         "Amnesic Counterfactuals.</i> TACL. (decodable is not used)",
         "https://arxiv.org/abs/2006.00995"),
    ],

    idea=r"""
<p>Does a language model &ldquo;know&rdquo; the part of speech of each word? Does a chess network
&ldquo;know&rdquo; where the king is? A <b>probe</b> is the simplest way to ask. You freeze the network,
record its hidden activations, and train a tiny, separate classifier to predict the concept from them.
If even a <b>linear</b> classifier succeeds, people say the concept is <b>linearly represented</b> at
that layer. Probing is cheap, works on any layer, and has been run on almost every large model.</p>
<p>Two traps make the answer less clear than it looks. First, a probe that is too flexible can
<b>memorise</b> labels instead of finding them. Hewitt and Liang's <b>control task</b> fixes that: give
the same probe a nonsense labelling and see how well it does anyway. Second, and deeper: a probe tells
you a concept is <b>present</b>, that is, readable from the activations. It does not tell you the network
<b>uses</b> it. Information can sit in a layer as a by-product, read by no later computation. This page
is built around that present-versus-used gap.</p>""",

    nn=r"""
<p>Let \(h \in \mathbb{R}^d\) be the frozen activation for one input and \(c \in \{1,\dots,K\}\) the
concept label. A linear probe predicts</p>
\[ \hat c = \arg\max_k \,(W h + b)_k , \]
<p>with \(W, b\) fit on labelled examples and accuracy measured on held-out ones. The <b>control
task</b> keeps the inputs but replaces the labels: every distinct input type (in NLP, every word) gets a
<b>fixed random</b> label. A probe can only score well on that by memorising. The honest number is the
<b>selectivity</b>:</p>
\[ \text{selectivity} = \text{acc}_{\text{real}} - \text{acc}_{\text{control}} . \]
<p><b>Toy example (present but unused).</b> A two-unit hidden layer gets an input number \(n\). Unit
\(h_1 = n\) and unit \(h_2 = n \bmod 2\) (the parity). The output is \(y = 2h_1\); nothing ever reads
\(h_2\). A probe on \(h_2\) predicts &ldquo;is \(n\) odd?&rdquo; with 100&nbsp;% accuracy, and on a
control task it scores only chance, so selectivity is high. A naive reading says &ldquo;the network
computes parity&rdquo;. Yet setting \(h_2\) to any value leaves \(y\) unchanged: for \(n=3\), \(y=6\)
whether \(h_2\) is 0 or 1. Decodable, selective, and causally irrelevant.</p>""",

    recipe=[
        "Record frozen activations and the concept label for many inputs; split into train and test.",
        "Fit a linear classifier (logistic or least-squares, usually regularised) and measure test accuracy.",
        "Build a control task: map each input type to a fixed random label of the same cardinality, and train the same probe on it.",
        "Report selectivity = real accuracy minus control accuracy, not raw accuracy.",
        "To ask whether the concept is <i>used</i>, intervene: remove or change it in the activations and check the output (e.g. amnesic probing).",
    ],

    atari_map=[
        ("Hidden activation \\(h\\)", "The 128-byte RAM at one frame, minus the one byte being probed (127 inputs, z-scored, plus a bias)."),
        ("The concept", "A known game variable in its verified RAM cell, e.g. ball x in <code>RAM[49]</code> for Pong."),
        ("Class label \\(c\\)", "The cell's value split into up to 4 equal-width bins over the range seen in the run."),
        ("Control task", "Each distinct raw byte value of the cell gets a fixed random label among the same number of classes."),
        ("&ldquo;Is it used?&rdquo; (rarely testable)", "The oracle's flag: cells whose intervention is overwritten by the program within the horizon, so it never reaches the output."),
    ],

    atari=r"""
<p>Our runner <code>linear_probing.jl</code> needs labels, because it
decodes <i>named</i> concepts. Per game it does this:</p>
<ul>
<li><b>Data.</b> A 600-frame RAM tape. The first 105 frames follow the shared seeded gameplay stream; the
rest continue with NOOP. 600 frames give 420 training rows for 127 inputs, enough that the linear
system cannot simply memorise.</li>
<li><b>Hold out the target.</b> To probe cell \(c\), the input is every <i>other</i> byte. Otherwise the
probe could copy the answer. So the real question is: is this variable <b>redundantly</b> encoded in the
rest of the state?</li>
<li><b>Probe.</b> Labels are 4 equal-width bins of the cell's observed range (empty bins dropped). The
probe is a one-vs-rest least-squares classifier with a small ridge penalty (\(\lambda = 0.01\)), solved
in closed form. Frames are shuffled with a fixed seed and split 70/30.</li>
<li><b>Control.</b> Every distinct raw byte value gets a random label. Keying on raw values, not on the 4
bins, makes the control genuinely hard: a 4-way random map would be easy to memorise.</li>
<li><b>Decodable.</b> A cell counts as decodable (present) if test accuracy is at least 0.5 <i>and</i>
selectivity is at least 0.1.</li>
<li><b>Used?</b> Each probed cell is then cross-checked against the oracle's list of
&ldquo;transient&rdquo; cells from the DAS run (<code>das_&lt;game&gt;.json</code>). These are cells the
program recomputes within the horizon, so writing a new value into them has no downstream effect. A cell
that is decodable <b>and</b> transient is counted as <b>decodable-but-not-causal</b>: present, not
used.</li>
</ul>
<p>The headline is mean selectivity over all probeable cells, next to the count of decodable-but-not-causal
cells. Note the dependency: if the DAS output is missing, the &ldquo;used&rdquo; cross-check is marked as
unavailable rather than guessed.</p>""",

    code=[
        dict(path="tools/xai_study/phaseC_mechanistic/linear_probing.jl", find="function probe_accuracy(Xtr, ytr, Xte, yte, nclass",
             role="xai", desc="The probe: closed-form ridge least squares, one-vs-rest, accuracy on held-out frames."),
        dict(path="tools/xai_study/phaseC_mechanistic/linear_probing.jl", find="function quantise(values::AbstractVector{<:Real}",
             role="xai", desc="Turns a byte series into up to 4 class labels (equal-width bins)."),
        dict(path="tools/xai_study/phaseC_mechanistic/linear_probing.jl", find="function oracle_transient_cells(game::AbstractString)",
             role="xai", desc="Reads the oracle's &ldquo;not causally used&rdquo; cells from the DAS output."),
        dict(path="tools/xai_study/phaseC_mechanistic/linear_probing.jl", find="function record_ram_tape(game::AbstractString",
             role="xai", desc="Records the 600-frame RAM tape the probes are trained on."),
        dict(path="tools/xai_study/phaseC_mechanistic/linear_probing.jl", find="function run_game(; game::AbstractString",
             role="xai", desc="Per game: hold out each cell, fit probe and control, selectivity, present-vs-used cross-check."),
        dict(path="tools/xai_study/phaseC_mechanistic/linear_probing.jl", find="const DECODE_SELECTIVITY_MIN",
             role="xai", desc="The decodability thresholds (accuracy &ge; 0.5, selectivity &ge; 0.1)."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function candidate_ram_indices(candidates_path)",
             role="substrate", desc="Loads the verified concept&harr;cell map: the probe targets."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function build_shared_testbed(",
             role="substrate", desc="The shared seeded gameplay stream the tape starts from."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_reset!(env::StellaEnvironment",
             role="substrate", desc="Boots the cartridge (xitari's construction sequence: a 60-frame probe, then boots of 60 NOOP + 4 RESET) before recording."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="get_ram(env::StellaEnvironment)",
             role="substrate", desc="The live 128-byte RAM read after every frame to fill the tape."),
    ],

    reading=r"""
<p>Each bar is one labelled cell: its selectivity (probe accuracy minus control accuracy); red bars are
cells the oracle flags as not causally used. Expect most game variables to be decodable above the control. The VCS state is small and full of
copies and derived values, so the rest of RAM often predicts a variable well. Then look for the red cells
that are still decodable: present but not used. They are the key result: a probe would &ldquo;find&rdquo; these concepts,
a naive reading would say the program relies on them, and the intervention shows that, within the
window, it does not.</p>""",

    pitfalls=[
        "<b>Present is not used.</b> The central lesson. A probe answers &ldquo;can it be read?&rdquo;. Only an intervention answers &ldquo;does changing it change the output?&rdquo;.",
        "<b>Probe capacity.</b> A powerful probe can learn the task itself. Use a control task and report selectivity; keep the probe linear and regularised.",
        "<b>Neighbouring frames leak.</b> We shuffle frames before splitting, and consecutive frames are nearly identical. Test accuracy can therefore be optimistic compared with a split by time.",
        "<b>Binning shapes the task.</b> Four equal-width bins over the observed range are a choice; a different binning changes both accuracies.",
        "<b>Redundancy, not location.</b> Because the target cell is held out, a high score means the variable can be inferred from <i>other</i> bytes. It does not say where the program stores it.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseC_mechanistic/linear_probing.jl --games core",
    try_note="Writes <code>linear_probing_&lt;game&gt;.{json,npz}</code> and "
             "<code>linear_probing_core_summary.json</code> into "
             "<code>tools/xai_study/phaseC_mechanistic/out/</code>. Run <code>das.jl</code> first so the "
             "present-vs-used cross-check has the oracle's transient-cell list.",

    next=["logit_lens", "das", "sae", "A3_tuning"],
)
