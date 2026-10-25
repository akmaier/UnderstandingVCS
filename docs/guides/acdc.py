# Beginner's guide — ACDC (Automatic Circuit DisCovery).
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="acdc",
    level="Mechanistic family · circuit discovery, automated",

    wiki=[
        ("Mechanistic interpretability", "https://en.wikipedia.org/wiki/Mechanistic_interpretability"),
        ("Pruning (artificial neural network)", "https://en.wikipedia.org/wiki/Pruning_(artificial_neural_network)"),
        ("Directed acyclic graph", "https://en.wikipedia.org/wiki/Directed_acyclic_graph"),
        ("Precision and recall", "https://en.wikipedia.org/wiki/Precision_and_recall"),
        ("Ablation (artificial intelligence)", "https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence)"),
    ],
    papers=[
        ("Conmy, Mavor-Parker, Lynch, Heimersheim &amp; Garriga-Alonso (2023). <i>Towards Automated "
         "Circuit Discovery for Mechanistic Interpretability.</i> NeurIPS.",
         "https://arxiv.org/abs/2304.14997"),
        ("Syed, Rager &amp; Conmy (2023). <i>Attribution Patching Outperforms Automated Circuit "
         "Discovery.</i>",
         "https://arxiv.org/abs/2310.10348"),
        ("Miller, Chughtai &amp; Saunders (2024). <i>Transformer Circuit Faithfulness Metrics are not "
         "Robust.</i> (critique of how circuits are scored)",
         "https://arxiv.org/abs/2407.08734"),
    ],

    idea=r"""
<p>A trained network is a huge web of computations. Mechanistic interpretability hopes that a
<b>specific behaviour</b>, such as &ldquo;complete the name in this sentence&rdquo;, is carried by a
<b>small sub-web</b> of that computation. That sub-web is called a <b>circuit</b>. For years, people
found circuits by hand: guess a component, knock it out, look at the output, repeat. That is slow and
depends on the researcher's intuition.</p>
<p><b>ACDC</b> automates the search. Draw the whole model as a <b>graph</b>: nodes are components
(attention heads, MLP blocks), and an <b>edge</b> means &ldquo;this node's output is an input to that
node&rdquo;. Then walk over the edges and <b>cut</b> each one in turn. If cutting an edge barely changes
the model's output, the edge is not needed and is pruned for good. If it changes the output by more than
a threshold \(\tau\), it stays. What survives is the discovered circuit. Turning the dial \(\tau\)
gives a whole family of circuits, from tiny (large \(\tau\)) to almost the full model (small
\(\tau\)).</p>""",

    nn=r"""
<p>&ldquo;Cutting&rdquo; an edge does not mean setting it to zero. Zero is often an input the model has
never seen. ACDC instead uses <b>resample ablation</b>: the signal on the edge is replaced by the value
it takes on a <b>corrupted input</b> (for example, the same sentence with a different name). Let
\(G\) be the full model, \(H\) the current candidate circuit, and \(H \setminus e\) the circuit with
edge \(e\) cut. ACDC measures how far the output distribution moves, with the
<b>Kullback&ndash;Leibler divergence</b> \(D_{\mathrm{KL}}\) (a standard distance between two
probability distributions), and prunes \(e\) if</p>
\[ D_{\mathrm{KL}}\big(G \,\|\, H\setminus e\big) \;-\; D_{\mathrm{KL}}\big(G \,\|\, H\big) \;<\; \tau . \]
<p>Edges are visited from the output backwards, and every prune is <b>kept</b>, so later decisions are
made on an already-thinned graph.</p>
<p><b>Toy example.</b> A network has input \(x\), two hidden units \(A\) and \(B\), and output \(y\).
The four edges are \(x\!\to\!A\), \(x\!\to\!B\), \(A\!\to\!y\), \(B\!\to\!y\). Resample-ablating each
one moves the output by 1.4 (for \(A\!\to\!y\)), 0.02 (\(B\!\to\!y\)), 1.0 (\(x\!\to\!A\)) and 0.5
(\(x\!\to\!B\)). With \(\tau = 0.1\), \(B\!\to\!y\) is pruned. Now \(B\) feeds nothing, so cutting
\(x\!\to\!B\) changes nothing either, and it is pruned too. The circuit is
\(x \to A \to y\). Note how the second decision depended on the first: that is the iterative part.</p>""",

    recipe=[
        "Choose a behaviour, a set of clean inputs, matching corrupted inputs, and an output metric (KL divergence or a logit difference).",
        "Write the model as a graph of components and start with every edge present.",
        "Visit edges from the output backwards. For each edge, replace its signal by the corrupted-input value and re-run the model.",
        "If the metric moves by less than \\(\\tau\\), delete the edge permanently; otherwise keep it.",
        "Repeat for several \\(\\tau\\) values. Each gives one candidate circuit.",
        "Judge the circuits: compare to a known circuit (precision/recall of edges), or check that the circuit alone reproduces the behaviour.",
    ],

    atari_map=[
        ("The model's computational graph", "The data flow between <b>candidate RAM cells</b> of the cartridge program. For Pong these are the 8 cells the labelling step named: the two scores, the paddle and ball positions."),
        ("A node (attention head, MLP)", "One candidate RAM byte, e.g. <code>RAM[49]</code> = ball x."),
        ("An edge \\(i \\to j\\)", "&ldquo;The value of cell \\(i\\) at frame \\(f^*\\) influences the value of cell \\(j\\) 15 frames later&rdquo;. This is a <i>total</i> effect over the horizon, not one direct wire."),
        ("Corrupted input / resample ablation", "Overwrite cell \\(i\\) at \\(f^*\\) with a corrupted value (from a donor run if it differs, else the clean value + 17), then re-run the real ROM."),
        ("Output metric (KL divergence)", "The absolute change of child cell \\(j\\)'s byte after the horizon: \\(|y_j(\\text{ablated}) - y_j(\\text{clean})|\\)."),
        ("The true circuit (unknown for a network)", "The <b>exhaustive</b> data-flow graph: edge \\(i\\to j\\) exists iff setting cell \\(i\\) to 0, base+17 or base+37 changes cell \\(j\\) on the bit-exact re-run."),
    ],

    atari=r"""
<p>On a language model nobody knows the &ldquo;real&rdquo; circuit, so ACDC can only be judged against
other human guesses. On the VCS the program's data flow can be measured exactly, because
<code>jutari</code> re-runs the ROM bit-for-bit. Our runner <code>acdc.jl</code> does the following:</p>
<ol>
<li><b>State.</b> It boots the game, plays the shared seeded random-action stream for 90 frames, and
checkpoints the machine at \(f^*\). The remaining 15 frames of the stream are the <b>tail</b>. It
asserts that two fresh boots give identical RAM and screen before scoring anything.</li>
<li><b>Edge effects.</b> For every candidate cell \(i\), it writes one corrupted value into cell \(i\),
plays the tail once, and reads <i>every</i> candidate cell \(j\). That fills a whole row of the effect
matrix \(E[i,j]\) with one re-run. The corrupted value comes from a donor run that switches to the
RIGHT action at \(f^*\); because that donor is identical to the clean run up to \(f^*\), the synthetic
fallback base+17 is what is normally used. The record counts how often the donor really diverged.</li>
<li><b>Threshold sweep.</b> The discovered circuit at \(\tau\) is every off-diagonal pair with
\(E[i,j] &gt; \tau\). Thresholds are placed between consecutive distinct effect sizes, so the sweep
walks from &ldquo;every edge with any effect&rdquo; down to &ldquo;nothing&rdquo;.</li>
<li><b>Scoring.</b> Each circuit is compared with the exhaustive true graph: precision, recall, F1,
plus an ROC curve and its area. The headline is the <b>best F1</b> over the sweep.</li>
<li><b>Sufficiency.</b> At the best \(\tau\), every source cell with no surviving outgoing edge is
resample-ablated <i>at the same time</i>, the tail is replayed, and we count the fraction of candidate
cells that still match the clean run (&ldquo;scrubbing-preserved performance&rdquo;).</li>
<li><b>Positive control.</b> Running the same keep-rule with the full oracle value set and \(\tau=0\)
reproduces the true graph exactly. This proves the scorer can reward a perfect discovery.</li>
</ol>
<p><b>Where we differ from the original.</b> Our ACDC is <b>single-shot</b>: one ablation per source,
then pure thresholding. It does not prune iteratively, so it skips the &ldquo;dangling node&rdquo; step
of the toy example. The metric is a byte difference, not a KL divergence. And an edge is a total
effect over 15 frames, because RAM cells interact through many CPU instructions, not through one
weight. These choices make it cheap and comparable with the oracle. They also explain a typical failure:
a single corrupted value can miss edges that only fire for other values, which costs recall.</p>""",

    code=[
        dict(path="tools/xai_study/phaseC_mechanistic/acdc.jl", find="function acdc_edge_effects(",
             role="xai", desc="The method: one resample-ablation and real-ROM re-run per source cell fills a row of \\(E[i,j]\\)."),
        dict(path="tools/xai_study/phaseC_mechanistic/acdc.jl", find="function resample_value(clean_val::Int, donor_val::Int)",
             role="xai", desc="Chooses the corrupted value: the donor's byte if it differs, else base+17."),
        dict(path="tools/xai_study/phaseC_mechanistic/acdc.jl", find="function discovered_at(E::Matrix{Float64}, tau::Float64)",
             role="xai", desc="The keep rule: edge \\(i\\to j\\) survives iff its effect exceeds \\(\\tau\\)."),
        dict(path="tools/xai_study/phaseC_mechanistic/acdc.jl", find="function tau_sweep(E::Matrix{Float64})",
             role="xai", desc="The threshold schedule: one \\(\\tau\\) between each pair of distinct effect sizes."),
        dict(path="tools/xai_study/phaseC_mechanistic/acdc.jl", find="function true_dataflow_graph(checkpoint, tail",
             role="xai", desc="The ground-truth circuit: exhaustive interventions {0, base+17, base+37} on each cell."),
        dict(path="tools/xai_study/phaseC_mechanistic/acdc.jl", find="function scrubbing_preserved(checkpoint, tail",
             role="xai", desc="Sufficiency check: scrub all non-circuit sources at once and count preserved cells."),
        dict(path="tools/xai_study/phaseC_mechanistic/acdc.jl", find="function run_game(spec::GameSpec",
             role="xai", desc="One game end to end: bit-exact check, checkpoint, truth, ACDC, sweep, positive control, triad."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function build_shared_testbed(",
             role="substrate", desc="The shared gameplay state every method sits on (seeded actions, \\(f^*\\)=90, 15-frame horizon)."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(env::StellaEnvironment",
             role="substrate", desc="One video frame of the bit-exact machine; every re-run in ACDC is a loop of these."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function run_intervention(",
             role="substrate", desc="The exact intervention oracle used to build the shared state and its cause-density gate."),
    ],

    reading=r"""
<p>The figure shows the threshold sweep: precision and recall of the discovered edges against the true
data-flow graph as \(\tau\) falls. A perfect method would hit the top-right corner at some \(\tau\);
the positive control does exactly that. Expect ACDC's curve to sit lower. Precision is usually decent,
because an edge with a visible effect under one corruption is usually real. Recall is the weak side:
an edge that only shows up for a different value, or that the program re-derives from another cell,
looks dead under a single resample and gets pruned. Read the scrubbing-preserved number beside it: a
circuit can score a good F1 and still fail to reproduce the behaviour on its own.</p>""",

    pitfalls=[
        "<b>One corruption is not all corruptions.</b> An edge whose effect depends on the value chosen can be invisible under the one resample used. The exhaustive oracle tries several values; ACDC tries one.",
        "<b>The threshold is a free parameter.</b> There is no correct \\(\\tau\\). Reporting only the best-F1 point uses the ground truth to pick it, which a real network user cannot do.",
        "<b>Total vs direct effects.</b> On the VCS an edge means &ldquo;influences after 15 frames&rdquo;. Indirect paths (\\(i\\to k\\to j\\)) appear as edges \\(i\\to j\\) too, in both the discovered and the true graph.",
        "<b>Minimal is not sufficient.</b> A small, clean-looking circuit can still fail when everything outside it is scrubbed. Always check sufficiency separately (see causal scrubbing).",
        "<b>Metric choice changes the circuit.</b> Miller et al. show that small changes in how ablations and metrics are defined change which circuit is found. Our byte-difference metric is one choice among many.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseC_mechanistic/acdc.jl --games core",
    try_note="Writes <code>acdc_&lt;game&gt;.{json,npz}</code> and <code>acdc_core_summary.json</code> into "
             "<code>tools/xai_study/phaseC_mechanistic/out/</code>. Add <code>--selftest</code> to run the "
             "positive control on the pilot game without writing anything.",

    next=["causal_scrubbing", "path_patching", "activation_patching", "A1_connectomics"],
)
