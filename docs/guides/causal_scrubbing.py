# Beginner's guide — causal scrubbing.
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="causal_scrubbing",
    level="Mechanistic family · testing a circuit hypothesis",

    wiki=[
        ("Statistical hypothesis test", "https://en.wikipedia.org/wiki/Statistical_hypothesis_test"),
        ("Resampling (statistics)", "https://en.wikipedia.org/wiki/Resampling_(statistics)"),
        ("Causal model", "https://en.wikipedia.org/wiki/Causal_model"),
        ("Ablation (artificial intelligence)", "https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence)"),
        ("Mechanistic interpretability", "https://en.wikipedia.org/wiki/Mechanistic_interpretability"),
    ],
    papers=[
        ("Chan, Garriga-Alonso, Goldowsky-Dill, Greenblatt, Nitishinskaya, Radhakrishnan, Shlegeris "
         "&amp; Thomas (2022). <i>Causal Scrubbing: a method for rigorously testing interpretability "
         "hypotheses.</i> Redwood Research / Alignment Forum.",
         "https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing"),
        ("Geiger, Ibeling, Zur, Chaudhary, Chauhan, Huang, Arora, Wu, Goodman, Potts &amp; Icard (2023). "
         "<i>Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability.</i> "
         "(follow-up: puts scrubbing, patching and DAS in one framework)",
         "https://arxiv.org/abs/2301.04709"),
        ("Miller, Chughtai &amp; Saunders (2024). <i>Transformer Circuit Faithfulness Metrics are not "
         "Robust.</i> (critique: ablation details change the verdict)",
         "https://arxiv.org/abs/2407.08734"),
    ],

    idea=r"""
<p>Most interpretability methods <i>produce</i> explanations. Causal scrubbing <i>checks</i> one. You
bring a <b>hypothesis</b>: &ldquo;for this behaviour, only these parts of the network matter, and they
compute this&rdquo;. Causal scrubbing turns that sentence into an experiment that the hypothesis can
fail.</p>
<p>The trick is a simple consequence of the claim. If a part really does not matter, then replacing its
activation with the activation it had on a <b>different, random input</b> should not change the
behaviour. So: take everything the hypothesis calls irrelevant, <b>resample</b> it from other inputs
(this is the &ldquo;scrubbing&rdquo;), run the model, and measure how much of the original performance
survives. If it all survives, the hypothesis passes. If performance collapses, something the hypothesis
called irrelevant was in fact doing work. It tests a whole explanation at once, not one component at a
time, which makes it one of the strictest checks in mechanistic interpretability.</p>""",

    nn=r"""
<p>A hypothesis has three parts: the model's computational graph \(G\), a simpler &ldquo;story&rdquo;
graph \(I\), and a map \(c\) saying which nodes of \(G\) implement which nodes of \(I\). Scrubbing
rewrites the forward pass on input \(x\). Every activation <b>not</b> claimed by the hypothesis is taken
from a random other input \(x'\). Every activation that <b>is</b> claimed may be taken from any input
\(x''\) that agrees with \(x\) on the feature the story says it computes. The score is the share of
the loss gap that survives:</p>
\[ \text{loss recovered} \;=\;
   \frac{\mathbb{E}[L_{\text{random}}] - \mathbb{E}[L_{\text{scrubbed}}]}
        {\mathbb{E}[L_{\text{random}}] - \mathbb{E}[L_{\text{model}}]} . \]
<p>1 means the hypothesis explains all of the behaviour; 0 means it explains none of it.
\(L_{\text{random}}\) is the loss when <i>everything</i> is resampled.</p>
<p><b>Toy example.</b> A model computes \(y = 3x_1 + 0\cdot x_2\). The clean input is
\(x=(1,4)\), so \(y=3\). A random other input is \(x'=(2,7)\).
<b>Hypothesis A</b>: &ldquo;only \(x_1\) matters&rdquo;. Scrub \(x_2\) to 7: \(y = 3\cdot 1 + 0
\cdot 7 = 3\). Unchanged, so A passes. <b>Hypothesis B</b>: &ldquo;only \(x_2\) matters&rdquo;.
Scrub \(x_1\) to 2: \(y = 6\). The output moved, so B fails. A real test resamples many \(x'\) and
averages, because one lucky resample can hide a dependence.</p>""",

    recipe=[
        "Write the hypothesis as a set of components (and, ideally, what each one computes).",
        "Pick a behaviour metric and a pool of other inputs to resample from.",
        "For each test input, replace every out-of-hypothesis activation with its value on a random other input; keep in-hypothesis activations consistent with the claim.",
        "Run the model and record the metric; repeat over many resamples.",
        "Report loss recovered. Also scrub a deliberately wrong hypothesis: a test that cannot fail a wrong story proves nothing.",
    ],

    atari_map=[
        ("Components of the network", "The candidate RAM cells of the game (for Pong: the two scores, paddle positions, ball position)."),
        ("The behaviour", "The value of <b>output cells</b> after the 15-frame horizon: cells that are driven by other cells but drive none themselves."),
        ("The hypothesis", "A set of cells claimed to carry the behaviour. The TRUE one is the outputs plus all their ancestors in the exact data-flow graph."),
        ("Resampling from another input", "Writing a different value into an out-of-circuit cell at \\(f^*\\): each value from the oracle set {0, base+17, base+37} except the current one."),
        ("Loss recovered", "Scrubbing-preserved performance: the share of (trial, output) pairs whose output byte equals the clean run."),
        ("A wrong hypothesis", "The true circuit with one demonstrably live ancestor swapped for an irrelevant cell, keeping the size the same."),
    ],

    atari=r"""
<p>On a network you must <i>guess</i> the hypothesis. On the VCS we can <b>compute</b> the true one and
then check that scrubbing rewards it and punishes a false one. Our runner
<code>causal_scrubbing.jl</code> does this per game:</p>
<ol>
<li><b>State.</b> The shared gameplay checkpoint at \(f^*\) (90 seeded random-action frames) and the
15-frame action tail.</li>
<li><b>True data-flow graph.</b> For each candidate cell \(i\), write 0, base+17 and base+37 in turn,
replay the tail, and mark an edge \(i\to j\) if any value changes cell \(j\). This is the same exact
graph the connectomics and ACDC pages are scored against.</li>
<li><b>Behaviour and hypotheses.</b> The outputs are the sinks of that graph that have at least one
ancestor. The <b>true</b> circuit is the outputs plus every cell with a path to them. The <b>wrong</b>
circuit drops one ancestor that is confirmed live (changing it alone changes an output) and adds an
outside cell instead.</li>
<li><b>Scrubbing.</b> For each hypothesis, every cell outside the circuit is overwritten with a resample
value, while circuit cells keep their clean value. The trial set walks through every resample value of
every outside cell (the others held at their first resample value), plus one trial with all outside
cells at their first resample value. After each trial the
tail is replayed and the output cells are compared with the clean run.</li>
<li><b>Verdict.</b> A hypothesis <b>passes</b> only if every output is preserved in every trial (score
exactly 1). The run also reports whether the true circuit scores higher than the wrong one, and a graded
similarity over all candidate cells and over the whole screen.</li>
</ol>
<p><b>Where we differ from the original.</b> The resample values are not activations from other real
inputs. They come from the oracle's fixed value set, the same set used to build the true graph. The
hypothesis is also only a <i>set of cells</i>: it makes no claim about what each cell computes, so
there is no &ldquo;consistent&rdquo; resampling of in-circuit cells. Because the true circuit is
built from the same interventions, its pass is close to guaranteed; it can still fail if several
scrubbed cells interact. The informative half of the experiment is the <b>wrong</b> hypothesis: it must
fail, and it shows that the test can tell a right story from a wrong one.</p>""",

    code=[
        dict(path="tools/xai_study/phaseC_mechanistic/causal_scrubbing.jl", find="function scrub_assign(base_ckpt, tail",
             role="xai", desc="The scrub: overwrite every out-of-circuit cell, keep circuit cells, replay the tail."),
        dict(path="tools/xai_study/phaseC_mechanistic/causal_scrubbing.jl", find="function resample_values(at_target::Snapshot",
             role="xai", desc="The resample distribution per cell: {0, base+17, base+37} minus the current value."),
        dict(path="tools/xai_study/phaseC_mechanistic/causal_scrubbing.jl", find="function true_dataflow_graph(base_ckpt, tail",
             role="xai", desc="The exact data-flow graph the true hypothesis is derived from."),
        dict(path="tools/xai_study/phaseC_mechanistic/causal_scrubbing.jl", find="function ancestors(A::BitMatrix, out::Int)",
             role="xai", desc="Transitive closure: every cell with a path to an output joins the true circuit."),
        dict(path="tools/xai_study/phaseC_mechanistic/causal_scrubbing.jl", find="function pick_outputs(A::BitMatrix)",
             role="xai", desc="Chooses the behaviour: sink cells of the true graph that have at least one ancestor."),
        dict(path="tools/xai_study/phaseC_mechanistic/causal_scrubbing.jl", find="function _run_game_body(",
             role="xai", desc="Builds true and wrong hypotheses, runs every scrub trial, computes pass/fail and the triad."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function build_shared_testbed(",
             role="substrate", desc="The shared gameplay checkpoint, action tail and clean continuation."),
        dict(path="tools/xai_study/common/jutari_oracle.jl", find="function intervene_ram!(env::StellaEnvironment",
             role="substrate", desc="Writes one byte into emulated RAM: the physical act of scrubbing a cell."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(env::StellaEnvironment",
             role="substrate", desc="One bit-exact video frame; each scrub trial replays 15 of them."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function run_intervention(",
             role="substrate", desc="The exact intervention oracle behind the shared state's cause-density gate."),
    ],

    reading=r"""
<p>The figure shows two preservation maps for Pong (output cell &times; scrub trial): one for the true
hypothesis and one for the wrong one. Expect the true map fully preserved and the wrong map broken in
some trials, and the same split game after game in the scores. That pattern
means two things. The test is <b>sensitive</b>: it catches a story that wrongly calls a live cell
irrelevant. And it is <b>exact</b> here, because each resample is a real intervention on a fully
observable machine, not an approximation. What a pass does <i>not</i> tell you is how the circuit
computes the behaviour. It confirms the wiring, not the algorithm.</p>""",

    pitfalls=[
        "<b>A passing test needs a failing control.</b> If the resampling is too weak (for example, the pool of other inputs is too similar), every hypothesis passes. Always scrub a wrong hypothesis too.",
        "<b>Over-large hypotheses pass trivially.</b> &ldquo;Everything matters&rdquo; never fails a scrub. Pair the score with a measure of how small the hypothesis is.",
        "<b>Resample distribution matters.</b> Chan et al. resample from real inputs; we use a fixed oracle value set. Different distributions can expose different dependences, and Miller et al. show such details can flip verdicts.",
        "<b>Wiring is not meaning.</b> A set-of-cells hypothesis that passes says <i>where</i> the computation happens, not <i>what</i> it is. Testing a semantic claim needs a richer hypothesis (see DAS).",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseC_mechanistic/causal_scrubbing.jl --games core",
    try_note="Writes <code>causal_scrubbing_&lt;game&gt;.{json,npz}</code> and "
             "<code>causal_scrubbing_core_summary.json</code> into "
             "<code>tools/xai_study/phaseC_mechanistic/out/</code>. <code>--selftest</code> runs a quick check on Pong.",

    next=["acdc", "das", "path_patching", "activation_patching"],
)
