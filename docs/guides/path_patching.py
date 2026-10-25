# Beginner's guide — path patching / IOI-style circuit recovery.
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="path_patching",
    level="Mechanistic family · patching the wiring, not the nodes",

    wiki=[
        ("Mechanistic interpretability", "https://en.wikipedia.org/wiki/Mechanistic_interpretability"),
        ("Mediation (statistics)", "https://en.wikipedia.org/wiki/Mediation_(statistics)"),
        ("Directed graph", "https://en.wikipedia.org/wiki/Directed_graph"),
        ("Precision and recall", "https://en.wikipedia.org/wiki/Precision_and_recall"),
    ],
    papers=[
        ("Wang, Variengien, Conmy, Shlegeris &amp; Steinhardt (2022). <i>Interpretability in the Wild: a "
         "Circuit for Indirect Object Identification in GPT-2 small.</i> ICLR 2023.",
         "https://arxiv.org/abs/2211.00593"),
        ("Goldowsky-Dill, MacLeod, Sato &amp; Arora (2023). <i>Localizing Model Behavior with Path "
         "Patching.</i>",
         "https://arxiv.org/abs/2304.05969"),
        ("Miller, Chughtai &amp; Saunders (2024). <i>Transformer Circuit Faithfulness Metrics are not "
         "Robust.</i> (Shows that circuit scores depend heavily on ablation choices.)",
         "https://arxiv.org/abs/2407.08734"),
    ],

    idea=r"""
<p>Activation patching tells you that a component matters. It does not tell you <b>how</b> its
influence reaches the output. Component A might act on the output directly, or it might only matter
because it changes component B, which then changes the output. A <b>circuit</b> is the answer to that
question: the small set of components and the <b>edges</b> (connections) between them that carry out a
behaviour. To find edges you need a sharper tool.</p>
<p><b>Path patching</b> is that tool. You change a <b>sender</b> component, but you let the change reach
only one chosen <b>receiver</b>. Every other component downstream of the sender is held at the value it
had in the normal run. If the output still moves, the signal must have travelled along the
sender&rarr;receiver edge. This is how Wang et al. mapped the <b>indirect-object-identification
(IOI)</b> circuit in GPT-2: which attention heads send which information to which other heads to
complete &ldquo;When Mary and John went to the store, John gave a drink to&hellip;&rdquo; with
&ldquo;Mary&rdquo;. On the VCS the true wiring between memory cells can be measured exactly, so we can
check how much of it a path patch actually recovers.</p>""",

    nn=r"""
<p>Think of the network as a directed graph of components. A transformer&rsquo;s components all read
from and write to the <b>residual stream</b>, so every earlier head can influence every later one.
Pick a sender \(s\) and a receiver \(r\). Path patching runs the clean input \(x\) with three rules:
\(s\) takes its value from the corrupted input \(\tilde x\); the receiver \(r\) is recomputed
normally, so it sees the corrupted \(s\); and every other component that \(s\) could reach is
<b>frozen</b> to its clean value. Then</p>
\[ \Delta_{s \to r} \;=\; y\big(x \,\big|\, s := s(\tilde x),\; \text{others frozen},\; r \text{ recomputed}\big) \;-\; y(x). \]
<p>A non-zero \(\Delta_{s\to r}\) means the edge \(s \to r\) carries part of the behaviour. Repeating
this over pairs, working backwards from the output, builds the circuit. The result is scored as a set
of edges.</p>
<p><b>Toy example.</b> Let \(b = 2a\), \(c = a + b\), and output \(y = c\). Clean \(a = 1\): \(b = 2\),
\(c = 3\). Corrupt \(a := 2\). Plain activation patching of \(a\) gives \(b = 4\), \(c = 6\), so
\(\Delta_a = 3\). Now path-patch the edge \(a \to b\): let \(b\) see \(a = 2\) (so \(b = 4\)), but
freeze \(a\)&rsquo;s direct input to \(c\) at its clean value 1. Then \(c = 1 + 4 = 5\) and
\(\Delta_{a\to b} = 2\). Path-patch the direct edge \(a \to c\) instead: \(b\) stays at 2, \(c = 2 + 2
= 4\), \(\Delta_{a\to c} = 1\). The total effect of 3 splits into an indirect part (2, via \(b\)) and a
direct part (1). That split is what node-level patching cannot give you.</p>""",

    recipe=[
        "Choose clean and corrupted inputs and a metric, as for activation patching.",
        "Pick a sender component \\(s\\) and a receiver \\(r\\).",
        "Run the clean input with \\(s\\) set to its corrupted value; freeze every other component downstream of \\(s\\) at its clean value; let only \\(r\\) recompute.",
        "Run the rest of the network and record \\(\\Delta_{s\\to r}\\).",
        "Repeat over sender/receiver pairs, starting from the output and moving backwards; keep the edges with large effects.",
        "Score the recovered edge set (precision, recall, F1) against a reference, or test it with ablations.",
    ],

    atari_map=[
        ("A component (head)", "A candidate RAM cell: a byte the game uses as a variable."),
        ("Layers", "Video frames. The sender lives at frame \\(f^*\\), the receiver at frame \\(f^*{+}1\\)."),
        ("Sender corruption", "Write <code>base+17</code> into the sender cell at \\(f^*\\)."),
        ("Freeze all other components", "After one frame, overwrite every other candidate cell with its "
         "clean value at \\(f^*{+}1\\); only the receiver keeps the sender&rsquo;s signal."),
        ("Metric", "Does the receiver cell&rsquo;s own value differ from the clean run at the end of the "
         "15-frame horizon?"),
        ("Reference circuit", "The <b>true routine</b>: the same path patch with an exhaustive set of "
         "sender values (0, <code>base+17</code>, <code>base+37</code>). An edge is true if any value "
         "fires it."),
    ],

    atari=r"""
<p>The nodes are the game&rsquo;s candidate RAM cells. Everything starts from the shared gameplay
checkpoint at \(f^*\) with a 15-frame horizon. The first frame of the horizon is the
sender&rarr;receiver hop, using the gameplay stream&rsquo;s own next action.</p>
<ol>
<li><b>One path patch</b> \(i \to j\): copy the checkpoint, write the sender value into cell \(i\),
step one frame, then restore <i>every</i> candidate cell except \(j\) to its clean frame-\(f^*{+}1\)
value. Run the remaining 14 frames. The edge fires if cell \(j\) ends up different from the clean
run.</li>
<li><b>Recovered circuit.</b> One sender value per edge (<code>base+17</code>), the classical
single-shot path patch, over all ordered pairs of cells.</li>
<li><b>True routine.</b> The same edge-restricted operation, but with three sender values. This is the
faithful <b>direct</b> circuit the method targets. A separate transitive graph (perturb a cell, run the
full horizon, see which cells change) is recorded for context only.</li>
<li><b>Score.</b> Edge precision, recall and F1 of the recovered circuit against the true routine,
with self-edges excluded.</li>
<li><b>Positive control.</b> Re-deriving the circuit with the exhaustive value set must give F1 = 1,
proving the scorer rewards a faithful circuit.</li>
</ol>
<p>Two differences from the transformer original matter. First, our &ldquo;layer&rdquo; is a whole
frame of thousands of CPU instructions, and a cell can be read and rewritten many times inside it.
Second, we freeze only the <i>candidate</i> cells, not every byte, CPU register or TIA register. A
signal can also leak through those unfrozen parts. The runner also counts <b>transient receivers</b>:
cells the program recomputes from scratch, so no single-frame path into them can survive to the end of
the horizon.</p>""",

    code=[
        dict(path="tools/xai_study/phaseC_mechanistic/path_patching.jl", find="function path_patch(",
             role="xai", desc="One path patch: corrupt the sender, step one frame, freeze every other candidate cell, run the rest."),
        dict(path="tools/xai_study/phaseC_mechanistic/path_patching.jl", find="function recovered_path_circuit(",
             role="xai", desc="The method: single-shot path patch with one sender value per edge."),
        dict(path="tools/xai_study/phaseC_mechanistic/path_patching.jl", find="function true_routine_graph(",
             role="xai", desc="The reference: the same edge-restricted patch with the exhaustive value set."),
        dict(path="tools/xai_study/phaseC_mechanistic/path_patching.jl", find="function node_dataflow_graph(",
             role="xai", desc="The transitive (multi-hop) data-flow graph, recorded for context."),
        dict(path="tools/xai_study/phaseC_mechanistic/path_patching.jl", find="function score_circuit(",
             role="xai", desc="Circuit precision, recall and F1 over off-diagonal edges."),
        dict(path="tools/xai_study/phaseC_mechanistic/path_patching.jl", find="hop_action = Int(tail[1])",
             role="xai", desc="The sender&rarr;receiver hop uses the gameplay stream&rsquo;s next action."),
        dict(path="tools/xai_study/common/jutari_oracle.jl", find="function intervene_ram!(",
             role="substrate", desc="The byte write used for both the sender corruption and the freeze-restore."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function build_shared_testbed(",
             role="substrate", desc="The shared gameplay checkpoint and action stream."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(",
             role="substrate", desc="One bit-exact frame: the one-frame hop and the remaining horizon."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function candidate_ram_indices(",
             role="substrate", desc="Reads the candidate cells that become the circuit&rsquo;s nodes."),
    ],

    reading=r"""
<p>Each game&rsquo;s figure compares two adjacency matrices: the recovered edges and the true routine.
Expect low recall on most games. A single sender value often fails to trip an edge that another value
would. Many cells are also rebuilt by the program within the frame through other cells, so the
one-step path carries no surviving signal. On games where the direct circuit is genuinely empty the
method trivially matches it. On others, including the Pong state shown here, it can recover almost
nothing. The positive control shows this is a real limit of the single-shot method, not a scoring
bug.</p>""",

    pitfalls=[
        "<b>Which nodes you freeze defines the answer.</b> Freezing too few lets the signal leak around the chosen path; freezing too many can break the computation. On the VCS we freeze only candidate cells.",
        "<b>One corruption value is not enough.</b> An edge can carry a signal only for some values (a branch that triggers above a threshold). Single-shot recall is limited by that.",
        "<b>Coarse layers.</b> A VCS frame contains many reads and writes of the same cell. A one-frame hop merges several real edges into one, or hides them.",
        "<b>Metric fragility.</b> Miller et al. show that circuit faithfulness scores in transformers change a lot with the ablation type and the metric. Report your choices.",
        "<b>Cost grows with pairs.</b> One re-run per sender&ndash;receiver pair and value: quadratic in the number of nodes. This is why ACDC and edge attribution patching exist.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseC_mechanistic/path_patching.jl --games core",
    try_note="Writes <code>path_patching_&lt;game&gt;.json/.npz</code> and a core summary into "
             "<code>tools/xai_study/phaseC_mechanistic/out/</code>. The exhaustive reference needs up to "
             "three re-runs per ordered cell pair, so this is the slowest of the patching runners; it can "
             "be sharded per game on the cluster (<code>--shard i --nshards n</code>).",

    next=["activation_patching", "acdc", "attribution_patching", "A1_connectomics"],
)
