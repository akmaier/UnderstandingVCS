# Beginner's guide — A1 connectomics / data-flow graph recovery.
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="A1_connectomics",
    level="Phase A · neuroscience battery · wiring from perturbation",

    wiki=[
        ("Connectomics", "https://en.wikipedia.org/wiki/Connectomics"),
        ("Connectome", "https://en.wikipedia.org/wiki/Connectome"),
        ("Directed graph", "https://en.wikipedia.org/wiki/Directed_graph"),
        ("Data-flow analysis", "https://en.wikipedia.org/wiki/Data-flow_analysis"),
        ("F-score (precision, recall, F1)", "https://en.wikipedia.org/wiki/F-score"),
    ],
    papers=[
        ("Jonas &amp; Kording (2017). <i>Could a Neuroscientist Understand a Microprocessor?</i> "
         "PLOS Computational Biology 13(1): e1005268.",
         "https://doi.org/10.1371/journal.pcbi.1005268"),
        ("Sporns, Tononi &amp; K&ouml;tter (2005). <i>The Human Connectome: A Structural Description "
         "of the Human Brain.</i> PLOS Computational Biology 1(4): e42.",
         "https://doi.org/10.1371/journal.pcbi.0010042"),
        ("Bargmann &amp; Marder (2013). <i>From the connectome to brain function.</i> Nature Methods "
         "10: 483&ndash;490. (Why a wiring diagram alone does not explain behaviour.)",
         "https://doi.org/10.1038/nmeth.2451"),
    ],

    idea=r"""
<p>A <b>connectome</b> is a wiring diagram: a list of which parts of a system talk to which other
parts. Neuroscientists build it for brains. They draw one dot per neuron and one arrow per
connection. The hope is simple: <i>if we know who is wired to whom, we can read off how the system
computes.</i> One way to find the arrows is to look at the anatomy. Another is to <b>perturb</b>:
nudge one unit, watch which other units react, and draw an arrow from the nudged unit to every unit
that moved. The result is a <b>directed graph</b> of influence.</p>
<p>For a neural network you already have the anatomy: the weight matrices <i>are</i> the
connectome, and they are fully known. Yet nobody claims that printing all the weights explains the
network. That is the lesson of Jonas &amp; Kording, who mapped a whole microprocessor and still could
not say how it ran a game. On the Atari we ask a sharper question. If we recover the wiring by
cheap perturbations, the way it is done in practice, <b>how many of the arrows do we get right?</b>
Because the machine can be re-run exactly, we can count every right and every wrong arrow.</p>""",

    nn=r"""
<p><b>Neuroscience origin.</b> The first complete connectome was that of the worm
<i>C.&nbsp;elegans</i>: 302 neurons, about 7000 synapses, mapped by electron microscopy in the 1980s.
Its behaviour is still not fully explained by that map. Perturbation-based mapping (stimulate one
cell, record the others) gives the <i>effective</i> connectivity instead of the anatomical one.</p>
<p><b>Neural-network analogue.</b> In a network the anatomical connectome is the set of non-zero
weights. The perturbation version is what interpretability calls <i>circuit discovery</i>: change
one activation \(h_i\), see which downstream activations \(h_j\) change. Write the intervention as
\(\mathrm{do}(h_i := v)\) (&ldquo;force \(h_i\) to the value \(v\), leave everything else to the
network&rdquo;). The recovered edge set is</p>
\[ E_{\text{rec}} = \{\, i\to j \;:\; h_j^{(L)}\big(\mathrm{do}(h_i := h_i+\delta)\big) \neq h_j^{(L)} \,\}, \]
<p>where \(L\) is the latency (how many steps later you look) and \(\delta\) is the size of the
nudge. Given a true edge set \(E_{\text{true}}\) you score it like any detector:</p>
\[ P = \frac{|E_{\text{rec}}\cap E_{\text{true}}|}{|E_{\text{rec}}|},\quad
   R = \frac{|E_{\text{rec}}\cap E_{\text{true}}|}{|E_{\text{true}}|},\quad
   F_1 = \frac{2PR}{P+R}. \]
<p><b>Toy example.</b> Three units in a chain, each copying the previous one with a one-step
delay: \(a \to b \to c\). Nudge \(a\) and look one step later: only \(b\) moved. Nudge \(b\): only
\(c\) moved. So \(E_{\text{rec}} = \{a\to b,\; b\to c\}\). Now look many steps later instead:
nudging \(a\) moves \(b\) <i>and</i> \(c\). The long-horizon truth is
\(E_{\text{true}} = \{a\to b,\; b\to c,\; a\to c\}\). Then \(P = 2/2 = 1\), \(R = 2/3\) and
\(F_1 = 0.8\). Make it harder: let \(b\) copy \(a\) only when \(a &gt; 100\). With \(a = 50\) and a
nudge of \(+17\), the gate never opens and even \(a\to b\) is missed. A single small nudge with a
short wait sees only part of the wiring.</p>""",

    recipe=[
        r"Choose the units (here: candidate RAM bytes) and freeze the system in one state.",
        r"Record a baseline: let the system run unperturbed for \(L\) steps and store every unit's value.",
        r"For each unit \(i\): restore the frozen state, apply one perturbation \(\mathrm{do}(i := \text{base}_i+\delta)\), run \(L\) steps.",
        r"Draw an arrow \(i\to j\) for every unit \(j\) whose value now differs from the baseline. Ignore the self-arrow \(i\to i\).",
        r"Build the reference graph the same way but exhaustively (several values, full horizon), or take it from the known anatomy.",
        r"Compare the two edge sets: precision, recall, \(F_1\), and the graph-edit distance (number of wrong plus missing arrows).",
    ],

    atari_map=[
        ("The neuron", "One <b>candidate RAM byte</b>. The VCS has 128 bytes of working memory "
         "(RAM). Each game's candidate list (e.g. ball&nbsp;x, paddle&nbsp;y, score) comes from "
         "published RAM annotations (OCAtari / AtariARI)."),
        ("The synapse", "A <b>data-flow edge</b> \\(i\\to j\\): the game program (the ROM, the "
         "read-only cartridge code) reads byte \\(i\\) and, directly or through other bytes, "
         "writes a value into byte \\(j\\)."),
        ("The stimulation electrode", "<code>intervene_ram!</code>: overwrite one RAM byte in the "
         "emulator at frame \\(f^*\\). A <i>frame</i> is one 1/60&nbsp;s video picture, the unit "
         "of game time."),
        ("The recording", "A full snapshot of RAM (and screen) after re-running the bit-exact "
         "emulator <code>jutari</code> from that frozen state."),
        ("The true connectome", "The <b>oracle</b> graph: the same probe done exhaustively (three "
         "values, full 15-frame horizon). Each probe is a real re-run of the real ROM, so every "
         "edge in it is a true causal dependency."),
    ],

    atari=r"""
<p>The runner <code>A1_connectomics.jl</code> does the whole experiment on one frozen game state.
That state is shared with every other method in the study:</p>
<ol>
<li><b>The state.</b> The emulator boots the cartridge (60 no-op frames plus 4 reset frames, the
same boot as the C++ reference), then plays 90 frames of <b>seeded random joystick input</b>
(18 possible actions, seed&nbsp;0). That frame is \(f^*\). The 15 frames after it are the
<i>horizon</i>. A <i>cause-density gate</i> checks with the oracle that at least four candidate
causes really move the screen and records whether the state passed, so a frozen title screen
is flagged. Two fresh boots must give
byte-identical RAM and screen, or the runner refuses to score.</li>
<li><b>The recovered graph (method under test).</b> For each candidate byte \(i\) the runner copies
the frozen machine, adds 17 to byte \(i\) (wrapping at 256), steps <b>one frame</b>, and draws
\(i\to j\) for every candidate byte \(j\) that differs from an unperturbed one-frame run. One value,
one frame: a thin, realistic probe.</li>
<li><b>The true graph (oracle).</b> The same loop, but byte \(i\) is set to <b>each</b> of
\(0\), base\(+17\) and base\(+37\), and the machine runs the <b>full 15 frames</b>. An edge exists if
<i>any</i> of the three values changes \(j\) at the end. Note what this truth is: a measured
<i>total-effect</i> graph over the horizon. It includes indirect paths (\(a\to b\to c\) also counts as
\(a\to c\)), exactly like the toy example. It is measured by re-running the ROM, not parsed from the
disassembly listing.</li>
<li><b>Scores.</b> Precision, recall, \(F_1\) and graph-edit distance over the off-diagonal edges.
A positive control runs the oracle procedure as if it were the method; it must score a perfect
\(F_1\), which proves the scorer can reward a correct graph.</li>
<li><b>Generalisation probe.</b> The runner also builds a true graph from base\(+37\) alone, a value
the recovery never used, and checks how well the recovered graph predicts it (edge agreement). A
minimality term divides the number of true edges by the number of claimed edges.</li>
</ol>
<p>Why the gap? The VCS program updates many bytes only on certain frames, behind comparisons and
timers. A one-frame wait misses every influence that takes longer to arrive. A single nudge of
\(+17\) misses every influence that only fires for other values. And an effect that shows up after
one frame but is undone by frame 15 becomes an arrow the truth does not have. The method sees a
small, partly wrong slice of a richly connected machine.</p>""",

    code=[
        dict(path="tools/xai_study/phaseA_kording/A1_connectomics.jl",
             find="function recovered_dataflow_graph(",
             role="xai", desc="The method under test: one <code>do(i := base+17)</code> per cell, one-frame latency, edge if another cell differs."),
        dict(path="tools/xai_study/phaseA_kording/A1_connectomics.jl",
             find="function true_dataflow_graph(",
             role="xai", desc="The ground-truth graph: exhaustive do-set {0, +17, +37}, full horizon, edge if <i>any</i> value moves the cell."),
        dict(path="tools/xai_study/phaseA_kording/A1_connectomics.jl",
             find="function intervene_continue(",
             role="xai", desc="One probe: deep-copy the frozen machine, poke one RAM byte, step the tail, snapshot."),
        dict(path="tools/xai_study/phaseA_kording/A1_connectomics.jl",
             find="function score_graph(",
             role="xai", desc="Precision, recall, F1 and graph-edit distance over off-diagonal edges."),
        dict(path="tools/xai_study/phaseA_kording/A1_connectomics.jl",
             find="function heldout_true_graph(",
             role="xai", desc="The held-out base+37 graph used to test whether the recovery generalises."),
        dict(path="tools/xai_study/phaseA_kording/A1_connectomics.jl",
             find="function run_game(",
             role="xai", desc="One game end to end: shared state, bit-exact check, both graphs, positive control, triad."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl",
             find="function shared_gameplay_actions(",
             role="substrate", desc="The seeded random joystick stream (90 + 15 frames) every method shares."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl",
             find="function build_shared_testbed(",
             role="substrate", desc="Builds the shared state and runs the oracle cause-density gate."),
        dict(path="tools/xai_study/common/jutari_oracle.jl",
             find="function intervene_ram!(",
             role="substrate", desc="The &ldquo;electrode&rdquo;: write one byte into the console's RAM (the A1 runner carries an identical local copy)."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(",
             role="substrate", desc="One video frame of the bit-exact emulated VCS: CPU, TIA video chip and RIOT timer."),
    ],

    reading=r"""
<p>The figure shows the game frame and two square matrices over the same candidate bytes. Rows are
the <i>cause</i> byte, columns the <i>effect</i> byte, and a bright square is an arrow. The left
matrix is the oracle's true graph, the right one the recovered graph. The diagonal (a poked byte
changing itself, unless the program overwrites it) is not scored. Read the error as the squares
that are bright in one matrix but dark in the other. In Pong the true graph has whole bright rows,
because the score bytes and the ball bytes reach many other cells within 15 frames. The one-frame
recovery is empty off the diagonal. It misses the wiring rather than inventing it, so recall
collapses and \(F_1\) lands at the bottom of the battery.</p>""",

    pitfalls=[
        "<b>Latency decides the answer.</b> A one-step readout finds only direct, fast edges. A long readout finds every indirect path. Neither is &ldquo;the&rdquo; connectome; say which one you mean.",
        "<b>One perturbation value is not enough.</b> Programs (and ReLU networks) are full of thresholds. A nudge that does not cross the threshold reports no edge, even though a different value would.",
        "<b>Transient effects create false arrows.</b> A change that appears one frame later but is overwritten before the horizon counts as an edge for the method and not for the truth.",
        "<b>Wiring is not function.</b> Even a perfect graph does not say <i>what</i> is computed. The worm's complete connectome did not explain its behaviour, and our oracle graph does not explain Pong.",
        "<b>Our truth is a total-effect graph, measured at one state.</b> It is not the static read/write graph of the whole program. Another game state can light up other edges.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseA_kording/A1_connectomics.jl --games core",
    try_note="Writes <code>A1_&lt;game&gt;.{json,npz}</code> per game plus the combined "
             "<code>A1_connectomics.{json,npz}</code> into <code>tools/xai_study/phaseA_kording/out/</code>. "
             "Use <code>--game pong</code> for one game, or <code>--selftest</code> to run the checks "
             "without writing. Each game costs a few hundred short emulator re-runs (minutes on one core).",

    next=["A2_lesions", "A6_granger", "A4_correlations", "acdc"],
)
